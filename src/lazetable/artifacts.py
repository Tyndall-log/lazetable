"""Artifact metadata, storage, and payload-lowering helpers."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .core import (
    ArtifactId,
    LazyTable,
    SandboxRegistry,
    SandboxRoot,
    infer_resolver_context,
    make_artifact_ref,
    make_source_ref,
)


Scalar = str | int | float | bool | None
INTERNAL_SOURCE_ROW_IDX = "__lz_source_row_idx__"
LAZETABLE_STORAGE_FORMAT_VERSION = 1


class ArtifactKeyError(ValueError):
    """Base error for artifact unique-key validation failures."""


class MissingArtifactKeyFieldError(ArtifactKeyError):
    """Raised when a declared unique-key field is missing from an output row."""


class DuplicateArtifactKeyError(ArtifactKeyError):
    """Raised when output rows violate declared artifact uniqueness."""


@dataclass(frozen=True)
class ArtifactJoin:
    child_on: tuple[str, ...]
    parent_on: tuple[str, ...]


@dataclass(frozen=True)
class ArtifactDependency:
    artifact_name: str
    artifact_version: str = "latest"
    join: ArtifactJoin | None = None


@dataclass
class Payload:
    payload_type: "PayloadType"
    data: Any
    metadata: dict[str, Any] | None = None


@dataclass
class ExternalPayloadRef:
    """Lazy field declaration for an already-existing external payload."""

    payload_type: "PayloadType"
    ref: str
    metadata: dict[str, Any] | None = None


@dataclass
class SourcePayloadRef:
    """Lazy field declaration backed by a registered sandbox root."""

    payload_type: "PayloadType"
    sandbox_root: SandboxRoot
    relative_path: str
    metadata: dict[str, Any] | None = None


class PayloadType(Protocol):
    payload_id: str
    default_extension: str

    def dump_to_path(
        self,
        data: Any,
        path: Path,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Scalar]: ...

    def load_from_path(
        self,
        path: Path,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> Any: ...

    def payload(self, data: Any, **kwargs: Any) -> Payload: ...

    def external_ref(self, ref: str, **kwargs: Any) -> ExternalPayloadRef: ...

    def source_path(
        self, sandbox_root: SandboxRoot, relative_path: str, **kwargs: Any
    ) -> SourcePayloadRef: ...


class PayloadTypeBase:
    """Shared convenience base class for PayloadType implementations."""

    payload_id: str
    default_extension: str

    def payload(self, data: Any, **kwargs: Any) -> Payload:
        return Payload(self, data, metadata=kwargs or None)

    def external_ref(self, ref: str, **kwargs: Any) -> ExternalPayloadRef:
        return ExternalPayloadRef(self, ref=ref, metadata=kwargs or None)

    def source_path(
        self, sandbox_root: SandboxRoot, relative_path: str, **kwargs: Any
    ) -> SourcePayloadRef:
        return SourcePayloadRef(
            self,
            sandbox_root=sandbox_root,
            relative_path=relative_path,
            metadata=kwargs or None,
        )


@dataclass(frozen=True)
class ArtifactStorePaths:
    artifact_roots: dict[str, Path]


class ArtifactStore:
    def __init__(self, paths: ArtifactStorePaths):
        self.paths = paths

    def artifact_dir(self, root_name: str, artifact_name: str, artifact_version: str) -> Path:
        return self.paths.artifact_roots[root_name] / artifact_name / artifact_version

    def artifact_root_map(self) -> dict[str, str]:
        return {name: str(path) for name, path in self.paths.artifact_roots.items()}

    def open(
        self,
        artifact_name: str,
        artifact_version: str = "latest",
        *,
        root_name: str = "default",
        join_parents: bool = False,
        sandbox_registry: SandboxRegistry | None = None,
    ) -> LazyTable:
        root = self.paths.artifact_roots[root_name]
        if artifact_version == "latest":
            base = root / artifact_name
            candidates = [path for path in base.iterdir() if path.is_dir()] if base.exists() else []
            if not candidates:
                raise FileNotFoundError(f"artifact not found: {artifact_name}@latest")
            artifact_dir = max(candidates, key=lambda path: path.stat().st_mtime)
            artifact_version = artifact_dir.name
        else:
            artifact_dir = root / artifact_name / artifact_version
        return open_artifact(
            artifact_dir,
            store=self,
            join_parents=join_parents,
            sandbox_registry=sandbox_registry,
        )


def open_artifact(
    path: str | Path,
    *,
    store: ArtifactStore | None = None,
    join_parents: bool = False,
    sandbox_registry: SandboxRegistry | None = None,
) -> LazyTable:
    artifact_dir = Path(path)
    manifest_path = artifact_dir / "manifest.parquet"
    meta_path = artifact_dir / "meta.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"meta not found: {meta_path}")
    manifest = pq.read_table(manifest_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    artifact_id = ArtifactId(
        name=str(meta.get("artifact_name") or artifact_dir.parent.name),
        version=str(meta.get("artifact_version") or artifact_dir.name),
    )
    table = _table_from_artifact_parts(
        table=manifest,
        artifact_id=artifact_id,
        path=artifact_dir,
        meta=meta,
        sandbox_registry=sandbox_registry,
    )
    if not join_parents:
        return table
    return join_parent_tables(table, store=store, sandbox_registry=sandbox_registry)


@dataclass
class ArtifactWriteResult:
    table: LazyTable
    lowered_rows: list[dict[str, Any]]


@dataclass
class ArtifactWriter:
    artifact_name: str
    artifact_version: str
    out_dir: Path
    artifact_roots: dict[str, str] = field(default_factory=dict)
    unique_key_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.payload_dir.mkdir(parents=True, exist_ok=True)

    @property
    def artifact_id(self) -> ArtifactId:
        return ArtifactId(self.artifact_name, self.artifact_version)

    @property
    def payload_dir(self) -> Path:
        return self.out_dir / "payload"

    def write_payload_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        row_written_callback: Any | None = None,
    ) -> list[dict[str, Any]]:
        lowered_rows: list[dict[str, Any]] = []
        seen_keys: set[tuple[Any, ...]] = set()
        for row_idx, row in enumerate(rows):
            if self.unique_key_fields:
                key = self._extract_unique_key(row)
                if key in seen_keys:
                    raise DuplicateArtifactKeyError(
                        "duplicate artifact unique key detected: "
                        f"fields={self.unique_key_fields}, value={key}"
                    )
                seen_keys.add(key)
            lowered_row, source_row_idx = self._lower_row(row, row_idx=row_idx)
            lowered_rows.append(lowered_row)
            if row_written_callback is not None and source_row_idx is not None:
                row_written_callback(int(source_row_idx))
        return lowered_rows

    def _extract_unique_key(self, row: dict[str, Any]) -> tuple[Any, ...]:
        values: list[Any] = []
        for field_name in self.unique_key_fields:
            if field_name not in row:
                raise MissingArtifactKeyFieldError(
                    "artifact unique key field missing from output row: "
                    f"{field_name!r} (declared key={self.unique_key_fields})"
                )
            values.append(row[field_name])
        return tuple(values)

    def extract_unique_key(self, row: dict[str, Any]) -> tuple[Any, ...]:
        return self._extract_unique_key(row)

    def lower_row(self, row: dict[str, Any], *, row_idx: int) -> dict[str, Any]:
        lowered, _ = self._lower_row(row, row_idx=row_idx)
        return lowered

    def _lower_row(self, row: dict[str, Any], *, row_idx: int) -> tuple[dict[str, Any], int | None]:
        row_id = make_row_id(row, row_idx=row_idx, unique_key_fields=self.unique_key_fields)
        lowered: dict[str, Any] = {}
        source_row_idx = row.get(INTERNAL_SOURCE_ROW_IDX)
        for key, value in row.items():
            if key == INTERNAL_SOURCE_ROW_IDX:
                continue
            if isinstance(value, Payload):
                lowered.update(self._lower_payload_field(key, value, row_id=row_id))
            elif isinstance(value, ExternalPayloadRef):
                lowered.update(self._lower_external_payload_field(key, value))
            elif isinstance(value, SourcePayloadRef):
                lowered.update(self._lower_source_payload_field(key, value))
            else:
                lowered[key] = value
        return lowered, int(source_row_idx) if source_row_idx is not None else None

    def _lower_payload_field(
        self, field_name: str, payload: Payload, *, row_id: str
    ) -> dict[str, Any]:
        payload_type = payload.payload_type
        filename = f"{field_name}.{payload_type.default_extension.lstrip('.')}"
        path = self.payload_dir / row_id / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.stem}.tmp-{uuid.uuid4().hex}{path.suffix}")
        try:
            extra_meta = payload_type.dump_to_path(
                payload.data,
                temporary,
                metadata=dict(payload.metadata or {}),
            )
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        relative_path = f"{row_id}/{filename}"
        payload_ref = make_artifact_ref(
            artifact_id=self.artifact_id.text,
            relative_path=f"payload/{relative_path}",
        )
        lowered = {
            f"[payload_id]{field_name}": str(payload_type.payload_id),
            f"[payload_ref]{field_name}": payload_ref,
            f"[payload_status]{field_name}": "present",
            f"[payload_sha256]{field_name}": digest.hexdigest(),
        }
        for meta_key, meta_value in extra_meta.items():
            lowered[f"[payload_{meta_key}]{field_name}"] = meta_value
        return lowered

    def _lower_external_payload_field(
        self, field_name: str, payload: ExternalPayloadRef
    ) -> dict[str, Any]:
        lowered = {
            f"[payload_id]{field_name}": str(payload.payload_type.payload_id),
            f"[payload_ref]{field_name}": str(payload.ref),
            f"[payload_status]{field_name}": "present",
        }
        for meta_key, meta_value in dict(payload.metadata or {}).items():
            lowered[f"[payload_{meta_key}]{field_name}"] = meta_value
        return lowered

    def _lower_source_payload_field(
        self, field_name: str, payload: SourcePayloadRef
    ) -> dict[str, Any]:
        lowered = {
            f"[payload_id]{field_name}": str(payload.payload_type.payload_id),
            f"[payload_ref]{field_name}": make_source_ref(
                sandbox_name=payload.sandbox_root.name,
                relative_path=payload.relative_path,
            ),
            f"[payload_status]{field_name}": "present",
        }
        for meta_key, meta_value in dict(payload.metadata or {}).items():
            lowered[f"[payload_{meta_key}]{field_name}"] = meta_value
        return lowered


def make_row_id(
    row: dict[str, Any],
    *,
    row_idx: int,
    unique_key_fields: tuple[str, ...] = (),
) -> str:
    if unique_key_fields:
        parts: list[str] = []
        for field_name in unique_key_fields:
            if field_name not in row:
                raise MissingArtifactKeyFieldError(
                    "cannot build row_id because unique-key field is missing: "
                    f"{field_name!r} (declared key={unique_key_fields})"
                )
            parts.append(f"{field_name}={_safe_row_token(str(row[field_name]))}")
        return "__".join(parts)

    key_candidates = ("tooth_uid", "arch_uid", "case_uid", "id")
    for key in key_candidates:
        value = row.get(key)
        if value is not None:
            return f"{key}={_safe_row_token(str(value))}"
    payload = json.dumps(row_idx, ensure_ascii=False)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"row_{row_idx:06d}_{digest}"


def _safe_row_token(value: str) -> str:
    return value.replace("\\", "_").replace("/", "_").replace(":", "__").replace(" ", "_")


def write_artifact(
    *,
    artifact_name: str,
    artifact_version: str,
    rows: list[dict[str, Any]],
    out_dir: Path,
    table_meta: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
    artifact_roots: dict[str, str] | None = None,
    row_written_callback: Any | None = None,
    unique_key_fields: tuple[str, ...] = (),
) -> ArtifactWriteResult:
    effective_roots = dict(artifact_roots or {})
    effective_roots.setdefault(f"{artifact_name}@{artifact_version}", str(out_dir))

    writer = ArtifactWriter(
        artifact_name=artifact_name,
        artifact_version=artifact_version,
        out_dir=out_dir,
        artifact_roots=effective_roots,
        unique_key_fields=tuple(unique_key_fields),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    lowered_rows = writer.write_payload_rows(rows, row_written_callback=row_written_callback)
    manifest = pa.Table.from_pylist(lowered_rows)
    schema_meta = {
        "artifact_name": artifact_name,
        "artifact_version": artifact_version,
        "storage_format_version": LAZETABLE_STORAGE_FORMAT_VERSION,
    }
    if table_meta:
        schema_meta.update(_normalize_metadata_dict(table_meta))
    if effective_roots:
        schema_meta["artifact_roots"] = effective_roots
    encoded = {
        str(key).encode("utf-8"): json.dumps(value, ensure_ascii=False).encode("utf-8")
        for key, value in schema_meta.items()
    }
    manifest = manifest.replace_schema_metadata(encoded)
    manifest_path = out_dir / "manifest.parquet"
    pq.write_table(manifest, manifest_path, compression="zstd")
    final_meta = {
        "artifact_name": artifact_name,
        "artifact_version": artifact_version,
        "storage_format_version": LAZETABLE_STORAGE_FORMAT_VERSION,
        "row_count": manifest.num_rows,
        "artifact_roots": effective_roots,
    }
    if meta:
        final_meta.update(_normalize_metadata_dict(meta))
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(final_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    table = _table_from_artifact_parts(
        table=manifest,
        artifact_id=ArtifactId(artifact_name, artifact_version),
        path=out_dir,
        meta=final_meta,
        sandbox_registry=None,
    )
    return ArtifactWriteResult(table=table, lowered_rows=lowered_rows)


def write_lowered_artifact(
    *,
    artifact_name: str,
    artifact_version: str,
    lowered_rows: list[dict[str, Any]],
    out_dir: Path,
    table_meta: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
    artifact_roots: dict[str, str] | None = None,
) -> ArtifactWriteResult:
    """이미 payload가 저장된 lowered row로 manifest와 metadata만 기록한다."""

    effective_roots = dict(artifact_roots or {})
    effective_roots.setdefault(f"{artifact_name}@{artifact_version}", str(out_dir))
    columns = sorted({key for row in lowered_rows for key in row})
    normalized_rows = [{key: row.get(key) for key in columns} for row in lowered_rows]
    manifest = pa.Table.from_pylist(normalized_rows)
    schema_meta = {
        "artifact_name": artifact_name,
        "artifact_version": artifact_version,
        "storage_format_version": LAZETABLE_STORAGE_FORMAT_VERSION,
    }
    if table_meta:
        schema_meta.update(_normalize_metadata_dict(table_meta))
    if effective_roots:
        schema_meta["artifact_roots"] = effective_roots
    encoded = {
        str(key).encode("utf-8"): json.dumps(value, ensure_ascii=False).encode("utf-8")
        for key, value in schema_meta.items()
    }
    manifest = manifest.replace_schema_metadata(encoded)
    pq.write_table(manifest, out_dir / "manifest.parquet", compression="zstd")
    final_meta = {
        "artifact_name": artifact_name,
        "artifact_version": artifact_version,
        "storage_format_version": LAZETABLE_STORAGE_FORMAT_VERSION,
        "row_count": manifest.num_rows,
        "artifact_roots": effective_roots,
    }
    if meta:
        final_meta.update(_normalize_metadata_dict(meta))
    (out_dir / "meta.json").write_text(
        json.dumps(final_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    table = _table_from_artifact_parts(
        table=manifest,
        artifact_id=ArtifactId(artifact_name, artifact_version),
        path=out_dir,
        meta=final_meta,
        sandbox_registry=None,
    )
    return ArtifactWriteResult(table=table, lowered_rows=lowered_rows)


def _normalize_metadata_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _normalize_metadata_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_metadata_value(v) for v in value]
    if isinstance(value, tuple):
        return [_normalize_metadata_value(v) for v in value]
    return value


def _normalize_metadata_dict(payload: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _normalize_metadata_value(value) for key, value in payload.items()}


def serialize_dependencies(dependencies: list[ArtifactDependency]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for dep in dependencies:
        item: dict[str, Any] = {
            "artifact_name": dep.artifact_name,
            "artifact_version": dep.artifact_version,
        }
        if dep.join is not None:
            item["join"] = {
                "child_on": list(dep.join.child_on),
                "parent_on": list(dep.join.parent_on),
            }
        payload.append(item)
    return payload


def _dependency_from_meta(meta: dict[str, Any]) -> ArtifactDependency:
    join_meta = meta.get("join")
    join = None
    if isinstance(join_meta, dict):
        join = ArtifactJoin(
            child_on=tuple(str(value) for value in join_meta.get("child_on", [])),
            parent_on=tuple(str(value) for value in join_meta.get("parent_on", [])),
        )
    return ArtifactDependency(
        artifact_name=str(meta["artifact_name"]),
        artifact_version=str(meta.get("artifact_version", "latest")),
        join=join,
    )


def join_parent_tables(
    table: LazyTable,
    *,
    store: ArtifactStore | None = None,
    sandbox_registry: SandboxRegistry | None = None,
) -> LazyTable:
    dependency_payload = table.meta.get("dependencies")
    if not isinstance(dependency_payload, list) or not dependency_payload:
        return table

    current = table
    for dependency_meta in dependency_payload:
        if not isinstance(dependency_meta, dict):
            continue
        dependency = _dependency_from_meta(dependency_meta)
        parent = _open_dependency_artifact(
            dependency,
            table=table,
            store=store,
            sandbox_registry=sandbox_registry,
        )
        current = _join_parent_child(
            parent=parent, child=current, dependency=dependency, sandbox_registry=sandbox_registry
        )
    return current


def _open_dependency_artifact(
    dependency: ArtifactDependency,
    *,
    table: LazyTable,
    store: ArtifactStore | None,
    sandbox_registry: SandboxRegistry | None,
) -> LazyTable:
    if store is not None:
        return store.open(
            dependency.artifact_name,
            dependency.artifact_version,
            join_parents=True,
            sandbox_registry=sandbox_registry,
        )

    artifact_roots = table.meta.get("artifact_roots", {})
    if not isinstance(artifact_roots, dict):
        raise FileNotFoundError(
            "artifact_roots metadata missing; cannot resolve dependency without store"
        )
    default_root = artifact_roots.get("default")
    if not default_root:
        raise FileNotFoundError(
            "default artifact root missing; cannot resolve dependency without store"
        )
    path = Path(str(default_root)) / dependency.artifact_name / dependency.artifact_version
    return open_artifact(path, join_parents=True, sandbox_registry=sandbox_registry)


def _join_parent_child(
    *,
    parent: LazyTable,
    child: LazyTable,
    dependency: ArtifactDependency,
    sandbox_registry: SandboxRegistry | None,
) -> LazyTable:
    if dependency.join is None:
        raise ValueError(
            f"dependency join is required for auto-join: {dependency.artifact_name}@{dependency.artifact_version}"
        )

    child_on = list(dependency.join.child_on)
    parent_on = list(dependency.join.parent_on)
    joined_manifest = _join_two_tables_child_wins(
        parent.table,
        child.table,
        child_on=child_on,
        parent_on=parent_on,
    )
    return _table_from_artifact_parts(
        table=joined_manifest,
        artifact_id=child.artifact_id,
        path=child.path,
        meta=child.meta,
        sandbox_registry=sandbox_registry or child.resolver_context.sandbox_registry,
    )


def _join_two_tables_child_wins(
    parent: pa.Table,
    child: pa.Table,
    *,
    child_on: list[str],
    parent_on: list[str],
) -> pa.Table:
    if len(child_on) != len(parent_on):
        raise ValueError("child_on and parent_on must have the same length")

    parent_idx_name = "__parent_idx"
    child_idx_name = "__child_idx"
    parent_key_table = pa.table(
        [parent[field] for field in parent_on]
        + [pa.array(range(parent.num_rows), type=pa.int64())],
        names=parent_on + [parent_idx_name],
    )
    child_key_table = pa.table(
        [child[field] for field in child_on] + [pa.array(range(child.num_rows), type=pa.int64())],
        names=child_on + [child_idx_name],
    )
    joined = parent_key_table.join(
        child_key_table,
        keys=parent_on,
        right_keys=child_on,
        join_type="inner",
        coalesce_keys=True,
        use_threads=True,
    )

    parent_indices = joined[parent_idx_name]
    child_indices = joined[child_idx_name]
    child_names = set(child.column_names)
    child_join_names = set(child_on)

    arrays: list[pa.Array | pa.ChunkedArray] = []
    names: list[str] = []

    for name in parent.column_names:
        if name in child_names:
            continue
        if name in parent_on and name not in child_join_names:
            arrays.append(joined[name])
        else:
            arrays.append(pc.take(parent[name], parent_indices))
        names.append(name)

    for name in child.column_names:
        if name in child_join_names and name in joined.column_names:
            arrays.append(joined[name])
        else:
            arrays.append(pc.take(child[name], child_indices))
        names.append(name)

    out = pa.table(arrays, names=names)
    if child.schema.metadata:
        out = out.replace_schema_metadata(child.schema.metadata)
    return out


def _table_from_artifact_parts(
    *,
    table: pa.Table,
    artifact_id: ArtifactId,
    path: Path,
    meta: dict[str, Any],
    sandbox_registry: SandboxRegistry | None,
) -> LazyTable:
    merged_context = {
        "artifact_root": str(path.parent.parent),
    }
    resolver_context = infer_resolver_context(
        schema_metadata=table.schema.metadata,
        artifact_meta=meta,
        context=merged_context,
        sandbox_registry=sandbox_registry,
    )
    return LazyTable.from_table(
        table,
        artifact_id=artifact_id,
        path=path,
        meta=meta,
        resolver_context=resolver_context,
        context=merged_context,
    )
