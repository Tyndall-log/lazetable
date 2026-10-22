"""Artifact metadata, storage, and payload-lowering helpers."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import pyarrow as pa
import pyarrow.compute as pc

from .core import (
    ArtifactId,
    LazyTable,
    PathResolution,
    ResolverContext,
    RootOverrides,
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


class ArtifactPathFallbackWarning(UserWarning):
    """상대 위치에 artifact 디렉터리가 없어 저장된 절대 경로로 대체할 때 발생한다."""


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
    """Store 루트와 이 store로 여는 artifact의 기본 경로 해석 규칙."""

    artifact_roots: dict[str, Path]
    path_resolution: PathResolution = "relative"
    root_overrides: RootOverrides | None = None


def _normalize_root_overrides(overrides: RootOverrides | None, base: Path) -> dict[str, str]:
    """상대 경로 override를 ``base`` 기준으로 결합해 절대 경로 문자열로 만든다."""

    normalized: dict[str, str] = {}
    for key, value in (overrides or {}).items():
        path = Path(value)
        if not path.is_absolute():
            path = (base / path).resolve()
        normalized[str(key)] = str(path)
    return normalized


class ArtifactStore:
    def __init__(self, paths: ArtifactStorePaths):
        self.paths = paths

    def artifact_dir(self, root_name: str, artifact_name: str, artifact_version: str) -> Path:
        return self.paths.artifact_roots[root_name] / artifact_name / artifact_version

    def artifact_root_map(self) -> dict[str, str]:
        return {name: str(path) for name, path in self.paths.artifact_roots.items()}

    def merged_root_overrides(
        self, root_overrides: RootOverrides | None = None, *, base: Path | None = None
    ) -> dict[str, str]:
        """Store 기본 override와 호출 override를 합친다.

        호출 쪽이 키 단위로 우선한다. 상대 경로 값은 ``base``(생략 시 ``default`` 루트) 기준이다.
        """

        merged = {**(self.paths.root_overrides or {}), **(root_overrides or {})}
        if base is None:
            base = Path(self.paths.artifact_roots.get("default", Path.cwd()))
        return _normalize_root_overrides(merged, base)

    def open(
        self,
        artifact_name: str,
        artifact_version: str = "latest",
        *,
        root_name: str = "default",
        join_parents: bool = False,
        sandbox_registry: SandboxRegistry | None = None,
        path_resolution: PathResolution | None = None,
        root_overrides: RootOverrides | None = None,
    ) -> LazyTable:
        root = Path(self.paths.artifact_roots[root_name])
        overrides = self.merged_root_overrides(root_overrides, base=root)
        if artifact_version == "latest":
            base = root / artifact_name
            candidates = (
                [path for path in base.iterdir() if path.is_dir() and not path.name.startswith(".")]
                if base.exists()
                else []
            )
            if not candidates:
                raise FileNotFoundError(f"artifact not found: {artifact_name}@latest")
            artifact_dir = max(candidates, key=lambda path: path.stat().st_mtime)
            artifact_version = artifact_dir.name
        else:
            artifact_dir = root / artifact_name / artifact_version
        artifact_dir = Path(overrides.get(f"{artifact_name}@{artifact_version}", artifact_dir))
        return open_artifact(
            artifact_dir,
            store=self,
            join_parents=join_parents,
            sandbox_registry=sandbox_registry,
            path_resolution=path_resolution or self.paths.path_resolution,
            root_overrides=overrides,
        )


def open_artifact(
    path: str | Path,
    *,
    store: ArtifactStore | None = None,
    join_parents: bool = False,
    sandbox_registry: SandboxRegistry | None = None,
    path_resolution: PathResolution | None = None,
    root_overrides: RootOverrides | None = None,
) -> LazyTable:
    """완성 SQLite artifact를 연다.

    ``path_resolution="relative"``(store가 없을 때의 기본값)는 payload ref를 실제로 연
    artifact 디렉터리와 그 저장 루트 기준으로 해석하고, ``"absolute"``는 artifact에
    저장된 ``artifact_roots``를 그대로 쓴다. ``root_overrides``(``"<name>@<version>"`` 또는
    ``"default"`` → 경로)는 두 모드 모두에서 최우선이며, 상대 경로 값은 store 루트(store가
    없으면 이 artifact의 저장 루트) 기준으로 결합한다. ``join_parents``로 여는 부모에도 같은
    규칙을 전달한다.
    """

    artifact_dir = Path(path)
    resolution: PathResolution = path_resolution or (
        store.paths.path_resolution if store is not None else "relative"
    )
    storage_root = artifact_dir.parent.parent
    overrides = (
        store.merged_root_overrides(
            root_overrides, base=Path(store.paths.artifact_roots.get("default", storage_root))
        )
        if store is not None
        else _normalize_root_overrides(root_overrides, storage_root)
    )
    sqlite_path = artifact_dir / "artifact.sqlite"
    if sqlite_path.is_file():
        from .sqlite_store import SQLITE_SCHEMA_VERSION, SQLiteArtifactStore

        sqlite_store = SQLiteArtifactStore(sqlite_path)
        sqlite_store.validate_schema(read_only=True)
        stored = sqlite_store.artifact(read_only=True)
        if stored is None or stored.status != "completed":
            raise ValueError(f"artifact is not completed: {artifact_dir}")
        rows = sqlite_store.final_rows()
        columns = sorted({key for row in rows for key in row if key != "__unique_key__"})
        normalized = [{key: row.get(key) for key in columns} for row in rows]
        manifest = pa.Table.from_pylist(normalized)
        meta = {
            **stored.metadata,
            "artifact_name": stored.artifact_name,
            "artifact_version": stored.artifact_version,
            "storage_format": "sqlite",
            "storage_format_version": SQLITE_SCHEMA_VERSION,
            "row_count": stored.output_row_count,
            "content_hash": stored.content_hash,
            "dependencies": sqlite_store.dependencies(),
            "statistics": sqlite_store.statistics(),
        }
        artifact_id = ArtifactId(stored.artifact_name, stored.artifact_version)
        table = _table_from_artifact_parts(
            table=manifest,
            artifact_id=artifact_id,
            path=artifact_dir,
            meta=meta,
            sandbox_registry=sandbox_registry,
            path_resolution=resolution,
            root_overrides=overrides,
        )
        if not join_parents:
            return table
        return join_parent_tables(
            table,
            store=store,
            sandbox_registry=sandbox_registry,
            path_resolution=resolution,
            root_overrides=overrides,
        )
    raise FileNotFoundError(f"SQLite artifact not found: {sqlite_path}")


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
            f"[payload_size_bytes]{field_name}": path.stat().st_size,
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
    return write_lowered_artifact(
        artifact_name=artifact_name,
        artifact_version=artifact_version,
        lowered_rows=lowered_rows,
        out_dir=out_dir,
        table_meta=table_meta,
        meta=meta,
        artifact_roots=effective_roots,
        unique_key_fields=unique_key_fields,
    )


def write_lowered_artifact(
    *,
    artifact_name: str,
    artifact_version: str,
    lowered_rows: list[dict[str, Any]],
    out_dir: Path,
    table_meta: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
    artifact_roots: dict[str, str] | None = None,
    unique_key_fields: tuple[str, ...] = (),
) -> ArtifactWriteResult:
    """이미 payload가 저장된 lowered row를 완성 SQLite artifact로 기록한다."""

    from .sqlite_store import PayloadRecord, SQLiteArtifactStore

    effective_roots = dict(artifact_roots or {})
    effective_roots.setdefault(f"{artifact_name}@{artifact_version}", str(out_dir))
    normalized_rows = [_normalize_metadata_dict(row) for row in lowered_rows]
    final_meta = {
        "artifact_name": artifact_name,
        "artifact_version": artifact_version,
        "storage_format_version": LAZETABLE_STORAGE_FORMAT_VERSION,
        "row_count": len(normalized_rows),
        "artifact_roots": effective_roots,
    }
    if table_meta:
        final_meta.update(_normalize_metadata_dict(table_meta))
    if meta:
        final_meta.update(_normalize_metadata_dict(meta))
    signature_payload = {
        "artifact_name": artifact_name,
        "artifact_version": artifact_version,
        "rows": normalized_rows,
    }
    build_signature = hashlib.sha256(
        json.dumps(signature_payload, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    store = SQLiteArtifactStore(out_dir / "artifact.sqlite")
    store.initialize_build(
        artifact_name=artifact_name,
        artifact_version=artifact_version,
        build_signature=build_signature,
        signature_payload=signature_payload,
        dependencies=list(final_meta.get("dependencies", [])),
    )
    store.ensure_source_rows(len(normalized_rows))
    final_rows: list[dict[str, Any]] = []
    for row_idx, row in enumerate(normalized_rows):
        key_values = [row[field] for field in unique_key_fields] if unique_key_fields else [row_idx]
        unique_key = json.dumps(key_values, ensure_ascii=False, sort_keys=True)
        payload_records: list[PayloadRecord] = []
        for key, reference in row.items():
            if not key.startswith("[payload_ref]") or reference is None:
                continue
            field = key[len("[payload_ref]") :]
            raw_reference = str(reference)
            marker = "/payload/"
            relative_path = (
                f"payload/{raw_reference.split(marker, 1)[1]}"
                if raw_reference.startswith("art://") and marker in raw_reference
                else raw_reference
            )
            payload_records.append(
                PayloadRecord(
                    field_name=field,
                    payload_id=str(row.get(f"[payload_id]{field}") or ""),
                    relative_path=relative_path,
                    status=str(row.get(f"[payload_status]{field}") or "present"),
                    size_bytes=int(row.get(f"[payload_size_bytes]{field}") or 0),
                    sha256=str(row.get(f"[payload_sha256]{field}") or ""),
                )
            )
        store.commit_source(row_idx, [(0, unique_key, row, payload_records)])
        final_rows.append({**row, "__unique_key__": key_values})
    content_hash = hashlib.sha256(
        json.dumps(final_rows, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    store.finalize(
        rows=final_rows,
        metadata=final_meta,
        statistics={},
        content_hash=content_hash,
    )
    store.checkpoint_and_close()
    return ArtifactWriteResult(
        table=open_artifact(out_dir),
        lowered_rows=lowered_rows,
    )


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
    path_resolution: PathResolution | None = None,
    root_overrides: RootOverrides | None = None,
) -> LazyTable:
    """부모 artifact를 열어 child-wins 규칙으로 join한다.

    ``path_resolution``과 ``root_overrides``를 생략하면 ``table``을 열 때 쓴 규칙을 따른다.
    """

    dependency_payload = table.meta.get("dependencies")
    if not isinstance(dependency_payload, list) or not dependency_payload:
        return table

    context = table.resolver_context
    resolution: PathResolution = path_resolution or context.path_resolution
    storage_root = table.path.parent.parent if table.path is not None else Path.cwd()
    if store is not None:
        added = store.merged_root_overrides(
            root_overrides, base=Path(store.paths.artifact_roots.get("default", storage_root))
        )
    else:
        added = _normalize_root_overrides(root_overrides, storage_root)
    overrides = {**context.root_overrides, **added}

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
            path_resolution=resolution,
            root_overrides=overrides,
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
    path_resolution: PathResolution = "absolute",
    root_overrides: Mapping[str, str] | None = None,
) -> LazyTable:
    overrides = dict(root_overrides or {})
    options: dict[str, Any] = {
        "join_parents": True,
        "sandbox_registry": sandbox_registry,
        "path_resolution": path_resolution,
        "root_overrides": overrides,
    }
    namespace = f"{dependency.artifact_name}@{dependency.artifact_version}"
    if namespace in overrides:
        return open_artifact(overrides[namespace], store=store, **options)
    if store is not None:
        return store.open(dependency.artifact_name, dependency.artifact_version, **options)
    if path_resolution == "relative" and table.path is not None:
        storage_root = Path(overrides.get("default", table.path.parent.parent))
        candidate = storage_root / dependency.artifact_name / dependency.artifact_version
        if candidate.is_dir():
            return open_artifact(candidate, **options)
        warnings.warn(
            f"부모 artifact {namespace}가 상대 위치 {candidate}에 없어 저장된 artifact_roots로 대체합니다",
            ArtifactPathFallbackWarning,
            stacklevel=2,
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
    return open_artifact(path, **options)


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
    child_context = child.resolver_context
    joined_roots = {
        **parent.resolver_context.artifact_roots,
        **child_context.artifact_roots,
    }
    joined_meta = child.meta
    joined_meta["artifact_roots"] = joined_roots
    # 부모와 자식의 root는 각자 열 때 이미 해석되었으므로 다시 추론하지 않는다.
    return LazyTable.from_table(
        joined_manifest,
        artifact_id=child.artifact_id,
        path=child.path,
        meta=joined_meta,
        resolver_context=ResolverContext(
            sandbox_registry or child_context.sandbox_registry,
            joined_roots,
            child_context.path_resolution,
            child_context.root_overrides,
        ),
        context=child.context,
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
    path_resolution: PathResolution = "relative",
    root_overrides: Mapping[str, str] | None = None,
) -> LazyTable:
    merged_context = {
        "artifact_root": str(path.parent.parent),
    }
    stored = infer_resolver_context(
        schema_metadata=table.schema.metadata,
        artifact_meta=meta,
        context=merged_context,
        sandbox_registry=sandbox_registry,
    )
    overrides = dict(root_overrides or {})
    if path_resolution == "absolute":
        roots = {**stored.artifact_roots, **overrides}
    elif path_resolution == "relative":
        roots = _relative_artifact_roots(
            stored.artifact_roots, artifact_id=artifact_id, path=path, overrides=overrides
        )
    else:
        raise ValueError(f"unknown path_resolution: {path_resolution!r}")
    resolver_context = ResolverContext(stored.sandbox_registry, roots, path_resolution, overrides)
    return LazyTable.from_table(
        table,
        artifact_id=artifact_id,
        path=path,
        meta=meta,
        resolver_context=resolver_context,
        context=merged_context,
    )


def _relative_artifact_roots(
    stored_roots: Mapping[str, str],
    *,
    artifact_id: ArtifactId,
    path: Path,
    overrides: Mapping[str, str],
) -> dict[str, str]:
    """실제로 연 artifact 위치를 기준으로 namespace별 root를 다시 계산한다.

    자기 namespace는 연 디렉터리, ``<name>@<version>`` 형식의 다른 namespace는 저장 루트
    아래 ``<name>/<version>``으로 계산한다. 그 디렉터리가 없을 때만 저장된 절대 경로로
    대체하고 경고한다. ``overrides``는 마지막에 덮어써 최우선으로 적용한다.
    """

    storage_root = Path(overrides.get("default", path.parent.parent))
    roots: dict[str, str] = {"default": str(storage_root)}
    for namespace, stored_dir in stored_roots.items():
        if namespace in ("default", artifact_id.text) or namespace in overrides:
            continue
        name, at, version = namespace.rpartition("@")
        if not at or not name:
            roots[namespace] = stored_dir
            continue
        candidate = storage_root / name / version
        if candidate.is_dir():
            roots[namespace] = str(candidate)
            continue
        warnings.warn(
            f"artifact {namespace}가 상대 위치 {candidate}에 없어 저장된 경로 {stored_dir}로 대체합니다",
            ArtifactPathFallbackWarning,
            stacklevel=3,
        )
        roots[namespace] = stored_dir
    roots[artifact_id.text] = str(path)
    roots.update(overrides)
    return roots
