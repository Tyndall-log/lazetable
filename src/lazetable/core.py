"""Core runtime primitives for lazetable."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq


PathResolution = Literal["relative", "absolute"]
RootOverrides = Mapping[str, str | PathLike[str]]
PAYLOAD_COL_RE = re.compile(r"^\[payload_(?P<meta>[^\]]+)\](?P<field>.+)$")
ROW_VIEW_COL_RE = re.compile(r"^\[row_view_(?P<meta>[^\]]+)\](?P<field>.+)$")


@dataclass(frozen=True)
class SandboxRoot:
    """Named sandbox root used for source-backed lazy refs."""

    name: str
    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))


@dataclass(frozen=True)
class ArtifactId:
    """Logical identifier for a persisted table artifact."""

    name: str
    version: str

    @property
    def text(self) -> str:
        return f"{self.name}@{self.version}"


class SandboxRegistry:
    """Registry of allowed source sandboxes for `src://...` resolution."""

    def __init__(self, roots: Mapping[str, str | Path | SandboxRoot]) -> None:
        normalized: dict[str, SandboxRoot] = {}
        for key, value in roots.items():
            if isinstance(value, SandboxRoot):
                normalized[str(value.name)] = SandboxRoot(str(value.name), value.root.resolve())
            else:
                normalized[str(key)] = SandboxRoot(str(key), Path(value).resolve())
        self._roots = normalized

    def names(self) -> list[str]:
        return sorted(self._roots)

    def sandbox_for(self, name: str) -> SandboxRoot:
        try:
            return self._roots[name]
        except KeyError as exc:
            raise KeyError(f"unknown sandbox root: {name}") from exc

    def resolve(self, name: str, relative_path: str) -> Path:
        root = self.sandbox_for(name).root
        rel = relative_path.replace("\\", "/").lstrip("/")
        if not rel:
            raise ValueError("sandbox relative path must not be empty")
        candidate = root.joinpath(*[part for part in rel.split("/") if part]).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"resolved path escapes sandbox root: sandbox={name}, relative_path={relative_path!r}"
            ) from exc
        return candidate


@dataclass(frozen=True)
class ResolverContext:
    """Absolute-root mapping used for logical ref resolution.

    ``path_resolution``과 ``root_overrides``는 이 context로 연 table이 나중에
    ``join_parents``할 때 같은 규칙을 물려주기 위해 보관한다. ``relative``에서는
    ``artifact_roots``에 없는 ``<name>@<version>`` namespace를 ``default`` 저장
    루트 아래 ``<name>/<version>``으로 계산한다.
    """

    sandbox_registry: SandboxRegistry | None
    artifact_roots: Mapping[str, str]
    path_resolution: PathResolution = "absolute"
    root_overrides: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ParseHandle:
    """Normalized lazy-resolution request for a payload-backed field."""

    payload_id: str
    field: str
    ref: str
    metadata: Mapping[str, str | int | float | bool | None]


class PayloadManager:
    """Payload type registry and lazy field resolver."""

    def __init__(self) -> None:
        self._payload_types: dict[str, Any] = {}

    def register(self, payload_type: Any) -> Any:
        payload_id = str(payload_type.payload_id)
        if payload_id in self._payload_types:
            raise ValueError(f"duplicate payload_id: {payload_id}")
        self._payload_types[payload_id] = payload_type
        return payload_type

    def get(self, payload_id: str) -> Any:
        try:
            return self._payload_types[payload_id]
        except KeyError as exc:
            raise KeyError(f"unknown payload_id: {payload_id}") from exc

    def known_payload_ids(self) -> list[str]:
        return sorted(self._payload_types)

    def resolve(self, handle: ParseHandle, resolver_context: ResolverContext) -> Any:
        payload_type = self.get(handle.payload_id)
        path = resolve_ref(str(handle.ref), resolver_context)
        return payload_type.load_from_path(path, metadata=dict(handle.metadata))


class RowViewManager:
    """Row-view registry and dispatcher."""

    def __init__(self) -> None:
        self._views: dict[str, Any] = {}

    def register(self, row_view: Any) -> Any:
        view_id = str(row_view.view_id)
        if view_id in self._views:
            raise ValueError(f"duplicate row_view_id: {view_id}")
        self._views[view_id] = row_view
        return row_view

    def get(self, view_id: str) -> Any:
        try:
            return self._views[view_id]
        except KeyError as exc:
            raise KeyError(f"unknown row_view_id: {view_id}") from exc

    def resolve(self, *, field: str, row: "ParsedRow", view_id: str) -> Any:
        row_view = self.get(view_id)
        return row_view.compute(field=field, row=row, row_view_manager=self)


DEFAULT_PAYLOAD_MANAGER = PayloadManager()
DEFAULT_ROW_VIEW_MANAGER = RowViewManager()


def resolve_ref(ref: str, context: ResolverContext) -> Path:
    """Resolve logical refs into absolute filesystem paths."""

    raw = str(ref)
    if raw.startswith("src://"):
        payload = raw[len("src://") :]
        namespace, sep, rel_path = payload.partition("/")
        if not sep:
            raise ValueError(f"invalid source ref: {ref}")
        if context.sandbox_registry is None:
            raise KeyError(f"no sandbox registry available for source ref: {ref}")
        return context.sandbox_registry.resolve(namespace, rel_path)

    if raw.startswith("art://"):
        payload = raw[len("art://") :]
        namespace, sep, rel_path = payload.partition("/")
        if not sep:
            raise ValueError(f"invalid artifact ref: {ref}")
        root = context.artifact_roots.get(namespace)
        if root is None:
            name, at, version = namespace.rpartition("@")
            default_root = context.artifact_roots.get("default")
            if context.path_resolution != "relative" or not at or not name or not default_root:
                raise KeyError(f"unknown artifact root namespace: {namespace}")
            root = str(Path(default_root) / name / version)
        return Path(root).joinpath(*[part for part in rel_path.split("/") if part])

    return Path(raw)


def make_artifact_ref(*, artifact_id: str, relative_path: str) -> str:
    rel = relative_path.replace("\\", "/").lstrip("/")
    return f"art://{artifact_id}/{rel}"


def make_source_ref(*, sandbox_name: str, relative_path: str) -> str:
    rel = relative_path.replace("\\", "/").lstrip("/")
    return f"src://{sandbox_name}/{rel}"


def _decode_schema_metadata(metadata: Mapping[bytes, bytes] | None) -> dict[str, Any]:
    if not metadata:
        return {}
    decoded: dict[str, Any] = {}
    for key, value in metadata.items():
        text_key = key.decode("utf-8")
        text_value = value.decode("utf-8")
        try:
            decoded[text_key] = json.loads(text_value)
        except json.JSONDecodeError:
            decoded[text_key] = text_value
    return decoded


def infer_resolver_context(
    *,
    schema_metadata: Mapping[bytes, bytes] | None = None,
    artifact_meta: Mapping[str, Any] | None = None,
    context: Mapping[str, Any] | None = None,
    sandbox_registry: SandboxRegistry | None = None,
) -> ResolverContext:
    """명시적 context와 artifact metadata에서 resolver context를 구성한다."""

    schema_meta = _decode_schema_metadata(schema_metadata)
    merged_artifact_roots: dict[str, str] = {}
    active_sandbox_registry = sandbox_registry

    for payload in (schema_meta, artifact_meta or {}, context or {}):
        sandbox_payload = payload.get("sandbox_registry")
        if isinstance(sandbox_payload, SandboxRegistry):
            active_sandbox_registry = sandbox_payload
        src = payload.get("source_roots")
        if active_sandbox_registry is None and isinstance(src, dict):
            active_sandbox_registry = SandboxRegistry({str(k): str(v) for k, v in src.items()})
        arts = payload.get("artifact_roots")
        if isinstance(arts, dict):
            merged_artifact_roots.update({str(k): str(v) for k, v in arts.items()})
        default_artifact_root = payload.get("artifact_root")
        if default_artifact_root and "default" not in merged_artifact_roots:
            merged_artifact_roots["default"] = str(default_artifact_root)

    return ResolverContext(
        sandbox_registry=active_sandbox_registry,
        artifact_roots=merged_artifact_roots,
    )


class ParsedRow:
    """Row facade with payload-backed lazy fields and row-view support."""

    def __init__(
        self,
        values: Mapping[str, Any],
        *,
        payload_manager: PayloadManager | None = None,
        row_view_manager: RowViewManager | None = None,
        resolver_context: ResolverContext | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        self._values = dict(values)
        self._payload_manager = payload_manager or DEFAULT_PAYLOAD_MANAGER
        self._row_view_manager = row_view_manager or DEFAULT_ROW_VIEW_MANAGER
        self._resolver_context = resolver_context or ResolverContext(
            sandbox_registry=None, artifact_roots={}
        )
        self._context = dict(context or {})
        self._resolved_fields: dict[str, Any] = {}
        self._payload_meta = self._collect_payload_meta(self._values)
        self._row_view_meta = self._collect_row_view_meta(self._values)

    @staticmethod
    def _collect_payload_meta(values: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        parsed: dict[str, dict[str, Any]] = {}
        for key, value in values.items():
            match = PAYLOAD_COL_RE.match(key)
            if match is None:
                continue
            meta_name = match.group("meta")
            field_name = match.group("field")
            parsed.setdefault(field_name, {})[meta_name] = value
        return parsed

    @staticmethod
    def _collect_row_view_meta(values: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        parsed: dict[str, dict[str, Any]] = {}
        for key, value in values.items():
            match = ROW_VIEW_COL_RE.match(key)
            if match is None:
                continue
            meta_name = match.group("meta")
            field_name = match.group("field")
            parsed.setdefault(field_name, {})[meta_name] = value
        return parsed

    def raw(self, key: str, default: Any = None) -> Any:
        if key in self._values:
            return self._values[key]
        return self._context.get(key, default)

    def as_dict(self) -> dict[str, Any]:
        return dict(self._values)

    def fields(self) -> list[str]:
        base_fields = {key for key in self._values if not PAYLOAD_COL_RE.match(key)}
        return sorted(base_fields | set(self._payload_meta) | set(self._row_view_meta))

    def payload_fields(self) -> list[str]:
        return sorted(self._payload_meta)

    def row_view_fields(self) -> list[str]:
        return sorted(self._row_view_meta)

    @property
    def resolver_context(self) -> ResolverContext:
        return self._resolver_context

    @property
    def context(self) -> dict[str, Any]:
        return dict(self._context)

    def payload_handle(self, field: str) -> ParseHandle | None:
        meta = self._payload_meta.get(field)
        if meta is None:
            return None
        payload_id = meta.get("id")
        payload_ref = meta.get("ref")
        if payload_id is None or payload_ref is None:
            return None
        extra = {k: v for k, v in meta.items() if k not in {"id", "ref"}}
        return ParseHandle(
            payload_id=str(payload_id),
            field=field,
            ref=str(payload_ref),
            metadata=extra,
        )

    def row_view_id(self, field: str) -> str | None:
        meta = self._row_view_meta.get(field)
        if meta is None:
            return None
        view_id = meta.get("id")
        return None if view_id is None else str(view_id)

    def resolve(self, field: str) -> Any:
        if field in self._resolved_fields:
            return self._resolved_fields[field]

        if field in self._values and self._values[field] is not None:
            value = self._values[field]
            self._resolved_fields[field] = value
            return value

        handle = self.payload_handle(field)
        if handle is not None:
            value = self._payload_manager.resolve(handle, self._resolver_context)
            self._resolved_fields[field] = value
            return value

        view_id = self.row_view_id(field)
        if view_id is not None:
            value = self._row_view_manager.resolve(field=field, row=self, view_id=view_id)
            self._resolved_fields[field] = value
            return value

        raise AttributeError(field)

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self._values or key in self._payload_meta or key in self._row_view_meta

    def __getstate__(self) -> dict[str, Any]:
        return {
            "_values": self._values,
            "_payload_manager": self._payload_manager,
            "_row_view_manager": self._row_view_manager,
            "_resolver_context": self._resolver_context,
            "_context": self._context,
            "_resolved_fields": self._resolved_fields,
            "_payload_meta": self._payload_meta,
            "_row_view_meta": self._row_view_meta,
        }

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self._values = dict(state["_values"])
        self._payload_manager = state["_payload_manager"]
        self._row_view_manager = state["_row_view_manager"]
        self._resolver_context = state["_resolver_context"]
        self._context = dict(state["_context"])
        self._resolved_fields = dict(state["_resolved_fields"])
        self._payload_meta = {
            str(field): dict(meta) for field, meta in dict(state["_payload_meta"]).items()
        }
        self._row_view_meta = {
            str(field): dict(meta) for field, meta in dict(state["_row_view_meta"]).items()
        }

    def __getattr__(self, field: str) -> Any:
        return self.resolve(field)


class LazyTable:
    """Unified table object for persisted artifacts and joined logical views."""

    def __init__(
        self,
        table: pa.Table,
        *,
        artifact_id: ArtifactId | None = None,
        path: Path | None = None,
        meta: Mapping[str, Any] | None = None,
        payload_manager: PayloadManager | None = None,
        row_view_manager: RowViewManager | None = None,
        resolver_context: ResolverContext | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        self._table = table
        self._artifact_id = artifact_id
        self._path = Path(path) if path is not None else None
        self._meta = dict(meta or {})
        self._payload_manager = payload_manager or DEFAULT_PAYLOAD_MANAGER
        self._row_view_manager = row_view_manager or DEFAULT_ROW_VIEW_MANAGER
        self._resolver_context = resolver_context or ResolverContext(
            sandbox_registry=None, artifact_roots={}
        )
        self._context = dict(context or {})

    @classmethod
    def from_table(
        cls,
        table: pa.Table,
        *,
        artifact_id: ArtifactId | None = None,
        path: Path | None = None,
        meta: Mapping[str, Any] | None = None,
        payload_manager: PayloadManager | None = None,
        row_view_manager: RowViewManager | None = None,
        resolver_context: ResolverContext | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> "LazyTable":
        return cls(
            table,
            artifact_id=artifact_id,
            path=path,
            meta=meta,
            payload_manager=payload_manager,
            row_view_manager=row_view_manager,
            resolver_context=resolver_context,
            context=context,
        )

    @classmethod
    def from_parquet(
        cls,
        path: str | Path,
        *,
        payload_manager: PayloadManager | None = None,
        row_view_manager: RowViewManager | None = None,
        context: Mapping[str, Any] | None = None,
        columns: list[str] | None = None,
        sandbox_registry: SandboxRegistry | None = None,
    ) -> "LazyTable":
        parquet_path = Path(path)
        table = pq.read_table(parquet_path, columns=columns)
        artifact_meta: dict[str, Any] = {}
        meta_path = parquet_path.parent / "meta.json"
        if meta_path.exists():
            artifact_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        merged_context = dict(context or {})
        merged_context.setdefault("dataset_root", str(parquet_path.parent))
        resolver_context = infer_resolver_context(
            schema_metadata=table.schema.metadata,
            artifact_meta=artifact_meta,
            context=merged_context,
            sandbox_registry=sandbox_registry,
        )
        return cls.from_table(
            table,
            artifact_id=ArtifactId(
                name=str(artifact_meta.get("artifact_name") or parquet_path.parent.parent.name),
                version=str(artifact_meta.get("artifact_version") or parquet_path.parent.name),
            ),
            path=parquet_path.parent,
            meta=artifact_meta,
            payload_manager=payload_manager,
            row_view_manager=row_view_manager,
            resolver_context=resolver_context,
            context=merged_context,
        )

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        join_parents: bool = False,
        sandbox_registry: SandboxRegistry | None = None,
        path_resolution: PathResolution | None = None,
        root_overrides: RootOverrides | None = None,
    ) -> "LazyTable":
        from .artifacts import open_artifact

        return open_artifact(
            path,
            join_parents=join_parents,
            sandbox_registry=sandbox_registry,
            path_resolution=path_resolution,
            root_overrides=root_overrides,
        )

    @property
    def table(self) -> pa.Table:
        return self._table

    @property
    def artifact_id(self) -> ArtifactId | None:
        return self._artifact_id

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def meta(self) -> dict[str, Any]:
        return dict(self._meta)

    @property
    def resolver_context(self) -> ResolverContext:
        return self._resolver_context

    @property
    def context(self) -> dict[str, Any]:
        return dict(self._context)

    def join_parents(
        self,
        *,
        sandbox_registry: SandboxRegistry | None = None,
        store: Any = None,
        path_resolution: PathResolution | None = None,
        root_overrides: RootOverrides | None = None,
    ) -> "LazyTable":
        from .artifacts import join_parent_tables

        return join_parent_tables(
            self,
            store=store,
            sandbox_registry=sandbox_registry,
            path_resolution=path_resolution,
            root_overrides=root_overrides,
        )

    def count_rows(self) -> int:
        return self._table.num_rows

    def iter_rows(self) -> Iterator[ParsedRow]:
        for row in self._table.to_pylist():
            yield ParsedRow(
                row,
                payload_manager=self._payload_manager,
                row_view_manager=self._row_view_manager,
                resolver_context=self._resolver_context,
                context=self._context,
            )

    def materialize_rows(self) -> list[ParsedRow]:
        return list(self.iter_rows())

    def transform(self, func: Any) -> "LazyTable":
        out_rows: list[dict[str, Any]] = []
        for row in self.iter_rows():
            result = func(row)
            if result is None:
                continue
            if isinstance(result, list):
                out_rows.extend(dict(item) for item in result)
            else:
                out_rows.append(dict(result))
        return LazyTable.from_table(
            pa.Table.from_pylist(out_rows),
            artifact_id=self._artifact_id,
            path=self._path,
            meta=self._meta,
            payload_manager=self._payload_manager,
            row_view_manager=self._row_view_manager,
            resolver_context=self._resolver_context,
            context=self._context,
        )

    def __len__(self) -> int:
        return self.count_rows()

    def __iter__(self) -> Iterator[ParsedRow]:
        return self.iter_rows()

    def __getitem__(self, index: int | slice) -> ParsedRow | "LazyTable":
        if isinstance(index, slice):
            start = 0 if index.start is None else index.start
            stop = len(self) if index.stop is None else index.stop
            step = 1 if index.step is None else index.step
            rows = (
                self._table.to_pylist()[index]
                if step != 1
                else self._table.slice(start, max(0, stop - start)).to_pylist()
            )
            return LazyTable.from_table(
                pa.Table.from_pylist(rows),
                artifact_id=self._artifact_id,
                path=self._path,
                meta=self._meta,
                payload_manager=self._payload_manager,
                row_view_manager=self._row_view_manager,
                resolver_context=self._resolver_context,
                context=self._context,
            )
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        row = self._table.slice(index, 1).to_pylist()[0]
        return ParsedRow(
            row,
            payload_manager=self._payload_manager,
            row_view_manager=self._row_view_manager,
            resolver_context=self._resolver_context,
            context=self._context,
        )

    def first_row(self) -> ParsedRow:
        if len(self) == 0:
            raise IndexError("table is empty")
        return self[0]

    def to_dataset_index(
        self,
        *,
        where: Mapping[str, Any] | None = None,
        fields: list[str] | None = None,
    ):
        from .datasets import DatasetIndex

        return DatasetIndex.from_table(self, where=where, fields=fields)

    def export_parquet(self, path: str | Path, *, compression: str = "zstd") -> Path:
        """완성 SQLite artifact의 현재 logical table을 Parquet snapshot으로 내보낸다."""

        import json

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        metadata = dict(self._table.schema.metadata or {})
        metadata[b"lazetable_artifact_meta"] = json.dumps(
            self._meta,
            ensure_ascii=False,
        ).encode("utf-8")
        table = self._table.replace_schema_metadata(metadata)
        pq.write_table(table, target, compression=compression)
        return target


def payload_columns_for(field_names: Iterable[str], all_columns: Iterable[str]) -> list[str]:
    targets = set(field_names)
    selected: list[str] = []
    for name in all_columns:
        payload_match = PAYLOAD_COL_RE.match(name)
        row_view_match = ROW_VIEW_COL_RE.match(name)
        if payload_match is None and row_view_match is None:
            if name in targets:
                selected.append(name)
            continue
        if payload_match is not None and payload_match.group("field") in targets:
            selected.append(name)
        if row_view_match is not None and row_view_match.group("field") in targets:
            selected.append(name)
    return selected
