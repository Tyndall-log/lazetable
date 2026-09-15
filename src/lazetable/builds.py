"""재사용, 재개 및 background payload 저장을 위한 build runtime."""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import shutil
import threading
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .artifacts import ArtifactWriter
from .sqlite_store import PayloadRecord, SQLiteArtifactStore


@dataclass(frozen=True, slots=True)
class BuildPolicy:
    reuse_completed: bool = True
    resume_partial: bool = True
    retry_failed: bool = True


@dataclass(frozen=True, slots=True)
class WriterConfig:
    payload_workers: int = 2
    max_pending_writes: int = 8

    def validate(self) -> None:
        if self.payload_workers <= 0:
            raise ValueError("payload_workers must be positive")
        if self.max_pending_writes < self.payload_workers:
            raise ValueError("max_pending_writes must be >= payload_workers")


@dataclass(frozen=True, slots=True)
class BuildResult:
    table: Any
    status: str
    processed_source_rows: int
    skipped_source_rows: int
    failed_source_rows: int


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    item = getattr(value, "item", None)
    if callable(item):
        return jsonable(item())
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(jsonable(payload), ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def manifest_sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    sqlite_path = path / "artifact.sqlite"
    if sqlite_path.is_file():
        artifact = SQLiteArtifactStore(sqlite_path).artifact(read_only=True)
        return None if artifact is None else artifact.content_hash
    manifest = path / "manifest.parquet"
    if not manifest.is_file():
        return None
    digest = hashlib.sha256()
    with manifest.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_signature(
    *,
    builder: Any,
    build_context: dict[str, Any],
    parent_tables: list[Any],
    seed_plan: Any,
    runtime_parameters: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    builder_source = inspect.getsourcefile(type(builder))
    payload = {
        "lazetable_version": importlib.metadata.version("lazetable"),
        "builder": f"{type(builder).__module__}.{type(builder).__qualname__}",
        "builder_source_sha256": _file_sha256(
            Path(builder_source) if builder_source is not None else None
        ),
        "artifact_name": builder.artifact_name,
        "artifact_version": builder.artifact_version,
        "unique_keys": builder.unique_keys,
        "dependencies": [asdict(item) for item in builder.dependencies],
        "build_context": build_context,
        "seed_plan": None if seed_plan is None else asdict(seed_plan),
        "runtime_parameters": runtime_parameters,
        "parents": [
            {
                "artifact_id": None if table.artifact_id is None else table.artifact_id.text,
                "manifest_sha256": manifest_sha256(table.path),
            }
            for table in parent_tables
        ],
        "custom": builder.signature_data(build_context=build_context),
    }
    return stable_hash(payload), jsonable(payload)


def _file_sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class StreamingArtifactWriter:
    """변환 결과를 bounded background thread pool에서 즉시 저장한다."""

    def __init__(
        self,
        *,
        writer: ArtifactWriter,
        store: SQLiteArtifactStore,
        config: WriterConfig,
    ) -> None:
        config.validate()
        self.writer = writer
        self.store = store
        self.executor = ThreadPoolExecutor(
            max_workers=config.payload_workers,
            thread_name_prefix="lazetable-payload",
        )
        self.capacity = threading.Semaphore(config.max_pending_writes)
        self.lock = threading.Lock()
        self.futures: set[Future[Any]] = set()
        self.source_futures: dict[int, list[Future[Any]]] = {}
        self.source_results: dict[
            int, list[tuple[int, str, dict[str, Any], list[PayloadRecord]]]
        ] = {}
        self.closed_sources: set[int] = set()
        self.marked_sources: set[int] = set(store.completed_sources())
        self.errors: list[BaseException] = []
        self.seen_keys: set[str] = set(store.completed_unique_keys())

    def submit(self, source_idx: int, output_idx: int, row: dict[str, Any]) -> None:
        key = json.dumps(
            jsonable(self.writer.extract_unique_key(row)),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self.lock:
            if key in self.seen_keys:
                from .artifacts import DuplicateArtifactKeyError

                raise DuplicateArtifactKeyError(f"duplicate artifact unique key detected: {key}")
            self.seen_keys.add(key)
        self.capacity.acquire()
        future = self.executor.submit(self._write_row, source_idx, output_idx, row)
        with self.lock:
            self.futures.add(future)
            self.source_futures.setdefault(source_idx, []).append(future)
        future.add_done_callback(lambda completed: self._on_done(source_idx, completed))

    def _write_row(
        self, source_idx: int, output_idx: int, row: dict[str, Any]
    ) -> tuple[int, str, dict[str, Any], list[PayloadRecord]]:
        lowered = self.writer.lower_row(row, row_idx=output_idx)
        unique_key = json.dumps(
            jsonable(tuple(lowered[field] for field in self.writer.unique_key_fields)),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        payloads: list[PayloadRecord] = []
        for key, value in lowered.items():
            if not key.startswith("[payload_ref]") or value is None:
                continue
            field = key[len("[payload_ref]") :]
            reference = str(value)
            marker = "/payload/"
            relative = (
                f"payload/{reference.split(marker, 1)[1]}"
                if reference.startswith("art://") and marker in reference
                else reference
            )
            payloads.append(
                PayloadRecord(
                    field_name=field,
                    payload_id=str(lowered.get(f"[payload_id]{field}") or ""),
                    relative_path=relative,
                    status=str(lowered.get(f"[payload_status]{field}") or "present"),
                    size_bytes=int(lowered.get(f"[payload_size_bytes]{field}") or 0),
                    sha256=str(lowered.get(f"[payload_sha256]{field}") or ""),
                )
            )
        return output_idx, unique_key, jsonable(lowered), payloads

    def _on_done(self, source_idx: int, future: Future[Any]) -> None:
        self.capacity.release()
        error = future.exception()
        result = None if error is not None else future.result()
        with self.lock:
            self.futures.discard(future)
            if error is not None:
                self.errors.append(error)
            elif result is not None:
                self.source_results.setdefault(source_idx, []).append(result)
            source_futures = self.source_futures.get(source_idx, [])
            source_results = self.source_results.get(source_idx, [])
            should_complete = (
                source_idx in self.closed_sources
                and all(item.done() and item.exception() is None for item in source_futures)
                and len(source_results) == len(source_futures)
            )
        if should_complete:
            self._mark_complete(source_idx)

    def finish_source(self, source_idx: int) -> None:
        with self.lock:
            self.closed_sources.add(source_idx)
            source_futures = self.source_futures.get(source_idx, [])
            source_results = self.source_results.get(source_idx, [])
            should_complete = not source_futures or (
                len(source_results) == len(source_futures)
                and all(item.done() and item.exception() is None for item in source_futures)
            )
        if should_complete:
            self._mark_complete(source_idx)

    def _mark_complete(self, source_idx: int) -> None:
        with self.lock:
            if source_idx in self.marked_sources:
                return
            self.marked_sources.add(source_idx)
            rows = sorted(self.source_results.pop(source_idx, []), key=lambda item: item[0])
        try:
            self.store.commit_source(source_idx, rows)
        except BaseException as exc:
            with self.lock:
                self.marked_sources.discard(source_idx)
                self.errors.append(exc)
            try:
                self.store.mark_failed(source_idx, exc)
            except BaseException:
                pass

    def mark_failed(self, source_idx: int, error: BaseException) -> None:
        self.store.mark_failed(source_idx, error)

    def close(self) -> None:
        with self.lock:
            pending = list(self.futures)
        if pending:
            wait(pending)
        self.executor.shutdown(wait=True)
        if self.errors:
            raise self.errors[0]


def reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
