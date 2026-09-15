"""재사용, 재개 및 background payload 저장을 위한 build runtime."""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import os
import shutil
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .artifacts import ArtifactWriter


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


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def manifest_sha256(path: Path | None) -> str | None:
    if path is None:
        return None
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
        state_dir: Path,
        config: WriterConfig,
        completed_sources: set[int],
    ) -> None:
        config.validate()
        self.writer = writer
        self.state_dir = state_dir
        self.rows_dir = state_dir / "rows"
        self.sources_dir = state_dir / "sources"
        self.failures_dir = state_dir / "failures"
        self.executor = ThreadPoolExecutor(
            max_workers=config.payload_workers,
            thread_name_prefix="lazetable-payload",
        )
        self.capacity = threading.Semaphore(config.max_pending_writes)
        self.lock = threading.Lock()
        self.futures: set[Future[Any]] = set()
        self.source_futures: dict[int, list[Future[Any]]] = {}
        self.closed_sources: set[int] = set()
        self.marked_sources: set[int] = set(completed_sources)
        self.errors: list[BaseException] = []
        self.seen_keys: set[tuple[Any, ...]] = set()
        self._load_completed_keys(completed_sources)

    def _load_completed_keys(self, completed_sources: set[int]) -> None:
        for source_idx in sorted(completed_sources):
            source_dir = self.rows_dir / f"{source_idx:012d}"
            for path in sorted(source_dir.glob("*.json")):
                row = json.loads(path.read_text(encoding="utf-8"))
                key = tuple(row[field] for field in self.writer.unique_key_fields)
                self.seen_keys.add(key)

    def submit(self, source_idx: int, output_idx: int, row: dict[str, Any]) -> None:
        key = self.writer.extract_unique_key(row)
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

    def _write_row(self, source_idx: int, output_idx: int, row: dict[str, Any]) -> None:
        lowered = self.writer.lower_row(row, row_idx=output_idx)
        target = self.rows_dir / f"{source_idx:012d}" / f"{output_idx:08d}.json"
        atomic_write_json(target, lowered)

    def _on_done(self, source_idx: int, future: Future[Any]) -> None:
        self.capacity.release()
        error = future.exception()
        with self.lock:
            self.futures.discard(future)
            if error is not None:
                self.errors.append(error)
            should_complete = source_idx in self.closed_sources and all(
                item.done() and item.exception() is None
                for item in self.source_futures.get(source_idx, [])
            )
        if should_complete:
            self._mark_complete(source_idx)

    def finish_source(self, source_idx: int) -> None:
        with self.lock:
            self.closed_sources.add(source_idx)
            source_futures = self.source_futures.get(source_idx, [])
            should_complete = all(
                item.done() and item.exception() is None for item in source_futures
            )
        if should_complete:
            self._mark_complete(source_idx)

    def _mark_complete(self, source_idx: int) -> None:
        with self.lock:
            if source_idx in self.marked_sources:
                return
            self.marked_sources.add(source_idx)
        atomic_write_json(
            self.sources_dir / f"{source_idx:012d}.json",
            {"source_row_idx": source_idx, "status": "completed"},
        )
        failure = self.failures_dir / f"{source_idx:012d}.json"
        failure.unlink(missing_ok=True)

    def mark_failed(self, source_idx: int, error: BaseException) -> None:
        atomic_write_json(
            self.failures_dir / f"{source_idx:012d}.json",
            {
                "source_row_idx": source_idx,
                "status": "failed",
                "error_type": type(error).__name__,
                "message": str(error),
            },
        )

    def close(self) -> None:
        with self.lock:
            pending = list(self.futures)
        if pending:
            wait(pending)
        self.executor.shutdown(wait=True)
        if self.errors:
            raise self.errors[0]


def completed_source_indices(state_dir: Path) -> set[int]:
    sources_dir = state_dir / "sources"
    if not sources_dir.is_dir():
        return set()
    return {int(path.stem) for path in sources_dir.glob("*.json")}


def failed_source_indices(state_dir: Path) -> set[int]:
    failures_dir = state_dir / "failures"
    if not failures_dir.is_dir():
        return set()
    return {int(path.stem) for path in failures_dir.glob("*.json")}


def read_lowered_rows(state_dir: Path, completed_sources: set[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_idx in sorted(completed_sources):
        source_dir = state_dir / "rows" / f"{source_idx:012d}"
        rows.extend(
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(source_dir.glob("*.json"))
        )
    return rows


def reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
