"""Transform contracts and builder-driven artifact execution."""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .artifacts import (
    ArtifactDependency,
    ArtifactWriter,
    INTERNAL_SOURCE_ROW_IDX,
    serialize_dependencies,
)
from .builds import (
    BuildPolicy,
    BuildResult,
    StreamingArtifactWriter,
    WriterConfig,
    build_signature,
    reset_directory,
    stable_hash,
)
from .core import LazyTable, ParsedRow, SandboxRegistry
from .sqlite_store import SQLiteArtifactStore

MutableBuildContext = MutableMapping[str, Any]
BuildContextView = Mapping[str, Any]


class BatchCardinalityError(ValueError):
    """Raised when transform_batch changes item cardinality."""


class MissingSandboxKeyError(KeyError):
    """Raised when a builder requires sandbox keys that are not registered."""


@dataclass(frozen=True)
class SeedPlan:
    """Deterministic seed container for staged transforms.

    This is intentionally small in the first draft. More detailed per-row and
    per-batch derivation should be added once the queued executor is finalized.
    """

    global_seed: int | None = None


@dataclass(frozen=True)
class ExecutorConfig:
    """Queued transform executor settings.

    Notes:
    - `row_executor="process"` requires the transform object and its
      dependencies to be pickleable under the platform's multiprocessing
      start method.
    """

    row_workers: int = 1
    batch_size: int = 32
    max_inflight: int | None = None
    row_executor: str = "process"


@dataclass(frozen=True)
class FinalizeResult:
    rows: list[dict[str, Any]]
    metadata: dict[str, Any]
    statistics: dict[str, dict[str, Any]]


class ArtifactBuilderBase:
    """Base class for four-stage artifact builders.

    The build contract is:
    1. `prepare_table`
    2. `transform_row`
    3. `transform_batch`
    4. payload lowering/save
    5. `finalize_table`

    Notes:
    - `transform_batch` is intended to be cardinality-preserving. A stronger
      enforcement pass will be added in a follow-up step.
    - `unique_keys` is required and is treated as a declaration contract for
      artifact identity. Duplicate enforcement is intentionally deferred to the
      next implementation step.
    """

    artifact_name: str = ""
    artifact_version: str = ""
    unique_keys: list[tuple[str, ...]] = []
    dependencies: list[ArtifactDependency] = []
    required_sandbox_keys: tuple[str, ...] = ()
    executor_config: ExecutorConfig = ExecutorConfig()

    @classmethod
    def validate_definition(cls) -> None:
        if not cls.artifact_name:
            raise ValueError(f"{cls.__name__} must declare artifact_name")
        if not cls.artifact_version:
            raise ValueError(f"{cls.__name__} must declare artifact_version")
        if not cls.unique_keys:
            raise ValueError(f"{cls.__name__} must declare unique_keys")
        for key in cls.unique_keys:
            if not key:
                raise ValueError(f"{cls.__name__} has an empty unique key declaration")

    @classmethod
    def primary_unique_key(cls) -> tuple[str, ...]:
        cls.validate_definition()
        return cls.unique_keys[0]

    @classmethod
    def validate_sandbox_registry(cls, sandbox_registry: SandboxRegistry | None) -> None:
        if not cls.required_sandbox_keys:
            return
        if sandbox_registry is None:
            raise MissingSandboxKeyError(
                f"{cls.__name__} requires sandbox keys {cls.required_sandbox_keys}, but no sandbox_registry was provided"
            )
        available = set(sandbox_registry.names())
        missing = [key for key in cls.required_sandbox_keys if key not in available]
        if missing:
            raise MissingSandboxKeyError(
                f"{cls.__name__} is missing required sandbox keys: {missing}. available={sorted(available)}"
            )

    def prepare_table(
        self,
        *,
        parent_rows: list[ParsedRow] | None = None,
        build_context: MutableBuildContext | None = None,
        sandbox_registry: SandboxRegistry | None = None,
        seed_plan: SeedPlan | None = None,
    ) -> list[dict[str, Any]]:
        return list(parent_rows or [])

    def transform_row(
        self,
        row: dict[str, Any],
        *,
        row_idx: int,
        parent_rows: list[ParsedRow],
        build_context: BuildContextView,
        sandbox_registry: SandboxRegistry | None,
        seed_plan: SeedPlan | None = None,
    ) -> Any:
        return row

    def transform_batch(
        self,
        items: list[dict[str, Any]],
        *,
        batch_idx: int,
        parent_rows: list[ParsedRow],
        build_context: BuildContextView,
        sandbox_registry: SandboxRegistry | None,
        seed_plan: SeedPlan | None = None,
    ) -> list[dict[str, Any]]:
        return [dict(item) for item in items]

    def finalize_manifest(
        self,
        rows: list[dict[str, Any]],
        *,
        parent_rows: list[ParsedRow],
        build_context: MutableBuildContext,
        sandbox_registry: SandboxRegistry | None,
        seed_plan: SeedPlan | None = None,
    ) -> FinalizeResult:
        return FinalizeResult(rows=rows, metadata={}, statistics={})

    def signature_data(self, *, build_context: BuildContextView) -> dict[str, Any]:
        """Builder가 build signature에 추가할 source fingerprint를 반환한다."""

        return {}

    def artifact_dir(self, storage_root: Path) -> Path:
        return Path(storage_root) / self.artifact_name / self.artifact_version

    def build(
        self,
        *,
        storage_root: str | Path,
        parent_tables: list[LazyTable] | None = None,
        build_context: MutableBuildContext | None = None,
        sandbox_registry: SandboxRegistry | None = None,
        policy: BuildPolicy,
        seed_plan: SeedPlan | None = None,
        transform_limit: int | None = None,
        executor_config: ExecutorConfig | None = None,
        writer_config: WriterConfig | None = None,
    ) -> BuildResult:
        self.validate_definition()
        self.validate_sandbox_registry(sandbox_registry)
        storage_root_path = Path(storage_root)
        out_dir = self.artifact_dir(storage_root_path)
        staging_dir = out_dir.with_name(f".{out_dir.name}.building")
        parent_tables_list = list(parent_tables or [])
        parent_rows: list[ParsedRow] = []
        for table in parent_tables_list:
            parent_rows.extend(list(table))
        context = dict(build_context or {})
        cfg = executor_config or self.executor_config
        signature, signature_payload = build_signature(
            builder=self,
            build_context=context,
            parent_tables=parent_tables_list,
            seed_plan=seed_plan,
            runtime_parameters={
                "transform_limit": transform_limit,
                "executor_config": {
                    "row_workers": cfg.row_workers,
                    "batch_size": cfg.batch_size,
                    "max_inflight": cfg.max_inflight,
                    "row_executor": cfg.row_executor,
                },
            },
        )
        if out_dir.exists():
            completed_store = SQLiteArtifactStore(out_dir / "artifact.sqlite")
            completed_store.validate_schema(read_only=True)
            artifact = completed_store.artifact(read_only=True)
            if artifact is None or artifact.status != "completed":
                raise ValueError(f"completed artifact metadata is invalid: {out_dir}")
            if policy.reuse_completed:
                if artifact.build_signature != signature:
                    raise ValueError(f"completed artifact signature mismatch: {out_dir}")
                return BuildResult(
                    table=LazyTable.open(out_dir, sandbox_registry=sandbox_registry),
                    status="reused",
                    processed_source_rows=0,
                    skipped_source_rows=artifact.source_row_count,
                    failed_source_rows=0,
                )
            reset_directory(out_dir)
            out_dir.rmdir()

        if staging_dir.exists():
            if policy.resume_partial:
                partial_store = SQLiteArtifactStore(staging_dir / "artifact.sqlite")
                partial_store.validate_schema()
                artifact = partial_store.artifact()
                if artifact is None or artifact.build_signature != signature:
                    raise ValueError(f"partial artifact signature mismatch: {staging_dir}")
            else:
                reset_directory(staging_dir)
        else:
            staging_dir.mkdir(parents=True)

        store = SQLiteArtifactStore(staging_dir / "artifact.sqlite")
        dependency_records = []
        for position, dependency in enumerate(serialize_dependencies(self.dependencies)):
            parent_hash = None
            if position < len(parent_tables_list):
                parent = parent_tables_list[position]
                if parent.path is not None:
                    parent_artifact = SQLiteArtifactStore(parent.path / "artifact.sqlite").artifact(
                        read_only=True
                    )
                    parent_hash = None if parent_artifact is None else parent_artifact.content_hash
            dependency_records.append({**dependency, "manifest_hash": parent_hash})
        store.initialize_build(
            artifact_name=self.artifact_name,
            artifact_version=self.artifact_version,
            build_signature=signature,
            signature_payload=signature_payload,
            dependencies=dependency_records,
        )

        prepared_rows = self.prepare_table(
            parent_rows=parent_rows,
            build_context=context,
            sandbox_registry=sandbox_registry,
            seed_plan=seed_plan,
        )
        if transform_limit is not None:
            prepared_rows = prepared_rows[: max(0, int(transform_limit))]
        store.ensure_source_rows(len(prepared_rows))
        store.remove_orphan_payloads(staging_dir)
        completed = store.completed_sources()
        failures = store.failed_sources()
        if failures and not policy.retry_failed:
            raise RuntimeError(f"partial build contains failed source rows: {sorted(failures)}")
        for failed_idx in failures:
            store.clear_failure(failed_idx)

        artifact_roots = {
            "default": str(storage_root_path),
            f"{self.artifact_name}@{self.artifact_version}": str(out_dir),
        }
        payload_writer = StreamingArtifactWriter(
            writer=ArtifactWriter(
                artifact_name=self.artifact_name,
                artifact_version=self.artifact_version,
                out_dir=staging_dir,
                artifact_roots=artifact_roots,
                unique_key_fields=self.primary_unique_key(),
            ),
            store=store,
            config=writer_config or WriterConfig(),
        )
        processed: set[int] = set()
        try:
            with tqdm(
                total=len(prepared_rows),
                initial=len(completed),
                desc="artifact_build",
                unit="row",
            ) as progress:

                def emit(source_idx: int, output_idx: int, row: dict[str, Any]) -> None:
                    payload_writer.submit(source_idx, output_idx, row)

                def finish(source_idx: int) -> None:
                    payload_writer.finish_source(source_idx)
                    processed.add(source_idx)
                    progress.update(1)

                _run_streaming_pipeline(
                    self,
                    prepared_rows,
                    parent_rows=parent_rows,
                    build_context=context,
                    sandbox_registry=sandbox_registry,
                    seed_plan=seed_plan,
                    executor_config=cfg,
                    skip_source_indices=completed,
                    emit=emit,
                    finish_source=finish,
                    fail_source=payload_writer.mark_failed,
                )
            payload_writer.close()
        except BaseException:
            try:
                payload_writer.close()
            except BaseException:
                pass
            raise

        final_completed = store.completed_sources()
        if len(final_completed) != len(prepared_rows):
            raise RuntimeError(
                "build did not complete every source row: "
                f"completed={len(final_completed)}, total={len(prepared_rows)}"
            )
        store.set_phase("finalizing")
        finalize_result = self.finalize_manifest(
            store.raw_rows(),
            parent_rows=parent_rows,
            build_context=context,
            sandbox_registry=sandbox_registry,
            seed_plan=seed_plan,
        )
        rows_with_keys = []
        for row in finalize_result.rows:
            keyed = dict(row)
            keyed["__unique_key__"] = [keyed[field] for field in self.primary_unique_key()]
            rows_with_keys.append(keyed)
        store.set_phase("validating")
        store.validate_payloads(staging_dir, verify_hash=True)
        for position, parent in enumerate(parent_tables_list):
            if parent.path is None or position >= len(dependency_records):
                continue
            parent_artifact = SQLiteArtifactStore(parent.path / "artifact.sqlite").artifact(
                read_only=True
            )
            expected_hash = dependency_records[position].get("manifest_hash")
            actual_hash = None if parent_artifact is None else parent_artifact.content_hash
            if parent_artifact is None or parent_artifact.status != "completed":
                raise ValueError(f"parent artifact is not completed: {parent.path}")
            if expected_hash != actual_hash:
                raise ValueError(f"parent artifact changed during build: {parent.path}")
        content_hash = stable_hash(rows_with_keys)
        store.set_phase("committing")
        store.finalize(
            rows=rows_with_keys,
            metadata={
                **finalize_result.metadata,
                "dependencies": serialize_dependencies(self.dependencies),
                "artifact_roots": artifact_roots,
                "build_signature": signature,
            },
            statistics=finalize_result.statistics,
            content_hash=content_hash,
        )
        store.checkpoint_and_close()
        os.replace(staging_dir, out_dir)
        return BuildResult(
            table=LazyTable.open(out_dir, sandbox_registry=sandbox_registry),
            status="resumed" if completed else "built",
            processed_source_rows=len(processed),
            skipped_source_rows=len(completed),
            failed_source_rows=0,
        )


def _process_worker_transform_row(
    transform: ArtifactBuilderBase,
    row: dict[str, Any],
    row_idx: int,
    parent_rows: list[ParsedRow],
    build_context: BuildContextView,
    sandbox_registry: SandboxRegistry | None,
    seed_plan: SeedPlan | None,
) -> Any:
    """Top-level worker entrypoint for process-based row execution."""
    return transform.transform_row(
        row,
        row_idx=row_idx,
        parent_rows=parent_rows,
        build_context=build_context,
        sandbox_registry=sandbox_registry,
        seed_plan=seed_plan,
    )


_PROCESS_WORKER_CONTEXT: tuple[Any, ...] | None = None


def _initialize_process_worker(
    transform: ArtifactBuilderBase,
    parent_rows: list[ParsedRow],
    build_context: BuildContextView,
    sandbox_registry: SandboxRegistry | None,
    seed_plan: SeedPlan | None,
) -> None:
    global _PROCESS_WORKER_CONTEXT
    _PROCESS_WORKER_CONTEXT = (
        transform,
        parent_rows,
        build_context,
        sandbox_registry,
        seed_plan,
    )


def _process_initialized_row(row: dict[str, Any], row_idx: int) -> Any:
    if _PROCESS_WORKER_CONTEXT is None:
        raise RuntimeError("process worker context was not initialized")
    transform, parent_rows, build_context, sandbox_registry, seed_plan = _PROCESS_WORKER_CONTEXT
    return transform.transform_row(
        row,
        row_idx=row_idx,
        parent_rows=parent_rows,
        build_context=build_context,
        sandbox_registry=sandbox_registry,
        seed_plan=seed_plan,
    )


def _run_streaming_pipeline(
    transform: ArtifactBuilderBase,
    rows: list[dict[str, Any]],
    *,
    parent_rows: list[ParsedRow],
    build_context: BuildContextView,
    sandbox_registry: SandboxRegistry | None,
    seed_plan: SeedPlan | None,
    executor_config: ExecutorConfig,
    skip_source_indices: set[int],
    emit: Any,
    finish_source: Any,
    fail_source: Any,
) -> None:
    """변환 결과를 누적하지 않고 bounded writer로 전달한다."""

    row_workers = int(executor_config.row_workers)
    worker_pool_size = max(1, row_workers)
    batch_size = max(1, int(executor_config.batch_size))
    max_inflight = int(executor_config.max_inflight or max(worker_pool_size * 2, 1))
    if executor_config.row_executor not in {"thread", "process"}:
        raise ValueError("row_executor must be 'thread' or 'process'")

    ready_items: list[dict[str, Any]] = []
    remaining_by_source: dict[int, int] = {}
    output_ordinal: dict[int, int] = {}
    batch_idx = 0

    def drain(*, flush: bool = False) -> None:
        nonlocal batch_idx, ready_items
        while len(ready_items) >= batch_size or (flush and ready_items):
            current = ready_items[:batch_size]
            ready_items = ready_items[len(current) :]
            try:
                batch_rows = transform.transform_batch(
                    current,
                    batch_idx=batch_idx,
                    parent_rows=parent_rows,
                    build_context=build_context,
                    sandbox_registry=sandbox_registry,
                    seed_plan=seed_plan,
                )
            except BaseException as exc:
                for source_idx in {int(item[INTERNAL_SOURCE_ROW_IDX]) for item in current}:
                    fail_source(source_idx, exc)
                raise
            if len(batch_rows) != len(current):
                error = BatchCardinalityError(
                    "transform_batch must preserve item cardinality exactly: "
                    f"input={len(current)}, output={len(batch_rows)}, batch_idx={batch_idx}"
                )
                for source_idx in {int(item[INTERNAL_SOURCE_ROW_IDX]) for item in current}:
                    fail_source(source_idx, error)
                raise error
            for original, result in zip(current, batch_rows, strict=True):
                source_idx = int(original[INTERNAL_SOURCE_ROW_IDX])
                normalized = _attach_source_row_idx(result, source_row_idx=source_idx)
                ordinal = output_ordinal.get(source_idx, 0)
                output_ordinal[source_idx] = ordinal + 1
                emit(source_idx, ordinal, normalized)
                remaining_by_source[source_idx] -= 1
                if remaining_by_source[source_idx] == 0:
                    finish_source(source_idx)
            batch_idx += 1

    def accept(source_idx: int, value: Any) -> None:
        items = _normalize_stage_items(value, source_row_idx=source_idx)
        remaining_by_source[source_idx] = len(items)
        if not items:
            finish_source(source_idx)
            return
        ready_items.extend(items)
        drain()

    if row_workers <= 0:
        for source_idx, row in enumerate(rows):
            if source_idx in skip_source_indices:
                continue
            try:
                value = transform.transform_row(
                    row,
                    row_idx=source_idx,
                    parent_rows=parent_rows,
                    build_context=build_context,
                    sandbox_registry=sandbox_registry,
                    seed_plan=seed_plan,
                )
            except BaseException as exc:
                fail_source(source_idx, exc)
                raise
            accept(source_idx, value)
        drain(flush=True)
        return

    pending_by_row_idx: dict[int, Any] = {}
    futures: dict[Future[Any], int] = {}
    next_submit_idx = 0
    next_release_idx = 0

    def submit_next(pool: ThreadPoolExecutor | ProcessPoolExecutor) -> bool:
        nonlocal next_submit_idx
        while next_submit_idx < len(rows) and next_submit_idx in skip_source_indices:
            next_submit_idx += 1
        if next_submit_idx >= len(rows):
            return False
        source_idx = next_submit_idx
        row = rows[source_idx]
        if executor_config.row_executor == "thread":
            future = pool.submit(
                transform.transform_row,
                row,
                row_idx=source_idx,
                parent_rows=parent_rows,
                build_context=build_context,
                sandbox_registry=sandbox_registry,
                seed_plan=seed_plan,
            )
        else:
            future = pool.submit(
                _process_initialized_row,
                row,
                source_idx,
            )
        futures[future] = source_idx
        next_submit_idx += 1
        return True

    def release_ready() -> None:
        nonlocal next_release_idx
        while next_release_idx < len(rows):
            if next_release_idx in skip_source_indices:
                next_release_idx += 1
                continue
            if next_release_idx not in pending_by_row_idx:
                break
            accept(next_release_idx, pending_by_row_idx.pop(next_release_idx))
            next_release_idx += 1

    if executor_config.row_executor == "thread":
        pool = ThreadPoolExecutor(max_workers=worker_pool_size)
    else:
        pool = ProcessPoolExecutor(
            max_workers=worker_pool_size,
            initializer=_initialize_process_worker,
            initargs=(
                transform,
                parent_rows,
                build_context,
                sandbox_registry,
                seed_plan,
            ),
        )
    with pool:
        while len(futures) < max_inflight and submit_next(pool):
            pass
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                source_idx = futures.pop(future)
                try:
                    pending_by_row_idx[source_idx] = future.result()
                except BaseException as exc:
                    fail_source(source_idx, exc)
                    raise
            release_ready()
            while len(futures) < max_inflight and submit_next(pool):
                pass
    release_ready()
    drain(flush=True)


def _run_row_batch_pipeline(
    transform: ArtifactBuilderBase,
    rows: list[dict[str, Any]],
    *,
    parent_rows: list[ParsedRow],
    build_context: BuildContextView,
    sandbox_registry: SandboxRegistry | None,
    seed_plan: SeedPlan | None,
    executor_config: ExecutorConfig,
    row_completed_callback: Any | None = None,
) -> list[dict[str, Any]]:
    row_workers = int(executor_config.row_workers)
    worker_pool_size = max(1, row_workers)
    batch_size = max(1, int(executor_config.batch_size))
    max_inflight = int(executor_config.max_inflight or max(worker_pool_size * 2, 1))

    if executor_config.row_executor not in {"thread", "process"}:
        raise ValueError("row_executor must be 'thread' or 'process'")

    if row_workers <= 0:
        staged_items: list[Any] = []
        out_rows: list[dict[str, Any]] = []
        batch_idx = 0
        for row_idx, row in enumerate(rows):
            staged = transform.transform_row(
                row,
                row_idx=row_idx,
                parent_rows=parent_rows,
                build_context=build_context,
                sandbox_registry=sandbox_registry,
                seed_plan=seed_plan,
            )
            normalized_items = _normalize_stage_items(staged, source_row_idx=row_idx)
            if not normalized_items:
                if row_completed_callback is not None:
                    row_completed_callback(row_idx)
                continue
            for item in normalized_items:
                staged_items.append(item)
                if len(staged_items) >= batch_size:
                    current = staged_items[:batch_size]
                    batch_rows = transform.transform_batch(
                        current,
                        batch_idx=batch_idx,
                        parent_rows=parent_rows,
                        build_context=build_context,
                        sandbox_registry=sandbox_registry,
                        seed_plan=seed_plan,
                    )
                    if len(batch_rows) != len(current):
                        raise BatchCardinalityError(
                            "transform_batch must preserve item cardinality exactly: "
                            f"input={len(current)}, output={len(batch_rows)}, batch_idx={batch_idx}"
                        )
                    out_rows.extend(batch_rows)
                    batch_idx += 1
                    staged_items = staged_items[batch_size:]
        if staged_items:
            batch_rows = transform.transform_batch(
                staged_items,
                batch_idx=batch_idx,
                parent_rows=parent_rows,
                build_context=build_context,
                sandbox_registry=sandbox_registry,
                seed_plan=seed_plan,
            )
            if len(batch_rows) != len(staged_items):
                raise BatchCardinalityError(
                    "transform_batch must preserve item cardinality exactly: "
                    f"input={len(staged_items)}, output={len(batch_rows)}, batch_idx={batch_idx}"
                )
            out_rows.extend(batch_rows)
        return out_rows

    out_rows: list[dict[str, Any]] = []
    next_submit_idx = 0
    next_release_idx = 0
    batch_idx = 0
    pending_by_row_idx: dict[int, list[Any]] = {}
    ready_items: list[Any] = []
    futures: dict[Future[Any], int] = {}

    def _submit_next_thread(pool: ThreadPoolExecutor) -> bool:
        nonlocal next_submit_idx
        if next_submit_idx >= len(rows):
            return False
        row_idx = next_submit_idx
        row = rows[row_idx]
        future = pool.submit(
            transform.transform_row,
            row,
            row_idx=row_idx,
            parent_rows=parent_rows,
            build_context=build_context,
            sandbox_registry=sandbox_registry,
            seed_plan=seed_plan,
        )
        futures[future] = row_idx
        next_submit_idx += 1
        return True

    def _submit_next_process(pool: ProcessPoolExecutor) -> bool:
        nonlocal next_submit_idx
        if next_submit_idx >= len(rows):
            return False
        row_idx = next_submit_idx
        row = rows[row_idx]
        future = pool.submit(
            _process_worker_transform_row,
            transform,
            row,
            row_idx,
            parent_rows,
            build_context,
            sandbox_registry,
            seed_plan,
        )
        futures[future] = row_idx
        next_submit_idx += 1
        return True

    def _release_ready_rows() -> None:
        nonlocal next_release_idx
        while next_release_idx in pending_by_row_idx:
            ready_items.extend(pending_by_row_idx.pop(next_release_idx))
            next_release_idx += 1

    def _drain_ready_batches() -> None:
        nonlocal batch_idx, ready_items
        while len(ready_items) >= batch_size:
            current = ready_items[:batch_size]
            ready_items = ready_items[batch_size:]
            batch_rows = transform.transform_batch(
                current,
                batch_idx=batch_idx,
                parent_rows=parent_rows,
                build_context=build_context,
                sandbox_registry=sandbox_registry,
                seed_plan=seed_plan,
            )
            if len(batch_rows) != len(current):
                raise BatchCardinalityError(
                    "transform_batch must preserve item cardinality exactly: "
                    f"input={len(current)}, output={len(batch_rows)}, batch_idx={batch_idx}"
                )
            out_rows.extend(batch_rows)
            batch_idx += 1

    if executor_config.row_executor == "thread":
        with ThreadPoolExecutor(max_workers=worker_pool_size) as pool:
            while len(futures) < min(len(rows), max_inflight) and _submit_next_thread(pool):
                pass

            while futures:
                done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
                for future in done:
                    row_idx = futures.pop(future)
                    staged = future.result()
                    normalized_items = _normalize_stage_items(staged, source_row_idx=row_idx)
                    pending_by_row_idx[row_idx] = normalized_items
                    if not normalized_items and row_completed_callback is not None:
                        row_completed_callback(row_idx)
                _release_ready_rows()
                _drain_ready_batches()
                while len(futures) < max_inflight and _submit_next_thread(pool):
                    pass
    else:
        with ProcessPoolExecutor(max_workers=worker_pool_size) as pool:
            while len(futures) < min(len(rows), max_inflight) and _submit_next_process(pool):
                pass

            while futures:
                done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
                for future in done:
                    row_idx = futures.pop(future)
                    staged = future.result()
                    normalized_items = _normalize_stage_items(staged, source_row_idx=row_idx)
                    pending_by_row_idx[row_idx] = normalized_items
                    if not normalized_items and row_completed_callback is not None:
                        row_completed_callback(row_idx)
                _release_ready_rows()
                _drain_ready_batches()
                while len(futures) < max_inflight and _submit_next_process(pool):
                    pass

    _release_ready_rows()
    if ready_items:
        batch_rows = transform.transform_batch(
            ready_items,
            batch_idx=batch_idx,
            parent_rows=parent_rows,
            build_context=build_context,
            sandbox_registry=sandbox_registry,
            seed_plan=seed_plan,
        )
        if len(batch_rows) != len(ready_items):
            raise BatchCardinalityError(
                "transform_batch must preserve item cardinality exactly: "
                f"input={len(ready_items)}, output={len(batch_rows)}, batch_idx={batch_idx}"
            )
        out_rows.extend(batch_rows)
    return out_rows


def _normalize_stage_items(value: Any, *, source_row_idx: int) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, list):
        return [_attach_source_row_idx(item, source_row_idx=source_row_idx) for item in value]
    return [_attach_source_row_idx(value, source_row_idx=source_row_idx)]


def _attach_source_row_idx(item: Any, *, source_row_idx: int) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise TypeError(
            "transform_row must return None, dict[str, Any], or list[dict[str, Any]]: "
            f"got {type(item).__name__}"
        )
    out = dict(item)
    out.setdefault(INTERNAL_SOURCE_ROW_IDX, int(source_row_idx))
    return out
