"""Transform contracts and builder-driven artifact execution."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any

from tqdm import tqdm

from .artifacts import ArtifactDependency, INTERNAL_SOURCE_ROW_IDX, serialize_dependencies, write_artifact
from .core import LazyTable, ParsedRow, SandboxRegistry

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

    def finalize_table(
        self,
        rows: list[dict[str, Any]],
        *,
        parent_rows: list[ParsedRow],
        build_context: MutableBuildContext,
        sandbox_registry: SandboxRegistry | None,
        seed_plan: SeedPlan | None = None,
    ) -> list[dict[str, Any]]:
        return rows

    def artifact_dir(self, storage_root: Path) -> Path:
        return Path(storage_root) / self.artifact_name / self.artifact_version

    def _build_to_directory(
        self,
        *,
        parent_rows: list[ParsedRow],
        out_dir: Path,
        storage_root: Path,
        seed_plan: SeedPlan | None = None,
        transform_limit: int | None = None,
        executor_config: ExecutorConfig,
        build_context: MutableBuildContext,
        sandbox_registry: SandboxRegistry | None,
    ) -> LazyTable:
        """Build an artifact via the staged transform contract."""

        prepared_rows = self.prepare_table(
            parent_rows=parent_rows,
            build_context=build_context,
            sandbox_registry=sandbox_registry,
            seed_plan=seed_plan,
        )
        if transform_limit is not None:
            prepared_rows = prepared_rows[: max(0, int(transform_limit))]
        completed_source_rows: set[int] = set()

        with tqdm(total=len(prepared_rows), desc="artifact_build", unit="row") as pbar:
            def _progress_complete(row_idx: int) -> None:
                if row_idx in completed_source_rows:
                    return
                completed_source_rows.add(int(row_idx))
                pbar.update(1)

            out_rows = _run_row_batch_pipeline(
                self,
                prepared_rows,
                parent_rows=parent_rows,
                build_context=build_context,
                sandbox_registry=sandbox_registry,
                seed_plan=seed_plan,
                executor_config=executor_config,
                row_completed_callback=_progress_complete,
            )

            out_rows = self.finalize_table(
                out_rows,
                parent_rows=parent_rows,
                build_context=build_context,
                sandbox_registry=sandbox_registry,
                seed_plan=seed_plan,
            )

            result = write_artifact(
                artifact_name=str(self.artifact_name),
                artifact_version=str(self.artifact_version),
                rows=out_rows,
                out_dir=out_dir,
                table_meta={},
                meta={
                    "dependencies": serialize_dependencies(self.dependencies),
                },
                artifact_roots={"default": str(storage_root)},
                row_written_callback=_progress_complete,
                unique_key_fields=self.primary_unique_key(),
            )
            if pbar.n < pbar.total:
                remaining = [idx for idx in range(len(prepared_rows)) if idx not in completed_source_rows]
                if remaining:
                    pbar.update(len(remaining))
            if sandbox_registry is not None:
                return LazyTable.from_table(
                    result.table.table,
                    artifact_id=result.table.artifact_id,
                    path=result.table.path,
                    meta=result.table.meta,
                    payload_manager=result.table._payload_manager,  # type: ignore[attr-defined]
                    row_view_manager=result.table._row_view_manager,  # type: ignore[attr-defined]
                    resolver_context=result.table.resolver_context.__class__(
                        sandbox_registry=sandbox_registry,
                        artifact_roots=result.table.resolver_context.artifact_roots,
                    ),
                    context=result.table.context,
                )
        return result.table

    def build(
        self,
        *,
        storage_root: str | Path,
        parent_tables: list[LazyTable] | None = None,
        build_context: MutableBuildContext | None = None,
        sandbox_registry: SandboxRegistry | None = None,
        overwrite: bool = False,
        seed_plan: SeedPlan | None = None,
        transform_limit: int | None = None,
        batch_size: int | None = None,
        executor_config: ExecutorConfig | None = None,
    ) -> LazyTable:
        self.validate_definition()
        self.validate_sandbox_registry(sandbox_registry)
        storage_root_path = Path(storage_root)
        out_dir = self.artifact_dir(storage_root_path)
        if out_dir.exists():
            if not overwrite:
                raise FileExistsError(f"artifact already exists: {out_dir}")
            shutil.rmtree(out_dir)
        parent_tables_list = list(parent_tables or [])
        parent_rows: list[ParsedRow] = []
        for table in parent_tables_list:
            parent_rows.extend(list(table))
        cfg = executor_config or self.executor_config
        if batch_size is not None and int(batch_size) != cfg.batch_size:
            cfg = ExecutorConfig(
                row_workers=cfg.row_workers,
                batch_size=int(batch_size),
                max_inflight=cfg.max_inflight,
                row_executor=cfg.row_executor,
            )
        return self._build_to_directory(
            parent_rows=parent_rows,
            out_dir=out_dir,
            storage_root=storage_root_path,
            seed_plan=seed_plan,
            transform_limit=transform_limit,
            executor_config=cfg,
            build_context=dict(build_context or {}),
            sandbox_registry=sandbox_registry,
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
