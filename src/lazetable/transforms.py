"""Transform contracts and builder-driven artifact execution."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, MutableMapping, Sequence
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
    INTERNAL_SOURCE_ROW_IDX,
    ArtifactDependency,
    ArtifactWriter,
    ColumnConflictError,
    JoinKeyError,
    check_field_conflicts,
    joined_parent_fields,
    logical_field,
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
from .grouping import Group, Grouping, ItemMeta, meta_values, plan_groups
from .pipeline import (
    BatchCardinalityError,
    Node,
    pipeline_signature,
    run_pipeline,
    split_stages,
    validate_nodes,
)
from .sqlite_store import SQLiteArtifactStore
from .stages import StageStore, StageUnits

SOURCES_FIELD = "__sources__"  # 결과 행의 원본 행 번호(묶음 구성원을 원본까지 거슬러 올라간 것)
_GROUPING = Grouping()

MutableBuildContext = MutableMapping[str, Any]
BuildContextView = Mapping[str, Any]


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
    - For a builder with `pipeline()`, `row_workers` and `row_executor` set the
      CPU workers; `batch_size` and `max_inflight` are not used. GPU nodes always
      run on one dedicated thread.
    """

    row_workers: int = 1
    batch_size: int = 32
    max_inflight: int | None = None
    row_executor: str = "process"
    stage_cache_mb: int = 256  # 장벽 단계 저장소(SQLite) 페이지 캐시 상한
    keep_stages: bool = False  # 빌드가 끝난 뒤 장벽 단계 저장소(stages.sqlite)를 남길지


class OutputFieldGuard:
    """빌더 출력 행이 부모 조인으로 붙을 열과 겹치지 않는지, 조인 키를 갖는지 검사한다.

    부모 열(``prefix``·``rename`` 적용 후)끼리 겹치면 만들 때 바로 오류다. 출력 행 검사는 행마다
    논리 필드 이름 집합만 본다.
    """

    def __init__(self, dependencies: list[ArtifactDependency], parent_tables: list[LazyTable]) -> None:
        self.added: dict[str, str] = {}  # 붙는 이름 → "부모@판"
        self.required: dict[str, str] = {}  # 자식 조인 키 → "부모@판"
        for position, dependency in enumerate(dependencies):
            if dependency.join is None or position >= len(parent_tables):
                continue
            where = f"{dependency.artifact_name}@{dependency.artifact_version}"
            mapping = joined_parent_fields(parent_tables[position].table.column_names, dependency.join)
            check_field_conflicts(set(self.added), mapping, where=f"부모 {where}와 앞선 부모")
            for new in mapping.values():
                self.added[new] = where
            for key in dependency.join.child_on:
                self.required.setdefault(key, where)

    def check(self, row: Any) -> None:
        if not self.added and not self.required:
            return
        keys = row.keys() if isinstance(row, Mapping) else row.as_dict().keys()
        fields = {logical_field(str(key)) for key in keys if key != INTERNAL_SOURCE_ROW_IDX}
        conflicts = sorted(fields & set(self.added))
        if conflicts:
            detail = ", ".join(f"{name}(부모 {self.added[name]})" for name in conflicts)
            raise ColumnConflictError(
                f"빌더 출력 열이 부모 조인으로 붙는 열과 겹침: {detail}. "
                "출력 이름을 바꾸거나 ArtifactJoin의 prefix·rename을 쓸 것"
            )
        missing = sorted(set(self.required) - fields)
        if missing:
            detail = ", ".join(f"{name}(부모 {self.required[name]})" for name in missing)
            raise JoinKeyError(f"빌더 출력 행에 조인 키가 없음: {detail}")


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
    - A builder may override `pipeline()` to stack CPU/GPU nodes instead of using
      `transform_row` and `transform_batch`.
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

    def pipeline(self) -> list[Node] | None:
        """CPU/GPU 노드 목록. None이면 ``transform_row``와 ``transform_batch``를 쓴다."""

        return None

    @property
    def grouping(self) -> Grouping:
        """노드 ``groups=``에 쓰는 묶음 규칙 도우미(``by_columns``, ``by_list_column``, ``by_key``)."""

        return _GROUPING

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
        output_guard = OutputFieldGuard(list(self.dependencies), parent_tables_list)
        cfg = executor_config or self.executor_config
        nodes = self.pipeline()
        if nodes is not None:
            validate_nodes(nodes)
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
            }
            if nodes is None
            else {"transform_limit": transform_limit, "pipeline": pipeline_signature(nodes)},
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
        parents = parent_table_map(list(self.dependencies), parent_tables_list)

        # 단계 실행: 마지막 단계의 단위(묶음)가 커밋 단위다. 앞 단계(장벽)는 디스크 저장소에 결과를 둔다.
        stages = split_stages(nodes) if nodes is not None else []
        stage_store: StageStore | None = None
        stage_stats: dict[str, dict[str, Any]] = {}
        if nodes is None:
            final_units: Sequence[Any] = prepared_rows
            unit_sources: Callable[[int], tuple[int, ...]] = lambda index: (index,)
        else:
            final_units, unit_sources = _stage_zero_units(nodes[0], prepared_rows, store)
            if len(stages) > 1:
                stage_store = StageStore(staging_dir / "stages.sqlite", cache_mb=cfg.stage_cache_mb)
            try:
                for stage_idx, (offset, stage_nodes) in enumerate(stages[:-1]):
                    assert stage_store is not None
                    stats = _run_barrier_stage(
                        self,
                        stage_idx,
                        offset,
                        stage_nodes,
                        final_units,
                        unit_sources,
                        stage_store=stage_store,
                        policy=policy,
                        parent_rows=parent_rows,
                        parents=parents,
                        build_context=context,
                        sandbox_registry=sandbox_registry,
                        seed_plan=seed_plan,
                        executor_config=cfg,
                    )
                    stage_stats.update(stats)
                    next_spec = stages[stage_idx + 1][1][0].groups
                    planned, plan_hash = plan_groups(next_spec, stage_store.item_metas(stage_idx))
                    store.check_plan(stage_idx + 1, plan_hash, len(planned))
                    stage_units = StageUnits(stage_store, planned)
                    final_units, unit_sources = stage_units, stage_units.sources
            except BaseException:
                if stage_store is not None:
                    stage_store.close()
                raise

        store.ensure_source_rows(len(final_units))
        store.remove_orphan_payloads(staging_dir)
        completed = store.completed_sources()
        failures = store.failed_sources()
        if failures and not policy.retry_failed:
            if stage_store is not None:
                stage_store.close()
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
        node_stats = None
        node_failures: dict[int, BaseException] = {}
        try:
            with tqdm(
                total=len(final_units),
                initial=len(completed),
                desc="artifact_build" if len(stages) < 2 else f"artifact_build:{stages[-1][1][0].name}",
                unit="unit",
            ) as progress:

                def emit(source_idx: int, output_idx: int, row: dict[str, Any]) -> None:
                    output_guard.check(row)
                    row = dict(row)
                    row[SOURCES_FIELD] = list(unit_sources(source_idx))
                    payload_writer.submit(source_idx, output_idx, row)

                def finish(source_idx: int) -> None:
                    payload_writer.finish_source(source_idx)
                    processed.add(source_idx)
                    progress.update(1)

                def fail_node_row(source_idx: int, error: BaseException) -> None:
                    payload_writer.mark_failed(source_idx, error)
                    progress.update(1)

                if nodes is None:
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
                else:
                    offset, stage_nodes = stages[-1]
                    node_stats, node_failures = run_pipeline(
                        self,
                        stage_nodes,
                        final_units,
                        parent_rows=parent_rows,
                        build_context=context,
                        sandbox_registry=sandbox_registry,
                        seed_plan=seed_plan,
                        cpu_workers=int(cfg.row_workers),
                        cpu_executor=cfg.row_executor,
                        skip_source_indices=completed,
                        emit=emit,
                        finish_source=finish,
                        fail_source=fail_node_row,
                        node_offset=offset,
                        parents=parents,
                    )
                    node_stats = {**stage_stats, **node_stats}
            payload_writer.close()
        except BaseException:
            try:
                payload_writer.close()
            except BaseException:
                pass
            if stage_store is not None:
                stage_store.close()
            raise

        try:
            if node_failures:
                first = min(node_failures)
                raise RuntimeError(
                    f"{len(node_failures)} units failed; first is unit {first}: "
                    f"{type(node_failures[first]).__name__}: {node_failures[first]}"
                )
            final_completed = store.completed_sources()
            if len(final_completed) != len(final_units):
                raise RuntimeError(
                    "build did not complete every unit: "
                    f"completed={len(final_completed)}, total={len(final_units)}"
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
            if stage_store is not None:
                stage_store.close()
                if not cfg.keep_stages:
                    for suffix in ("", "-wal", "-shm"):
                        stage_path = staging_dir / f"stages.sqlite{suffix}"
                        if stage_path.exists():
                            stage_path.unlink()
        except BaseException:
            if stage_store is not None:
                stage_store.close()
            raise
        os.replace(staging_dir, out_dir)
        return BuildResult(
            table=LazyTable.open(out_dir, sandbox_registry=sandbox_registry),
            status="resumed" if completed else "built",
            processed_source_rows=len(processed),
            skipped_source_rows=len(completed),
            failed_source_rows=0,
            node_stats=node_stats,
        )


def parent_table_map(
    dependencies: list[ArtifactDependency], parent_tables: list[LazyTable]
) -> dict[str, LazyTable]:
    """부모 표를 이름으로: ``"이름@판"``과, 이름이 하나뿐이면 ``"이름"``."""

    out: dict[str, LazyTable] = {}
    for position, table in enumerate(parent_tables):
        if position < len(dependencies):
            dependency = dependencies[position]
            full = f"{dependency.artifact_name}@{dependency.artifact_version}"
        elif table.artifact_id is not None:
            full = table.artifact_id.text
        else:
            continue
        out.setdefault(full, table)
    names: dict[str, set[str]] = {}
    for full in list(out):
        names.setdefault(full.rsplit("@", 1)[0], set()).add(full)
    for name, fulls in names.items():
        if len(fulls) == 1:
            out.setdefault(name, out[next(iter(fulls))])
    return out


def _stage_zero_units(
    first: Node, prepared_rows: list[Any], store: SQLiteArtifactStore
) -> tuple[list[Group], Callable[[int], tuple[int, ...]]]:
    """첫 단계 단위: 묶음이 없으면 원본 행 하나씩, 있으면 원본 행 메타 값으로 실행 전에 묶는다."""

    if first.groups is None:
        units = [
            Group.single(row, member_id=f"s{index}", sources=(index,))
            for index, row in enumerate(prepared_rows)
        ]
        return units, lambda index: (index,)
    metas = [
        ItemMeta(id=f"s{index}", values=meta_values(row), sources=(index,))
        for index, row in enumerate(prepared_rows)
    ]
    planned, plan_hash = plan_groups(first.groups, metas)
    store.check_plan(0, plan_hash, len(planned))
    units = [
        Group(
            [prepared_rows[int(member_id[1:])] for member_id in group.member_ids],
            key=group.key,
            member_ids=group.member_ids,
            sources=group.sources,
        )
        for group in planned
    ]
    return units, lambda index: planned[index].sources


def _run_barrier_stage(
    builder: ArtifactBuilderBase,
    stage_idx: int,
    offset: int,
    stage_nodes: list[Node],
    units: Sequence[Group],
    unit_sources: Callable[[int], tuple[int, ...]],
    *,
    stage_store: StageStore,
    policy: BuildPolicy,
    parent_rows: list[ParsedRow],
    parents: Mapping[str, LazyTable],
    build_context: BuildContextView,
    sandbox_registry: SandboxRegistry | None,
    seed_plan: SeedPlan | None,
    executor_config: ExecutorConfig,
) -> dict[str, dict[str, Any]]:
    """뒤에 묶음이 오는 단계: 결과를 단위마다 디스크에 쓴다. 실패한 단위가 있으면 빌드를 멈춘다.

    뒤 단계 묶음은 이 단계 결과 전체를 봐야 정해지므로, 일부 단위가 빠진 채로 묶지 않는다.
    """

    completed = stage_store.completed_units(stage_idx)
    if policy.retry_failed:
        stage_store.clear_failed(stage_idx)
    buffers: dict[int, list[dict[str, Any]]] = {}

    with tqdm(
        total=len(units),
        initial=len(completed),
        desc=f"stage{stage_idx}:{stage_nodes[0].name}",
        unit="unit",
    ) as progress:

        def emit(unit_idx: int, output_idx: int, row: dict[str, Any]) -> None:
            buffers.setdefault(unit_idx, []).append(row)

        def finish(unit_idx: int) -> None:
            sources = unit_sources(unit_idx)
            rows = buffers.pop(unit_idx, [])
            stage_store.commit_unit(
                stage_idx, unit_idx, [(meta_values(row), sources, row) for row in rows]
            )
            progress.update(1)

        def fail(unit_idx: int, error: BaseException) -> None:
            buffers.pop(unit_idx, None)
            stage_store.mark_failed(stage_idx, unit_idx, error)
            progress.update(1)

        stats, failures = run_pipeline(
            builder,
            stage_nodes,
            units,
            parent_rows=parent_rows,
            build_context=build_context,
            sandbox_registry=sandbox_registry,
            seed_plan=seed_plan,
            cpu_workers=int(executor_config.row_workers),
            cpu_executor=executor_config.row_executor,
            skip_source_indices=completed,
            emit=emit,
            finish_source=finish,
            fail_source=fail,
            node_offset=offset,
            parents=parents,
        )
    if failures:
        first = min(failures)
        raise RuntimeError(
            f"stage {stage_idx}({stage_nodes[0].name}): {len(failures)} units failed; first is unit "
            f"{first}: {type(failures[first]).__name__}: {failures[first]}. "
            "뒤 단계 묶음은 이 단계 결과가 모두 있어야 정해지므로 빌드를 멈춤(다시 실행하면 실패 단위만 다시 함)"
        )
    return stats


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
