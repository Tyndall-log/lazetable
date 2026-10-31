from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from lazetable import ArtifactBuilderBase, BuildPolicy, ExecutorConfig, Node, WriterConfig
from lazetable.pipeline import run_pipeline, validate_nodes


def _add_cpu(items, context):
    return [{**item, "cpu": item["id"] * 2} for item in items]


def _add_gpu(items, context):
    return [{**item, "gpu_thread": threading.current_thread().name, "size": len(items)} for item in items]


def _finish(items, context):
    return [{**item, "done": True} for item in items]


class NodeBuilder(ArtifactBuilderBase):
    artifact_name = "nodes"
    artifact_version = "v1"
    unique_keys = [("id",)]

    def prepare_table(self, *, build_context=None, **kwargs):
        return [{"id": i} for i in range(build_context["count"])]

    def pipeline(self):
        return [
            Node("first", _add_cpu),
            Node("gpu", _add_gpu, device="gpu", min_rows=4, max_rows=4),
            Node("last", _finish),
        ]


@pytest.mark.parametrize(
    "config",
    [
        ExecutorConfig(row_workers=0),
        ExecutorConfig(row_workers=3, row_executor="thread"),
        ExecutorConfig(row_workers=2, row_executor="process"),
    ],
)
def test_builds_through_cpu_and_gpu_nodes(tmp_path: Path, config) -> None:
    result = NodeBuilder().build(
        storage_root=tmp_path,
        build_context={"count": 10},
        policy=BuildPolicy(),
        executor_config=config,
    )
    rows = list(result.table)
    assert result.status == "built"
    assert sorted(int(r.id) for r in rows) == list(range(10))
    assert all(r.done and int(r.cpu) == int(r.id) * 2 for r in rows)
    assert {str(r.gpu_thread).rsplit("_", 1)[0] for r in rows} == {"lazetable-gpu"}
    # 4행씩 묶고, 더 올 행이 없는 마지막 2행은 min_rows보다 적어도 실행한다
    assert sorted(int(r.size) for r in rows) == [2, 2, 4, 4, 4, 4, 4, 4, 4, 4]
    assert result.node_stats["gpu"]["rows"] == 10
    assert result.node_stats["gpu"]["calls"] == 3


def test_signature_ignores_workers_and_batch_limits(tmp_path: Path) -> None:
    first = NodeBuilder().build(
        storage_root=tmp_path,
        build_context={"count": 3},
        policy=BuildPolicy(),
        executor_config=ExecutorConfig(row_workers=0),
    )
    second = NodeBuilder().build(
        storage_root=tmp_path,
        build_context={"count": 3},
        policy=BuildPolicy(),
        executor_config=ExecutorConfig(row_workers=4, row_executor="thread"),
    )
    assert first.status == "built" and second.status == "reused"


def _run(nodes, count, *, workers=0, executor="thread", skip=()):
    emitted, finished, failed = [], [], {}
    stats, failures = run_pipeline(
        None,
        nodes,
        [{"id": i} for i in range(count)],
        parent_rows=[],
        build_context={},
        sandbox_registry=None,
        seed_plan=None,
        cpu_workers=workers,
        cpu_executor=executor,
        skip_source_indices=set(skip),
        emit=lambda source, ordinal, row: emitted.append((source, ordinal, row)),
        finish_source=finished.append,
        fail_source=lambda source, error: failed.__setitem__(source, error),
    )
    return emitted, finished, failed, stats, failures


def test_later_node_runs_first() -> None:
    order = []

    def record(name):
        def fn(items, context):
            order.append(name)
            return items

        return fn

    _run([Node("a", record("a")), Node("b", record("b"))], 3)
    assert order == ["a", "b", "a", "b", "a", "b"]


def test_queue_cap_limits_rows_ahead_of_slow_gpu() -> None:
    seen = {"gpu_calls": 0, "ahead": 0, "cpu_done": 0}
    lock = threading.Lock()

    def cpu(items, context):
        with lock:
            seen["cpu_done"] += len(items)
            seen["ahead"] = max(seen["ahead"], seen["cpu_done"] - seen["gpu_calls"])
        return items

    def gpu(items, context):
        time.sleep(0.01)
        with lock:
            seen["gpu_calls"] += len(items)
        return items

    nodes = [Node("cpu", cpu), Node("gpu", gpu, device="gpu", max_queue=3)]
    emitted, *_ = _run(nodes, 30, workers=4)
    assert len(emitted) == 30
    # 대기열 3 + 처리 중 CPU 워커 4 + GPU 1
    assert seen["ahead"] <= 3 + 4 + 1


def test_failed_batch_retries_one_by_one_and_keeps_others() -> None:
    sizes = []

    def gpu(items, context):
        sizes.append(len(items))
        if any(item["id"] == 2 for item in items):
            raise ValueError("bad row")
        return items

    nodes = [Node("gpu", gpu, device="gpu", min_rows=4, max_rows=4), Node("last", _finish)]
    emitted, finished, failed, stats, failures = _run(nodes, 4)
    assert sizes == [4, 1, 1, 1, 1]
    assert sorted(finished) == [0, 1, 3] and sorted(failed) == [2]
    assert sorted(failures) == [2] and stats["gpu"]["errors"] == 2
    assert sorted(source for source, _, _ in emitted) == [0, 1, 3]


def test_fan_out_and_drop_keep_deterministic_order() -> None:
    def split(items, context):
        return [None if item["id"] == 0 else [{"id": item["id"], "n": n} for n in range(3)] for item in items]

    def slow_first(items, context):
        return [{**item, "m": 1} for item in items]

    emitted, finished, *_ = _run([Node("split", split), Node("tail", slow_first)], 2, workers=3)
    assert sorted(finished) == [0, 1]
    assert [(s, o, r["n"]) for s, o, r in emitted] == [(1, 0, 0), (1, 1, 1), (1, 2, 2)]


def test_wrong_result_count_fails_the_row() -> None:
    nodes = [Node("bad", lambda items, context: [])]
    _, finished, failed, *_ = _run(nodes, 1)
    assert finished == [] and list(failed) == [0]


def test_build_reports_failed_rows_and_retries_them(tmp_path: Path) -> None:
    class Flaky(ArtifactBuilderBase):
        artifact_name = "flaky"
        artifact_version = "v1"
        unique_keys = [("id",)]

        def prepare_table(self, *, build_context=None, **kwargs):
            return [{"id": "a"}, {"id": "b"}]

        def pipeline(self):
            return [Node("step", self.step)]

        def step(self, items, context):
            if Path(context.build_context["sentinel"]).exists() and items[0]["id"] == "b":
                raise ValueError("fail once")
            return items

    sentinel = tmp_path / "sentinel"
    sentinel.touch()
    context = {"sentinel": str(sentinel)}
    config = ExecutorConfig(row_workers=0)
    with pytest.raises(RuntimeError, match="1 source rows failed"):
        Flaky().build(storage_root=tmp_path, build_context=context, policy=BuildPolicy(), executor_config=config)
    sentinel.unlink()
    result = Flaky().build(storage_root=tmp_path, build_context=context, policy=BuildPolicy(), executor_config=config)
    assert result.status == "resumed" and len(result.table) == 2


def test_validate_rejects_bad_nodes() -> None:
    for nodes in (
        [],
        [Node("a", _finish), Node("a", _finish)],
        [Node("a", _finish, device="npu")],
        [Node("a", _finish, min_rows=0)],
        [Node("a", _finish, min_rows=4, max_rows=2)],
        [Node("a", _finish, min_rows=4, max_queue=2)],
    ):
        with pytest.raises(ValueError):
            validate_nodes(nodes)


def test_builder_without_pipeline_uses_row_transform(tmp_path: Path) -> None:
    class Legacy(ArtifactBuilderBase):
        artifact_name = "legacy"
        artifact_version = "v1"
        unique_keys = [("id",)]

        def prepare_table(self, *, build_context=None, **kwargs):
            return [{"id": "a"}]

    result = Legacy().build(
        storage_root=tmp_path,
        policy=BuildPolicy(),
        executor_config=ExecutorConfig(row_workers=0),
        writer_config=WriterConfig(payload_workers=1, max_pending_writes=1),
    )
    assert result.status == "built" and result.node_stats is None
