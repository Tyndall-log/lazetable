"""CPU/GPU 노드를 쌓아 행을 처리하는 build 스케줄러.

builder가 ``pipeline()``으로 ``Node`` 목록을 돌려주면 ``transform_row``/``transform_batch`` 대신
이 스케줄러가 돈다.

- CPU 노드는 ``row_workers``개 워커가, GPU 노드는 메인 프로세스의 전용 스레드 하나가 실행한다.
  GPU 초기화는 노드 함수 안에서 하면 항상 그 스레드에서 일어난다.
- 노드는 입력 대기열에 ``min_items``개(묶음)가 차면 실행할 수 있고, 한 번에 ``max_items``개까지 가져간다.
  앞 노드가 모두 끝나 더 올 묶음이 없으면 ``min_items``보다 적어도 실행한다.
- 실행할 수 있는 노드가 여럿이면 뒤쪽 노드부터 고른다. 다음 노드 대기열이 ``max_queue``에 차면
  앞 노드를 멈춰 메모리를 제한한다.
- 여러 행 묶음이 실패하면 한 행씩 다시 실행해, 실제로 실패한 행만 실패로 기록하고 나머지는 계속한다.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import (
    FIRST_COMPLETED,
    BrokenExecutor,
    Executor,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import dataclass, field
from typing import Any

from .grouping import Group, as_group_spec

DEVICES = ("cpu", "gpu")


class BatchCardinalityError(ValueError):
    """Raised when a batch step changes item cardinality."""


@dataclass(frozen=True)
class NodeContext:
    node: str
    batch_idx: int
    parent_rows: list[Any]
    build_context: Mapping[str, Any]
    sandbox_registry: Any
    seed_plan: Any
    parents: Mapping[str, Any] = field(default_factory=dict)  # 부모 표: "이름@판"과(겹치지 않으면) "이름"


@dataclass(frozen=True)
class Node:
    """파이프라인 블록.

    ``fn(items, context)``의 ``items`` 원소 하나는 ``Group``(목록처럼 쓰고 ``.key``) 하나다.
    ``groups=None``이면 앞 노드 결과(첫 노드는 원본 행) 한 행짜리 묶음이고, ``groups=...``이면 앞 결과를
    그 규칙으로 묶은 것이다(``lazetable.grouping`` 참고). 결과는 입력과 같은 길이의 목록이고, 각 원소는
    그 묶음의 결과로 ``None``(없음), ``dict``(한 행), ``list[dict]``(여러 행) 중 하나다.

    ``min_items``/``max_items``는 한 번에 보낼 묶음 수(스케줄링 값)이고 묶음을 나누거나 합치지 않는다.
    """

    name: str
    fn: Callable[[list[Any], NodeContext], list[Any]]
    device: str = "cpu"
    groups: Any = None
    min_items: int = 1
    max_items: int | None = None  # 기본값은 min_items
    max_queue: int | None = None  # 입력 대기열 상한. 기본값은 max_items의 2배와 CPU 워커 2배 중 큰 값

    def __post_init__(self) -> None:
        object.__setattr__(self, "groups", as_group_spec(self.groups))

    @property
    def batch_limit(self) -> int:
        return self.min_items if self.max_items is None else self.max_items


def validate_nodes(nodes: list[Node]) -> None:
    if not nodes:
        raise ValueError("pipeline must contain at least one node")
    names = [node.name for node in nodes]
    if len(set(names)) != len(names):
        raise ValueError(f"pipeline node names must be unique: {names}")
    for node in nodes:
        if node.device not in DEVICES:
            raise ValueError(f"{node.name}: device must be one of {DEVICES}")
        if node.min_items < 1:
            raise ValueError(f"{node.name}: min_items must be positive")
        if node.batch_limit < node.min_items:
            raise ValueError(f"{node.name}: max_items must be >= min_items")
        if node.max_queue is not None and node.max_queue < node.batch_limit:
            raise ValueError(f"{node.name}: max_queue must be >= max_items")


def split_stages(nodes: list[Node]) -> list[tuple[int, list[Node]]]:
    """첫 노드가 아닌 ``groups`` 노드에서 나눈 단계들((시작 위치, 노드 목록))."""

    stages: list[tuple[int, list[Node]]] = []
    for index, node in enumerate(nodes):
        if index == 0 or node.groups is not None:
            stages.append((index, [node]))
        else:
            stages[-1][1].append(node)
    return stages


def pipeline_signature(nodes: list[Node]) -> list[dict[str, Any]]:
    """결과에 영향을 주는 구성만 담는다. 묶음 크기와 워커 수는 속도에만 영향을 줘야 한다."""
    return [
        {
            "name": node.name,
            "device": node.device,
            "groups": None if node.groups is None else node.groups.signature(),
        }
        for node in nodes
    ]


def _call_node(node: Node, items: list[Any], context: NodeContext) -> tuple[list[Any], float]:
    started = time.perf_counter()
    outputs = node.fn(items, context)
    seconds = time.perf_counter() - started
    if not isinstance(outputs, list) or len(outputs) != len(items):
        count = len(outputs) if isinstance(outputs, list) else type(outputs).__name__
        raise BatchCardinalityError(
            f"{node.name} must return one result per input: input={len(items)}, output={count}"
        )
    return outputs, seconds


_WORKER_STATE: tuple[list[Node], tuple[Any, ...]] | None = None


def _initialize_worker(builder: Any, parent_rows, build_context, sandbox_registry, seed_plan, parents):
    global _WORKER_STATE
    _WORKER_STATE = (
        builder.pipeline(),
        (parent_rows, build_context, sandbox_registry, seed_plan, parents),
    )


def _run_in_worker(index: int, items: list[Any], batch_idx: int) -> tuple[list[Any], float]:
    """``index``는 빌더 ``pipeline()`` 전체에서의 위치다."""
    if _WORKER_STATE is None:
        raise RuntimeError("process worker was not initialized")
    nodes, shared = _WORKER_STATE
    return _call_node(nodes[index], items, NodeContext(nodes[index].name, batch_idx, *shared))


class _InlineExecutor(Executor):
    """``row_workers=0``: 스케줄러 스레드에서 바로 실행한다."""

    def submit(self, fn, /, *args, **kwargs):
        future: Future[Any] = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as error:
            future.set_exception(error)
        return future


def _normalize_outputs(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    for item in values:
        if not isinstance(item, dict):
            raise TypeError(f"node result must be None, dict or list[dict]: {type(item).__name__}")
    return values


def run_pipeline(
    builder: Any,
    nodes: list[Node],
    rows: Sequence[Group],
    *,
    parent_rows: list[Any],
    build_context: Mapping[str, Any],
    sandbox_registry: Any,
    seed_plan: Any,
    cpu_workers: int,
    cpu_executor: str,
    skip_source_indices: set[int],
    emit: Callable[[int, int, dict[str, Any]], None],
    finish_source: Callable[[int], None],
    fail_source: Callable[[int, BaseException], None],
    node_offset: int = 0,
    parents: Mapping[str, Any] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[int, BaseException]]:
    """단계 하나의 노드를 실행하고 (노드별 통계, 실패한 단위)를 돌려준다.

    ``rows``의 원소 하나(묶음 하나)가 단위다. 결과·완료·실패는 단위 번호로 알린다. 첫 노드는 그 묶음을,
    뒤 노드는 앞 노드 결과를 한 행짜리 묶음으로 받는다. ``node_offset``은 이 단계 첫 노드의 빌더
    ``pipeline()`` 안 위치다(프로세스 워커가 노드를 찾는 데 쓴다).
    """
    validate_nodes(nodes)
    for node in nodes[1:]:
        if node.groups is not None:
            raise ValueError(f"{node.name}: groups는 단계 첫 노드에만 올 수 있음")
    shared_parents = dict(parents or {})
    if cpu_executor not in {"thread", "process"}:
        raise ValueError("row_executor must be 'thread' or 'process'")
    count = len(nodes)
    worker_slots = max(1, cpu_workers)
    caps = [
        node.max_queue
        if node.max_queue is not None
        else max(2 * node.batch_limit, 2 * worker_slots)
        for node in nodes
    ]
    queues: list[deque[tuple[int, tuple[int, ...], Any]]] = [deque() for _ in nodes]
    retries: list[deque[tuple[int, tuple[int, ...], Any]]] = [deque() for _ in nodes]
    inflight_rows = [0] * count
    inflight_by_device = {"cpu": 0, "gpu": 0}
    pending_sources = deque(i for i in range(len(rows)) if i not in skip_source_indices)
    live: dict[int, int] = {}
    outputs: dict[int, list[tuple[tuple[int, ...], dict[str, Any]]]] = {}
    failures: dict[int, BaseException] = {}
    futures: dict[Future[Any], tuple[int, list[tuple[int, tuple[int, ...], Any]]]] = {}
    batch_idx = 0
    stats = {
        node.name: {"device": node.device, "rows": 0, "calls": 0, "seconds": 0.0, "errors": 0}
        for node in nodes
    }

    def context(index: int, number: int) -> NodeContext:
        return NodeContext(
            nodes[index].name,
            number,
            parent_rows,
            build_context,
            sandbox_registry,
            seed_plan,
            shared_parents,
        )

    def refill() -> None:
        while pending_sources and len(queues[0]) + len(retries[0]) + inflight_rows[0] < caps[0]:
            source = pending_sources.popleft()
            live[source] = 1
            outputs[source] = []
            queues[0].append((source, (), rows[source]))

    def upstream_done(index: int) -> bool:
        if pending_sources:
            return False
        return not any(queues[j] or retries[j] or inflight_rows[j] for j in range(index))

    def blocked(index: int) -> bool:
        if index + 1 >= count:
            return False
        following = index + 1
        waiting = len(queues[following]) + len(retries[following])
        return waiting + inflight_rows[index] >= caps[following]

    def take(index: int) -> list[tuple[int, tuple[int, ...], Any]] | None:
        if blocked(index):
            return None
        if retries[index]:
            return [retries[index].popleft()]
        queue = queues[index]
        if not queue or (len(queue) < nodes[index].min_items and not upstream_done(index)):
            return None
        return [queue.popleft() for _ in range(min(nodes[index].batch_limit, len(queue)))]

    def fail(source: int, error: BaseException) -> None:
        if source in failures:
            return
        failures[source] = error
        live.pop(source, None)
        outputs.pop(source, None)
        for group in (*queues, *retries):
            kept = [tagged for tagged in group if tagged[0] != source]
            group.clear()
            group.extend(kept)
        fail_source(source, error)

    def complete(source: int) -> None:
        del live[source]
        finished = sorted(outputs.pop(source), key=lambda entry: entry[0])
        for ordinal, (_, row) in enumerate(finished):
            emit(source, ordinal, row)
        finish_source(source)

    def accept(index: int, tagged: tuple[int, tuple[int, ...], Any], value: Any) -> None:
        source, path, _ = tagged
        if source in failures:
            return
        try:
            results = _normalize_outputs(value)
        except TypeError as error:
            fail(source, error)
            return
        last = index + 1 == count
        live[source] += (0 if last else len(results)) - 1  # 아직 노드를 지나야 할 행 수
        for position, result in enumerate(results):
            entry = (*path, position)
            if last:
                outputs[source].append((entry, result))
            else:
                queues[index + 1].append((source, entry, Group.single(result)))
        if live[source] == 0:
            complete(source)

    if cpu_workers <= 0:
        cpu_pool: Executor = _InlineExecutor()
    elif cpu_executor == "thread":
        cpu_pool = ThreadPoolExecutor(cpu_workers, thread_name_prefix="lazetable-cpu")
    else:
        cpu_pool = ProcessPoolExecutor(
            cpu_workers,
            initializer=_initialize_worker,
            initargs=(builder, parent_rows, build_context, sandbox_registry, seed_plan, shared_parents),
        )
    gpu_pool = None
    if any(node.device == "gpu" for node in nodes):
        gpu_pool = ThreadPoolExecutor(1, thread_name_prefix="lazetable-gpu")

    def dispatch(device: str, limit: int) -> None:
        nonlocal batch_idx
        while inflight_by_device[device] < limit:
            for index in range(count - 1, -1, -1):
                if nodes[index].device != device:
                    continue
                tagged = take(index)
                if tagged is None:
                    continue
                items = [entry[2] for entry in tagged]
                number = batch_idx
                batch_idx += 1
                inflight_rows[index] += len(tagged)
                inflight_by_device[device] += 1
                if device == "gpu":
                    future = gpu_pool.submit(_call_node, nodes[index], items, context(index, number))
                elif cpu_workers > 0 and cpu_executor == "process":
                    future = cpu_pool.submit(_run_in_worker, node_offset + index, items, number)
                else:
                    future = cpu_pool.submit(_call_node, nodes[index], items, context(index, number))
                futures[future] = (index, tagged)
                break
            else:
                return

    try:
        while True:
            refill()
            dispatch("gpu", 1)
            dispatch("cpu", worker_slots)
            if not futures:
                if live:
                    raise RuntimeError(f"pipeline scheduler stalled with {len(live)} live rows")
                return stats, failures
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                index, tagged = futures.pop(future)
                node = nodes[index]
                inflight_rows[index] -= len(tagged)
                inflight_by_device[node.device] -= 1
                record = stats[node.name]
                record["calls"] += 1
                try:
                    results, seconds = future.result()
                except BrokenExecutor:
                    raise
                except Exception as error:
                    record["errors"] += 1
                    if len(tagged) > 1:
                        retries[index].extend(e for e in tagged if e[0] not in failures)
                    else:
                        fail(tagged[0][0], error)
                    continue
                record["rows"] += len(tagged)
                record["seconds"] += seconds
                for entry, value in zip(tagged, results, strict=True):
                    accept(index, entry, value)
    finally:
        cpu_pool.shutdown(wait=True, cancel_futures=True)
        if gpu_pool is not None:
            gpu_pool.shutdown(wait=True, cancel_futures=True)
