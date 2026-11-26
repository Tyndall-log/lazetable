"""노드 묶음(n:1·n:n): 계획 단계 묶음, 장벽 단계(디스크 저장)와 재개, 겹침, 오류, __sources__."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from lazetable import (
    ArtifactBuilderBase,
    BuildPolicy,
    ExecutorConfig,
    Group,
    GroupingError,
    LazyTable,
    Node,
    write_artifact,
)

PIECES = [
    # 조각으로 저장된 악궁: 조각 하나에 치아 여러 개, 치아 하나가 조각 둘에 걸칠 수 있음
    {"piece": "p1", "case_key": "c1", "jaw": "upper", "teeth": [11, 12, 13]},
    {"piece": "p2", "case_key": "c1", "jaw": "upper", "teeth": [13, 14]},
    {"piece": "p3", "case_key": "c1", "jaw": "lower", "teeth": [31]},
    {"piece": "p4", "case_key": "c2", "jaw": "upper", "teeth": [21]},
]


def _split_teeth(groups, context):
    out = []
    for g in groups:
        case_key, jaw = g.key
        by_fdi: dict[int, list[str]] = {}
        for piece in g:
            for fdi in piece["teeth"]:
                by_fdi.setdefault(fdi, []).append(piece["piece"])
        out.append(
            [
                {"tooth_key": f"{case_key}:{jaw}:{fdi}", "fdi": fdi, "pieces": pieces}
                for fdi, pieces in sorted(by_fdi.items())
            ]
        )
    return out


class TeethFromPieces(ArtifactBuilderBase):
    """계획 단계 묶음: 첫 노드가 원본 행(조각)을 악궁별로 묶어 받는다(n:n)."""

    artifact_name = "teeth"
    artifact_version = "v1"
    unique_keys = [("tooth_key",)]

    def prepare_table(self, **kwargs):
        return [dict(row) for row in PIECES]

    def pipeline(self):
        return [Node("split", _split_teeth, groups=self.grouping.by_columns("case_key", "jaw"))]


CONFIGS = [
    ExecutorConfig(row_workers=0),
    ExecutorConfig(row_workers=2, row_executor="thread"),
    ExecutorConfig(row_workers=2, row_executor="process"),
]


@pytest.mark.parametrize("config", CONFIGS)
def test_plan_time_groups_feed_first_node(tmp_path: Path, config) -> None:
    table = TeethFromPieces().build(storage_root=tmp_path, policy=BuildPolicy(), executor_config=config).table
    rows = {row.tooth_key: row for row in table}
    assert sorted(rows) == [
        "c1:lower:31", "c1:upper:11", "c1:upper:12", "c1:upper:13", "c1:upper:14", "c2:upper:21",
    ]
    assert list(rows["c1:upper:13"].pieces) == ["p1", "p2"]  # 두 조각에 걸친 치아
    assert list(rows["c1:upper:13"].get("__sources__")) == [0, 1]  # 묶음 구성원의 원본 행
    assert list(rows["c2:upper:21"].get("__sources__")) == [3]


def _decode(groups, context):
    # 무거운 계산 자리: 내용 해시(mesh_id)는 계산해 봐야 안다
    return [{**g[0], "mesh_id": f"m{g[0]['content']}"} for g in groups]


def _encode_first(groups, context):
    return [{"mesh_id": g.key, "files": [m["file"] for m in g], "copies": len(g)} for g in groups]


def _decoded(log_dir: Path) -> list[str]:
    return sorted(path.name.rsplit(".", 1)[0] for path in log_dir.iterdir())


def _counted_decode(groups, context):
    # 스레드·프로세스 워커가 동시에 써도 깨지지 않게 디코드마다 표시 파일 하나
    log_dir = Path(context.build_context["decode_log"])
    log_dir.mkdir(parents=True, exist_ok=True)
    for g in groups:
        (log_dir / f"{g[0]['file']}.{uuid.uuid4().hex}").touch()
    fail_on = context.build_context.get("fail_file")
    if fail_on and Path(context.build_context["fail_flag"]).exists():
        if any(g[0]["file"] == fail_on for g in groups):
            raise ValueError("decode failed once")
    return _decode(groups, context)


FILES = [
    {"file": "a.stl", "content": 1},
    {"file": "b.stl", "content": 2},
    {"file": "a_copy.stl", "content": 1},
    {"file": "c.stl", "content": 3},
    {"file": "a_backup.stl", "content": 1},
]


class MeshDedupe(ArtifactBuilderBase):
    """장벽 묶음: 앞 노드가 계산한 mesh_id로 묶어 같은 메쉬 사본 n개를 한 행으로(n:1)."""

    artifact_name = "mesh"
    artifact_version = "v1"
    unique_keys = [("mesh_id",)]

    def prepare_table(self, **kwargs):
        return [dict(row) for row in FILES]

    def pipeline(self):
        return [
            Node("decode", _counted_decode),
            Node("encode", _encode_first, groups=self.grouping.by_columns("mesh_id")),
        ]


@pytest.mark.parametrize("config", CONFIGS)
def test_barrier_groups_on_computed_column(tmp_path: Path, config) -> None:
    log = tmp_path / "decode.log"
    result = MeshDedupe().build(
        storage_root=tmp_path / "out",
        build_context={"decode_log": str(log)},
        policy=BuildPolicy(),
        executor_config=config,
    )
    rows = {row.mesh_id: row for row in result.table}
    assert sorted(rows) == ["m1", "m2", "m3"]
    assert list(rows["m1"].files) == ["a.stl", "a_copy.stl", "a_backup.stl"]
    assert list(rows["m1"].get("__sources__")) == [0, 2, 4]
    assert _decoded(log) == sorted(f["file"] for f in FILES)  # 파일마다 한 번
    assert not (tmp_path / "out" / "mesh" / "v1" / "stages.sqlite").exists()  # 끝나면 지움
    assert set(result.node_stats) == {"decode", "encode"}


def test_barrier_stage_failure_stops_and_resume_skips_finished_units(tmp_path: Path) -> None:
    log = tmp_path / "decode.log"
    flag = tmp_path / "fail.flag"
    flag.touch()
    context = {
        "decode_log": str(log),
        "fail_file": "c.stl",
        "fail_flag": str(flag),
    }
    config = ExecutorConfig(row_workers=0)
    with pytest.raises(RuntimeError, match="stage 0\\(decode\\): 1 units failed"):
        MeshDedupe().build(
            storage_root=tmp_path / "out", build_context=context, policy=BuildPolicy(), executor_config=config
        )
    assert _decoded(log) == sorted(f["file"] for f in FILES)
    flag.unlink()
    for marker in log.iterdir():
        marker.unlink()

    result = MeshDedupe().build(
        storage_root=tmp_path / "out", build_context=context, policy=BuildPolicy(), executor_config=config
    )
    assert _decoded(log) == ["c.stl"]  # 끝난 단위는 다시 계산하지 않음
    assert sorted(row.mesh_id for row in result.table) == ["m1", "m2", "m3"]


def _overlap_groups(items):
    # 자유 묶음: 같은 행이 여러 묶음에 들어감
    by_case: dict[str, list] = {}
    for item in items:
        for case in item["cases"]:
            by_case.setdefault(case, []).append(item)
    return [Group(members, key=case) for case, members in sorted(by_case.items())]


def _summarize(groups, context):
    return [{"case": g.key, "scans": sorted(m["scan"] for m in g)} for g in groups]


class SharedScans(ArtifactBuilderBase):
    artifact_name = "shared"
    artifact_version = "v1"
    unique_keys = [("case",)]

    def prepare_table(self, **kwargs):
        return [
            {"scan": "s1", "cases": ["A", "B"]},
            {"scan": "s2", "cases": ["A"]},
            {"scan": "s3", "cases": ["B", "C"]},
        ]

    def pipeline(self):
        return [
            Node("pass", lambda groups, context: [dict(g[0]) for g in groups]),
            Node("summarize", _summarize, groups=_overlap_groups),
        ]


def test_free_function_groups_allow_overlap(tmp_path: Path) -> None:
    table = SharedScans().build(
        storage_root=tmp_path, policy=BuildPolicy(), executor_config=ExecutorConfig(row_workers=0)
    ).table
    assert {row.case: list(row.scans) for row in table} == {"A": ["s1", "s2"], "B": ["s1", "s3"], "C": ["s3"]}


def test_by_list_column_overlap_at_plan_time(tmp_path: Path) -> None:
    class ListGroups(SharedScans):
        artifact_name = "list_groups"

        def pipeline(self):
            return [Node("summarize", _summarize, groups=self.grouping.by_list_column("cases"))]

    table = ListGroups().build(
        storage_root=tmp_path, policy=BuildPolicy(), executor_config=ExecutorConfig(row_workers=0)
    ).table
    assert {row.case: list(row.scans) for row in table} == {"A": ["s1", "s2"], "B": ["s1", "s3"], "C": ["s3"]}


@pytest.mark.parametrize(
    ("groups", "message"),
    [
        (lambda items: [Group(items[:1], key="x"), Group(items[1:], key="x")], "같은 키"),
        (lambda items: [Group([], key="x")], "빈 묶음"),
        (lambda items: [Group([items[0], items[0]], key="x")], "두 번"),
        (lambda items: [Group(["nope"], key="x")], "입력에 없는"),
    ],
)
def test_grouping_rule_violations_are_errors(tmp_path: Path, groups, message) -> None:
    class Bad(SharedScans):
        artifact_name = "bad"

        def pipeline(self):
            return [Node("summarize", _summarize, groups=groups)]

    with pytest.raises(GroupingError, match=message):
        Bad().build(storage_root=tmp_path, policy=BuildPolicy(), executor_config=ExecutorConfig(row_workers=0))


def test_node_context_exposes_parents_by_name(tmp_path: Path) -> None:
    from lazetable import ArtifactDependency

    mesh = write_artifact(
        artifact_name="meshes",
        artifact_version="v1",
        rows=[{"mesh_id": "m1", "faces": 7}],
        out_dir=tmp_path / "meshes" / "v1",
        unique_key_fields=("mesh_id",),
    ).table

    def lookup(groups, context):
        faces = {row.raw("mesh_id"): row.raw("faces") for row in context.parents["meshes"]}
        assert context.parents["meshes@v1"] is context.parents["meshes"]
        return [{"unit": g[0]["unit"], "n": faces[g[0]["mesh_id"]]} for g in groups]

    class UsesParent(ArtifactBuilderBase):
        artifact_name = "uses_parent"
        artifact_version = "v1"
        unique_keys = [("unit",)]
        dependencies = [ArtifactDependency("meshes", "v1")]

        def prepare_table(self, **kwargs):
            return [{"unit": "u1", "mesh_id": "m1"}]

        def pipeline(self):
            return [Node("lookup", lookup)]

    table = UsesParent().build(
        storage_root=tmp_path, parent_tables=[mesh], policy=BuildPolicy(), executor_config=ExecutorConfig(row_workers=0)
    ).table
    assert table.first_row().n == 7


def test_keep_stages_leaves_stage_store(tmp_path: Path) -> None:
    MeshDedupe().build(
        storage_root=tmp_path,
        build_context={"decode_log": str(tmp_path / "log")},
        policy=BuildPolicy(),
        executor_config=ExecutorConfig(row_workers=0, keep_stages=True),
    )
    assert (tmp_path / "mesh" / "v1" / "stages.sqlite").exists()
    assert LazyTable.open(tmp_path / "mesh" / "v1").count_rows() == 3
