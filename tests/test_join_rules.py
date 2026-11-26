"""조인 규칙: 열 이름이 겹치면 오류, prefix·rename, how(strict·inner·left), 빌드 출력 검사."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lazetable import (
    ArtifactBuilderBase,
    ArtifactDependency,
    ArtifactJoin,
    BuildPolicy,
    ColumnConflictError,
    ExecutorConfig,
    JoinKeyError,
    LazyTable,
    Payload,
    register_payload_type,
    write_artifact,
)


@register_payload_type
class JoinJsonPayloadType:
    payload_id = "join_rules_json_v1"
    default_extension = "json"

    def payload(self, data, **kwargs):
        return Payload(self, data, metadata=kwargs or None)

    def dump_to_path(self, data, path: Path, *, metadata=None):
        path.write_text(json.dumps(data), encoding="utf-8")
        return {}

    def load_from_path(self, path: Path, *, metadata=None):
        return json.loads(path.read_text(encoding="utf-8"))


JSON = JoinJsonPayloadType()


def _mesh_table(root: Path) -> LazyTable:
    return write_artifact(
        artifact_name="mesh",
        artifact_version="v1",
        rows=[
            {"mesh_id": "m1", "faces": 10, "mesh": JSON.payload({"name": "upper"})},
            {"mesh_id": "m2", "faces": 20, "mesh": JSON.payload({"name": "lower"})},
        ],
        out_dir=root / "mesh" / "v1",
        artifact_roots={"default": str(root)},
        unique_key_fields=("mesh_id",),
    ).table


def _unit_table(root: Path, rows: list[dict]) -> LazyTable:
    return write_artifact(
        artifact_name="unit",
        artifact_version="v1",
        rows=rows,
        out_dir=root / "unit" / "v1",
        artifact_roots={"default": str(root)},
        unique_key_fields=("unit",),
    ).table


def test_join_conflict_is_an_error_not_an_overwrite(tmp_path: Path) -> None:
    mesh = _mesh_table(tmp_path)
    units = _unit_table(tmp_path, [{"unit": "u1", "jaw_mesh_id": "m1", "faces": 3}])

    with pytest.raises(ColumnConflictError, match="faces"):
        units.join(mesh, left_on="jaw_mesh_id", right_on="mesh_id")


def test_prefix_and_rename_carry_payload_fields(tmp_path: Path) -> None:
    mesh = _mesh_table(tmp_path)
    units = _unit_table(
        tmp_path, [{"unit": "u1", "jaw_mesh_id": "m1", "antagonist_mesh_id": "m2", "faces": 3}]
    )

    joined = units.join(mesh, left_on="jaw_mesh_id", right_on="mesh_id", prefix="jaw_").join(
        mesh,
        left_on="antagonist_mesh_id",
        right_on="mesh_id",
        rename={"mesh": "antagonist_mesh", "faces": "antagonist_faces"},
    )

    row = joined.first_row()
    assert row.faces == 3  # 자식 열은 그대로
    assert (row.jaw_faces, row.antagonist_faces) == (10, 20)
    assert row.jaw_mesh == {"name": "upper"}  # payload 메타 열이 함께 이름이 바뀌어 그대로 읽힘
    assert row.antagonist_mesh == {"name": "lower"}
    assert "mesh_id" not in joined.table.column_names  # 부모 키는 붙지 않는다


def test_rename_unknown_field_and_duplicate_target_are_errors(tmp_path: Path) -> None:
    mesh = _mesh_table(tmp_path)
    units = _unit_table(tmp_path, [{"unit": "u1", "jaw_mesh_id": "m1"}])

    with pytest.raises(ColumnConflictError, match="rename"):
        units.join(mesh, left_on="jaw_mesh_id", right_on="mesh_id", rename={"nope": "x"})
    with pytest.raises(ColumnConflictError, match="같은 이름"):
        units.join(mesh, left_on="jaw_mesh_id", right_on="mesh_id", rename={"faces": "mesh"})


def test_how_strict_inner_left_and_child_order(tmp_path: Path) -> None:
    mesh = _mesh_table(tmp_path)
    units = _unit_table(
        tmp_path,
        [
            {"unit": "u3", "jaw_mesh_id": "m2"},
            {"unit": "u1", "jaw_mesh_id": "missing"},
            {"unit": "u2", "jaw_mesh_id": "m1"},
        ],
    )

    with pytest.raises(JoinKeyError, match="부모가 없는 자식 행 1개"):
        units.join(mesh, left_on="jaw_mesh_id", right_on="mesh_id")

    inner = units.join(mesh, left_on="jaw_mesh_id", right_on="mesh_id", how="inner")
    assert [row.unit for row in inner] == ["u3", "u2"]  # 자식 순서 유지
    assert [row.faces for row in inner] == [20, 10]

    left = units.join(mesh, left_on="jaw_mesh_id", right_on="mesh_id", how="left")
    assert [row.unit for row in left] == ["u3", "u1", "u2"]
    assert [row.get("faces") for row in left] == [20, None, 10]


def test_duplicate_parent_key_is_an_error(tmp_path: Path) -> None:
    parent = write_artifact(
        artifact_name="dup",
        artifact_version="v1",
        rows=[{"k": "a", "v": 1, "n": 0}, {"k": "a", "v": 2, "n": 1}],
        out_dir=tmp_path / "dup" / "v1",
        unique_key_fields=("n",),
    ).table
    child = _unit_table(tmp_path, [{"unit": "u1", "k": "a"}])

    with pytest.raises(JoinKeyError, match="중복"):
        child.join(parent, on="k")


class TwoRoleChild(ArtifactBuilderBase):
    """같은 부모(mesh)를 역할별로 두 번 잇는다."""

    artifact_name = "two_role"
    artifact_version = "v1"
    unique_keys = [("unit",)]
    dependencies = [
        ArtifactDependency("mesh", "v1", join=ArtifactJoin(("jaw_mesh_id",), ("mesh_id",), prefix="jaw_")),
        ArtifactDependency("mesh", "v1", join=ArtifactJoin(("ant_mesh_id",), ("mesh_id",), prefix="ant_")),
    ]

    def prepare_table(self, **kwargs):
        return [{"unit": "u1", "jaw_mesh_id": "m1", "ant_mesh_id": "m2"}]

    def transform_row(self, row, **kwargs):
        return dict(row)


def test_same_parent_twice_through_join_parents(tmp_path: Path) -> None:
    mesh = _mesh_table(tmp_path)
    table = (
        TwoRoleChild()
        .build(
            storage_root=tmp_path,
            parent_tables=[mesh, mesh],
            policy=BuildPolicy(),
            executor_config=ExecutorConfig(row_workers=0, batch_size=1),
        )
        .table
    )

    row = LazyTable.open(tmp_path / "two_role" / "v1", join_parents=True).first_row()
    assert (row.jaw_mesh, row.ant_mesh) == ({"name": "upper"}, {"name": "lower"})
    assert table.meta["dependencies"][0]["join"]["prefix"] == "jaw_"


class CollidingChild(ArtifactBuilderBase):
    artifact_name = "colliding"
    artifact_version = "v1"
    unique_keys = [("mesh_id",)]
    dependencies = [ArtifactDependency("mesh", "v1", join=ArtifactJoin(("mesh_id",), ("mesh_id",)))]

    def transform_row(self, row, **kwargs):
        return {"mesh_id": row.raw("mesh_id"), "faces": 1}  # 부모 faces와 겹침


class KeylessChild(CollidingChild):
    artifact_name = "keyless"
    unique_keys = [("other",)]

    def transform_row(self, row, **kwargs):
        return {"other": row.raw("mesh_id")}


def test_build_rejects_output_that_collides_with_parent_or_lacks_join_key(tmp_path: Path) -> None:
    mesh = _mesh_table(tmp_path)
    config = ExecutorConfig(row_workers=0, batch_size=1)

    with pytest.raises(ColumnConflictError, match="faces"):
        CollidingChild().build(
            storage_root=tmp_path, parent_tables=[mesh], policy=BuildPolicy(), executor_config=config
        )
    with pytest.raises(JoinKeyError, match="mesh_id"):
        KeylessChild().build(
            storage_root=tmp_path, parent_tables=[mesh], policy=BuildPolicy(), executor_config=config
        )


class ParentsCollide(ArtifactBuilderBase):
    artifact_name = "parents_collide"
    artifact_version = "v1"
    unique_keys = [("unit",)]
    dependencies = [
        ArtifactDependency("mesh", "v1", join=ArtifactJoin(("a",), ("mesh_id",))),
        ArtifactDependency("mesh", "v1", join=ArtifactJoin(("b",), ("mesh_id",))),
    ]


def test_build_rejects_parents_that_would_collide(tmp_path: Path) -> None:
    mesh = _mesh_table(tmp_path)
    with pytest.raises(ColumnConflictError, match="앞선 부모"):
        ParentsCollide().build(
            storage_root=tmp_path,
            parent_tables=[mesh, mesh],
            policy=BuildPolicy(),
            executor_config=ExecutorConfig(row_workers=0, batch_size=1),
        )
