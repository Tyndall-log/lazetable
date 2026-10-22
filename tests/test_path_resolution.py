from __future__ import annotations

import hashlib
import json
import os
import shutil
import warnings
from pathlib import Path

import pytest

from lazetable import (
    ArtifactBuilderBase,
    ArtifactDependency,
    ArtifactJoin,
    ArtifactPathFallbackWarning,
    ArtifactStore,
    ArtifactStorePaths,
    BuildPolicy,
    ExecutorConfig,
    LazyTable,
    Payload,
    ResolverContext,
    register_payload_type,
    resolve_ref,
    write_artifact,
)


@register_payload_type
class PathJsonPayloadType:
    payload_id = "path_resolution_json_v1"
    default_extension = "json"

    def payload(self, data, **kwargs):
        return Payload(self, data, metadata=kwargs or None)

    def dump_to_path(self, data, path: Path, *, metadata=None):
        path.write_text(json.dumps(data), encoding="utf-8")
        return {"encoding": "json"}

    def load_from_path(self, path: Path, *, metadata=None):
        return json.loads(path.read_text(encoding="utf-8"))


class PathChildTransform(ArtifactBuilderBase):
    artifact_name = "child"
    artifact_version = "v1"
    unique_keys = [("id",)]
    dependencies = [
        ArtifactDependency("parent", "v1", join=ArtifactJoin(child_on=("id",), parent_on=("id",)))
    ]

    def transform_row(self, row, **kwargs):
        return {"id": row.raw("id"), "child_payload": PathJsonPayloadType().payload(7)}


def _build(storage_root: Path) -> None:
    """``storage_root``에 절대 artifact_roots가 저장된 parent/child artifact를 만든다."""

    parent = write_artifact(
        artifact_name="parent",
        artifact_version="v1",
        rows=[{"id": "a", "parent_payload": PathJsonPayloadType().payload(11)}],
        out_dir=storage_root / "parent" / "v1",
        artifact_roots={"default": str(storage_root)},
        unique_key_fields=("id",),
    ).table
    PathChildTransform().build(
        storage_root=storage_root,
        parent_tables=[parent],
        policy=BuildPolicy(),
        executor_config=ExecutorConfig(row_workers=0, batch_size=1),
    )


def _sha256_tree(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _assert_joined(table: LazyTable) -> None:
    row = table.first_row()
    assert row.child_payload == 7
    assert row.parent_payload == 11


@pytest.fixture
def built(tmp_path: Path) -> Path:
    root = tmp_path / "built"
    _build(root)
    return root


def test_local_open_uses_relative_resolution_by_default(built: Path) -> None:
    table = LazyTable.open(built / "child" / "v1", join_parents=True)

    assert table.resolver_context.path_resolution == "relative"
    _assert_joined(table)


def test_copied_artifacts_resolve_relative_without_modifying_files(
    built: Path, tmp_path: Path
) -> None:
    moved = tmp_path / "moved"
    shutil.copytree(built, moved)
    shutil.rmtree(built)
    before = _sha256_tree(moved)

    with warnings.catch_warnings():
        warnings.simplefilter("error", ArtifactPathFallbackWarning)
        _assert_joined(LazyTable.open(moved / "child" / "v1", join_parents=True))
        store = ArtifactStore(ArtifactStorePaths(artifact_roots={"default": moved}))
        _assert_joined(store.open("child", "v1", join_parents=True))
        _assert_joined(LazyTable.open(moved / "child" / "v1").join_parents())

    assert _sha256_tree(moved) == before


def test_absolute_option_keeps_stored_roots(built: Path, tmp_path: Path) -> None:
    moved = tmp_path / "moved"
    shutil.copytree(built, moved)

    absolute = LazyTable.open(moved / "child" / "v1", join_parents=True, path_resolution="absolute")
    assert Path(absolute.resolver_context.artifact_roots["child@v1"]) == built / "child" / "v1"
    _assert_joined(absolute)

    store = ArtifactStore(
        ArtifactStorePaths(artifact_roots={"default": moved}, path_resolution="absolute")
    )
    shutil.rmtree(built)
    with pytest.raises(FileNotFoundError):
        store.open("child", "v1", join_parents=True).first_row().child_payload
    _assert_joined(store.open("child", "v1", join_parents=True, path_resolution="relative"))


@pytest.mark.parametrize("override_kind", ["absolute", "relative"])
def test_root_overrides_locate_parent_moved_elsewhere(
    built: Path, tmp_path: Path, override_kind: str
) -> None:
    store_root = tmp_path / "new" / "store"
    parent_dir = tmp_path / "new" / "parents" / "parent-v1"
    shutil.copytree(built / "child" / "v1", store_root / "child" / "v1")
    shutil.copytree(built / "parent" / "v1", parent_dir)
    shutil.rmtree(built)
    value = str(parent_dir) if override_kind == "absolute" else "../parents/parent-v1"

    with warnings.catch_warnings():
        warnings.simplefilter("error", ArtifactPathFallbackWarning)
        _assert_joined(
            LazyTable.open(
                store_root / "child" / "v1",
                join_parents=True,
                root_overrides={"parent@v1": value},
            )
        )
        store = ArtifactStore(
            ArtifactStorePaths(
                artifact_roots={"default": store_root}, root_overrides={"parent@v1": value}
            )
        )
        _assert_joined(store.open("child", "v1", join_parents=True))


def test_root_overrides_win_in_absolute_mode(built: Path, tmp_path: Path) -> None:
    moved = tmp_path / "moved"
    shutil.copytree(built, moved)
    shutil.rmtree(built)

    table = LazyTable.open(
        moved / "child" / "v1",
        join_parents=True,
        path_resolution="absolute",
        root_overrides={
            "child@v1": moved / "child" / "v1",
            "parent@v1": moved / "parent" / "v1",
        },
    )

    _assert_joined(table)


def test_missing_relative_parent_falls_back_to_stored_root_with_warning(
    built: Path, tmp_path: Path
) -> None:
    store_root = tmp_path / "child-only"
    shutil.copytree(built / "child" / "v1", store_root / "child" / "v1")

    with pytest.warns(ArtifactPathFallbackWarning, match="parent@v1"):
        table = LazyTable.open(store_root / "child" / "v1", join_parents=True)
    _assert_joined(table)

    shutil.rmtree(built)
    with pytest.warns(ArtifactPathFallbackWarning), pytest.raises(FileNotFoundError):
        LazyTable.open(store_root / "child" / "v1", join_parents=True)


def test_relative_context_derives_unknown_namespace_from_default(tmp_path: Path) -> None:
    relative = ResolverContext(None, {"default": str(tmp_path)}, "relative")
    absolute = ResolverContext(None, {"default": str(tmp_path)})

    assert resolve_ref("art://other@v2/payload/x.json", relative) == (
        tmp_path / "other" / "v2" / "payload" / "x.json"
    )
    with pytest.raises(KeyError, match="other@v2"):
        resolve_ref("art://other@v2/payload/x.json", absolute)


@pytest.mark.skipif(os.name != "nt", reason="UNC 경로는 Windows 전용입니다")
def test_copied_artifacts_open_through_windows_unc_path(built: Path, tmp_path: Path) -> None:
    moved = tmp_path / "moved"
    shutil.copytree(built, moved)
    shutil.rmtree(built)
    resolved = moved.resolve()
    unc = Path(rf"\\localhost\{resolved.drive.rstrip(':')}$") / resolved.relative_to(
        resolved.anchor
    )
    if not unc.exists():
        pytest.skip("관리 공유(\\\\localhost\\<drive>$)에 접근할 수 없습니다")

    with warnings.catch_warnings():
        warnings.simplefilter("error", ArtifactPathFallbackWarning)
        table = LazyTable.open(unc / "child" / "v1", join_parents=True)
    assert table.resolver_context.artifact_roots["parent@v1"].startswith("\\\\localhost\\")
    _assert_joined(table)
