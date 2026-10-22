from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest

from lazetable.sqlite_store import PayloadRecord, SQLiteArtifactStore, _sqlite_uri


def _store(tmp_path: Path) -> SQLiteArtifactStore:
    store = SQLiteArtifactStore(tmp_path / "artifact.sqlite")
    store.initialize_build(
        artifact_name="toy",
        artifact_version="v1",
        build_signature="signature",
        signature_payload={"builder": "toy"},
        dependencies=[],
    )
    store.ensure_source_rows(2)
    return store


def test_source_commit_is_atomic_and_queryable(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.commit_source(
        0,
        [
            (
                0,
                '["row-a"]',
                {"id": "row-a", "value": 3},
                [PayloadRecord("data", "json", "payload/a.json", "present", 3, "abc")],
            )
        ],
    )

    assert store.completed_sources() == {0}
    assert store.raw_rows() == [{"id": "row-a", "value": 3}]


def test_source_transaction_rolls_back_on_duplicate_key(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.commit_source(0, [(0, '["same"]', {"id": "a"}, [])])

    with pytest.raises(sqlite3.IntegrityError):
        store.commit_source(1, [(0, '["same"]', {"id": "b"}, [])])

    assert store.completed_sources() == {0}
    assert store.raw_rows() == [{"id": "a"}]


def test_finalize_replaces_statistics_transactionally(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.commit_source(0, [(0, '["a"]', {"id": "a"}, [])])
    store.commit_source(1, [(0, '["b"]', {"id": "b"}, [])])
    store.finalize(
        rows=[
            {"__unique_key__": ["a"], "id": "a", "split": "train"},
            {"__unique_key__": ["b"], "id": "b", "split": "validation"},
        ],
        metadata={"purpose": "test"},
        statistics={"split": {"train": 1, "validation": 1}},
        content_hash="content",
    )

    artifact = store.artifact(read_only=True)
    assert artifact is not None
    assert artifact.status == "completed"
    assert store.statistics() == {"split": {"train": 1, "validation": 1}}
    assert [row["id"] for row in store.final_rows()] == ["a", "b"]


def test_multiple_readers_can_open_completed_store(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.commit_source(0, [(0, '["a"]', {"id": "a"}, [])])
    store.commit_source(1, [])
    store.finalize(
        rows=[{"__unique_key__": ["a"], "id": "a"}],
        metadata={},
        statistics={},
        content_hash="content",
    )
    store.checkpoint_and_close()

    assert SQLiteArtifactStore(store.path).final_rows() == [{"__unique_key__": ["a"], "id": "a"}]
    assert SQLiteArtifactStore(store.path).final_rows() == [{"__unique_key__": ["a"], "id": "a"}]


def _completed_store(tmp_path: Path) -> SQLiteArtifactStore:
    store = _store(tmp_path)
    store.commit_source(0, [(0, '["a"]', {"id": "a"}, [])])
    store.commit_source(1, [])
    store.finalize(
        rows=[{"__unique_key__": ["a"], "id": "a"}],
        metadata={},
        statistics={},
        content_hash="content",
    )
    store.checkpoint_and_close()
    return store


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (PureWindowsPath(r"C:\data\a.sqlite"), "file:///C:/data/a.sqlite?mode=ro"),
        (PureWindowsPath(r"\\nas\share\a.sqlite"), "file:////nas/share/a.sqlite?mode=ro"),
        (PureWindowsPath(r"\\?\UNC\nas\share\a.sqlite"), "file:////nas/share/a.sqlite?mode=ro"),
        (PureWindowsPath(r"\\?\C:\data\a.sqlite"), "file:///C:/data/a.sqlite?mode=ro"),
        (PurePosixPath("/srv/data/a.sqlite"), "file:///srv/data/a.sqlite?mode=ro"),
        (PurePosixPath("data/a.sqlite"), "file:data/a.sqlite?mode=ro"),
        (
            PurePosixPath("/srv/a#b c%d?e/데이터.sqlite"),
            "file:///srv/a%23b%20c%25d%3Fe/%EB%8D%B0%EC%9D%B4%ED%84%B0.sqlite?mode=ro",
        ),
    ],
)
def test_sqlite_uri_keeps_path_out_of_authority_and_query(path, expected: str) -> None:
    assert _sqlite_uri(path, "mode=ro") == expected


def test_read_only_open_handles_uri_special_characters(tmp_path: Path) -> None:
    directory = tmp_path / "a#b c%d 데이터"
    directory.mkdir()
    store = _completed_store(directory)

    for immutable in (False, True):
        with closing(store._connect(read_only=True, immutable=immutable)) as connection:
            assert connection.execute("SELECT count(*) FROM final_output_rows").fetchone()[0] == 1
    assert SQLiteArtifactStore(store.path).artifact(read_only=True).status == "completed"


@pytest.mark.skipif(os.name != "nt", reason="UNC 경로는 Windows 전용입니다")
def test_read_only_open_supports_windows_unc_path(tmp_path: Path) -> None:
    store = _completed_store(tmp_path)
    resolved = store.path.resolve()
    drive = resolved.drive.rstrip(":")
    unc = Path(rf"\\localhost\{drive}$") / resolved.relative_to(resolved.anchor)
    if not unc.exists():
        pytest.skip("관리 공유(\\\\localhost\\<drive>$)에 접근할 수 없습니다")

    unc_store = SQLiteArtifactStore(unc)
    assert unc_store.artifact(read_only=True).status == "completed"
    with closing(unc_store._connect(read_only=True, immutable=True)) as connection:
        assert connection.execute("SELECT count(*) FROM final_output_rows").fetchone()[0] == 1


def test_orphan_cleanup_removes_unreferenced_payload(tmp_path: Path) -> None:
    store = _store(tmp_path)
    payload_dir = tmp_path / "payload"
    payload_dir.mkdir()
    orphan = payload_dir / "orphan.bin"
    orphan.write_bytes(b"orphan")

    assert store.remove_orphan_payloads(tmp_path) == 1
    assert not orphan.exists()


def test_payload_validation_detects_corruption(tmp_path: Path) -> None:
    store = _store(tmp_path)
    payload = tmp_path / "payload" / "data.bin"
    payload.parent.mkdir()
    payload.write_bytes(b"actual")
    store.commit_source(
        0,
        [
            (
                0,
                '["a"]',
                {"id": "a"},
                [PayloadRecord("data", "binary", "payload/data.bin", "present", 6, "wrong")],
            )
        ],
    )

    with pytest.raises(ValueError, match="hash mismatch"):
        store.validate_payloads(tmp_path)
