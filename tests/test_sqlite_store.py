from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from lazetable.sqlite_store import PayloadRecord, SQLiteArtifactStore


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
