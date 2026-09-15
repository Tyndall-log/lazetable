from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from lazetable import (
    ArtifactDependency,
    ArtifactBuilderBase,
    ArtifactJoin,
    BuildPolicy,
    ExecutorConfig,
    FinalizeResult,
    LazyTable,
    Payload,
    WriterConfig,
    register_payload_type,
    write_artifact,
)


@register_payload_type
class ThreadPayloadType:
    payload_id = "thread_payload_v1"
    default_extension = "json"

    def payload(self, data, **kwargs):
        return Payload(self, data, metadata=kwargs or None)

    def dump_to_path(self, data, path: Path, *, metadata=None):
        path.write_text(
            json.dumps({"value": data, "thread": threading.current_thread().name}),
            encoding="utf-8",
        )
        return {"encoding": "json"}

    def load_from_path(self, path: Path, *, metadata=None):
        return json.loads(path.read_text(encoding="utf-8"))


class StreamingTransform(ArtifactBuilderBase):
    artifact_name = "streaming"
    artifact_version = "v1"
    unique_keys = [("id",)]

    def prepare_table(self, *, build_context=None, **kwargs):
        return list(build_context["rows"])

    def transform_row(self, row, **kwargs):
        return {"id": row["id"], "payload": ThreadPayloadType().payload(row["value"])}


def test_payload_is_written_by_background_writer_and_reused(tmp_path: Path) -> None:
    builder = StreamingTransform()
    context = {"rows": [{"id": "a", "value": 3}]}
    first = builder.build(
        storage_root=tmp_path,
        build_context=context,
        policy=BuildPolicy(),
        executor_config=ExecutorConfig(row_workers=0, batch_size=1),
        writer_config=WriterConfig(payload_workers=1, max_pending_writes=1),
    )
    loaded = first.table.first_row().payload

    assert first.status == "built"
    assert loaded["value"] == 3
    assert loaded["thread"].startswith("lazetable-payload")

    second = builder.build(
        storage_root=tmp_path,
        build_context=context,
        policy=BuildPolicy(),
        executor_config=ExecutorConfig(row_workers=0, batch_size=1),
    )
    assert second.status == "reused"
    assert second.skipped_source_rows == 1


class ResumeTransform(ArtifactBuilderBase):
    artifact_name = "resume"
    artifact_version = "v1"
    unique_keys = [("id",)]

    def prepare_table(self, *, build_context=None, **kwargs):
        return [{"id": "a"}, {"id": "b"}, {"id": "c"}]

    def transform_row(self, row, *, build_context, **kwargs):
        sentinel = Path(build_context["sentinel"])
        if row["id"] == "b" and not sentinel.exists():
            sentinel.write_text("failed once", encoding="utf-8")
            raise RuntimeError("intentional interruption")
        return {"id": row["id"], "value": row["id"].upper()}


def test_partial_build_resumes_completed_source_rows(tmp_path: Path) -> None:
    builder = ResumeTransform()
    context = {"sentinel": str((tmp_path / "sentinel").resolve())}
    with pytest.raises(RuntimeError, match="intentional interruption"):
        builder.build(
            storage_root=tmp_path,
            build_context=context,
            policy=BuildPolicy(),
            executor_config=ExecutorConfig(row_workers=0, batch_size=1),
        )

    with pytest.raises(ValueError, match="not completed"):
        LazyTable.open(tmp_path / "resume" / ".v1.building")

    resumed = builder.build(
        storage_root=tmp_path,
        build_context=context,
        policy=BuildPolicy(reuse_completed=True, resume_partial=True, retry_failed=True),
        executor_config=ExecutorConfig(row_workers=0, batch_size=1),
    )

    assert resumed.status == "resumed"
    assert resumed.skipped_source_rows == 1
    assert resumed.processed_source_rows == 2
    assert [row.id for row in resumed.table] == ["a", "b", "c"]


def test_completed_signature_mismatch_is_rejected(tmp_path: Path) -> None:
    builder = StreamingTransform()
    builder.build(
        storage_root=tmp_path,
        build_context={"rows": [{"id": "a", "value": 1}]},
        policy=BuildPolicy(),
    )

    with pytest.raises(ValueError, match="signature mismatch"):
        builder.build(
            storage_root=tmp_path,
            build_context={"rows": [{"id": "a", "value": 2}]},
            policy=BuildPolicy(),
        )


class StatisticalTransform(ArtifactBuilderBase):
    artifact_name = "statistics"
    artifact_version = "v1"
    unique_keys = [("id",)]

    def prepare_table(self, **kwargs):
        return [{"id": "a", "value": 2}, {"id": "b", "value": 5}]

    def transform_row(self, row, **kwargs):
        return dict(row)

    def finalize_manifest(self, rows, **kwargs):
        return FinalizeResult(
            rows=[{**row, "normalized": row["value"] / 5} for row in rows],
            metadata={"normalization": "max"},
            statistics={"value": {"count": len(rows), "max": 5}},
        )


def test_finalize_rows_metadata_and_statistics_are_persisted(tmp_path: Path) -> None:
    result = StatisticalTransform().build(
        storage_root=tmp_path,
        policy=BuildPolicy(),
        executor_config=ExecutorConfig(row_workers=0, batch_size=1),
    )

    assert [row.normalized for row in result.table] == [0.4, 1.0]
    assert result.table.meta["normalization"] == "max"
    assert result.table.meta["statistics"] == {"value": {"count": 2, "max": 5}}


class ChildTransform(ArtifactBuilderBase):
    artifact_name = "child"
    artifact_version = "v1"
    unique_keys = [("id",)]
    dependencies = [
        ArtifactDependency(
            "parent",
            "v1",
            join=ArtifactJoin(child_on=("id",), parent_on=("id",)),
        )
    ]

    def transform_row(self, row, **kwargs):
        return {"id": row.raw("id"), "child_value": 7}


def test_sqlite_parent_join_preserves_parent_payload_resolution(tmp_path: Path) -> None:
    parent = write_artifact(
        artifact_name="parent",
        artifact_version="v1",
        rows=[{"id": "a", "payload": ThreadPayloadType().payload(11)}],
        out_dir=tmp_path / "parent" / "v1",
        artifact_roots={"default": str(tmp_path)},
        unique_key_fields=("id",),
    ).table
    child = (
        ChildTransform()
        .build(
            storage_root=tmp_path,
            parent_tables=[parent],
            policy=BuildPolicy(),
            executor_config=ExecutorConfig(row_workers=0, batch_size=1),
        )
        .table
    )

    joined = child.join_parents()

    assert joined.first_row().child_value == 7
    assert joined.first_row().payload["value"] == 11


class FinalizeResumeTransform(StatisticalTransform):
    artifact_name = "finalize_resume"

    def finalize_manifest(self, rows, *, build_context, **kwargs):
        sentinel = Path(build_context["finalize_sentinel"])
        if not sentinel.exists():
            sentinel.write_text("failed once", encoding="utf-8")
            raise RuntimeError("finalize interrupted")
        return super().finalize_manifest(rows, **kwargs)


def test_finalize_is_recomputed_from_all_raw_rows_after_resume(tmp_path: Path) -> None:
    builder = FinalizeResumeTransform()
    context = {"finalize_sentinel": str((tmp_path / "finalize-sentinel").resolve())}
    with pytest.raises(RuntimeError, match="finalize interrupted"):
        builder.build(
            storage_root=tmp_path,
            build_context=context,
            policy=BuildPolicy(),
            executor_config=ExecutorConfig(row_workers=0, batch_size=1),
        )

    resumed = builder.build(
        storage_root=tmp_path,
        build_context=context,
        policy=BuildPolicy(),
        executor_config=ExecutorConfig(row_workers=0, batch_size=1),
    )

    assert resumed.status == "resumed"
    assert resumed.processed_source_rows == 0
    assert resumed.skipped_source_rows == 2
    assert [row.normalized for row in resumed.table] == [0.4, 1.0]
    assert resumed.table.meta["statistics"]["value"] == {"count": 2, "max": 5}
