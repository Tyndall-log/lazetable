from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from lazetable import (
    ArtifactBuilderBase,
    BuildPolicy,
    ExecutorConfig,
    Payload,
    WriterConfig,
    register_payload_type,
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
