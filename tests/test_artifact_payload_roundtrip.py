from __future__ import annotations

import time
from pathlib import Path

from lazetable import (
    ArtifactBuilderBase,
    BatchCardinalityError,
    BuildPolicy,
    DuplicateArtifactKeyError,
    ExecutorConfig,
    ExternalPayloadRef,
    LazyTable,
    Payload,
    ResolverContext,
    SandboxRegistry,
    register_payload_type,
    resolve_ref,
    write_artifact,
)


class JsonPayloadType:
    payload_id = "json_payload_v1"
    default_extension = "json"

    def dump_to_path(self, data, path: Path, *, metadata=None):
        import json

        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return {"encoding": "json"}

    def load_from_path(self, path: Path, *, metadata=None):
        import json

        return json.loads(path.read_text(encoding="utf-8"))

    def payload(self, data, **kwargs):
        return Payload(self, data, metadata=kwargs or None)

    def external_ref(self, ref: str, **kwargs):
        return ExternalPayloadRef(self, ref=ref, metadata=kwargs or None)


register_payload_type(JsonPayloadType)


def test_write_artifact_lowers_payloads(tmp_path: Path) -> None:
    out_dir = tmp_path / "toy" / "v1"
    result = write_artifact(
        artifact_name="toy",
        artifact_version="v1",
        rows=[
            {
                "id": "row-1",
                "payload_field": JsonPayloadType().payload({"value": 3}),
            }
        ],
        out_dir=out_dir,
        artifact_roots={"default": str(tmp_path)},
        unique_key_fields=("id",),
    )
    row = result.lowered_rows[0]
    assert (out_dir / "artifact.sqlite").is_file()
    assert not (out_dir / "manifest.parquet").exists()
    assert not (out_dir / "meta.json").exists()
    assert row["[payload_id]payload_field"] == "json_payload_v1"
    assert row["[payload_status]payload_field"] == "present"
    assert row["[payload_ref]payload_field"].startswith("art://toy@v1/payload/id=row-1/")

    table = LazyTable.open(out_dir)
    loaded = table.first_row()
    assert loaded.payload_field == {"value": 3}


def test_resolve_ref_for_artifact_scheme(tmp_path: Path) -> None:
    artifact_root = tmp_path / "toy" / "v1"
    context = ResolverContext(
        sandbox_registry=SandboxRegistry({}),
        artifact_roots={"toy@v1": str(artifact_root)},
    )
    path = resolve_ref("art://toy@v1/row-1/payload.json", context)
    assert path == artifact_root / "row-1" / "payload.json"


def test_write_artifact_allows_external_payload_ref(tmp_path: Path) -> None:
    src_root = tmp_path / "raw"
    src_root.mkdir(parents=True, exist_ok=True)
    src_file = src_root / "sample.json"
    src_file.write_text('{"value": 7}', encoding="utf-8")

    out_dir = tmp_path / "toy_external" / "v1"
    result = write_artifact(
        artifact_name="toy_external",
        artifact_version="v1",
        rows=[
            {
                "id": "row-1",
                "payload_field": JsonPayloadType().external_ref(
                    "src://demo/sample.json",
                    status="present",
                ),
            }
        ],
        out_dir=out_dir,
        artifact_roots={"default": str(tmp_path)},
        unique_key_fields=("id",),
    )
    row = result.lowered_rows[0]
    assert row["[payload_ref]payload_field"] == "src://demo/sample.json"

    table = LazyTable.open(
        out_dir,
        sandbox_registry=SandboxRegistry({"demo": src_root}),
    )
    loaded = table.first_row()
    assert loaded.payload_field == {"value": 7}


class CopyTransform(ArtifactBuilderBase):
    artifact_name = "copy_rows"
    artifact_version = "v1"
    unique_keys = [("id",)]

    def prepare_table(
        self, *, parent_rows=None, build_context=None, sandbox_registry=None, seed_plan=None
    ):
        provided_rows = list((build_context or {}).get("input_rows") or [])
        return provided_rows

    def transform_row(
        self, row, *, row_idx, parent_rows, build_context, sandbox_registry=None, seed_plan=None
    ):
        return {"id": row["id"], "x2": int(row["x"]) * 2}


def test_build_artifact_reference_path(tmp_path: Path) -> None:
    artifact = CopyTransform().build(
        storage_root=tmp_path,
        policy=BuildPolicy(),
        build_context={"input_rows": [{"id": "a", "x": 2}, {"id": "b", "x": 5}]},
    )
    rows = artifact.table.table.to_pylist()
    assert rows == [{"id": "a", "x2": 4}, {"id": "b", "x2": 10}]


class QueuedTransform(ArtifactBuilderBase):
    artifact_name = "queued_rows"
    artifact_version = "v1"
    unique_keys = [("source", "value")]

    def prepare_table(
        self, *, parent_rows=None, build_context=None, sandbox_registry=None, seed_plan=None
    ):
        provided_rows = list((build_context or {}).get("input_rows") or [])
        return provided_rows

    def transform_row(
        self, row, *, row_idx, parent_rows, build_context, sandbox_registry=None, seed_plan=None
    ):
        # Reverse sleep order to ensure completion order differs from input order.
        time.sleep(0.01 * (3 - row_idx))
        return [
            {"source": row["id"], "value": int(row["x"])},
            {"source": row["id"], "value": int(row["x"]) + 10},
        ]

    def transform_batch(
        self, items, *, batch_idx, parent_rows, build_context, sandbox_registry=None, seed_plan=None
    ):
        return [
            {
                "batch_idx": batch_idx,
                "source": item["source"],
                "value": item["value"],
            }
            for item in items
        ]


def test_build_artifact_queued_pipeline_preserves_row_order(tmp_path: Path) -> None:
    artifact = QueuedTransform().build(
        storage_root=tmp_path,
        policy=BuildPolicy(),
        build_context={
            "input_rows": [{"id": "a", "x": 1}, {"id": "b", "x": 2}, {"id": "c", "x": 3}]
        },
        executor_config=ExecutorConfig(row_workers=3, batch_size=2),
    )
    rows = artifact.table.table.to_pylist()
    assert [(row["source"], row["value"]) for row in rows] == [
        ("a", 1),
        ("a", 11),
        ("b", 2),
        ("b", 12),
        ("c", 3),
        ("c", 13),
    ]
    batch_indices = [int(row["batch_idx"]) for row in rows]
    assert batch_indices == sorted(batch_indices)


def test_build_artifact_process_pipeline_preserves_row_order(tmp_path: Path) -> None:
    artifact = QueuedTransform().build(
        storage_root=tmp_path,
        policy=BuildPolicy(),
        build_context={
            "input_rows": [{"id": "a", "x": 1}, {"id": "b", "x": 2}, {"id": "c", "x": 3}]
        },
        executor_config=ExecutorConfig(row_workers=2, batch_size=2, row_executor="process"),
    )
    rows = artifact.table.table.to_pylist()
    assert [(row["source"], row["value"]) for row in rows] == [
        ("a", 1),
        ("a", 11),
        ("b", 2),
        ("b", 12),
        ("c", 3),
        ("c", 13),
    ]
    batch_indices = [int(row["batch_idx"]) for row in rows]
    assert batch_indices == sorted(batch_indices)


class BadBatchTransform(ArtifactBuilderBase):
    artifact_name = "bad_batch"
    artifact_version = "v1"
    unique_keys = [("id",)]

    def prepare_table(
        self, *, parent_rows=None, build_context=None, sandbox_registry=None, seed_plan=None
    ):
        provided_rows = list((build_context or {}).get("input_rows") or [])
        return provided_rows

    def transform_row(
        self, row, *, row_idx, parent_rows, build_context, sandbox_registry=None, seed_plan=None
    ):
        return {"id": row["id"], "x": int(row["x"])}

    def transform_batch(
        self, items, *, batch_idx, parent_rows, build_context, sandbox_registry=None, seed_plan=None
    ):
        return items[:-1]


def test_transform_batch_cardinality_violation_raises(tmp_path: Path) -> None:
    try:
        BadBatchTransform().build(
            storage_root=tmp_path,
            policy=BuildPolicy(),
            build_context={"input_rows": [{"id": "a", "x": 1}, {"id": "b", "x": 2}]},
            executor_config=ExecutorConfig(row_workers=1, batch_size=2),
        )
    except BatchCardinalityError:
        pass
    else:
        raise AssertionError("expected BatchCardinalityError")


class DuplicateKeyTransform(ArtifactBuilderBase):
    artifact_name = "dup_key"
    artifact_version = "v1"
    unique_keys = [("id",)]

    def prepare_table(
        self, *, parent_rows=None, build_context=None, sandbox_registry=None, seed_plan=None
    ):
        provided_rows = list((build_context or {}).get("input_rows") or [])
        return provided_rows

    def transform_row(
        self, row, *, row_idx, parent_rows, build_context, sandbox_registry=None, seed_plan=None
    ):
        return {"id": "dup", "x": int(row["x"])}


def test_duplicate_unique_key_raises(tmp_path: Path) -> None:
    try:
        DuplicateKeyTransform().build(
            storage_root=tmp_path,
            policy=BuildPolicy(),
            build_context={"input_rows": [{"id": "a", "x": 1}, {"id": "b", "x": 2}]},
            executor_config=ExecutorConfig(row_workers=1),
        )
    except DuplicateArtifactKeyError:
        pass
    else:
        raise AssertionError("expected DuplicateArtifactKeyError")


class ThreadRecordingTransform(ArtifactBuilderBase):
    artifact_name = "thread_record"
    artifact_version = "v1"
    unique_keys = [("id",)]

    def prepare_table(
        self, *, parent_rows=None, build_context=None, sandbox_registry=None, seed_plan=None
    ):
        provided_rows = list((build_context or {}).get("input_rows") or [])
        return provided_rows

    def transform_row(
        self, row, *, row_idx, parent_rows, build_context, sandbox_registry=None, seed_plan=None
    ):
        import threading

        return {
            "id": row["id"],
            "thread_name": threading.current_thread().name,
        }


def test_row_workers_zero_runs_transform_row_on_main_thread(tmp_path: Path) -> None:
    artifact = ThreadRecordingTransform().build(
        storage_root=tmp_path,
        policy=BuildPolicy(),
        build_context={"input_rows": [{"id": "a"}]},
        executor_config=ExecutorConfig(row_workers=0, batch_size=1, row_executor="thread"),
    )
    row = artifact.table.table.to_pylist()[0]
    assert row["thread_name"] == "MainThread"


def test_row_workers_one_runs_transform_row_on_worker_thread(tmp_path: Path) -> None:
    artifact = ThreadRecordingTransform().build(
        storage_root=tmp_path,
        policy=BuildPolicy(),
        build_context={"input_rows": [{"id": "a"}]},
        executor_config=ExecutorConfig(row_workers=1, batch_size=1, row_executor="thread"),
    )
    row = artifact.table.table.to_pylist()[0]
    assert row["thread_name"] != "MainThread"
