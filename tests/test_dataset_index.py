from __future__ import annotations

import pickle
import hashlib
from pathlib import Path

import pytest

from lazetable import (
    DatasetIndex,
    Payload,
    TorchDatasetAdapter,
    register_payload_type,
    write_artifact,
)


@register_payload_type
class TextPayloadType:
    payload_id = "dataset_index_text_v1"
    default_extension = "txt"

    def payload(self, data, **kwargs):
        return Payload(self, data, metadata=kwargs or None)

    def dump_to_path(self, data, path: Path, *, metadata=None):
        path.write_text(str(data), encoding="utf-8")
        return {"encoding": "utf-8"}

    def load_from_path(self, path: Path, *, metadata=None):
        return path.read_text(encoding="utf-8")


def _table(tmp_path: Path):
    return write_artifact(
        artifact_name="dataset",
        artifact_version="v1",
        rows=[
            {"id": "a", "split": "train", "fdi": 31, "data": TextPayloadType().payload("A")},
            {"id": "b", "split": "validation", "fdi": 32, "data": TextPayloadType().payload("B")},
            {"id": "c", "split": "train", "fdi": 33, "data": TextPayloadType().payload("C")},
        ],
        out_dir=tmp_path / "dataset" / "v1",
        artifact_roots={"default": str(tmp_path)},
        unique_key_fields=("id",),
    ).table


def test_dataset_index_filters_and_projects_payload_fields(tmp_path: Path) -> None:
    index = _table(tmp_path).to_dataset_index(
        where={"split": "train"},
        fields=["id", "fdi", "data"],
    )

    assert isinstance(index, DatasetIndex)
    assert len(index) == 2
    assert [row.raw("id") for row in index] == ["a", "c"]
    assert index[0].data == "A"
    assert "split" not in index[0].fields()


def test_dataset_index_is_pickle_safe(tmp_path: Path) -> None:
    index = _table(tmp_path).to_dataset_index(where={"split": "train"})
    restored = pickle.loads(pickle.dumps(index))

    assert len(restored) == 2
    assert restored[1].data == "C"


def test_torch_adapter_encodes_epoch_without_torch_dependency(tmp_path: Path) -> None:
    index = _table(tmp_path).to_dataset_index(where={"split": "train"})
    dataset = TorchDatasetAdapter(
        index,
        lambda row, *, index, epoch: {
            "id": row.raw("id"),
            "index": index,
            "epoch": epoch,
        },
    )

    encoded = dataset.encode_index(1, epoch=4)

    assert dataset[encoded] == {"id": "c", "index": 1, "epoch": 4}


def test_parquet_export_does_not_modify_sqlite_artifact(tmp_path: Path) -> None:
    table = _table(tmp_path)
    sqlite_path = table.path / "artifact.sqlite"
    before = hashlib.sha256(sqlite_path.read_bytes()).hexdigest()

    output = table.export_parquet(tmp_path / "exports" / "dataset.parquet")

    assert output.is_file()
    assert hashlib.sha256(sqlite_path.read_bytes()).hexdigest() == before


class EchoAdapter:
    def __call__(self, row, *, index: int, epoch: int):
        return {"id": row.raw("id"), "index": index, "epoch": epoch}


def test_torch_dataloader_workers_support_encoded_epoch(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    index = _table(tmp_path).to_dataset_index(where={"split": "train"})
    dataset = TorchDatasetAdapter(index, EchoAdapter())
    sampler = [dataset.encode_index(item, epoch=3) for item in range(len(dataset))]
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        sampler=sampler,
        num_workers=2,
        persistent_workers=True,
    )
    try:
        batch = next(iter(loader))
    finally:
        iterator = getattr(loader, "_iterator", None)
        if iterator is not None:
            iterator._shutdown_workers()

    assert batch["id"] == ["a", "c"]
    assert batch["epoch"].tolist() == [3, 3]
