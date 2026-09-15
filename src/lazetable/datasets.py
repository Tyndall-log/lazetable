"""학습 순회를 위한 불변 metadata index와 Torch 호환 adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Protocol

from .core import LazyTable, ParsedRow, ResolverContext, payload_columns_for


class SampleAdapter(Protocol):
    def __call__(
        self,
        row: ParsedRow,
        *,
        index: int,
        epoch: int,
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class DatasetRecord:
    values: Mapping[str, Any]


class DatasetIndex:
    """SQLite에서 한 번 preload한 뒤 메모리에서만 순회하는 불변 index."""

    def __init__(
        self,
        records: tuple[DatasetRecord, ...],
        *,
        resolver_context: ResolverContext,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        self._records = records
        self._resolver_context = resolver_context
        self._context = dict(context or {})

    @classmethod
    def from_table(
        cls,
        table: LazyTable,
        *,
        where: Mapping[str, Any] | None = None,
        fields: list[str] | None = None,
    ) -> DatasetIndex:
        filters = dict(where or {})
        selected_columns = None
        if fields is not None:
            selected_columns = set(payload_columns_for(fields, table.table.column_names))
        records: list[DatasetRecord] = []
        for values in table.table.to_pylist():
            if any(values.get(key) != expected for key, expected in filters.items()):
                continue
            projected = (
                values
                if selected_columns is None
                else {key: value for key, value in values.items() if key in selected_columns}
            )
            records.append(DatasetRecord(projected))
        return cls(
            tuple(records),
            resolver_context=table.resolver_context,
            context=table.context,
        )

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> ParsedRow:
        record = self._records[index]
        return ParsedRow(
            record.values,
            resolver_context=self._resolver_context,
            context=self._context,
        )

    def __iter__(self) -> Iterator[ParsedRow]:
        for index in range(len(self)):
            yield self[index]

    def to_torch_dataset(self, adapter: SampleAdapter) -> TorchDatasetAdapter:
        return TorchDatasetAdapter(self, adapter)


class TorchDatasetAdapter:
    """torch import 없이 `torch.utils.data.Dataset` protocol을 만족한다."""

    def __init__(self, index: DatasetIndex, adapter: SampleAdapter) -> None:
        self.index = index
        self.adapter = adapter

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, encoded_index: int) -> Any:
        epoch, index = self.decode_index(encoded_index)
        return self.adapter(self.index[index], index=index, epoch=epoch)

    def encode_index(self, index: int, *, epoch: int) -> int:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        return epoch * len(self) + index

    def decode_index(self, encoded_index: int) -> tuple[int, int]:
        if len(self) == 0:
            raise IndexError(encoded_index)
        if encoded_index < 0:
            raise IndexError(encoded_index)
        return divmod(encoded_index, len(self))


def identity_adapter(
    row: ParsedRow,
    *,
    index: int,
    epoch: int,
) -> dict[str, Any]:
    return {"row": row, "index": index, "epoch": epoch}
