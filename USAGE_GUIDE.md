# LazeTable Usage Guide

## What It Is

`lazetable` is a table-oriented artifact runtime.

It is designed for datasets where:

- row metadata should stay light
- large values should live outside the main table
- those large values should be loaded lazily
- derived artifacts should be built through a structured pipeline

At the moment, the core ideas are:

- `LazyTable`
- `Payload` / `PayloadType`
- `ArtifactBuilderBase`

---

## Install

This project currently uses `uv`.

```powershell
cd C:\ai\lazetable
uv sync --extra dev
```

Run tests:

```powershell
uv run pytest
```

---

## Artifact Layout

Every artifact uses the same layout:

```text
<artifact_dir>/
  artifact.sqlite
  payload/
```

Meaning:

- `artifact.sqlite`
  - row metadata, payload ref, dependency, build journal, statistics
- `payload/`
  - large payload files written outside the table

---

## Core Concepts

### `LazyTable`

`LazyTable` reads a SQLite-backed artifact table and gives row access through
`ParsedRow`.

Lazy payload fields are reconstructed from `[payload_*]field_name` metadata.

### `PayloadType`

A `PayloadType` defines how a large value is written to disk and later loaded
back.

It owns:

- `payload_id`
- `default_extension`
- `dump_to_path(...)`
- `load_from_path(...)`

### `Payload`

`Payload` is a declaration object used in output rows while building artifacts.

It tells the library:

- this field should be externalized
- which payload type should handle it
- what raw value should be saved

### `ArtifactBuilderBase`

`ArtifactBuilderBase` is the main build contract.

Required class fields:

- `artifact_name`
- `artifact_version`
- `unique_keys`

Optional class field:

- `executor_config`

Main entrypoint:

- `builder.build(...)`

Main hook methods:

1. `prepare_table(...)`
2. `transform_row(...)`
3. `transform_batch(...)`
4. payload lowering + save
5. `finalize_manifest(...)` → `FinalizeResult`

---

## Important Rules

### Unique Keys Are Required

Every builder must explicitly declare unique keys.

Example:

```python
unique_keys = [("id",)]
```

or

```python
unique_keys = [("arch_uid", "fdi")]
```

The first declared key is currently treated as the primary key for:

- duplicate checking
- deterministic row id generation
- payload directory naming

### `transform_batch(...)` Must Preserve Cardinality

`transform_batch(...)` is not allowed to expand or shrink item count.

If input length and output length differ, `lazetable` raises
`BatchCardinalityError`.

Use:

- `prepare_table(...)`
- `transform_row(...)`
- `finalize_manifest(...)`

for filtering, expansion, or reduction.

### Progress Tracking

The current build path uses a single progress bar:

- `total = len(prepared_rows)`
- progress is counted by completed input rows

## Build policy와 background writer

0.3부터 모든 builder 실행은 `BuildPolicy`를 명시한다.

```python
from lazetable import BuildPolicy, ExecutorConfig, WriterConfig

result = builder.build(
    storage_root=storage_root,
    parent_tables=[parent],
    policy=BuildPolicy(
        reuse_completed=True,
        resume_partial=True,
        retry_failed=True,
    ),
    executor_config=ExecutorConfig(row_workers=4, max_inflight=8),
    writer_config=WriterConfig(payload_workers=2, max_pending_writes=8),
)
```

- `reuse_completed`: signature가 같은 완성 artifact를 열고 전체 빌드를 건너뛴다.
- `resume_partial`: 같은 signature의 `.building` artifact에서 완료 source row를 건너뛴다.
- `retry_failed`: journal에 실패한 source row가 있으면 다시 실행한다.
- `payload_workers`: main process 내부의 background 저장 thread 수다.
- `max_pending_writes`: 메모리에 대기할 수 있는 최대 저장 작업 수이며 backpressure
  상한이 된다.

`BuildResult.status`는 `built`, `resumed`, `reused` 중 하나다. 실제 table은
`BuildResult.table`에서 얻는다.

후처리는 `FinalizeResult`를 반환한다.

```python
from lazetable import FinalizeResult


def finalize_manifest(self, rows, **kwargs):
    valid_count = sum(bool(row["is_valid"]) for row in rows)
    return FinalizeResult(
        rows=rows,
        metadata={"normalization": "unit-sphere"},
        statistics={"rows": {"total": len(rows), "valid": valid_count}},
    )
```

Transform 결과는 `raw_output_rows`에 그대로 남고, 후처리 결과와 통계는 하나의
SQLite transaction으로 `final_output_rows`와 `statistics`에 기록된다. 후처리 중
중단되면 raw output 전체에서 후처리를 다시 실행한다.

Payload는 transform worker 안에서 저장하지 않는다. process worker에서 저장하면
계산과 파일 생명주기가 결합되고 journal transaction을 중앙에서 보장하기 어렵기
때문이다. 별도 writer process도 Windows에서 ndarray를 한 번 더 직렬화하므로 기본
구현은 main process의 thread pool을 사용한다.

빌드 디렉터리:

```text
artifact-name/
├── .version.building/
│   ├── payload/
│   └── artifact.sqlite
└── version/                    # 완성 후 atomic 전환
    ├── artifact.sqlite
    └── payload/
```

---

## Example 1: Define A Payload Type

```python
from pathlib import Path
import json

from lazetable import Payload, register_payload_type


class JsonPayloadType:
    payload_id = "json_payload_v1"
    default_extension = "json"

    def dump_to_path(self, data, path: Path, *, metadata=None):
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return {"encoding": "json"}

    def load_from_path(self, path: Path, *, metadata=None):
        return json.loads(path.read_text(encoding="utf-8"))

    def payload(self, data, **kwargs):
        return Payload(self, data, metadata=kwargs or None)


register_payload_type(JsonPayloadType)
```

---

## Example 2: Write An Artifact Directly

```python
from pathlib import Path

from lazetable import write_artifact


out_dir = Path(r"C:\tmp\toy\v1")

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
    artifact_roots={"default": str(out_dir.parent.parent)},
    unique_key_fields=("id",),
)
```

What happens:

- the payload is written under `payload/id=row-1/`
- the row is lowered into metadata form
- `artifact.sqlite`에 row metadata와 artifact metadata가 기록됨

---

## Example 3: Read With `LazyTable`

```python
from lazetable import LazyTable


table = LazyTable.open(r"C:\tmp\toy\v1")
row = table.first_row()

print(row.id)
print(row.payload_field)
```

`row.payload_field` is loaded lazily from the payload file.

## 학습용 DatasetIndex

SQLite는 artifact 정본이지만 학습 epoch마다 query하지 않는다. 필요한 split과 필드를
한 번 preload하여 `DatasetIndex`를 만든다.

```python
index = table.to_dataset_index(
    where={"split": "train", "is_valid": True},
    fields=["id", "fdi", "mesh", "sdf_pool"],
)
```

Torch DataLoader protocol과 호환되는 wrapper:

```python
dataset = index.to_torch_dataset(sample_adapter)
encoded_index = dataset.encode_index(10, epoch=3)
sample = dataset[encoded_index]
```

`TorchDatasetAdapter` 자체는 torch를 import하지 않는다. 실제 DataLoader 통합을
검증하거나 사용할 때만 `torch` extra를 설치한다.

## Parquet export

Parquet은 artifact 정본이 아니라 선택적 snapshot이다.

```python
table.export_parquet(Path(r"C:\tmp\exports\artifact.parquet"))
```

export는 원본 `artifact.sqlite`를 수정하지 않는다.

---

## Example 4: Build An Artifact Through A Builder

```python
from pathlib import Path

from lazetable import ArtifactBuilderBase, BuildPolicy


class DoubleValueBuilder(ArtifactBuilderBase):
    artifact_name = "double_values"
    artifact_version = "v1"
    unique_keys = [("id",)]

    def transform_row(self, row, *, row_idx, parent_rows, build_context, seed_plan=None):
        return {
            "id": row["id"],
            "value_x2": int(row["value"]) * 2,
        }


result = DoubleValueBuilder().build(
    storage_root=Path(r"C:\tmp"),
    policy=BuildPolicy(),
    build_context={
        "input_rows": [
            {"id": "a", "value": 2},
            {"id": "b", "value": 5},
        ]
    },
)

artifact = result.table
```

---

## Example 5: Use `transform_batch(...)`

Use `transform_batch(...)` when:

- row items should be processed in batches
- main-process batch compute is needed
- item count stays exactly the same

```python
from lazetable import ArtifactBuilderBase


class BatchedBuilder(ArtifactBuilderBase):
    artifact_name = "batched_values"
    artifact_version = "v1"
    unique_keys = [("id",)]

    def transform_row(self, row, *, row_idx, parent_rows, build_context, seed_plan=None):
        return {
            "id": row["id"],
            "value": int(row["value"]),
        }

    def transform_batch(self, items, *, batch_idx, parent_rows, build_context, seed_plan=None):
        return [
            {
                "id": item["id"],
                "value_plus_one": item["value"] + 1,
            }
            for item in items
        ]
```

Important:

- input and output lengths must match exactly
- `items` is `list[dict[str, Any]]`

---

## Example 6: Configure The Executor

```python
from lazetable import ArtifactBuilderBase, ExecutorConfig


class ProcessBuilder(ArtifactBuilderBase):
    artifact_name = "process_builder"
    artifact_version = "v1"
    unique_keys = [("id",)]
    executor_config = ExecutorConfig(
        row_workers=4,
        batch_size=32,
        row_executor="process",
    )
```

Notes:

- default row executor is currently `process`
- process mode requires the builder object to be pickleable

---

## Current Limitations

These parts are still draft-level:

- payload save is not yet a fully separate public pipeline stage
- `transform_batch(...)` order preservation is assumed by the executor contract
- seed derivation policy is still minimal
- builder duplicate detection is implemented, but multi-key alternatives are not
  yet fully generalized beyond the primary key

---

## Recommended Mental Model

Think of `lazetable` as:

1. a lightweight row table
2. plus external payload files
3. plus a structured artifact builder runtime

Short version:

> Keep tables small, keep payloads external, and make artifact identity explicit.
