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
  manifest.parquet
  meta.json
  payload/
```

Meaning:

- `manifest.parquet`
  - lightweight row table
- `meta.json`
  - artifact metadata
- `payload/`
  - large payload files written outside the table

---

## Core Concepts

### `LazyTable`

`LazyTable` reads a Parquet-backed artifact table and gives row access through
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
5. `finalize_table(...)`

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
- `finalize_table(...)`

for filtering, expansion, or reduction.

### Progress Tracking

The current build path uses a single progress bar:

- `total = len(prepared_rows)`
- progress is counted by completed input rows

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
- `manifest.parquet` and `meta.json` are written

---

## Example 3: Read With `LazyTable`

```python
from lazetable import LazyTable


table = LazyTable.from_parquet(r"C:\tmp\toy\v1\manifest.parquet")
row = table.first_row()

print(row.id)
print(row.payload_field)
```

`row.payload_field` is loaded lazily from the payload file.

---

## Example 4: Build An Artifact Through A Builder

```python
from pathlib import Path

from lazetable import ArtifactBuilderBase


class DoubleValueBuilder(ArtifactBuilderBase):
    artifact_name = "double_values"
    artifact_version = "v1"
    unique_keys = [("id",)]

    def transform_row(self, row, *, row_idx, parent_rows, build_context, seed_plan=None):
        return {
            "id": row["id"],
            "value_x2": int(row["value"]) * 2,
        }


artifact = DoubleValueBuilder().build(
    storage_root=Path(r"C:\tmp"),
    build_context={
        "input_rows": [
            {"id": "a", "value": 2},
            {"id": "b", "value": 5},
        ]
    },
)
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
