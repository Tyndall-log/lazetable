# lazetable

`lazetable`은 다음 기능을 제공하는 범용 artifact 및 dataset 라이브러리다.

- lazy field resolution
- parser-aware rows
- SQLite 기반 재개 가능한 artifact
- explicit metadata-driven data access

대용량 값은 외부 payload 파일로 저장하고 SQLite에는 row metadata, payload 참조,
빌드 상태, dependency와 통계를 기록한다.

## Intended Scope

`lazetable` is meant to cover the reusable runtime layer, not dental-specific
algorithms or model-specific preprocessing steps.

Target capabilities:

- `LazyTable` / dataset-like table access
- `ParsedRow` style row facade
- metadata-driven field parsers
- row-level computed views
- artifact loading conventions for header/preprocess-style datasets

## Planned Package Surface

Initial public modules are expected to look like:

```python
from lazetable import (
    LazyTable,
    ParsedRow,
    ParseContext,
    ParserManager,
    RowViewManager,
    register_parser,
    register_row_view,
)
```

The implementation will be migrated in stages from:

- `C:\\ai\\data\\headers\\headerlib`
- `C:\\ai\\teeth_ai_v2\\teeth_ai_v2\\data\\preprocess`

## Development

## SQLite streaming build runtime

LazeTable 0.3의 builder는 `BuildPolicy`를 명시적으로 받는다. 이전 `overwrite`
기반 API는 지원하지 않는다.

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
    executor_config=ExecutorConfig(
        row_workers=4,
        max_inflight=8,
    ),
    writer_config=WriterConfig(
        payload_workers=2,
        max_pending_writes=8,
    ),
)
```

반환되는 `BuildResult.status`는 `built`, `resumed`, `reused` 중 하나다.

변환 process가 만든 결과는 main process로 한 번 전달된 뒤 bounded background writer
thread pool에서 즉시 payload로 저장된다. 저장 queue가 가득 차면 transform 쪽에
backpressure가 걸리므로 전체 payload가 메모리에 누적되지 않는다.

부분 빌드는 `<version>.building` 디렉터리의 `artifact.sqlite`에 저장되며 source row별 journal을 이용해
재개한다. 모든 output payload와 row metadata가 저장된 이후에만 해당 source row를
`completed`로 기록한다. 최종 SQLite artifact 검증이 끝나면 디렉터리를 원자적으로 정식
artifact 경로로 전환한다.

```text
<artifact-name>/
├── .<version>.building/
│   ├── artifact.sqlite
│   └── payload/
└── <version>/
    ├── artifact.sqlite
    └── payload/
```

## 학습용 index

완성 SQLite artifact는 `DatasetIndex`로 preload할 수 있다. 이후 epoch 순회에서는
SQL query를 실행하지 않는다.

```python
index = table.to_dataset_index(
    where={"split": "train", "is_valid": True},
    fields=["id", "fdi", "mesh", "sdf_pool"],
)
dataset = index.to_torch_dataset(sample_adapter)
```

Parquet이 필요하면 명시적으로 snapshot을 내보낸다.

```python
table.export_parquet(output_path)
```

```powershell
cd C:\ai\lazetable
python -m pip install -e .[dev]
pytest
```
