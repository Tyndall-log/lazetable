# lazetable

`lazetable` is a general-purpose Python library for table-shaped datasets with:

- lazy field resolution
- parser-aware rows
- Parquet-friendly artifact layouts
- explicit metadata-driven data access

The project is being extracted from an internal research codebase into a reusable
library. The current repository state is an initial scaffold for that migration.

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

## Streaming build runtime

LazeTable 0.2의 builder는 `BuildPolicy`를 명시적으로 받는다. 이전 `overwrite`
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

부분 빌드는 `<version>.building` 디렉터리에 저장되며 source row별 journal을 이용해
재개한다. 모든 output payload와 row metadata가 저장된 이후에만 해당 source row를
`completed`로 기록한다. 최종 manifest 검증이 끝나면 디렉터리를 원자적으로 정식
artifact 경로로 전환한다.

```powershell
cd C:\ai\lazetable
python -m pip install -e .[dev]
pytest
```
