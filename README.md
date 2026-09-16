# LazeTable

LazeTable은 대용량 payload를 지연 로딩하고 재개 가능한 전처리 artifact를 구축하기 위한
SQLite 기반 Python 라이브러리입니다.

행 metadata와 빌드 journal은 `artifact.sqlite`에 저장하고 mesh, 배열, tensor 같은 큰
값은 별도 payload 파일로 저장합니다. 완성 artifact는 query 결과를 `DatasetIndex`로 한
번 preload하여 PyTorch 학습 epoch 중 반복적인 SQLite query 없이 순회할 수 있습니다.

## 주요 기능

- `LazyTable`과 parser-aware row access
- 사용자 정의 `PayloadType` 등록과 지연 payload 로딩
- SQLite journal 기반 부분 빌드 재개·재사용·실패 재시도
- process/thread row executor와 batch transform
- bounded background payload writer와 backpressure
- 부모 artifact dependency와 명시적 join
- train/validation filtering용 `DatasetIndex`
- PyTorch `Dataset` adapter
- 필요 시 Parquet snapshot export

## 설치

```bash
pip install lazetable
```

PyTorch 선택 의존성까지 설치하려면 다음 extra를 사용합니다.

```bash
pip install "lazetable[torch]"
```

Python 3.11 이상을 지원합니다.

## Artifact 열기

```python
from pathlib import Path

from lazetable import LazyTable


artifact = LazyTable.open(Path("/absolute/path/to/artifact"))
print(len(artifact))

row = artifact[0]
print(row.raw("sample_id"))
mesh = row.mesh  # payload는 이 시점에 로드됩니다.
```

기본 artifact 구조는 다음과 같습니다.

```text
<artifact>/
├── artifact.sqlite
└── payload/
```

## 학습용 index

```python
index = artifact.to_dataset_index(
    where={"split": "train", "is_valid": True},
    fields=["sample_id", "label", "features"],
)

for row in index:
    features = row.features
    label = row.raw("label")
```

`DatasetIndex`는 선택된 row metadata를 메모리에 preload합니다. epoch 순회에서는 SQLite
query를 반복하지 않고 필요한 payload만 지연 로딩합니다.

## 재개 가능한 builder

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

print(result.status)  # built, resumed 또는 reused
```

부분 빌드는 다음 경로에 저장됩니다.

```text
<artifact-name>/
├── .<version>.building/
│   ├── artifact.sqlite
│   └── payload/
└── <version>/
    ├── artifact.sqlite
    └── payload/
```

모든 payload와 row metadata가 저장된 뒤에만 source row를 `completed`로 기록합니다.
최종 검증이 끝나면 building 디렉터리를 정식 version 경로로 원자적으로 전환합니다.

## 개발

```bash
uv sync --extra dev
uv run pytest
uv run ruff check src tests
```

더 자세한 builder, payload와 dependency 예제는 `USAGE_GUIDE.md`를 참고하세요.

## 라이선스

MIT
