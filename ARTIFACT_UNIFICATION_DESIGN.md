# Artifact Unification Design

## Goal

`lazetable`에서는 현재 작업 중인 `header`와 `preprocess`를 라이브러리 코어 개념에서
명시적으로 구분하지 않는다.

둘 다 같은 저장 규약을 가지며, 읽는 방식도 동일하다.
따라서 코어 런타임은 둘을 모두 그냥 **artifact**로 취급한다.

남는 차이는 다음뿐이다.

- 어떤 artifact는 raw 데이터에서 직접 생성된다
- 어떤 artifact는 다른 artifact를 입력으로 파생 생성된다

이 차이는 코어 타입 구분이 아니라 **metadata / build mode 차이**로 표현한다.

## Non-Goal

- raw source 스캐너/수집기 구현을 코어 라이브러리에 포함하지 않는다
- dental 전용 step, geometry, model-specific cache builder를 코어에 넣지 않는다
- 지금 단계에서 모든 API 이름을 최종 확정하지 않는다

---

## Core Position

`lazetable`의 중심 개념은 다음 하나다.

- `Artifact`

그리고 이 artifact를 다루는 공통 책임은 다음이다.

- artifact를 연다
- artifact metadata를 읽는다
- row를 lazy하게 접근한다
- parser / row view를 해석한다
- artifact lineage를 기록한다
- artifact끼리 join / bundle 한다
- artifact를 새로 쓴다

즉 `header`와 `preprocess`는 코어 내부 분기 조건이 아니라,
artifact 사용자의 도메인적 표현으로만 남는다.

---

## Unified Artifact Layout

모든 artifact는 동일한 폴더 구조를 가진다.

```text
<artifact_dir>/
  manifest.parquet
  meta.json
  data/
```

여기서:

- `manifest.parquet`
  - row 메타, parser metadata, row view metadata
- `meta.json`
  - artifact-level lineage, version, fingerprint, context
- `data/`
  - 큰 payload 파일

코어 런타임은 artifact가 어디에 놓였는지는 신경 쓰지 않는다.
중요한 것은 오직 위 구조를 만족하는가이다.

예:

```text
C:\ai\data\headers\v2\3dteethseg\
C:\ai\data\preprocess\root_teeth_curvature\v1_20260609\
```

둘 다 코어 입장에서는 그냥 artifact 디렉터리다.

---

## Terminology

기존 용어를 다음처럼 일반화한다.

| 기존 용어 | 새 코어 용어 |
|---|---|
| header | source artifact 또는 base artifact |
| preprocess | derived artifact |
| HeaderSource / PreprocessParent | ArtifactDependency |
| PreprocessArtifact | Artifact |
| PreprocessStore | ArtifactStore |

핵심 원칙:

- 코어 타입은 `Artifact` 하나
- source / derived 차이는 `meta.json`과 build mode가 표현

---

## Artifact Identity

코어에서는 버전을 다음 세 층으로 분리한다.

- `artifact_version`
- `schema_version`
- `producer_version`

의미:

- `artifact_version`
  - artifact 디렉터리 identity
  - 사람이 폴더만 보고 식별할 수 있는 값
- `schema_version`
  - manifest / metadata contract 버전
- `producer_version`
  - 이 artifact를 생성한 코드 또는 알고리즘 구현 버전

### Example

```json
{
  "artifact_name": "3dteethseg",
  "artifact_version": "v2",
  "schema_version": "arch-v2",
  "producer_version": "build_arch_dataset_3dteethseg@0.1.0"
}
```

또는

```json
{
  "artifact_name": "root_teeth_curvature",
  "artifact_version": "v1_20260609",
  "schema_version": "rtc-v2",
  "producer_version": "root_teeth_curvature@0.3.1"
}
```

---

## Artifact Metadata Shape

권장 `meta.json` 초안:

```json
{
  "artifact_name": "example",
  "artifact_version": "v1",
  "schema_version": "example-schema-v1",
  "producer_version": "example-builder@0.1.0",
  "created_at": "2026-06-22T00:00:00Z",
  "manifest_fingerprint": "sha256:...",
  "row_count": 1234,
  "resolver_context": {
    "source_roots": {
      "dataset_a": "C:\\data\\raw\\dataset_a"
    },
    "artifact_roots": {
      "default": "C:\\data\\artifacts"
    }
  },
  "dependencies": [
    {
      "artifact_name": "parent_a",
      "artifact_version": "v3",
      "manifest_ref": "art://parent_a/v3/manifest.parquet",
      "join": {
        "child_on": ["id"],
        "parent_on": ["id"]
      },
      "manifest_fingerprint": "sha256:..."
    }
  ],
  "table_meta": {}
}
```

핵심 포인트:

- source / derived를 강제 필드로 둘 필요는 없다
- `dependencies`가 비어 있으면 base/source artifact처럼 쓸 수 있다
- lineage는 parent 하나가 아니라 리스트로 확장 가능하게 둔다

---

## Reference Schemes

현재 `src://`, `pps://` 같은 구조는 유지하되, 코어에서는 더 일반화한다.

추천 scheme:

- `src://`
  - raw / external source root 기준 참조
- `art://`
  - artifact root 기준 참조

`pps://`는 preprocess 전용 이름이라 코어 범용성 측면에서는 피하는 편이 낫다.

### Example

```text
src://3dteethseg/3DTeethSeg_merged/lower/patient_001/case.obj
art://root_teeth_curvature/v1_20260609/data/369a__31.npz
```

---

## Resolver Model

기존 구현에서는:

- `src://`는 `ParseContext.source_roots`
- `pps://`는 `PreprocessStore.preprocess_root`

를 따로 사용했다.

`lazetable`에서는 이를 하나의 resolver 계층으로 통합하는 것을 추천한다.

### ResolverContext

```python
@dataclass(frozen=True)
class ResolverContext:
    source_roots: Mapping[str, str]
    artifact_roots: Mapping[str, str]
```

### Resolver API

```python
class RefResolver:
    def resolve(self, ref: str, context: ResolverContext) -> Path:
        if ref.startswith("src://"):
            namespace, rel_path = parse_src_ref(ref)
            return Path(context.source_roots[namespace]) / rel_path
        if ref.startswith("art://"):
            namespace, rel_path = parse_art_ref(ref)
            return Path(context.artifact_roots[namespace]) / rel_path
        return Path(ref)
```

### Notes

- 코어는 경로 prefix의 의미만 안다
- namespace별 실제 절대 루트는 `ResolverContext`가 들고 간다
- 이렇게 하면 source artifact, derived artifact 모두 같은 방식으로 ref 해석 가능하다

---

## Read Flow

artifact를 읽는 흐름은 `header`/`preprocess` 구분 없이 동일하다.

### Pseudocode

```python
def open_artifact(path: Path, *, store: ArtifactStore | None = None) -> Artifact:
    manifest_path = path / "manifest.parquet"
    meta_path = path / "meta.json"

    manifest = read_parquet(manifest_path)
    meta = read_json(meta_path)
    validate_manifest_fingerprint(manifest_path, meta)

    return Artifact(
        path=path,
        manifest=manifest,
        meta=meta,
        store=store,
    )
```

```python
def artifact_to_lazy_table(artifact: Artifact) -> LazyTable:
    context = infer_resolver_context(artifact.meta)
    return LazyTable.from_table(
        artifact.manifest,
        resolver_context=context,
        artifact_meta=artifact.meta,
    )
```

### Read Responsibilities

- fingerprint 검증
- metadata 복원
- resolver context 구성
- row runtime 생성

---

## Row Runtime

row runtime은 현재 `LazyDataset` / `ParsedRow` 핵심을 일반화해서 유지한다.

### Pseudocode

```python
class LazyTable:
    @classmethod
    def from_parquet(cls, path: Path) -> "LazyTable":
        table = read_parquet(path)
        meta = maybe_read_json(path.parent / "meta.json")
        context = infer_resolver_context(meta, table.schema.metadata)
        return cls(table=table, resolver_context=context, artifact_meta=meta)

    def __getitem__(self, index: int) -> "ParsedRow":
        row_dict = self.table.slice(index, 1).to_pylist()[0]
        return ParsedRow(
            row_dict,
            parser_manager=self.parser_manager,
            row_view_manager=self.row_view_manager,
            resolver_context=self.resolver_context,
            context=self.context,
        )
```

```python
class ParsedRow:
    def resolve(self, field: str) -> Any:
        if has_materialized_value(field):
            return value
        if has_parser_metadata(field):
            handle = make_parse_handle(field)
            return parser_manager.resolve(handle, resolver_context)
        if has_row_view_metadata(field):
            return row_view_manager.resolve(field, self)
        raise AttributeError(field)
```

### Keep

- raw field 우선
- parser field 다음
- row view 마지막
- row-level cache 유지

---

## Parser Model

parser는 artifact 종류를 몰라야 한다.
parser는 오직:

- parse handle
- resolver context
- parser manager

만 받아서 동작해야 한다.

### Pseudocode

```python
class FieldParser(Protocol):
    parser_id: str
    cache_enabled: bool

    def parse(
        self,
        *,
        handle: ParseHandle,
        resolver_context: ResolverContext,
        parser_manager: ParserManager,
    ) -> Any:
        ...
```

```python
class ParserManager:
    def resolve(self, handle: ParseHandle, context: ResolverContext) -> Any:
        if cache_hit(handle):
            return load_cache(handle)
        parser = self.get(handle.parser_id)
        value = parser.parse(
            handle=handle,
            resolver_context=context,
            parser_manager=self,
        )
        maybe_store_cache(handle, value)
        return value
```

### Core Rule

parser는 `header` 전용 또는 `preprocess` 전용이라고 가정하지 않는다.
그저 ref와 metadata를 해석해 값을 복원하는 함수다.

---

## Build Flow

`lazetable` 코어의 build는 artifact-to-artifact 파이프라인 중심으로 잡는다.
raw-to-artifact 빌드는 코어 바깥 책임으로 둔다.

### Pseudocode

```python
class ArtifactBuilder:
    def build(
        self,
        spec: BuildSpec,
        *,
        store: ArtifactStore,
    ) -> Artifact:
        input_artifacts = [
            store.open(dep.artifact_name, dep.artifact_version)
            for dep in spec.dependencies
        ]

        input_tables = [artifact_to_lazy_table(a) for a in input_artifacts]
        output_rows, extra_meta = spec.run(input_tables)

        out_dir = store.artifact_dir(spec.artifact_name, spec.artifact_version)
        write_manifest(out_dir / "manifest.parquet", output_rows)
        write_meta(out_dir / "meta.json", build_meta(spec, input_artifacts, extra_meta))

        return open_artifact(out_dir, store=store)
```

### Notes

- builder는 input이 source artifact인지 derived artifact인지 구분하지 않는다
- input은 그저 dependency list다
- 차이는 dependency가 없느냐 있느냐뿐이다

---

## Dependency Model

현재 `HeaderSource`, `PreprocessParent` 같은 구분은 제거한다.

### Pseudocode

```python
@dataclass(frozen=True)
class ArtifactJoin:
    child_on: list[str]
    parent_on: list[str]


@dataclass(frozen=True)
class ArtifactDependency:
    artifact_name: str
    artifact_version: str = "latest"
    join: ArtifactJoin | None = None
```

이 한 타입으로 충분하다.

### Why

- 코어 입장에서는 “입력 artifact”라는 의미만 필요
- source / derived 여부는 dependency 타입 분리로 표현할 가치가 작음

---

## Store Model

현재 `PreprocessStore`는 이름이 좁다.
이를 `ArtifactStore`로 일반화한다.

### Pseudocode

```python
@dataclass(frozen=True)
class ArtifactStorePaths:
    artifact_roots: Mapping[str, Path]


class ArtifactStore:
    def __init__(self, paths: ArtifactStorePaths):
        self.paths = paths

    def artifact_dir(self, root_name: str, artifact_name: str, artifact_version: str) -> Path:
        return self.paths.artifact_roots[root_name] / artifact_name / artifact_version

    def open(self, path_or_ref: str | Path) -> Artifact:
        path = self.resolve_artifact_ref(path_or_ref)
        return open_artifact(path, store=self)
```

### Comment

store가 꼭 `headers root`, `preprocess root`처럼 고정된 필드를 가져야 할 필요는 없다.
named root map이 더 일반적이다.

예:

```python
ArtifactStorePaths(
    artifact_roots={
        "headers": Path("..."),
        "derived": Path("..."),
        "cache": Path("..."),
    }
)
```

---

## Join / Bundle Model

join도 artifact 종류를 구분하지 않고 key 계약만 본다.

### Pseudocode

```python
def bundle(artifacts: list[Artifact]) -> ArtifactBundle:
    return ArtifactBundle(artifacts)


class ArtifactBundle:
    def to_lazy_table(self, *, key: str | list[str], join: str = "inner") -> LazyTable:
        merged = join_artifacts_by_key(self.artifacts, key=key, join=join)
        context = merge_resolver_contexts(self.artifacts)
        return LazyTable.from_table(merged, resolver_context=context)
```

핵심:

- bundle은 artifact type을 추론하지 않는다
- 명시적 key 계약만 사용한다

---

## Migration Impact

이 통합 설계를 따르면 현재 코드에서 다음을 유지할 수 있다.

- `manifest.parquet + meta.json + data/`
- parser metadata 규약
- row view 규약
- key 기반 join
- lineage / fingerprint

그리고 다음은 재설계된다.

- `HeaderSource` / `PreprocessParent` 분리
- `PreprocessStore` / `PreprocessPaths` naming
- `pps://` naming
- `src://`와 artifact ref 해석 경로의 이원화

---

## Recommended First Extraction Boundary

1차 이식 범위는 다음이 적당하다.

- `LazyTable`
- `ParsedRow`
- `ParseHandle`
- `ResolverContext`
- `ParserManager`
- `RowViewManager`
- `Artifact`
- `ArtifactDependency`
- `ArtifactStore`
- `open_artifact`
- `join_artifacts_by_key`

보류:

- raw source builder
- multiprocessing build executor
- GPU batch pipeline
- 도메인 parser / row view / preprocess step 구현

---

## Decision Summary

최종 제안:

1. `header`와 `preprocess`는 코어 구현에서 구분하지 않는다.
2. 둘 다 같은 artifact 규약을 사용한다.
3. 차이는 읽기 타입이 아니라 생성 방식과 lineage metadata가 표현한다.
4. ref 해석은 `src://`와 `art://`를 포함하는 통합 resolver 계층으로 정리한다.
5. 코어는 artifact runtime에 집중하고, raw-to-artifact 생성은 사용자 책임으로 둔다.

한 줄로 요약하면:

> `lazetable`의 코어는 "header runtime"이나 "preprocess runtime"이 아니라,
> 하나의 통합된 "artifact runtime"이어야 한다.
