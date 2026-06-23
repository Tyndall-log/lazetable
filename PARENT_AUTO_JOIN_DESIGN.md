# Parent Auto-Join Design Draft

## Goal

Introduce explicit parent dependency metadata and optional automatic parent
joining for derived artifacts.

This should allow a child artifact to store only its own fields while still
being opened as a combined `{parent | child}` view when desired.

The immediate motivation is to reduce duplicated columns in derived builders
such as:

- `3dteethseg_arch_binmesh`

which currently repeats many columns already present in its source artifact.

## Non-Goal

- Implement full multi-parent lineage resolution immediately
- Replace explicit duplicated columns in existing builders right away
- Introduce implicit join heuristics without explicit key contracts

---

## Current State

Today `lazetable` supports:

- `build_artifact(..., input_artifact=...)`
- artifact-local payloads
- external source refs
- unique-key enforcement

But it does **not** yet support:

- stored parent dependency metadata as a first-class concept
- automatic parent joining on open

As a result, a derived builder often copies parent fields into the child
artifact to keep the child independently usable.

---

## Proposed Model

Add explicit parent dependency metadata to artifact outputs.

Example conceptual metadata:

```json
{
  "artifact_name": "3dteethseg_arch_binmesh",
  "artifact_version": "v1_draft",
  "storage_format_version": 1,
  "dependencies": [
    {
      "artifact_name": "3dteethseg_arch",
      "artifact_version": "v1_draft",
      "artifact_ref": "art://3dteethseg_arch@v1_draft",
      "join": {
        "child_on": ["arch_uid"],
        "parent_on": ["arch_uid"]
      }
    }
  ]
}
```

Key principles:

- dependency metadata is explicit
- join keys are explicit
- no implicit matching by row order
- automatic join is opt-in at open time, but likely enabled by default later

---

## New Concepts

### `ArtifactJoin`

Already exists as a value shape and should continue to represent:

```python
ArtifactJoin(
    child_on=["arch_uid"],
    parent_on=["arch_uid"],
)
```

### `ArtifactDependency`

Should become a real dependency contract, not just a passive placeholder.

Possible shape:

```python
@dataclass(frozen=True)
class ArtifactDependency:
    artifact_name: str
    artifact_version: str = "latest"
    join: ArtifactJoin | None = None
```

### `parent_artifacts` / `dependencies`

Artifact metadata should store a normalized dependency list.

For the first implementation, a single direct parent is sufficient.

The metadata format should still use a list to keep the model extensible.

---

## Builder-Level Declaration

Derived builders should be able to declare their parent dependency in a
class-level field.

Example:

```python
class Source3DTeethSegArchBinMeshBuilder(ArtifactBuilderBase):
    artifact_name = "3dteethseg_arch_binmesh"
    artifact_version = "v1_draft"
    unique_keys = [("arch_uid",)]
    dependencies = [
        ArtifactDependency(
            artifact_name="3dteethseg_arch",
            artifact_version="v1_draft",
            join=ArtifactJoin(
                child_on=["arch_uid"],
                parent_on=["arch_uid"],
            ),
        )
    ]
```

This is a declaration-only layer.

At build time, the caller may still pass:

- `input_artifact`
- `input_rows`

But the dependency declaration becomes the source of truth for lineage metadata.

---

## Write-Time Metadata

When writing a derived artifact, `write_artifact(...)` or `build_artifact(...)`
should record dependency information in `meta.json`.

Example pseudocode:

```python
dependency_meta = [
    {
        "artifact_name": dep.artifact_name,
        "artifact_version": dep.artifact_version,
        "join": {
            "child_on": dep.join.child_on,
            "parent_on": dep.join.parent_on,
        },
    }
    for dep in builder.dependencies
]
```

Stored as:

```json
{
  "dependencies": [...]
}
```

---

## Read-Time Auto Join

`open_artifact(...)` should eventually support:

```python
artifact = open_artifact(path, join_parents=True)
```

Or:

```python
artifact = store.open("3dteethseg_arch_binmesh", "v1_draft", join_parents=True)
```

### Read-Time Behavior

1. open child manifest and child metadata
2. if `join_parents=False`, return child as-is
3. if dependency metadata exists:
   - resolve the parent artifact
   - open parent artifact
   - recursively join parent if needed
   - join parent and child tables by declared keys
4. return a combined artifact/table view

---

## Join Semantics

This should reuse the current explicit key-based join philosophy.

No row-order-based join should be allowed.

Preferred defaults:

- join mode: `inner`
- collision handling: parent columns kept unless child overrides explicitly

Possible policy:

```python
on_collision = "error" | "parent" | "child"
```

Recommended initial default:

- keep child columns if there is an overlap
- keep parent columns otherwise

This matches the intuition that the child may intentionally refine or replace a
parent field.

---

## Column Duplication Policy

Once parent auto-join exists, derived builders should be allowed to become much
thinner.

For example, a future `3dteethseg_arch_binmesh` output might only store:

- `arch_uid`
- `mesh`
- `vertex_fdi`
- `vertex_instance`

And rely on parent auto-join for:

- `patient_id_raw`
- `case_uid`
- `jaw`
- `available_fdis`
- and other inherited metadata

This is the main value of the feature.

---

## Artifact Independence Tradeoff

There is a real tradeoff:

- duplicated child rows are easier to inspect independently
- auto-joined children are cleaner and smaller

Recommended migration path:

1. implement parent metadata + auto join
2. keep duplicated columns for a while
3. once read paths are stable, remove duplicated inherited columns from children

This reduces migration risk.

---

## API Sketch

### Base class

```python
class ArtifactBuilderBase:
    artifact_name: str = ""
    artifact_version: str = ""
    unique_keys: list[tuple[str, ...]] = []
    dependencies: list[ArtifactDependency] = []
```

### Open

```python
artifact = open_artifact(path, join_parents=True)
```

### Store

```python
artifact = store.open(
    artifact_name="3dteethseg_arch_binmesh",
    artifact_version="v1_draft",
    join_parents=True,
)
```

### LazyTable conversion

```python
table = artifact.to_lazy_table(join_parents=True)
```

Or:

```python
table = artifact.to_lazy_table()
```

if `Artifact` already stores a joined manifest view.

---

## Suggested First Implementation Scope

Keep the first implementation narrow.

### In scope

- one direct parent
- explicit one-parent dependency metadata
- explicit key-based join
- `open_artifact(..., join_parents=True)`
- child-wins collision policy

### Out of scope for now

- many parents
- graph conflict resolution
- lazy on-demand join
- partial-field parent inheritance
- dependency version aliases

---

## Why This Is Worth Doing

Benefits:

- smaller child artifacts
- less duplicated metadata
- clearer artifact lineage
- better separation between source artifact and derived artifact concerns
- closer alignment with the original design intent from earlier preprocess code

---

## Recommended Next Step

Implement this in stages:

1. add `dependencies` field to `ArtifactBuilderBase`
2. write dependency metadata into `meta.json`
3. add `join_parents` option to artifact opening
4. test with `3dteethseg_arch` -> `3dteethseg_arch_binmesh`
5. only then remove duplicated inherited columns from binmesh rows

Short summary:

> Parent auto-join should be explicit, key-based, and lineage-driven, and it
> should be introduced before child builders stop duplicating inherited fields.
