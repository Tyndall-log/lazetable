# LazeTable Migration Plan

## Goal

Extract the reusable data runtime from the current workspace into a generic
library that can eventually be published on PyPI as `lazetable`.

## Source Areas

Primary migration sources:

- `C:\ai\data\headers\headerlib`
- `C:\ai\teeth_ai_v2\teeth_ai_v2\data\preprocess`

## Proposed Layering

### 1. Core runtime

Move first:

- `LazyDataset`-style table wrapper
- `ParsedRow`
- parser registry and parser metadata handling
- row view registry
- parser cache policy
- artifact metadata helpers that are dataset-agnostic

Candidate destination modules:

- `lazetable.core`
- `lazetable.parsers`
- `lazetable.row_views`
- `lazetable.artifacts`

### 2. Pipeline/runtime extension

Move next:

- `BasePreprocessStep`
- artifact store/path abstractions
- artifact open/build/bundle APIs
- step registry

This may live either:

- inside `lazetable.pipeline`, or
- in a companion package such as `lazetable-pipeline`

That decision is intentionally deferred until the core runtime is stabilized.

### 3. Domain-specific steps

Do not move into the generic core:

- tooth/root geometry generation
- curvature/SDF/sampling steps tied to dental meshes
- model-specific latent cache builders

These should remain in a project package or a plugin-style extension package.

## Immediate Cleanup Targets

Before migration, remove or redesign:

- hard-coded `C:\ai\data` assumptions
- `sys.path` injection for package discovery
- Python-version skew between current source packages
- tight imports from project-specific model code into reusable runtime layers

## Acceptance Criteria For First Real Extraction

- importable without path hacks
- unit tests for lazy field resolution
- unit tests for parser registration and cache behavior
- unit tests for Parquet roundtrip with metadata
- public API documented in `README.md`
