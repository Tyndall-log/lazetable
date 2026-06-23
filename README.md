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

```powershell
cd C:\ai\lazetable
python -m pip install -e .[dev]
pytest
```
