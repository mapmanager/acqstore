# Developer Guide

AcqStore is the scientific backend for acquisition-backed microscopy files:
loading, metadata, ROIs, analysis, and saved results.

```text
src/acqstore/
tests/acqstore/
docs/
examples/app/dff0_diameter_analysis/   # manually run demo (not packaged)
scripts/                      # developer diagnostic scripts
```

## Getting started

```bash
git clone https://github.com/mapmanager/acqstore.git
cd acqstore
uv sync --group dev
```

Run the test suite:

```bash
uv run pytest
```

Run the documentation locally:

```bash
uv sync --group docs
uv run mkdocs serve
```

## Package boundaries

AcqStore must not import:

- `cloudscope`
- `nicewidgets`
- `acqstore_server`

The NiceGUI demo under `examples/app/dff0_diameter_analysis/` may use NiceGUI and
NiceWidgets, but it is not part of the installable package and is not covered by
CI validation.

## Unit testing

Tests live under `tests/acqstore/`. Format-specific loader tests skip cleanly
when optional local fixtures under `tests/acqstore/data/` are unavailable.

```bash
uv run pytest
uv run pytest tests/acqstore/test_schema.py
```

## GitHub workflows

| Workflow | Purpose |
|---|---|
| `tests.yml` | Run unit tests. |
| `docs.yml` | Build and publish MkDocs documentation to GitHub Pages. |

## Sample data

Downloadable samples are provided by the
[cloudscope-data](https://github.com/mapmanager/cloudscope-data) repository and
accessed through `acqstore.sample_data`.

## More information

- [MkDocs style guide](mkdocs-style.md)
- [Contributing](contributing.md)
