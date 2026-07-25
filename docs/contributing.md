# Contributing

Contributions should preserve AcqStore's role as a GUI-independent scientific
backend. Do not introduce imports from `cloudscope`, `nicewidgets`, or
`acqstore_server` into `src/acqstore/`.

Before proposing a change, run tests locally and keep edits targeted to the
files actually changed.

## Development setup

```bash
git clone https://github.com/mapmanager/acqstore.git
cd acqstore
uv sync --group dev
uv run pytest
```

Documentation:

```bash
uv sync --group docs
uv run mkdocs serve
```

## Package layout

```text
src/acqstore/
tests/acqstore/
docs/
scripts/                      # developer diagnostic scripts
```

## Documentation

When editing pages under `docs/`, keep examples KISS: load sample data, construct
an `AcqImage`, then show the API under discussion. Prefer Google-style docstrings
in `src/` so mkdocstrings pages stay useful for both humans and LLMs.
