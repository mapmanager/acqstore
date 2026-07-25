# Install

AcqStore is a standalone Python package. Clone the repository and sync
dependencies with [uv](https://docs.astral.sh/uv/).

## Development install

```bash
git clone https://github.com/mapmanager/acqstore.git
cd acqstore
uv sync --group dev
```

## Tests

```bash
uv sync --group dev
uv run pytest
```

Format-specific loader tests skip cleanly when optional local fixtures under
`tests/acqstore/data/` are unavailable.

## Next

- [Sample data](sample-data.md)
- [Loading an image](loading.md)
