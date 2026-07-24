# AcqStore

[![Tests](https://github.com/mapmanager/acqstore/actions/workflows/tests.yml/badge.svg)](https://github.com/mapmanager/acqstore/actions/workflows/tests.yml)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL%203.0-blue.svg)](LICENSE)

AcqStore is a Python package for **acquisition-backed microscopy files**: discovery, loading, ROI models, metadata, and quantitative analysis of line-scan kymographs.

It does **not** depend on NiceGUI, NiceWidgets, or the CloudScope application.

## Install (development)

```bash
git clone https://github.com/mapmanager/acqstore.git
cd acqstore
uv sync --group dev
```

## Quick start

```python
from acqstore.acq_image.acq_image_list import AcqImageList

lst = AcqImageList('/path/to/folder', folder_depth=2)
single = AcqImageList('/path/to/file.tif')
```

## Documentation

Build and serve the MkDocs site locally:

```bash
uv sync --group docs
uv run mkdocs serve
```

## Tests

```bash
uv sync --group dev
uv run pytest
```

Format-specific loader tests skip cleanly when optional local fixtures under `tests/acqstore/data/` are unavailable.

## Example demo app

A manually run NiceGUI demo for ΔF/F0–diameter coupling analysis lives at:

```text
examples/app/dff0_diameter_analysis/
```

It is **not** part of the installable package and is not installed, tested, or validated by project CI. Run it only in an environment that already provides NiceGUI and NiceWidgets.

## License

GPL-3.0-only. Copyright (c) Robert Cudmore.
