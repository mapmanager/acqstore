---
hide:
  - toc
---

# AcqStore

AcqStore is a Python package for loading, annotating, and analyzing
acquisition-backed microscopy files. It provides the scientific backend used by
applications such as CloudScope, as well as by notebooks and scripts.

Current quantitative analysis workflows are designed for **line scan kymographs** and include:

- [blood flow velocity analysis](scientists/velocity-analysis.md) using a Radon-transform-based method
- [vessel diameter analysis](scientists/diameter-analysis.md)
- [peak detection / sum intensity analysis](scientists/sum-intensity-analysis.md) for functional fluorescence reporters (like GCaMP)
- [heart rate analysis](scientists/heart-rate-analysis.md) derived from velocity results

## Install

```bash
git clone https://github.com/mapmanager/acqstore.git
cd acqstore
uv sync
```

## Quick start

```python
from acqstore.acq_image.acq_image_list import AcqImageList

lst = AcqImageList('/path/to/folder', folder_depth=2)
```

## Supported file formats

Commercial microscopy formats:

- Olympus / Evident `.oir`
- Zeiss `.czi`
- Nikon `.nd2`

Open image formats:

- TIFF `.tif`
- OME-Zarr `.ome.zarr`

## Who is this documentation for?

<div class="grid cards" markdown>

-   :material-flask:{ .lg .middle } **Data Scientist**

    ---

    Understand `AcqImage`, `AcqImageList`, line scan kymograph analysis, saved files, metadata, and notebook workflows.

    [:octicons-arrow-right-24: Data Scientist Guide](scientists/index.md)

-   :material-code-braces:{ .lg .middle } **Developer**

    ---

    Clone the repository, run tests, build docs, and contribute to AcqStore.

    [:octicons-arrow-right-24: Developer Guide](developers/index.md)

-   :material-api:{ .lg .middle } **API Reference**

    ---

    mkdocstrings-generated API pages for AcqImage, analyses, pools, and batch APIs.

    [:octicons-arrow-right-24: API Reference](api/index.md)

</div>
