---
hide:
  - toc
---

# AcqStore

AcqStore is a Python package for **acquisition-backed microscopy files**: discovery,
loading, ROIs, metadata, and quantitative analysis of line-scan kymographs.

An *acquisition-backed* file is more than pixels. AcqStore keeps acquisition
context with the recording. That includes format-specific header and calibration,
optional ROIs, and analysis sidecars. Scripts and notebooks share one scientific
backend.

AcqStore uses **lazy loading** for image pixels and large analysis CSV tables.
You can browse many files without loading every pixel array into memory. See
[Loading an image](loading.md).

One example GUI that uses AcqStore is
[CloudScope](https://mapmanager.github.io/cloudscope/). AcqStore itself does not
depend on that application.

Current quantitative analysis workflows target **line scan kymographs**:

- [blood flow velocity](analysis/kymograph/velocity-analysis.md) (Radon transform)
- [vessel diameter](analysis/kymograph/diameter-analysis.md)
- [sum intensity / peak detection](analysis/kymograph/sum-intensity-analysis.md) for functional reporters (like GCaMP)
- [heart rate](analysis/kymograph/heart-rate-analysis.md) derived from velocity results

## Start here

1. [Install](install.md)
2. [Sample data](sample-data.md)
3. [Loading an image](loading.md)
4. [AcqImage](acqimage.md): header metadata, pixels, ROIs, analysis, sidecars
5. [ROIs](rois.md)
6. [Analysis](analysis/index.md)

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

See [Loading an image](loading.md) for how loaders work, including the native
`.cs.ome.zarr` format (in development).

## API reference

mkdocstrings-generated pages for `AcqImage`, analyses, pools, and batch APIs:

[:octicons-arrow-right-24: API Reference](api/index.md)
