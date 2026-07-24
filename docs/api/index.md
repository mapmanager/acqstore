---
search:
  exclude: true
---

# API Reference

The AcqStore API is organized around two primary concepts:

- `AcqImage`
- `AcqImageList`

Most workflows begin by loading data into an `AcqImage` or `AcqImageList` and then applying one or more analysis modules.

Current analysis modules include:

- velocity analysis: blood flow velocity estimation from line scan kymographs using a Radon-transform-based method
- diameter analysis: vessel diameter estimation from line scan kymographs
- sum intensity analysis: functional reporter fluorescence (like GCaMP) from normalized line intensity on line scan kymographs

The API pages are generated with mkdocstrings from Google-style docstrings in the source code.

## Main entry points

- [AcqImage](acq-image.md)
- [AcqImageList](acq-image-list.md)

## Analysis

- [Analysis Core](analysis-core.md)
- [Velocity Analysis](velocity-analysis.md)
- [Diameter Analysis](diameter-analysis.md)
- [Sum Intensity Analysis](sum-intensity-analysis.md)
- [Batch Analysis](batch-analysis.md)
