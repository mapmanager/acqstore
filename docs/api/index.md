---
search:
  exclude: true
---

# API Reference

The AcqStore API is organized around two primary concepts:

- `AcqImage`
- `AcqImageList`

Most workflows begin by loading data into an `AcqImage` or `AcqImageList` and
then applying one or more analysis modules.

Current analysis modules include:

- velocity analysis: blood flow velocity from line scan kymographs (Radon)
- diameter analysis: vessel diameter from line scan kymographs
- sum intensity analysis: functional reporter fluorescence from normalized line intensity
- heart rate analysis: periodic rate from a velocity time series

The API pages are generated with mkdocstrings from Google-style docstrings in the
source code.

## Main entry points

- [AcqImage](acq-image.md)
- [AcqImageList](acq-image-list.md)

## Analysis

- [Analysis Core](analysis-core.md)
- [Velocity Analysis](velocity-analysis.md)
- [Diameter Analysis](diameter-analysis.md)
- [Sum Intensity Analysis](sum-intensity-analysis.md)
- [Heart Rate Analysis](heart-rate-analysis.md)
- [Event Analysis](event-analysis.md)
- [Analysis Pools](analysis-pools.md)
- [Batch Analysis](batch-analysis.md)
