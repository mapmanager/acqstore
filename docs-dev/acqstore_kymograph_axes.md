# AcqStore kymograph axis convention

Line-scan kymograph images in CloudScope use a single numpy layout:

| Index | Name | Kymograph meaning | Typical physical unit |
|-------|------|-------------------|------------------------|
| dim0 / rows / `Y` | time | lines (frames) | seconds per line |
| dim1 / cols / `X` | space | pixels along vessel axis | µm per pixel |

## Rules

- 2D arrays are shaped `(time, space)` = `(rows, cols)`.
- `RectRoiBounds.dim0` spans **rows (time)**; `dim1` spans **columns (space)**.
- `AnalysisDataProvider.get_image_physical_units()` returns
  `(seconds_per_line, um_per_pixel)` aligned to `(dim0, dim1)`.
- The raster viewer maps **plot-x = row/time (dim0)** and **plot-y = col/space (dim1)**.

## ROI-local analysis coordinates

Analyses run on an ROI crop. Their tables and overlay traces use coordinates
relative to the ROI origin (first analyzed row/column). GUI views translate to
full-image Plotly coordinates using the selected `RectROI.bounds` and
`RasterGridSpec` spacing.

## See also

- `src/acqstore/acq_image/file_loaders/read_olympus_txt.py` — Olympus kymograph header
