# AcqStore Radon Velocity Analysis

## Purpose

This document describes the current backend Radon velocity analysis work in
`acqstore`.

The design separates the pure numerical algorithm from the acquisition model:

```text
radon_core.py             pure NumPy/scikit-image/multiprocessing algorithm
radon_velocity_analysis.py BaseAnalysis wrapper for AcqStore
AcqAnalysisSet            orchestration and persistence integration
AcqAnalysisBatch          batch runner across multiple AcqImage files
```

## Single-file flow

A single Radon velocity analysis is identified by:

```text
analysis_name = "radon_velocity"
channel
roi_id
```

The analysis obtains data only through `AnalysisDataProvider`:

```python
image = data_provider.get_roi_image(channel, roi_id)
physical_units = data_provider.get_image_physical_units()
```

The core algorithm does not import `acqstore`.

## Detection parameters

`RadonVelocityAnalysis` uses the existing `BaseAnalysis` detection schema.

Current parameter:

```text
window_width: int, default=64, choices=(16, 64, 128)
```

Script code should use:

```python
params = RadonVelocityAnalysis.get_default_detection_params()
params["window_width"] = 64
RadonVelocityAnalysis.validate_detection_params(params)
```

The detection schema is not serialized. Only `detection_params` are serialized
inside the AcqImage sidecar JSON.

## Result persistence

Small summary values are serialized under the `analysis` key in the AcqImage
sidecar JSON. Large table output is saved by `AcqAnalysisSet.save_results_df()`
to:

```text
<source_filename>.radon_velocity.csv
```

Rows include `channel` and `roi_id` bookkeeping columns.

## Result columns

The core result table columns are defined in one place:

```python
RADON_VELOCITY_COLUMNS
```

Current columns:

```text
time_s
time_index
theta_deg
velocity
```

`time_s` is computed from the Radon window center index and the image physical
unit along the time/line axis. It is not the raw image row index.

## Cancellation and progress

`RadonVelocityAnalysis` receives an `AnalysisRunContext`.

- `report_progress(fraction, message)` reports progress.
- `is_cancelled()` is passed to the core algorithm.
- cancellation raises `AnalysisCancelled`.
- cancelled runs do not overwrite result summary/table.

## Batch analysis

Batch Radon velocity analysis is a wrapper around the single-file analysis.

ROI selection is controlled by:

```python
class RoiBatchMode(StrEnum):
    ANALYZE_EXISTING_ROI = "analyze_existing_roi"
    ADD_NEW_ROI = "add_new_roi"
```

### ANALYZE_EXISTING_ROI

Uses the same configured `roi_id` for every file.

- missing ROI -> `SKIPPED_MISSING_ROI`
- existing Radon analysis for `(channel, roi_id)` is replaced
- no hidden save

### ADD_NEW_ROI

Creates a new default rectangular ROI in each file using:

```python
acq_image.rois.create_rect_roi()
```

The newly created ROI is then analyzed.

Repeated batch runs intentionally create additional ROIs.

## Scripts

Manual scripts are intentionally simple and hardcoded.

```text
scripts/try_radon_analysis.py
scripts/try_radon_batch_analysis.py
```

They are meant for local development before GUI integration.

`try_radon_analysis.py` also includes a modular matplotlib helper:

```python
plot_radon_velocity(analysis)
```

Matplotlib is used only in scripts, not in `acqstore`.

## Future GUI integration

The backend is prepared for GUI use because:

- detection schema can render a parameter card
- `AnalysisRunContext` supports progress and cancellation
- batch runner reports per-file results
- save remains explicit


## Canonical Plot API

`RadonVelocityAnalysis` exposes canonical velocity-versus-time plot data through:

```python
get_plot_data()
```

This returns an `AnalysisPlotData` instance.

Dependent analyses should consume this API rather than directly accessing dataframe columns.

Current dependent analyses include:

- `event`
