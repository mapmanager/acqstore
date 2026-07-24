# acqstore API Used by CloudScope

## Purpose

This document describes the acqstore public APIs that CloudScope relies on. It is not a full acqstore reference. It focuses on GUI-facing APIs that keep CloudScope thin.

CloudScope should not interpret acqstore internals. If CloudScope needs to understand acquisition details, ROI geometry, reference-image dimensionality, or analysis result columns, acqstore should provide a clearer public API.

## Design rule

```text
acqstore owns domain interpretation.
CloudScope consumes display-ready or workflow-ready APIs.
```

## AcqImage

CloudScope treats `AcqImage` as the per-file domain object.

Expected capabilities:

```text
- access images through image APIs
- access metadata through metadata APIs
- access ROIs through acq_image.rois
- access analyses through acq_image.analysis_set
- report dirty state
- save/load persisted sidecar/state
- produce file-list row dictionaries through get_schema_row()
```

CloudScope should not reach into private acqstore attributes.

## AcqImageList

CloudScope uses `AcqImageList` as the loaded file collection.

Expected capabilities:

```text
- load file/folder paths
- expose rows for file table display
- look up AcqImage by file_id
- save selected/all files
- support progress and cancellation for long-running load/save workflows
```

Load/save should expose cooperative cancellation. CloudScope can disable UI and request cancellation, but acqstore must check the cancellation callback.

## ReferenceImage

Reference image interpretation belongs in acqstore.

Current desired public API:

```python
@dataclass(frozen=True, eq=False)
class ReferenceImagePlane:
    array: np.ndarray
    dx: float
    dy: float
    x_unit: str
    y_unit: str
```

ReferenceImage should provide:

```python
get_plane(channel: int | None = None) -> ReferenceImagePlane
```

Future API:

```python
get_line_roi() -> tuple[float, float, float, float] | None
```

CloudScope should do:

```text
ref = acq_image.images.reference_image()
if ref is not None:
    plane = ref.get_plane(channel)
    viewer.set_data(plane.array, dx=plane.dx, dy=plane.dy, ...)
```

CloudScope should not inspect:

```text
ReferenceImage.array dimensionality
ReferenceImage.dims
ReferenceImage.coord_scales
ReferenceImage.coord_units
ReferenceImage.coords
```

If those need interpretation, improve acqstore.

## ROI API

ROIs are owned by acqstore.

Expected API:

```text
acq_image.rois.create_rect_roi(bounds=None)
acq_image.rois.edit_rect_roi(roi_id, bounds=...)
acq_image.rois.delete(roi_id)
acq_image.rois.get(roi_id)
acq_image.rois.get_roi_ids()
acq_image.rois.has_roi(roi_id)
```

Important domain rule:

```text
ROIs are not per-channel.
All channels in an AcqImage share the same image shape and ROI coordinate system.
```

Therefore deleting an ROI removes analysis for that ROI across all channels.

## ROI bounds

Rect ROI bounds use acqstore coordinate naming:

```text
dim0_start / dim0_stop = row axis
dim1_start / dim1_stop = column axis
```

CloudScope should not independently decide coordinate transforms. If a viewer needs display-ready coordinates, consider adding an acqstore helper or a dedicated conversion function near the API boundary.

## Analysis API

Analysis objects live in acqstore.

Expected concepts:

```text
analysis_name
channel
roi_id
analysis result summary
analysis result table
persistence to JSON/CSV
```

CloudScope should not know analysis storage details.

## Analysis plot API

CloudScope line plots should use analysis-provided plot data.

Expected dataclass:

```python
@dataclass(frozen=True)
class AnalysisPlotData:
    x: list[float]
    y: list[float]
    x_label: str
    y_label: str
```

Expected method:

```python
analysis.get_plot_data() -> AnalysisPlotData | None
```

For radon velocity:

```text
x = time_s
y = velocity
x_label = Time (s)
y_label = Velocity
```

CloudScope should not index analysis tables by hardcoded columns like `time_s` or `velocity`.

## Analysis dependencies for ROI mutation

When deleting or editing an ROI, dependent analysis must be removed.

Expected APIs:

```text
acq_image.analysis_set.delete_roi(roi_id) -> int
acq_image.analysis_set.edit_roi(roi_id) -> int
```

These should remove analysis for the ROI across all channels and return the number removed.

Future improvement:

```python
acq_image.analysis_set.get_roi_analysis_report(roi_id) -> list[dict]
```

This would support confirmation dialogs such as:

```text
This ROI has radon_velocity and diameter analysis. Editing/deleting the ROI will remove those results.
```

## Long-running analysis API

Analysis APIs should support progress and cancellation.

Expected pattern:

```python
run(
    progress_callback: Callable[[float, str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
)
```

The exact signature can vary, but the requirement is stable:

```text
- report progress from backend
- periodically check cancellation
- raise/return a clear cancelled state
- do not save automatically
```

## Load/save progress and cancellation

AcqImageList load/save operations can be long-running.

Expected behavior:

```text
load cancel:
  do not replace current CloudScope file list unless load completes successfully

save cancel:
  files already saved remain saved
  files not yet reached are not saved
```

acqstore should expose progress/cancel hooks. CloudScope should not fake progress in the GUI.

## Dirty state

CloudScope uses dirty state to enable/disable save buttons and show row state.

Expected:

```text
acq_image.is_dirty
```

If an operation changes ROI, metadata, analysis, or persistence-relevant state, acqstore should mark the object dirty.

## Stress points and improvement areas

### ReferenceImage API must mature

Reference-image support should be strengthened in acqstore before adding more GUI features. Required public APIs:

```text
get_plane(channel)
get_line_roi()
possibly get_display_extent()
```

### Analysis dependency reporting is still thin

Confirmation dialogs should eventually get a user-facing report from acqstore rather than inspecting analysis internals in CloudScope.

### Load/save cancellation needs rigorous tests

Cancellation must be tested at backend API level, not only GUI level.

### Analysis APIs should converge

All analysis types should expose consistent:

```text
params schema
run API
result API
plot data API
persistence API
```

This keeps CloudScope analysis views generic.


## Dependent Analysis Pattern

Analyses may depend on the canonical output of another analysis.

Dependent analyses should consume:

```python
BaseAnalysis.get_plot_data()
```

rather than directly inspecting analysis-specific dataframe columns.

Example:

- `event` analysis depends on `radon_velocity`
- `event` analysis consumes `AnalysisPlotData`
- GUI code should not need to know dataframe column names such as `time_s` or `velocity`
