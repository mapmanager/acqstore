# Sum intensity analysis architecture

This document captures the current backend design for CloudScope sum-intensity
analysis. The goal is to keep scientific logic in `acqstore` so future
CloudScope views are thin adapters over a stable backend API.

## Package boundary

Sum-intensity analysis belongs entirely in `src/acqstore/`. It must not import
from `cloudscope` or `nicewidgets`. GUI code should consume public analysis
methods and should not parse result tables or JSON summaries directly.

## Core inputs

The NumPy-only core algorithm operates on primitive scientific inputs:

```python
result = run_sum_intensity(
    image,
    detection_params=params,
    physical_units=(seconds_per_line, um_per_pixel),
)
```

The image is a two-dimensional ROI crop with shape `(time, space)`. Axis 0 is
line-scan time. Axis 1 is spatial samples along the scan line.

The `SumIntensityAnalysis` `BaseAnalysis` wrapper adapts `AcqImage` runtime data
to these primitive inputs by calling `AnalysisDataProvider.get_roi_image()` and
`AnalysisDataProvider.get_image_physical_units()`.

## Signal pipeline

The pipeline is intentionally explicit and serializable:

1. `sum_intensity`: row sum over spatial pixels.
2. `norm_sum_intensity`: `sum_intensity / image.shape[1]`, equivalent to mean
   line intensity.
3. `filtered_norm_sum_intensity`: optional median filtering, default kernel 3.
4. `detrended_norm_sum_intensity`: optional single-exponential photobleaching
   correction.
5. `df_f_signal`: delta-F over F0.
6. `d_df_f_signal`: derivative of `df_f_signal` in `1/s`.
7. Onset detection by derivative or absolute threshold.
8. Peak refinement within `peak_search_window_ms`.
9. Width measurements within `width_search_window_ms`.

The result table stores one row per time point. The summary stores compact
scalar values and serialized peak events.


### Detection presets

Built-in detection presets live in AcqStore and are exposed through the
sum-intensity backend API. They are intended as starting points for scripts and
GUI controls, not as hidden behavior. Selecting a preset should copy its complete
detection-parameter mapping into the parameter editor. The user may then edit
individual fields before running detection.

Initial built-in presets are:

- `fast` — short refractory/search windows for exploratory fast-event detection.
- `medium` — general-purpose starting point.
- `slow` — longer refractory/search windows for slower biological kinetics.

Manual F0 is not a preset. It is a normal parameter edit:

```python
baseline_method = "manual"
manual_f0_baseline = dragged_value
```

This keeps preset selection separate from the manual-F0 workflow used by the GUI.

## F0 baseline

F0 is a run-level result, not a peak-level result. Detection parameters specify
how F0 is calculated; the summary stores the actual calculated value.

Supported first-pass baseline methods:

- `percentile`: scalar percentile of the filtered/detrended normalized trace.
- `manual`: user-supplied scalar F0 value from `manual_f0_baseline`.

Relevant detection parameters:

- `baseline_method`
- `baseline_percentile`
- `manual_f0_baseline`
- `baseline_min_value`

The summary stores:

- `f0_baseline`
- `baseline_method`
- `baseline_percentile`
- `manual_f0_baseline`

A future CloudScope view may allow the user to drag a horizontal F0 line on the
filtered/detrended normalized-intensity plot, then write that value to
`manual_f0_baseline` and set `baseline_method='manual'` before rerunning the
analysis.

## Detection parameters

Detection parameters are flat backend-native values. Time-window parameters are
specified in physical units and converted to points internally.

Stable first-pass detection controls:

- `detection_source`: trace used by the detector.
- `detection_method`: `derivative_threshold` or `absolute_threshold`.
- `derivative_threshold_per_sec`: derivative threshold for selected source.
- `absolute_threshold`: absolute threshold for selected source.
- `refractory_period_ms`: onset-to-onset refractory period.
- `peak_search_window_ms`: forward window for peak refinement.
- `width_search_window_ms`: forward window for falling-side width crossings.

No legacy parameter aliases are supported unless explicitly requested.

## Peak events

`peak_events` is a list of one event record per accepted peak. Each event
contains:

- onset index, time, and value
- refined peak index, time, value, and amplitude
- interval values such as onset-to-onset and peak-to-peak interval
- fractional width measurements
- event-local warnings and status

`peak_events` means detected peak-like events, not only peak coordinates.

## Failure model

Expected scientific failures are stored as data rather than raised as runtime
exceptions. Examples:

- exponential detrend fit fails: analysis summary records an error and falls
  back to the previous valid trace
- F0 is near zero: summary records a warning and applies `baseline_min_value`
- width crossing not found before `width_search_window_ms`: level-crossing
  status records a normal measurement failure

Invalid API inputs still raise clear exceptions.

## Public plotting API

The backend exposes plotting primitives that are independent of Plotly,
NiceGUI, or ECharts.

Continuous traces:

```python
df_f = result.get_trace(SumIntensityTraceKey.DF_F_SIGNAL)
deriv = result.get_trace(SumIntensityTraceKey.D_DF_F_SIGNAL)
```

Event marker points:

```python
onsets = result.get_event_points(SumIntensityEventPointKey.ONSETS)
peaks = result.get_event_points(SumIntensityEventPointKey.PEAKS)
```

Width overlays:

```python
width_50 = result.get_width_trace(PeakWidthLevel.WIDTH_50)
all_widths = result.get_width_trace()
```

Width traces are NaN-separated line segments so frontends can draw all event
width overlays with ordinary line traces and `connectgaps=False`.

## GUI design intent

A future `SumIntensityPlotView` should contain no scientific logic. It should
map backend primitives to a child plotting widget:

- `ResultTrace` -> line trace
- `ResultPoints` -> scatter markers
- width `ResultTrace` -> line trace with gaps not connected
- F0 summary value -> horizontal measurement line only when the user is editing
  manual F0

A future `SumIntensityDetectionParamView` should edit detection params only. It
should not inspect raw result JSON.


## Concrete GUI integration examples

This section is intentionally explicit so future LLM-assisted tickets can write
CloudScope views without inventing backend APIs.

### SumIntensityParametersView contract

Responsibilities:

- Render detection controls from the backend detection-param schema.
- Let the user edit detection params.
- Run sum-intensity analysis from a **Detect** button.
- Dispatch analysis through a CloudScope/NiceGUI CPU-bound path.
- Emit or respond to a detection-finished event.

Non-responsibilities:

- Do not parse raw JSON summaries.
- Do not inspect private analysis attributes.
- Do not recalculate F0, derivative, onset, peak, or width measurements.

Pseudo-code:

```python
class SumIntensityParametersView:
    """GUI view for editing sum-intensity detection parameters.

    The view is a thin adapter over the acqstore analysis API. It renders
    controls from the detection parameter schema and dispatches analysis work
    through a CPU-bound execution helper so NiceGUI remains responsive.
    """

    def render(self, analysis: SumIntensityAnalysis) -> None:
        schema = analysis.get_detection_param_schema()
        params = analysis.get_detection_params()

        for field in schema.fields:
            self._add_control(
                name=field.name,
                label=field.display_name,
                description=field.description,
                value=params[field.name],
                value_type=field.value_type,
            )

        ui.button("Detect", on_click=lambda: self._detect(analysis))

    async def _detect(self, analysis: SumIntensityAnalysis) -> None:
        params = self._collect_params()

        # Use the real CloudScope CPU-bound helper/controller path here. Do not
        # call analysis.run() directly inside the NiceGUI event loop.
        await run_cpu_bound(lambda: analysis.set_detection_params(params))
        await run_cpu_bound(analysis.run)

        self._emit_detection_finished(analysis)
```

Stable backend methods used by this view:

```python
analysis.get_detection_param_schema()
analysis.get_detection_params()
analysis.set_detection_params(params)
analysis.run()
```

### Manual F0 interaction

Manual F0 is controlled by detection parameters:

```python
params["baseline_method"] = "manual"
params["manual_f0_baseline"] = dragged_f0_value
analysis.set_detection_params(params)
analysis.run()
```

Recommended GUI flow:

1. Plot `filtered_norm_sum_intensity` or `detrended_norm_sum_intensity`.
2. Add a draggable horizontal measurement line initialized from
   `SumIntensitySummaryKey.F0_BASELINE` when a previous result exists.
3. When the user accepts the line, set `baseline_method='manual'` and write the
   line value to `manual_f0_baseline`.
4. Rerun the analysis through the CPU-bound path.
5. Refresh all plotted traces from backend accessors.

F0 units are normalized-intensity units, because F0 is estimated from the
filtered/detrended normalized sum-intensity trace before `df_f_signal` is
computed.

### SumIntensityPlotView contract

Responsibilities:

- Display backend result primitives.
- Map `ResultTrace` to line traces.
- Map `ResultPoints` to scatter markers.
- Draw width overlays as ordinary line traces with gaps not connected.
- Optionally expose interactive measurement lines for parameter editing, but
  only by writing back to detection params and rerunning the backend analysis.

Non-responsibilities:

- Do not inspect the raw result table column names directly.
- Do not parse `peak_events` JSON directly for plotting when an accessor exists.
- Do not implement detection logic in the view.

Pseudo-code:

```python
class SumIntensityPlotView:
    """GUI view for plotting sum-intensity analysis results.

    The view contains a child PlotlyPlotWidget. It translates backend-native
    result primitives into widget calls and contains no scientific analysis
    logic.
    """

    def __init__(self) -> None:
        self._plot = PlotlyPlotWidget()

    def update_from_analysis(self, analysis: SumIntensityAnalysis) -> None:
        self._plot.clear_traces()

        df_f = analysis.get_trace(SumIntensityTraceKey.DF_F_SIGNAL)
        d_df_f = analysis.get_trace(SumIntensityTraceKey.D_DF_F_SIGNAL)
        onsets = analysis.get_event_points(SumIntensityEventPointKey.ONSETS)
        peaks = analysis.get_event_points(SumIntensityEventPointKey.PEAKS)
        widths = analysis.get_width_trace()

        self._plot.add_trace(name=df_f.display_name, x=df_f.x, y=df_f.y)
        self._plot.add_trace(name=d_df_f.display_name, x=d_df_f.x, y=d_df_f.y)
        self._plot.plot_scatter(name=onsets.display_name, x=onsets.x, y=onsets.y)
        self._plot.plot_scatter(name=peaks.display_name, x=peaks.x, y=peaks.y)
        self._plot.add_trace(
            name=widths.display_name,
            x=widths.x,
            y=widths.y,
            connectgaps=False,
        )
```

Stable backend methods used by this view:

```python
analysis.get_trace(SumIntensityTraceKey.DF_F_SIGNAL)
analysis.get_trace(SumIntensityTraceKey.D_DF_F_SIGNAL)
analysis.get_event_points(SumIntensityEventPointKey.ONSETS)
analysis.get_event_points(SumIntensityEventPointKey.PEAKS)
analysis.get_width_trace()
analysis.get_summary_value(SumIntensitySummaryKey.F0_BASELINE)
```

Recommended rendering details:

- Draw event markers after line traces so markers are not hidden by traces.
- Draw width overlays on top of `df_f_signal`, not as a separate width-vs-time
  line plot.
- Use `connectgaps=False` for width overlays because width traces are
  NaN-separated line segments.
- If derivative and df/f scales make one panel hard to read, place
  `d_df_f_signal` on a secondary y-axis or separate panel.

### Detection parameters for GUI controls

The GUI should expose the following first-pass parameters and labels:

| Parameter | Suggested label | Units | Notes |
| --- | --- | --- | --- |
| `detection_source` | Detection source | enum | Usually `df_f_signal`. |
| `detection_method` | Detection method | enum | `derivative_threshold` or `absolute_threshold`. |
| `derivative_threshold_per_sec` | Derivative threshold | `1/s` | Applied to derivative of selected source. |
| `absolute_threshold` | Absolute threshold | source units | Applied directly to selected source. |
| `refractory_period_ms` | Refractory period | ms | Onset-to-onset exclusion window. |
| `peak_search_window_ms` | Peak search window | ms | Forward onset-to-peak refinement window. |
| `width_search_window_ms` | Width search window | ms | Forward peak-to-falling-crossing window. |
| `baseline_method` | F0 method | enum | `percentile` or `manual`. |
| `baseline_percentile` | F0 percentile | percent | Used by percentile F0 mode. |
| `manual_f0_baseline` | Manual F0 | normalized intensity | Used by manual F0 mode. |
| `baseline_min_value` | F0 floor | normalized intensity | Avoids divide-by-zero. |

### NiceGUI CPU-bound execution note

Sum-intensity detection can perform filtering, curve fitting, event detection,
and feature extraction. It should be treated as CPU-bound from the GUI point of
view. The view should call through the project-standard CloudScope controller or
NiceGUI helper for CPU-bound work rather than running analysis directly in the
websocket/event-loop callback. The specific helper name belongs to CloudScope GUI
code; this architecture document only requires that `analysis.run()` is not
called directly from the UI event loop.

## Event features roadmap

The stable first-pass event object already supports onset, peak, amplitude,
widths, and simple event intervals. Next scientifically useful event features
include:

- time to peak: `peak_time_sec - onset_time_sec`
- pre-peak baseline/mean value
- area under curve (AUC), once event stop criteria are defined
- rise tau and decay tau from local fits

AUC and tau measurements require rigorous definitions of event start/stop and
fit windows. They should be added as first-class event features with status
fields for expected fit or boundary failures.

## Synthetic data

The public synthetic generator lives under
`src/acqstore/acq_image/analysis/sum_intensity_analysis/synthetic/`.
It creates actual `(time, space)` images, not just one-dimensional traces, so
unit tests and development scripts exercise the full core pipeline.

## Detection schema presentation metadata

The sum-intensity detection schema now carries enough metadata for a GUI to build
a mostly data-driven parameter editor. Each detection parameter has a scientific
`category` and a default `visible` flag.

The current categories are:

- `Preprocessing`: trace construction, filtering, detrending, and F0 baseline
  estimation.
- `Peak Detection`: onset detection, polarity, detection source, refractory
  behavior, peak refinement, and width measurement controls.

The schema order remains authoritative. A frontend should preserve the schema
order and insert section headings when the category changes. This keeps layout
decisions in the GUI while keeping scientific grouping in AcqStore.

Hidden fields remain valid backend parameters. For the current sum-intensity
schema, `baseline_min_value` and `level_fractions` are hidden by default because
they are advanced safety/configuration values. They should not be removed from
preset dictionaries or validation.

## Phase 2 event-level feature schema

Sum-intensity event features are now documented by a backend feature schema. The
feature schema is intentionally separate from detection-parameter schema and
continuous trace definitions.

- Detection schema documents inputs that control the analysis.
- Trace definitions document per-timepoint arrays.
- Feature schema documents per-event scalar results stored on each `PeakEvent`.

The public API is:

```python
SumIntensityAnalysis.get_feature_schema()
SumIntensityAnalysis.get_feature_schema_dataframe()
```

Each feature schema entry includes a human-readable `algorithm` field so reports,
notebooks, and GUIs can explain how a result was calculated without hardcoding
that documentation in the frontend.

The current phase-2 feature set is:

- `baseline_mean`
- `baseline_std`
- `prominence`
- `rise_10_90_sec`
- `decay_90_10_sec`
- `decay_time_sec`
- `max_rise_slope`
- `max_decay_slope`
- `auc`

All phase-2 features are stored as `EventFeature` records with value, status,
and reason. Expected scientific failures, such as a missing right 10% crossing,
are encoded in those fields rather than raised as runtime exceptions.

`auc` is currently defined as the area from left 10% crossing to right 10%
crossing above onset value. This avoids inventing a separate event-off detector
while providing a rigorous first-pass event area measurement.

`max_decay_slope` is measured only from peak to right 10% crossing. If the right
10% crossing is unavailable, the feature fails. The implementation deliberately
does not fall back to `peak_search_window_ms` or `width_search_window_ms` for
this slope measurement.
