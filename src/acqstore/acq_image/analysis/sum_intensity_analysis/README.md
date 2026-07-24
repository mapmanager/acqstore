# Sum intensity analysis and peak detection

This package implements the first backend-only pass of CloudScope sum-intensity
analysis. It is intentionally independent of CloudScope GUI code.

## Axis convention

The ROI image is a 2D array with shape `(time, space)`:

- dim 0 / rows: line-scan time, with spacing `seconds_per_line`.
- dim 1 / columns: spatial samples, with spacing `um_per_pixel`.

The analysis receives this image from `AnalysisDataProvider.get_roi_image()` and
physical spacing from `AnalysisDataProvider.get_image_physical_units()`.

## Signal pipeline

1. Compute row sums over the spatial dimension.
2. Apply optional rolling row-sum averaging using `window_radius_points`.
3. Compute `norm_sum_intensity = sum_intensity / image.shape[1]`.
4. Optionally median-filter the normalized trace. The default kernel is 3
   points, intended to remove one-line acquisition pops while preserving
   biological events when line-scan sampling is much faster than event kinetics.
5. Try single-exponential detrending for photobleaching:
   `a * exp(-b * t) + c`.
6. If detrending fails, record an analysis-level error and continue with the
   filtered normalized trace.
7. Estimate a scalar `F0` baseline from the filtered/detrended trace. The first
   pass supports percentile and manual F0 modes.
8. Compute `df_f_signal = (signal - F0) / F0`. Detection uses this trace by
   default, but `detection_source` may select another continuous trace.
9. Compute `d_df_f_signal = np.gradient(df_f_signal, time_sec)`, preserving one
   derivative value per original time point and expressing the derivative in
   `1 / second`.
10. Detect onsets using derivative threshold by default. Derivative-threshold
    detection is applied to the derivative of the selected `detection_source`.
11. Enforce onset-to-onset refractory period.
12. Search forward from each accepted onset within `peak_search_window_ms` to
    refine the peak.
13. Measure requested fractional widths such as 0.1, 0.2, 0.5, 0.8, and 0.9.
    Falling-side width crossings must occur within `width_search_window_ms`;
    otherwise the crossing status records a normal measurement failure.

## Delta-F over F0 baseline

Delta-F over F0 is useful because raw mean-line intensity still depends on
absolute fluorescence brightness, microscope settings, and image dtype. The
critical choice is how to estimate `F0`. This first pass supports a scalar
percentile baseline and a manual scalar baseline:

```python
F0 = percentile(filtered_or_detrended_norm_sum_intensity, baseline_percentile)
# or, when baseline_method == "manual":
F0 = manual_f0_baseline
df_f_signal = (filtered_or_detrended_norm_sum_intensity - F0) / F0
```

The default `baseline_percentile` is 20.0. This is not intended to be the final
scientific answer for all recordings; it is a simple starting point that is
easy to plot, inspect, and tune. For difficult traces, callers may set
`baseline_method="manual"` and provide `manual_f0_baseline`, for example from a
user-dragged horizontal measurement line in a future CloudScope view. The result
table stores `f0_baseline` and `df_f_signal`; the summary stores the baseline
method, percentile, manual F0 value, and actual F0 used. If F0 is zero or too
close to zero, the algorithm records a warning and uses `baseline_min_value` to
avoid division by zero.

## Vectorized row-sum performance

A direct implementation of `window_radius_points` could loop over every time row,
extract a 2D band, average rows, and then sum pixels. That repeats overlapping
work for adjacent rows and can be expensive for 30,000-50,000 line scans.

This implementation uses an equivalent 1D formulation:

```python
row_sum = image.sum(axis=1)
sum_intensity = rolling_mean(row_sum, radius=window_radius_points)
norm_sum_intensity = sum_intensity / image.shape[1]
```

The rolling mean uses a cumulative-sum implementation with clipped edge windows.
This keeps the first pass KISS, NumPy-vectorized, and single-process. Threading
or multiprocessing should only be added after a benchmark proves this step is a
bottleneck.

## Failure model

Expected scientific failures are stored as data rather than thrown as runtime
exceptions.

- Analysis-level failures, such as `curve_fit` not converging during detrending,
  are recorded in the summary `errors` list and the algorithm falls back to the
  previous valid signal.
- Event-local failures are recorded in event `warnings` or `status` fields.
- Level-crossing failures are recorded in each crossing's `status`, for example
  `right_not_found_within_width_search_window` when a peak does not decay below
  the requested fraction within the configured width-search window.

Invalid API inputs still raise clear exceptions.

## Result table columns

The analysis table has one row per time point and includes:

- `time_index`
- `time_sec`
- `sum_intensity`
- `norm_sum_intensity`
- `filtered_norm_sum_intensity`
- `detrended_norm_sum_intensity`
- `f0_baseline`
- `df_f_signal`
- `d_df_f_signal`
- `is_onset`
- `is_peak`
- `onset_index`
- `peak_id`

## Backend API primitives

The public backend API avoids raw DataFrame column names and raw JSON parsing.
Callers should use enums and accessors:

- `SumIntensityTraceKey` names continuous traces such as `DF_F_SIGNAL` and
  `D_DF_F_SIGNAL`.
- `SumIntensityEventPointKey` names sparse marker collections such as `ONSETS`
  and `PEAKS`.
- `PeakWidthLevel` names standard width levels: 10, 20, 50, 80, and 90 percent
  of peak amplitude.
- `SumIntensitySummaryKey` names scalar summary values such as `F0_BASELINE`,
  `DETECTION_SOURCE`, and `SECONDS_PER_LINE`.

Typical access pattern:

```python
df_f = analysis.get_trace(SumIntensityTraceKey.DF_F_SIGNAL)
deriv = analysis.get_trace(SumIntensityTraceKey.D_DF_F_SIGNAL)
onsets = analysis.get_event_points(SumIntensityEventPointKey.ONSETS)
peaks = analysis.get_event_points(SumIntensityEventPointKey.PEAKS)
width_50 = analysis.get_width_trace(PeakWidthLevel.WIDTH_50)
all_widths = analysis.get_width_trace()
f0 = analysis.get_summary_value(SumIntensitySummaryKey.F0_BASELINE)
```

`get_width_trace()` returns NaN-separated line segments so frontend widgets can
plot all event widths with one ordinary line trace and `connectgaps=False`.

## Summary contents

The analysis summary stores compact run-level metadata and scalar results. It
does not store full traces; those live in the result table. It includes:

- `status`
- `num_timepoints`
- `num_peaks`
- `num_space_pixels`
- `seconds_per_line`
- `f0_baseline`
- `baseline_method`
- `baseline_percentile`
- `manual_f0_baseline`
- `detection_source`
- `peak_search_window_ms`
- `width_search_window_ms`
- `peak_amplitude_mean`
- `peak_amplitude_median`
- `warnings`
- `errors`
- `peak_events`

`peak_events` is a list of one record per detected event. Each event contains
onset measurements, refined peak measurements, interval measurements, width
level crossings, event-local warnings, and event status. The name means
"detected peak-like events", not merely peak coordinate arrays.



## Detection presets

Sum-intensity detection presets are first-class AcqStore API, not CloudScope GUI
state. They provide complete detection-parameter dictionaries that callers may
copy into an analysis and then edit before running. Manual F0 is intentionally
not a preset; it remains a normal detection-parameter workflow using
`baseline_method="manual"` and `manual_f0_baseline`.

Built-in preset names are:

- `SumIntensityPresetName.FAST`
- `SumIntensityPresetName.MEDIUM`
- `SumIntensityPresetName.SLOW`

Typical usage:

```python
from acqstore.acq_image.analysis.sum_intensity_analysis.sum_intensity_analysis import (
    SumIntensityAnalysis,
)
from acqstore.acq_image.analysis.sum_intensity_analysis.sum_intensity_presets import (
    SumIntensityPresetName,
)

params = SumIntensityAnalysis.get_detection_preset_params(
    SumIntensityPresetName.MEDIUM
)
analysis = SumIntensityAnalysis(channel=0, roi_id=roi_id, detection_params=params)
```

GUI code can list preset descriptors for a select widget:

```python
presets = SumIntensityAnalysis.get_detection_presets()
options = {preset.name.value: preset.display_name for preset in presets}
```

Each preset descriptor contains `name`, `display_name`, `description`, and a
complete copied `params` mapping. Preset parameter dictionaries are independent
copies, so editing GUI controls does not mutate the built-in preset registry.

## CloudScope GUI integration examples

These examples show the intended public API for future CloudScope views. The
views should remain thin adapters: they edit detection parameters, trigger the
backend analysis, and map backend result primitives to plotting widgets. They
should not parse raw result JSON, inspect private fields, or reimplement
scientific logic.

### Detection parameter view

A future `SumIntensityParametersView` should read the analysis detection schema,
render editable controls, and dispatch analysis from a user action such as a
**Detect** button. The analysis must run away from the NiceGUI event loop so the
websocket and desktop UI remain responsive.

Pseudo-code:

```python
class SumIntensityParametersView:
    """Thin GUI adapter for editing sum-intensity detection parameters."""

    def __init__(self, controller: CloudScopeController) -> None:
        self._controller = controller
        self._params = {}

    def render(self, analysis: SumIntensityAnalysis) -> None:
        schema = analysis.get_detection_param_schema()
        self._params = analysis.get_detection_params()

        for field in schema.fields:
            # Render one GUI control per field using field.display_name,
            # field.description, field.value_type, default/current value,
            # and units when available.
            self._render_param_control(field, self._params[field.name])

        # Button callback should not run CPU-heavy analysis directly in the
        # NiceGUI event loop.
        ui.button("Detect", on_click=lambda: self._run_detection(analysis))

    async def _run_detection(self, analysis: SumIntensityAnalysis) -> None:
        params = self._collect_current_params()

        # Use the project-standard CPU-bound helper/wrapper used elsewhere in
        # CloudScope. Exact helper name belongs to CloudScope controller code;
        # the key requirement is that analysis.run() does not block the event
        # loop or websocket.
        await run_cpu_bound(lambda: analysis.set_detection_params(params))
        await run_cpu_bound(analysis.run)

        self._controller.emit_sum_intensity_detection_finished(analysis)
```

The exact controller/event names are CloudScope-level details. The stable
backend contract is:

```python
schema = analysis.get_detection_param_schema()
params = analysis.get_detection_params()
analysis.set_detection_params(params)
analysis.run()
```

### Detection parameters and units

The first-pass GUI should expose these detection parameters directly:

| Parameter | Units | Meaning |
| --- | --- | --- |
| `detection_source` | enum | Continuous trace used by the detector. Default is `df_f_signal`. |
| `detection_method` | enum | `derivative_threshold` or `absolute_threshold`. |
| `derivative_threshold_per_sec` | `1/s` for `df_f_signal` | Threshold applied to derivative of the selected detection source. |
| `absolute_threshold` | selected source units | Threshold applied directly to the selected detection source. |
| `refractory_period_ms` | ms | Minimum onset-to-onset interval for accepted events. |
| `peak_search_window_ms` | ms | Forward search window from onset to refined peak. |
| `width_search_window_ms` | ms | Forward search window from peak to falling-side width crossing. |
| `baseline_method` | enum | `percentile` or `manual`. |
| `baseline_percentile` | percent | Percentile used when `baseline_method='percentile'`. |
| `manual_f0_baseline` | normalized-intensity units | User-specified F0 value when `baseline_method='manual'`. |
| `baseline_min_value` | normalized-intensity units | Floor used to avoid division by zero. |

### Manual F0 workflow

Manual F0 is a detection-parameter workflow, not a plot-only annotation.

Recommended GUI flow:

1. Plot `filtered_norm_sum_intensity` or `detrended_norm_sum_intensity`.
2. Draw a draggable horizontal measurement line initialized to the current
   summary F0 value from `SumIntensitySummaryKey.F0_BASELINE`.
3. Let the user drag the line and click **Use as F0**.
4. Set detection params:

```python
params = analysis.get_detection_params()
params["baseline_method"] = "manual"
params["manual_f0_baseline"] = measurement_line.position
analysis.set_detection_params(params)
analysis.run()
```

5. Refresh the plot from backend result primitives.

The summary stores the actual F0 used:

```python
f0 = analysis.get_summary_value(SumIntensitySummaryKey.F0_BASELINE)
method = analysis.get_summary_value(SumIntensitySummaryKey.BASELINE_METHOD)
```

### Plot view

A future `SumIntensityPlotView` should consume backend primitives and map them
to a child plotting widget such as `PlotlyPlotWidget`. The backend primitives are
plotting-library independent.

Pseudo-code:

```python
class SumIntensityPlotView:
    """Thin GUI adapter that displays sum-intensity traces and events."""

    def __init__(self) -> None:
        self._plot = PlotlyPlotWidget()

    def update_from_analysis(self, analysis: SumIntensityAnalysis) -> None:
        self._plot.clear_traces()

        df_f = analysis.get_trace(SumIntensityTraceKey.DF_F_SIGNAL)
        deriv = analysis.get_trace(SumIntensityTraceKey.D_DF_F_SIGNAL)
        onsets = analysis.get_event_points(SumIntensityEventPointKey.ONSETS)
        peaks = analysis.get_event_points(SumIntensityEventPointKey.PEAKS)
        widths = analysis.get_width_trace()

        self._plot.add_trace(
            name=df_f.display_name,
            x=df_f.x,
            y=df_f.y,
        )
        self._plot.add_trace(
            name=deriv.display_name,
            x=deriv.x,
            y=deriv.y,
        )
        self._plot.plot_scatter(
            name=onsets.display_name,
            x=onsets.x,
            y=onsets.y,
        )
        self._plot.plot_scatter(
            name=peaks.display_name,
            x=peaks.x,
            y=peaks.y,
        )
        self._plot.add_trace(
            name=widths.display_name,
            x=widths.x,
            y=widths.y,
            connectgaps=False,
        )
```

Recommended first plot contents:

- `df_f_signal` line trace.
- `d_df_f_signal` line trace, ideally on a secondary y-axis or separate panel.
- onset markers from `get_event_points(SumIntensityEventPointKey.ONSETS)`.
- peak markers from `get_event_points(SumIntensityEventPointKey.PEAKS)`.
- width overlays from `get_width_trace()`, drawn on top of `df_f_signal` with
  gaps not connected.

Event markers should be drawn after line traces so they remain visible.

## References

This implementation is designed to be compatible with CloudScope analysis
patterns while taking conceptual cues from SanPy, IPFX, eFEL, and
`scipy.signal.find_peaks`. The first pass does not call `find_peaks`; it keeps
onset detection and feature extraction explicit so CloudScope can serialize and
inspect each event.

## Synthetic data utilities

The package includes a public synthetic-data utility under `synthetic/`. It is
intended for algorithm tests, interactive development scripts, and future demos.
The generator returns an actual image, not just a one-dimensional trace, so the
normal analysis pipeline is exercised end-to-end:

```python
synthetic = make_synthetic_sum_intensity_image(config)
result = run_sum_intensity(
    synthetic.image,
    detection_params=params,
    physical_units=(synthetic.seconds_per_line, synthetic.um_per_pixel),
)
```

The synthetic model is:

```python
F(t) = bleach(t) * (F0 + sum(events(t))) + noise + pop_artifacts
image[t, x] = F(t) * spatial_profile[x] + spatial_noise[t, x]
```

Events use unit-peak difference-of-exponentials kernels with configurable rise
and decay time constants. Event times can be supplied explicitly for
deterministic tests or generated from a Poisson process for exploratory data.
Optional event jitter is applied after event-time generation. Ground truth is
returned as a DataFrame with event onset time, approximate peak time, and event
amplitude.

The synthetic utilities deliberately live in `src/acqstore` rather than only in
`tests/` because they are useful as public development fixtures for validating
new detection parameters, plotting primitives, and CloudScope views.


## Architecture document

A longer developer-oriented design document lives at
`docs-dev/acqstore/analysis/sum_intensity_architecture.md`. It documents the
backend API, F0 baseline model, plotting primitives, GUI design intent, failure
model, and event-feature roadmap.

## Detection schema presentation metadata

Sum-intensity detection parameters include backend-owned presentation metadata so
frontends can build organized editors without hardcoding parameter names.

Each `DetectionParamSchema` entry has:

- `category`: scientific grouping for the parameter.
- `visible`: whether the parameter should be shown by default.

Current categories are:

- `DetectionParamCategory.PREPROCESSING` / `"Preprocessing"`
- `DetectionParamCategory.PEAK_DETECTION` / `"Peak Detection"`

The schema order remains the canonical linear order. GUI code can keep that order
and insert headings whenever `category` changes. Hidden parameters remain part of
the schema and detection params; they are hidden only from basic editors.

The currently hidden advanced fields are:

- `baseline_min_value`
- `level_fractions`

These fields are still validated by AcqStore and remain available for scripts,
advanced editors, and future expert-mode GUI controls.

## Event feature schema

Sum-intensity analysis now documents event-level result features with a backend
feature schema. This schema is separate from continuous trace definitions:

- traces describe per-timepoint arrays such as `df_f_signal` and
  `d_df_f_signal`.
- event features describe one scalar measurement per `PeakEvent`.

Use:

```python
schema = SumIntensityAnalysis.get_feature_schema()
df = SumIntensityAnalysis.get_feature_schema_dataframe()
```

The feature schema includes `name`, `display_name`, `value_type`, `unit`,
`description`, `algorithm`, and `category`. The `algorithm` field is intended for
human-readable reports and GUI tooltips that explain how each scalar was
calculated.

Current documented event-level features are:

| Feature | Category | Algorithm summary |
| --- | --- | --- |
| `baseline_mean` | Baseline | Mean detection-source value in `[onset - baseline_window_ms, onset)`. |
| `baseline_std` | Baseline | Sample standard deviation in the same pre-onset baseline window. |
| `prominence` | Amplitude | Peak value relative to `baseline_mean`, polarity-aware. |
| `rise_10_90_sec` | Kinetics | `left_90_time_sec - left_10_time_sec`. |
| `decay_90_10_sec` | Kinetics | `right_10_time_sec - right_90_time_sec`. |
| `decay_time_sec` | Kinetics | Alias of `decay_90_10_sec` in this implementation. |
| `max_rise_slope` | Slope | Maximum signed derivative from onset to peak. |
| `max_decay_slope` | Slope | Maximum signed derivative from peak to right 10% crossing. |
| `auc` | Area | Integral from left 10% to right 10% crossing above onset value. |

Each new feature is stored as an `EventFeature` record with:

- `value`
- `status`
- `reason`

This keeps expected scientific failures, such as missing right-side decay
crossings, as result data rather than runtime errors.

## Event-local baseline window

`baseline_window_ms` is a visible preprocessing detection parameter. It controls
the pre-onset window used to calculate `baseline_mean`, `baseline_std`, and
`prominence`. If the baseline window does not include enough samples for a
measurement, the affected feature is marked as failed with failure provenance.
