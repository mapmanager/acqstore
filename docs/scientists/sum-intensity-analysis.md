# Sum Intensity Analysis

Sum intensity analysis measures **normalized line intensity** along a line scan kymograph ROI
and detects transient peaks from a **functional reporter (like GCaMP)**. The analysis
operates on a rectangular ROI crop with time along rows and space along columns.

## Input data

Sum intensity analysis expects image data organized as a line scan kymograph. The ROI should
cover the spatial region used to estimate the mean line intensity trace. Peak detection uses
normalized intensity (`sum_intensity / spatial_pixel_count`) so ROIs with different widths
remain more comparable.

## Signal pipeline

The backend pipeline:

1. Compute row sums over the spatial dimension, with optional rolling averaging.
2. Normalize by spatial pixel count.
3. Optionally median-filter the normalized trace.
4. Optionally apply single-exponential detrending for photobleaching.
5. Estimate a scalar F0 baseline (percentile or manual).
6. Compute df/f0 and its time derivative.
7. Detect onsets (derivative threshold by default), enforce a refractory period, refine peaks,
   and measure fractional peak widths.

## Detection parameters

Detection parameters define preprocessing (filtering, detrending, F0 baseline) and peak
detection (threshold, refractory period, search windows). Parameters are grouped in the GUI
as **Preprocessing** and **Peak Detection**.

Built-in detection presets provide starting parameter sets:

| Preset | Typical use |
|---|---|
| **fast** | Rapid transients |
| **medium** | General-purpose starting point |
| **slow** | Slower rise and decay kinetics |

Presets are copied into the analysis when selected; editing controls afterward does not change
the built-in preset registry. Manual F0 uses `baseline_method="manual"` and a user-supplied
`manual_f0_baseline` value. Percentile F0 uses `baseline_method="percentile"` and
`baseline_percentile`.

In the GUI, after a run, the peak-detection plot context menu **Edit F0** opens an interactive
baseline editor: drag a Manual F0 line and click **Set Manual F0**, or choose a percentile and
click **Set Auto F0**. Both actions write the corresponding detection parameters and re-run
analysis.

For the full parameter schema and defaults, see the
[Sum Intensity Analysis API](../api/sum-intensity-analysis.md) (`get_detection_param_schema`).

## Results

Sum intensity analysis stores summary values in the `AcqImage` JSON sidecar and writes tabular
output to a CSV file.

For a source file named `my_file.tif`, sum intensity analysis saves:

```text
my_file.tif.json
my_file.tif.sum_intensity.csv
```

The JSON sidecar includes detection parameters and summary values for each analyzed ROI.
Typical summary values include:

- `num_peaks`
- `f0_baseline`
- `baseline_method`
- `detection_source`
- `peak_amplitude_mean`
- `peak_amplitude_median`
- `peak_events` (per-event onset, peak, width, and kinetics measurements)

The CSV file stores per-timepoint traces (for example `df_f_signal`, `d_df_f_signal`) and
onset/peak marker columns (`is_onset`, `is_peak`).

Event-level features (prominence, rise/decay kinetics, area under curve) are documented in the
backend feature schema (`get_feature_schema` on `SumIntensityAnalysis`).

## Programmatic use

Sum intensity analysis can be run from the GUI or from Python code using the same `acqstore`
backend.

See the [Sum Intensity Analysis API](../api/sum-intensity-analysis.md) and the
[Sum Intensity Analysis Notebook](../notebooks/sum-intensity-analysis.ipynb).
