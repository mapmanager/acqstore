# Sum Intensity Analysis

Sum intensity analysis measures **normalized line intensity** along a line-scan
kymograph ROI and detects transient peaks from a **functional reporter (like
GCaMP)**. The ROI crop uses time along rows and space along columns.

## Input data

Expects a line-scan kymograph and a rectangular ROI covering the spatial region
used for the mean line-intensity trace. Peak detection uses normalized intensity
(`sum_intensity / spatial_pixel_count`) so ROIs with different widths remain more
comparable.

## Signal pipeline

1. Compute row sums over the spatial dimension, with optional rolling averaging.
2. Normalize by spatial pixel count.
3. Optionally median-filter the normalized trace.
4. Optionally apply single-exponential detrending for photobleaching.
5. Estimate a scalar F0 baseline (percentile or manual).
6. Compute df/f0 and its time derivative.
7. Detect onsets (derivative threshold by default), enforce a refractory period,
   refine peaks, and measure fractional peak widths.

## Programmatic use

```python
from acqstore.acq_image import AcqImageList
from acqstore.acq_image.analysis.sum_intensity_analysis.sum_intensity_analysis import (
    SumIntensityAnalysis,
)
from acqstore.acq_image.analysis.sum_intensity_analysis.sum_intensity_presets import (
    SumIntensityPresetName,
)
from acqstore.sample_data import ensure_sample

folder = ensure_sample('diameter-sample-data')
acq = AcqImageList(str(folder)).get_files()[0]

channel = acq.images.channel_indices[0]
roi = acq.rois.create_rect_roi(name='sum_intensity')

detection_params = SumIntensityAnalysis.get_detection_preset_params(
    SumIntensityPresetName.MEDIUM
)

sum_intensity = acq.analysis_set.create_and_run(
    SumIntensityAnalysis,
    channel=channel,
    roi_id=roi.roi_id,
    detection_params=detection_params,
    replace_existing=True,
)
print(sum_intensity.result.summary)
acq.save()
```

## Detection parameters

Built-in presets provide starting parameter sets:

| Preset | Typical use |
|---|---|
| **fast** | Rapid transients |
| **medium** | General-purpose starting point |
| **slow** | Slower rise and decay kinetics |

Presets are copied into the analysis when selected; later edits do not change
the built-in preset registry. Manual F0 uses `baseline_method="manual"` and
`manual_f0_baseline`. Percentile F0 uses `baseline_method="percentile"` and
`baseline_percentile`.

For the full schema see the
[Sum Intensity Analysis API](../../api/sum-intensity-analysis.md)
(`get_detection_param_schema`).

## Results

For a source file `my_file.tif`:

```text
my_file.tif.json
my_file.tif.sum_intensity.csv
```

Typical summary fields include `num_peaks`, `f0_baseline`, `baseline_method`,
`detection_source`, `peak_amplitude_mean`, `peak_amplitude_median`, and
`peak_events`. The CSV stores per-timepoint traces and onset/peak markers.

See also the
[Sum Intensity Analysis notebook](../../notebooks/sum-intensity-analysis.ipynb).
