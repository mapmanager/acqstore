# Heart Rate Analysis

Heart rate analysis estimates a periodic **heartbeat frequency** from a velocity
time series and reports it in **beats-per-minute (bpm)** and **Hz**
(`bpm = 60 × Hz`).

## Input data

Heart rate is a **dependent analysis**. It is seeded by a `radon_velocity`
analysis on the *same* `(channel, roi_id)` and reads the parent velocity series
through `get_plot_data()`. Run [Velocity Analysis](velocity-analysis.md) first.

## Two estimators and quality control

Every run computes the dominant periodicity with **two** independent estimators:

- **Lomb-Scargle** periodogram
- **Welch** power spectral density

When the two estimates fall within `agree_tol_bpm` of each other the summary
`status` is `ok` (**accept**); when they diverge the status becomes
`method_disagree` (**reject / review**). Too few valid velocity samples yields
`insufficient_valid`.

## Programmatic use

```python
from acqstore.acq_image import AcqImage
from acqstore.acq_image.analysis import HeartRateAnalysis, RadonVelocityAnalysis
from acqstore.sample_data import ensure_sample_file

acq = AcqImage(str(ensure_sample_file('kymograph-flow')))

channel = acq.images.channel_indices[0]
roi_ids = acq.rois.get_roi_ids()
roi_id = roi_ids[0] if roi_ids else acq.rois.create_rect_roi(name='hr').roi_id

acq.analysis_set.create_and_run(
    RadonVelocityAnalysis,
    channel=channel,
    roi_id=roi_id,
    detection_params={'window_width': 64},
    replace_existing=True,
    execution_options={'use_multiprocessing': False},
)

heart_rate = acq.analysis_set.create_and_run(
    HeartRateAnalysis,
    channel=channel,
    roi_id=roi_id,
    replace_existing=True,
)
print(heart_rate.result.summary)
acq.save()
```

## Detection parameters

--8<-- "schemas/heart_rate_detection_parameters.md"

## Results

Heart rate stores a compact summary in the JSON sidecar. There is **no CSV table**
for heart rate.

Typical summary content includes per-estimator results (`lomb`, `welch`), rollup
`status`, and an `agreement` block (`abs_delta_bpm`, `agree_ok`).

See the [Heart Rate Analysis API](../../api/heart-rate-analysis.md) and the
[Heart Rate Analysis notebook](../../notebooks/heart-rate-analysis.ipynb).
The notebook uses the **folder** sample `velocity-sample-data` so it can show
both accept and reject outcomes on two files. For a one-file scripted demo, use
`ensure_sample_file('kymograph-flow')` as above.
