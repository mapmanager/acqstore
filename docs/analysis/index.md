# Analysis

Analysis modules take detection parameters as input and produce summary values
(and optional tabular CSV outputs) on an `AcqImage` for a given
**(channel, ROI)**.

All current quantitative workflows operate on **line scan kymographs**: repeated
sampling along a spatial line over time (typically Y = time, X = distance along
the line). Correct [physical units](../acqimage.md#physical-units-critical-for-analysis)
and a proper [rectangular ROI](../rois.md) are required for meaningful results.

## Kymograph analysis

| Analysis | What it measures |
|----------|------------------|
| [Velocity](kymograph/velocity-analysis.md) | Blood flow velocity (Radon transform) |
| [Diameter](kymograph/diameter-analysis.md) | Vessel diameter from intensity profiles |
| [Sum intensity](kymograph/sum-intensity-analysis.md) | Normalized line intensity + peak detection (e.g. GCaMP) |
| [Heart rate](kymograph/heart-rate-analysis.md) | Periodic rate from a velocity time series |

## Common pattern

Load one sample file, ensure a rectangular ROI, then `create_and_run`:

```python
from acqstore.acq_image import AcqImage
from acqstore.acq_image.analysis import RadonVelocityAnalysis
from acqstore.sample_data import ensure_sample_file

acq = AcqImage(str(ensure_sample_file('kymograph-flow')))

channel = acq.get_default_channel()
roi_id = acq.get_default_roi()
if roi_id is None:
    roi_id = acq.rois.create_rect_roi(name='analysis').roi_id

analysis = acq.analysis_set.create_and_run(
    RadonVelocityAnalysis,
    channel=channel,
    roi_id=roi_id,
    detection_params={'window_width': 64},
    replace_existing=True,
    execution_options={'use_multiprocessing': False},
)
print(analysis.result.summary)
```

Batch, pools, and event analysis remain documented under [API](../api/index.md).

## Next

Start with [Velocity](kymograph/velocity-analysis.md) or browse the
[notebooks](../notebooks.md).
