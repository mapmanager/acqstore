# Velocity Analysis

Velocity analysis estimates **blood flow velocity from line scan kymographs**
using a Radon-transform-based method.

## Input data

Expects a line-scan kymograph and a **rectangular ROI** covering the region used
to estimate flow. Physical X/Y calibration must be correct (see
[AcqImage physical units](../../acqimage.md#physical-units-critical-for-analysis)).

## Programmatic use

```python
from acqstore.acq_image import AcqImage
from acqstore.acq_image.analysis import RadonVelocityAnalysis
from acqstore.sample_data import ensure_sample_file

acq = AcqImage(str(ensure_sample_file('kymograph-flow')))

channel = acq.images.channel_indices[0]
roi_ids = acq.rois.get_roi_ids()
roi_id = roi_ids[0] if roi_ids else acq.rois.create_rect_roi(name='velocity').roi_id

velocity = acq.analysis_set.create_and_run(
    RadonVelocityAnalysis,
    channel=channel,
    roi_id=roi_id,
    detection_params={'window_width': 64},
    replace_existing=True,
    execution_options={'use_multiprocessing': False},
)
print(velocity.result.summary)
acq.save()
```

!!! tip "Notebooks / macOS"
    Pass `execution_options={'use_multiprocessing': False}` inside Jupyter so
    Radon velocity runs serially.

## Detection parameters

The primary parameter is the width of each Radon analysis window.

--8<-- "schemas/velocity_detection_parameters.md"

## Results

For a source file `my_file.tif`:

```text
my_file.tif.json
my_file.tif.radon_velocity.csv
```

Typical summary fields: `velocity_mean`, `velocity_median`, `velocity_cv`,
`num_windows`. The CSV stores per-window tabular velocity results.

See the [Velocity Analysis API](../../api/velocity-analysis.md) and the
[Velocity Analysis notebook](../../notebooks/velocity-analysis.ipynb).
