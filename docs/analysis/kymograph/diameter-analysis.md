# Diameter Analysis

Diameter analysis estimates **vessel diameter from line scan kymographs** using
intensity-profile measurements on a rectangular ROI.

## Input data

Expects a line-scan kymograph. The ROI should cover the spatial region used to
estimate vessel width. Diameter is reported in physical units when X calibration
is correct (see
[AcqImage physical units](../../acqimage.md#physical-units-critical-for-analysis)).

## Programmatic use

```python
from acqstore.acq_image import AcqImageList
from acqstore.acq_image.analysis import DiameterAnalysis
from acqstore.sample_data import ensure_sample

folder = ensure_sample('diameter-sample-data')
acq = AcqImageList(str(folder)).get_files()[0]

channel = acq.images.channel_indices[0]
roi_ids = acq.rois.get_roi_ids()
roi_id = roi_ids[0] if roi_ids else acq.rois.create_rect_roi(name='diameter').roi_id

diameter = acq.analysis_set.create_and_run(
    DiameterAnalysis,
    channel=channel,
    roi_id=roi_id,
    detection_params={'diameter_method': 'threshold_width'},
    replace_existing=True,
)
print(diameter.result.summary)
acq.save()
```

## Detection parameters

Parameters control profile aggregation, polarity, thresholding, gradient-based
edge detection, motion gating, and post-filtering.

--8<-- "schemas/diameter_detection_parameters.md"

## Results

For a source file `my_file.tif`:

```text
my_file.tif.json
my_file.tif.diameter.csv
```

Typical summary fields include `diameter_um_mean`, `diameter_um_median`,
`diameter_um_cv`, `num_rows`, `qc_score_mean`, and quality-control violation
counts.

See the [Diameter Analysis API](../../api/diameter-analysis.md) and the
[Diameter Analysis notebook](../../notebooks/diameter-analysis.ipynb).
