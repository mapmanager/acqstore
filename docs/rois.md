# ROIs

ROIs define the image region used for analysis. Every quantitative analysis run
is associated with an image, a **channel**, and an **ROI id**.

An ROI is defined in image pixel coordinates and is available across **all
channels** of an `AcqImage`. Analysis still picks one channel per run. The same
ROI can be reused for analysis on different channels.

AcqStore supports two ROI types:

- **Rectangular ROI** (`rectroi`): used by all current analysis modules
- **Line-segment ROI** (`linesegmentroi`): available in the API; not used by
  current kymograph analyses

Coordinates are pixel coordinates on the 2D image array (`row`, `column` /
dim0, dim1). Rectangular stop coordinates are exclusive (numpy-style slicing).

See `acqstore.acq_image.roi` and the [AcqImage API](api/acq-image.md).

## Create a rectangular ROI

With no bounds, the ROI covers the full image:

```python
from acqstore.acq_image import AcqImage
from acqstore.sample_data import ensure_sample_file

acq = AcqImage(str(ensure_sample_file('kymograph-diameter')))

roi = acq.rois.create_rect_roi(name='full', note='full-frame ROI')
print(roi.roi_id, roi.bounds)
```

## Set explicit bounds

```python
from acqstore.acq_image.roi import RectRoiBounds

bounds = RectRoiBounds(
    dim0_start=100,   # row start (inclusive)
    dim0_stop=2000,   # row stop (exclusive)
    dim1_start=40,    # column start
    dim1_stop=120,    # column stop
)
roi = acq.rois.create_rect_roi(bounds=bounds, name='vessel')
print(roi.roi_id, roi.bounds)
```

## Edit bounds

```python
acq.rois.edit_rect_roi(
    roi.roi_id,
    bounds=RectRoiBounds(
        dim0_start=100,
        dim0_stop=1800,
        dim1_start=50,
        dim1_stop=110,
    ),
    name='vessel-cropped',
)
```

## List and select ROIs

```python
print(acq.rois.get_roi_ids())
roi_id = acq.get_default_roi()  # first / default when present
channel = acq.get_default_channel()
```

## Line ROIs

```python
from acqstore.acq_image.roi import LineEndpoints

line = acq.rois.create_line_roi(
    endpoints=LineEndpoints(row0=10, col0=10, row1=10, col1=200),
    name='scan-line',
)
```

Current velocity, diameter, sum-intensity, and heart-rate analyses expect a
**rectangular** ROI. Prefer `create_rect_roi` for analysis workflows.

## Persist ROIs

```python
acq.save()  # writes ROIs into <source>.json
```

## Next

- [Analysis](analysis/index.md)
- [Kymograph velocity](analysis/kymograph/velocity-analysis.md)
