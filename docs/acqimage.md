# AcqImage

An `AcqImage` is one acquisition-backed microscopy recording plus its sidecar
state. It combines:

- **Image pixels** (optional until loaded; see lazy loading below)
- **Image header metadata** (shape, dims, dtype, calibration)
- **Experimental metadata** (user-editable sample / condition fields)
- **ROIs**
- **Analyses** (summaries and optional CSV tables)

See the [AcqImage API](api/acq-image.md).

## Anatomy

```text
AcqImage
├── path / name                 # source file
├── images (file loader)        # pixels + ImageHeader
├── image header metadata       # editable Y/X physical units
├── experimental metadata       # user fields (species, notes, ...)
├── rois (RoiSet)               # rect / line ROIs
├── analysis_set                # analyses keyed by (name, channel, roi_id)
└── sidecar files               # <file>.json + analysis CSVs
```

## Lazy loading

Pixels and analysis CSV tables can stay unloaded until needed. That is a key
AcqStore feature for large multi-file datasets. See
[Loading an image](loading.md#lazy-loading) for details and
`load_images` / `load_analysis_csv` / `load_lazy_data`.

## Load and inspect

```python
from acqstore.acq_image import AcqImage
from acqstore.sample_data import ensure_sample_file

acq = AcqImage(str(ensure_sample_file('kymograph-flow')))

print(acq.name)
print(acq.images.header.dims, acq.images.header.shape)
print(acq.get_image_physical_units())  # (step_y, step_x)
```

## Physical units (critical for analysis)

Quantitative analysis (velocity in physical units, diameter in µm, etc.) depends
on correct **Y/X physical pixel spacing**. If calibration is missing or wrong,
analysis numbers will be wrong even when the algorithm runs successfully.

Header fields (editable):

| Field | Meaning |
|-------|---------|
| `physical_unit_y` | Physical size of one pixel along Y (rows). For kymographs, Y is typically time. |
| `physical_unit_x` | Physical size of one pixel along X (columns). For kymographs, X is distance along the line. |
| `physical_label_y` | Unit label for Y (for example `s`, `ms`, or `Pixels`) |
| `physical_label_x` | Unit label for X (for example `um` or `Pixels`) |

Read current spacing:

```python
step_y, step_x = acq.get_image_physical_units()
print(step_y, step_x)
```

Inspect and set calibration through the header metadata section. **Always set
units and labels together:**

```python
header = acq.get_metadata_section('acq_image_header')
print(header.get_values())

header.update_values({
    'physical_unit_y': 0.002,   # example: seconds per row
    'physical_unit_x': 0.2,     # example: micrometers per column
    'physical_label_y': 's',
    'physical_label_x': 'um',
})

acq.save()  # persist calibration into the JSON sidecar
```

!!! warning "Wrong scale means wrong science"
    Always verify `physical_unit_x` / `physical_unit_y` (and their labels) before
    comparing analysis results across files or publishing numbers.

## Image header metadata

Full header field table (including read-only structural fields and editable
calibration):

--8<-- "schemas/header_metadata.md"

## Experimental metadata

User-editable fields describing the biological sample and condition. Saved in
the JSON sidecar.

--8<-- "schemas/experimental_metadata.md"

```python
exp = acq.get_metadata_section('experiment_metadata')
exp.update_values({'species': 'mouse', 'notes': 'demo'})
acq.save()
```

## Pixels and ROIs

- Pixel access goes through the loader / `AcqPixels` after images are loaded.
- ROIs define the region used by analysis. See [ROIs](rois.md).
- Analysis results are keyed by `(analysis_name, channel, roi_id)`.

## Sidecar files

For a source file `my_file.tif`, AcqStore may write:

```text
my_file.tif
my_file.tif.json
my_file.tif.radon_velocity.csv
my_file.tif.diameter.csv
my_file.tif.sum_intensity.csv
```

The JSON sidecar stores accept/reject state, experimental metadata, image header
calibration, ROIs, detection parameters, and analysis summaries. CSV files store
tabular outputs for analyses that export tables.

Reloading an `AcqImage` from the same path restores sidecar state when present.

## Current limitations

AcqStore can load the supported formats above. Current *quantitative analysis*
modules are designed for **line scan kymographs**, not general 2D segmentation
or tracking.

## Next

- [ROIs](rois.md)
- [Analysis](analysis/index.md)
