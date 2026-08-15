# AcqStore NWB import/export

This folder documents AcqStore's optional local NWB support. The implementation
lives in `src/acqstore/nwb_io.py`; NWB-specific pixel loading lives in
`src/acqstore/acq_image/file_loaders/nwb_file_loader.py`.

## Scope

Current NWB support is intentionally narrow:

- `AcqImage` primary pixels with axes `YX` or `CYX`.
- `AcqImageList` stored in one NWB file as multiple independent image members.
- Collection members may have unrelated YX shapes and channel counts.
- Each AcqImage uses one NWB `Images` acquisition container with one
  `GrayscaleImage` per channel.
- The existing AcqImage JSON payload is embedded in NWB and remains authoritative
  for ROI, metadata, analysis configuration/summary, peak detection, and contrast.
- Tabular analysis results are stored as NWB `DynamicTable` objects.
- Primary pixels and analysis tables are lazy on NWB import by default.
- Reference-image pixels, `ZYX`, and `ZCYX` are not implemented in this version.
  Export rejects reference-image sources rather than silently dropping pixels.

NWB import/export is separate from DANDI. DANDI upload/download is intentionally
not implemented here yet. Before DANDI publication work, re-evaluate whether the
reusable AcqStore JSON should remain in NWB scratch or move to a small NWB
extension/LabMetaData representation; current NWB guidance treats scratch as
non-standard exploratory storage even though PyNWB validates it.

## Optional installation

PyNWB is optional. Normal AcqStore imports do not require it.

From the AcqStore repository:

```bash
uv sync --extra nwb
```

On macOS the same command is used; no platform-specific switch is required.
Run scripts that use NWB with:

```bash
uv run --extra nwb python scripts/nwb/your_script.py
```

Calling an NWB API without the optional dependency installed raises an
`ImportError` with the installation command.

## Canonical module API

```python
from acqstore.nwb_io import load_nwb, save_nwb

img = load_nwb("image.nwb")
save_nwb(img, "copy.nwb")
```

Collection API:

```python
from acqstore.nwb_io import load_nwb_collection, save_nwb_collection

images = load_nwb_collection("dataset.nwb")
save_nwb_collection(images, "copy.nwb")
```

NWB import is lazy by default:

```python
images = load_nwb_collection("dataset.nwb")

img = images.get_files()[0]
assert not img.images_loaded
assert not img.analysis_csv_loaded

img.load_images()          # loads only this NWB member's primary pixels
img.load_analysis_csv()    # loads only this member's DynamicTables

img.unload_images()
img.unload_analysis_csv()
```

`load_analysis_csv()` retains its historical public name. For NWB-backed images
it loads NWB `DynamicTable` data rather than CSV files.

## Convenience API

```python
from acqstore.acq_image import AcqImage
from acqstore.acq_image.acq_image_list import AcqImageList

img = AcqImage.from_nwb("image.nwb")
img.save_as_nwb("copy.nwb")

images = AcqImageList.from_nwb("dataset.nwb")
images.save_as_nwb("copy.nwb")
```

## Metadata

Use `NwbMetadata` and `NwbSubjectMetadata` rather than adding many save-function
parameters:

```python
from acqstore.nwb_io import NwbMetadata, NwbSubjectMetadata, save_nwb

metadata = NwbMetadata(
    session_description="AcqStore acquisition",
    subject=NwbSubjectMetadata(
        subject_id="subject-001",
        species="Mus musculus",
        sex="U",
        age="P90D",
    ),
)

save_nwb(img, "image.nwb", metadata=metadata)
```

When `session_start_time` is omitted, AcqStore uses the current save time in
`America/New_York` (EST/EDT as appropriate). When `identifier` is omitted, a UUID
is generated. Biological subject values are never fabricated; subject metadata
is optional for local NWB export and can be supplied later for DANDI workflows.

## Lazy loading and memory behavior

`load_nwb()` and `load_nwb_collection()` read only the NWB manifest, embedded
AcqImage JSON, and small header metadata by default. They do not read primary
pixel datasets or convert analysis DynamicTables to DataFrames.

Collection export is also designed around AcqStore's lazy model. It writes one
AcqImage at a time using separate PyNWB `r+` transactions:

1. remember whether that member's pixels/tables were loaded;
2. materialize only that member if needed;
3. append that member to the NWB file;
4. close the PyNWB/HDF5 object graph;
5. restore the source member to its prior lazy state;
6. continue with the next member.

This keeps source-memory growth approximately bounded to one AcqImage plus its
analysis tables. If repeated `r+` writes later prove too slow for very large
collections, benchmark before considering `DataChunkIterator`/`H5DataIO`.

## Import/export boundary

NWB support currently implements import and explicit export only.

Runtime mutation of an NWB-backed `AcqImage` is allowed in memory, but
`AcqImage.save()` intentionally raises because in-place mutation of an existing
NWB container is not implemented. Use explicit export instead:

```python
save_nwb(img, "updated.nwb")
# or
save_nwb_collection(images, "updated_dataset.nwb")
```

Do not implement implicit NWB mutation by writing `.json` or `.csv` files next
to the `.nwb` source. A future atomic NWB-update feature, if needed, should be a
separate design.

## Tests

Run NWB-specific tests with the optional dependency enabled:

```bash
uv run --extra nwb pytest tests/acqstore/nwb/ -v
```

Regenerate the lock file after changing the optional dependency declaration:

```bash
uv lock
```

## Validate generated NWB files

PyNWB validation:

```bash
uv run --extra nwb pynwb-validate path/to/file.nwb
```

A successful validation prints `no errors found`.
