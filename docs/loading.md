# Loading an image

A core AcqStore feature is **file loading** across proprietary and open
microscopy formats. You load one recording with `AcqImage`, or discover many
files with `AcqImageList`.

## Lazy loading

AcqStore uses **lazy loading** for image pixels and for potentially large
analysis CSV tables. An `AcqImage` can open a file, read header metadata, and
restore ROIs and analysis summaries without materializing full pixel arrays or
CSV result tables. That lets you browse large folders of many files without
hitting memory limits. Load pixels or CSV data only when a script, notebook, or
analysis needs them (`load_images`, `load_analysis_csv`, or `load_lazy_data`).

## File loaders

Each supported extension has its own loader under
`acqstore.acq_image.file_loaders`. Loaders share a common base in
`base_file_loader.py`. Concrete loaders include, for example:

- `oir_file_loader.py` for `.oir`
- `czi_file_loader.py` for `.czi`
- `nd2_file_loader.py` for `.nd2`
- `tiff_file_loader.py` for `.tif` / `.tiff`
- `ome_zarr_file_loader.py` for `.ome.zarr` and related stores

`AcqImage` picks the loader from the file extension through a registry. An
unsupported extension raises `ValueError` at construction.

## Supported formats

Commercial microscopy formats:

| Extension | Vendor |
|-----------|--------|
| `.oir` | Olympus / Evident |
| `.czi` | Zeiss |
| `.nd2` | Nikon |

Open formats:

| Extension | Notes |
|-----------|--------|
| `.tif` / `.tiff` | Including Olympus header enrichment when present |
| `.ome.zarr` | OME-Zarr stores (also `.ome.zarr.zip`) |

### Native `.cs.ome.zarr` (in development)

AcqStore also defines a native **`.cs.ome.zarr`** (and `.cs.ome.zarr.zip`)
variant. It is **in development**. The goal is to store AcqStore metadata, ROIs,
and fully featured analysis together with the image pixels. Plain OME-Zarr does
not provide that for this workflow. Prefer the documented formats (`.oir`,
`.czi`, `.nd2`, `.tif`, `.ome.zarr`) for production scripts until the native
format is stabilized.

## Load one file with `AcqImage`

```python
from acqstore.acq_image import AcqImage

acq = AcqImage('/path/to/recording.oir')
print(acq.name)
print(acq.images.header.dims, acq.images.header.shape)
```

The constructor:

1. Opens the file through the matching loader
2. Reads image-header / calibration when available
3. Hydrates sidecar state from `<source>.json` when that file exists
4. Optionally loads primary pixels and analysis CSV tables (see lazy loading above)

To open metadata without pixels or analysis CSVs:

```python
acq = AcqImage('/path/to/recording.oir', load_images=False, load_analysis_csv=False)
# later, when needed:
acq.load_lazy_data(load_images=True, load_analysis_csv=True)
```

## Discover many files with `AcqImageList`

```python
from acqstore.acq_image import AcqImageList

# Folder of acquisitions (recurse with folder_depth)
acq_list = AcqImageList('/path/to/folder', folder_depth=2)

# Or a single file path
acq_list = AcqImageList('/path/to/file.tif')

for acq in acq_list.get_files():
    print(acq.name)
```

## Load from sample data

Until single-file catalog samples exist, use a folder sample and take the first
file:

```python
from acqstore.acq_image import AcqImageList
from acqstore.sample_data import ensure_sample

folder = ensure_sample('velocity-sample-data')
acq = AcqImageList(str(folder)).get_files()[0]
```

See [Sample data](sample-data.md).

## Next

- [AcqImage](acqimage.md): what an acquisition object contains
- [ROIs](rois.md)
