# Sample data

AcqStore downloads reusable sample datasets from the
[cloudscope-data](https://github.com/mapmanager/cloudscope-data) repository.
The live catalog is:

```text
https://raw.githubusercontent.com/mapmanager/cloudscope-data/main/catalog.json
```

Use `acqstore.sample_data` so docs, notebooks, and scripts share the same
download/cache path.

## List available samples

```python
from acqstore.sample_data import list_samples

for sample in list_samples():
    kind = 'file' if sample.is_single_file else 'folder'
    print(f'{sample.name} ({kind}) - {sample.label}')
```

## Download and load a single-file sample

Some catalog entries hold one representative recording. For those,
`ensure_sample_file(id)` returns the path to open with `AcqImage`:

```python
from acqstore.acq_image import AcqImage
from acqstore.sample_data import ensure_sample_file

acq = AcqImage(str(ensure_sample_file('kymograph-flow')))
print(acq.name, acq.path)
```

Current single-file IDs:

- `kymograph-flow`: one OIR line-scan kymograph (velocity / heart rate demos)
- `kymograph-diameter`: one TIFF line-scan kymograph (diameter / sum intensity demos)

`ensure_sample_file` raises `SampleDataError` for a folder sample. Use
`ensure_sample` + `AcqImageList` for those. The returned path is usually a
regular file (`.oir`, `.tif`, …). It can also be a directory-backed store such
as `*.ome.zarr`, which `AcqImage` opens like any other supported path.

## Download and load a folder sample

`ensure_sample(id)` downloads (if needed), extracts, and returns a **folder**
path suitable for `AcqImageList`:

```python
from acqstore.acq_image import AcqImageList
from acqstore.sample_data import ensure_sample

folder = ensure_sample('velocity-sample-data')
acq_list = AcqImageList(str(folder))
print(len(acq_list), 'files')
```

Current folder IDs:

- `velocity-sample-data`: OIR line-scan kymographs for velocity / heart rate
- `diameter-sample-data`: TIFF line-scan kymographs for diameter / sum intensity

Use folder samples for batch notebooks and multi-file workflows.

## Cache location

By default, archives land under the platform user-data directory for
`acqstore` / `sample-data` (on macOS typically
`~/Library/Application Support/acqstore/sample-data`).

Override the root with the environment variable `CLOUDSCOPE_SAMPLE_DATA_DIR`.

## Next

- [Loading an image](loading.md)
- [Notebooks](notebooks.md)
