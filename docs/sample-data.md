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
    print(sample.name, '-', sample.label)
```

## Download and load a folder sample

`ensure_sample(id)` downloads (if needed), extracts, and returns a **folder**
path suitable for `AcqImageList`:

```python
from acqstore.acq_image import AcqImageList
from acqstore.sample_data import ensure_sample

folder = ensure_sample('velocity-sample-data')
acq_list = AcqImageList(str(folder))
acq = acq_list.get_files()[0]
print(acq.name, acq.path)
```

Current catalog IDs include:

- `velocity-sample-data`: OIR line-scan kymographs for velocity / heart rate
- `diameter-sample-data`: TIFF line-scan kymographs for diameter / sum intensity

## Cache location

By default, archives land under the platform user-data directory for
`acqstore` / `sample-data` (on macOS typically
`~/Library/Application Support/acqstore/sample-data`).

Override the root with the environment variable `CLOUDSCOPE_SAMPLE_DATA_DIR`.

## Single-file samples

Folder samples are the supported path today. A future additive API for
single-file catalog entries (so docs can use `AcqImage(path)` directly) is
planned separately. Until that lands, prefer `ensure_sample` + `AcqImageList`
as above.

## Next

- [Loading an image](loading.md)
- [Notebooks](notebooks.md)
