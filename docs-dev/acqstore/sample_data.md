# AcqStore Sample Data

AcqStore owns reusable sample-data download helpers so scripts and CloudScope can
load the same datasets without GUI-specific code. The `cloudscope-data`
repository is the single source of truth for the catalog and dataset archives.

## Runtime contract

```python
from acqstore.acq_image import AcqImageList
from acqstore.sample_data import ensure_sample, get_sample, list_samples

samples = list_samples()
sample = get_sample(samples[0].name)
folder = ensure_sample(sample.name)
acq_list = AcqImageList(str(folder), path_kind="folder")
```

`list_samples()` returns entries in catalog display order. `get_sample()` returns
one entry by its stable catalog ID. `ensure_sample()` downloads, verifies, and
extracts the corresponding archive, then returns its loadable dataset folder.

## Default storage location

Sample data is stored under:

```text
platformdirs.user_data_dir("acqstore") / "sample-data"
```

On macOS this is usually:

```text
~/Library/Application Support/acqstore/sample-data
```

Deployments can override this with:

```bash
export CLOUDSCOPE_SAMPLE_DATA_DIR=/data/sample-data
```

For Docker or cloud deployments, mount that directory to persistent storage so
archives and extracted samples are reused.

## Catalog and archive contract

AcqStore downloads and caches:

```text
https://raw.githubusercontent.com/mapmanager/cloudscope-data/main/catalog.json
```

Each catalog entry provides:

- `id`: stable sample identifier and expected top-level extracted directory
- `label`: user-facing display label
- `description`: short user-facing description
- `url`: immutable release ZIP URL
- `sha256`: expected archive SHA-256 digest

The catalog list order is the client display order. Each ZIP must contain a
loadable top-level directory whose name exactly matches the entry's `id`.

## Architecture notes

- `cloudscope-data` owns the catalog and downloadable dataset archives.
- `acqstore.sample_data` owns catalog retrieval, local caching, hash validation,
  ZIP extraction, and the returned filesystem path.
- `acqstore.sample_data` must not import `cloudscope`, `nicegui`, or
  `nicewidgets`.
- CloudScope consumes only the public AcqStore sample-data API.
- `AcqImageList` remains path-based and does not know whether a path came from a
  native picker, mounted storage, or downloaded sample data.
