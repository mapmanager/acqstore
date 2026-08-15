# AcqStore NWB Import Integration — Developer Roadmap

## Goal

Make `.nwb` a first-class AcqStore import extension while preserving lazy pixels, lazy analysis tables, unique logical member identity, optional PyNWB installation, and the existing explicit-export boundary.

This roadmap is implementation-ready. Do not broaden scope into DANDI, BIDS, Z/T support, `ImageSeries`, in-place NWB mutation, or unrelated API cleanup.

## Current failure

This works:

```python
from acqstore.nwb_io import load_nwb

img = load_nwb("file.nwb")
```

and:

```python
img = AcqImage.from_nwb("file.nwb")
```

but this currently fails:

```python
img = AcqImage("file.nwb")
```

with:

```text
ValueError: Unsupported acquisition file extension 'nwb'
```

CloudScope file choosers driven by AcqStore's supported extensions therefore also omit `.nwb`.

### Root cause

AcqStore import extensions are registry-driven. The source-of-truth files are:

```text
src/acqstore/acq_image/file_loaders/loader_registry.py
src/acqstore/acq_image/file_loaders/file_loader_factory.py
src/acqstore/acq_image/supported_import_extensions.py
```

`loader_registry.py` currently registers TIFF/OIR/CZI/ND2/OME-Zarr variants, but not `nwb`.

## Fixed architectural decisions

### 1. `.nwb` must be registered

After the fix:

```python
assert "nwb" in get_registered_import_extensions()
```

and, under default configuration:

```python
assert "nwb" in get_allowed_import_extensions()
```

must pass.

Do not hard-code `.nwb` separately in CloudScope if CloudScope already consumes AcqStore's extension list.

### 2. Do not register `NwbFileLoader` directly

This is wrong:

```python
_FILE_LOADER_FACTORIES["nwb"] = NwbFileLoader
```

`NwbFileLoader` needs logical member metadata that cannot be derived from a filesystem path alone.

Register a manifest-aware factory instead:

```python
def create_single_member_nwb_file_loader(path: str) -> NwbFileLoader:
    ...
```

The factory must:

1. Require the optional PyNWB dependency.
2. Open the NWB.
3. Read only AcqStore manifest/lightweight metadata.
4. Determine how many logical AcqImages exist.
5. Require exactly one member.
6. Build a configured `NwbFileLoader`.
7. Close the NWB.
8. Never materialize pixel arrays.
9. Never convert DynamicTables to pandas DataFrames.

Recommended contract:

```python
def create_single_member_nwb_file_loader(path: str) -> NwbFileLoader:
    """Create a lazy NWB loader for a single-member AcqStore NWB file.

    Args:
        path: Path to an AcqStore NWB file.

    Returns:
        Configured lazy NWB loader.

    Raises:
        ImportError: If optional NWB dependencies are unavailable.
        ValueError: If the file is not an AcqStore NWB file.
        ValueError: If the file contains zero or multiple AcqImages.
    """
```

Use Google-style docstrings everywhere.

### 3. Single-member versus multi-member behavior

For a single-member NWB, all of these must work:

```python
img = AcqImage("single.nwb")
img = AcqImage.from_nwb("single.nwb")
img = load_nwb("single.nwb")
```

Normal constructor lazy flags must work:

```python
img = AcqImage(
    "single.nwb",
    load_images=False,
    load_analysis_csv=False,
)
```

For a multi-member NWB, this must fail:

```python
AcqImage("collection.nwb")
```

Do not silently select the first member.

Recommended error:

```text
NWB file contains multiple AcqImages; use
AcqImageList.from_nwb(...) or load_nwb_collection(...).
```

Supported collection APIs remain:

```python
images = AcqImageList.from_nwb("collection.nwb")
```

and:

```python
from acqstore.nwb_io import load_nwb_collection

images = load_nwb_collection("collection.nwb")
```

## Required production changes

### A. `nwb_file_loader.py`

Source:

```text
src/acqstore/acq_image/file_loaders/nwb_file_loader.py
```

Add or expose the manifest-aware factory described above.

`NwbFileLoader` remains responsible only for lazy pixel access for one logical NWB member.

### B. `loader_registry.py`

Edit:

```text
src/acqstore/acq_image/file_loaders/loader_registry.py
```

Register:

```python
_FILE_LOADER_FACTORIES["nwb"] = create_single_member_nwb_file_loader
```

Do not register the loader class directly.

### C. `supported_import_extensions.py`

Inspect:

```text
src/acqstore/acq_image/supported_import_extensions.py
```

Verify its defaults derive from `get_registered_import_extensions()`.

Do not guess that registry registration alone is enough. If this file caches or copies defaults, update it correctly.

### D. `file_loader_factory.py`

Preferred outcome: little or no NWB-specific branching.

Keep this architecture:

```python
suffix = normalize_import_extension_for_path(path)
...
return create_registered_file_loader(path, suffix)
```

Update docstrings/examples to mention `.nwb` where appropriate.

## Lazy-loading requirements

### Pixels

Opening:

```python
img = AcqImage(
    "single.nwb",
    load_images=False,
    load_analysis_csv=False,
)
```

must leave:

```python
assert img.images_loaded is False
```

Factory/loader construction must not retain pixel NumPy arrays.

`img.load_images()` must read only that logical NWB member.

`img.unload_images()` must release those arrays using the existing AcqImage/BaseFileLoader lazy contract.

For a collection, loading one member must not load any other member.

### Analysis tables

Do not load NWB DynamicTables during factory or collection construction.

Existing API remains:

```python
img.load_analysis_csv()
img.unload_analysis_csv()
img.analysis_csv_loaded
```

The name is historical; do not rename it in this work.

Use the persistence/backend seam already planned or implemented. Do not add a new:

```python
elif self.path.endswith(".nwb"):
```

inside `AcqImage.load_analysis_csv()`.

Only the selected member's DynamicTables should become DataFrames.

## `results_csv_loaded()` contract

Relevant source:

```text
src/acqstore/acq_image/acq_analysis_set.py
tests/acqstore/test_results_csv_loaded.py
```

Facts from the current source:

- `AcqImage.analysis_csv_loaded` delegates to `AcqAnalysisSet.results_csv_loaded()`.
- `_results_csv_loaded` is already a runtime state flag.
- `load_results_tables_by_name()` sets `_results_csv_loaded = True`.
- unloading result DataFrames resets the loaded state.
- existing tests include traditional CSV-sidecar expectations.

Decision:

- Do not casually redesign this method as part of the `.nwb` extension fix.
- If NWB works with the current state machine, keep behavior unchanged except for clearly incorrect documentation.
- If tests fail, first classify the failure as an API bug versus an outdated test assumption.
- Tests may change when the API intentionally changes; do not preserve a bad API solely for an old test.

## Logical identity

Multiple logical AcqImages may share one physical NWB path.

Required distinction:

```text
path
    /data/experiment.nwb

file_id
    /data/experiment.nwb#acqimage_0037
```

For ordinary formats:

```python
file_id == path
```

remains the default.

For NWB members, `file_id` must be unique so `AcqImageList._files_by_id` does not collapse members.

Also preserve a logical/display name per member so UI/tree rows do not show hundreds of identical `experiment.nwb` names.

Do not add production `getattr()` fallbacks merely to support tests that construct `AcqImage` using `__new__`. Repair those fixtures so they initialize required fields.

## Save/export boundary

NWB remains import + explicit export.

Supported:

```python
save_nwb(img, path)
save_nwb_collection(images, path)

img.save_as_nwb(path)
images.save_as_nwb(path)
```

Not supported for an NWB-backed AcqImage:

```python
img.save()
```

Reason: normal `AcqImage.save()` writes sidecar JSON and CSV state. On an NWB-backed member this could incorrectly create:

```text
collection.nwb.json
collection.nwb.velocity.csv
```

Therefore NWB-backed `img.save()` must raise a clear error directing users to `save_nwb()` or `save_nwb_collection()`.

README boundary:

> NWB support provides import and explicit export. It does not support in-place mutation or persistence back into an existing NWB file. Mutate AcqImage objects in memory and export a new NWB with `save_nwb()` or `save_nwb_collection()`.

## Optional dependency

PyNWB remains optional.

Install:

```bash
uv sync --extra nwb
```

Core AcqStore must import without PyNWB.

Calling NWB APIs or opening `.nwb` without the extra must raise a concise `ImportError` telling the user how to install support.

Validation command:

```bash
uv run --extra nwb pynwb-validate file.nwb
```

## Files to inspect and likely edit

Production:

```text
pyproject.toml

src/acqstore/nwb_io.py
src/acqstore/nwb/README.md

src/acqstore/acq_image/acq_image.py
src/acqstore/acq_image/acq_image_list.py
src/acqstore/acq_image/acq_analysis_set.py
src/acqstore/acq_image/persistence.py

src/acqstore/acq_image/file_loaders/nwb_file_loader.py
src/acqstore/acq_image/file_loaders/loader_registry.py
src/acqstore/acq_image/file_loaders/file_loader_factory.py
src/acqstore/acq_image/supported_import_extensions.py
```

Not every file must change. Inspect first and edit only where required.

Do not edit `__init__.py` files unless a demonstrated technical requirement exists.

Tests:

```text
tests/acqstore/test_file_loader_factory.py
tests/acqstore/test_results_csv_loaded.py

tests/acqstore/nwb/test_nwb_io.py
tests/acqstore/nwb/test_nwb_collection.py
```

Also search the full test tree for:

```text
DEFAULT_IMPORT_EXTENSIONS
get_registered_import_extensions
get_allowed_import_extensions
file_id
name
AcqImage.__new__
AcqImage.save
```

## Required tests

### Extension registration

```python
assert "nwb" in get_registered_import_extensions()
```

and under default configuration:

```python
assert "nwb" in get_allowed_import_extensions()
```

### Factory: single member

Create a valid synthetic single-member AcqStore NWB.

Verify:

```python
loader = create_file_loader(path)
assert isinstance(loader, NwbFileLoader)
```

and:

```python
img = AcqImage(
    path,
    load_images=False,
    load_analysis_csv=False,
)
```

succeeds.

### Factory: multi member

Create a valid two-member NWB.

Verify:

```python
with pytest.raises(ValueError, match="multiple AcqImages"):
    AcqImage(path, load_images=False, load_analysis_csv=False)
```

and:

```python
images = load_nwb_collection(path)
assert len(images) == 2
```

### Lazy pixels

Verify:

```python
assert img.images_loaded is False
img.load_images()
assert img.images_loaded is True
img.unload_images()
assert img.images_loaded is False
```

For collections, loading one member must leave all others unloaded.

### Lazy analysis tables

Verify:

```python
assert img.analysis_csv_loaded is False
img.load_analysis_csv()
assert img.analysis_csv_loaded is True
img.unload_analysis_csv()
assert img.analysis_csv_loaded is False
```

Include a summary-only analysis with no table.

For collections, loading one member's tables must not load another member's tables.

### Identity

For two members sharing one NWB path:

```python
assert img0.path == img1.path
assert img0.file_id != img1.file_id
assert img0.name != img1.name
```

Verify `_files_by_id` retains all members.

### Explicit save rejection

For NWB-backed members:

```python
with pytest.raises(..., match="save_nwb"):
    img.save()
```

Traditional file-backed `save()` behavior must remain unchanged.

### Full regression

Run:

```bash
uv sync --extra nwb
uv run pytest tests/
```

Classify every failure:

```text
real production regression
test fixture bypasses required initialization
test encodes intentionally changed API semantics
unrelated/environmental failure
```

Fix production only for genuine regressions.

## CloudScope acceptance check

After AcqStore tests pass, confirm AcqStore's default extension list includes `"nwb"`.

Then verify CloudScope's pywebview file chooser picks up `.nwb` through the existing AcqStore extension source.

Preferred outcome:

```text
no CloudScope-specific .nwb hard-code
```

If CloudScope has its own hard-coded list, call that out as a separate CloudScope integration issue.

## Non-goals

Do not implement here:

```text
DANDI upload/download
BIDS
Z/T NWB mappings
ImageSeries
in-place NWB mutation
atomic NWB rewrite
DataChunkIterator optimization
CloudScope runtime mutation persistence into existing NWB
renaming load_analysis_csv()
large unrelated persistence refactors
```

## Final review checklist

### Import/discovery

- [ ] `.nwb` appears in registered import extensions.
- [ ] `.nwb` appears in default allowed extensions.
- [ ] `create_file_loader("single.nwb")` works.
- [ ] `AcqImage("single.nwb")` works.
- [ ] multi-member NWB passed to `AcqImage(...)` raises a clear error.
- [ ] `load_nwb_collection()` handles multi-member NWB.

### Lazy behavior

- [ ] Lazy single-member open reads no pixels.
- [ ] Lazy single-member open reads no DynamicTable DataFrames.
- [ ] Collection open does not materialize all member images.
- [ ] Collection open does not materialize all analysis tables.
- [ ] Loading one member leaves others lazy.
- [ ] unload APIs restore lazy state.

### Identity/UI

- [ ] Shared-path members have unique `file_id`.
- [ ] Shared-path members preserve useful display names.
- [ ] `_files_by_id` contains every member.
- [ ] `__new__` test doubles initialize required internal fields.

### Persistence boundary

- [ ] Traditional `AcqImage.save()` behavior is unchanged.
- [ ] NWB-backed `AcqImage.save()` is rejected.
- [ ] Error points to `save_nwb()` / `save_nwb_collection()`.
- [ ] README states export-only/no in-place NWB mutation.

### Optional dependency

- [ ] Core AcqStore imports without PyNWB.
- [ ] Opening `.nwb` without extra gives a clear install instruction.
- [ ] `uv sync --extra nwb` enables support.
- [ ] `pynwb-validate` is documented.

### Regression

- [ ] NWB tests pass.
- [ ] loader registry/factory tests pass.
- [ ] lazy analysis tests pass.
- [ ] full `tests/` suite passes or unrelated failures are documented.
- [ ] CloudScope can see `.nwb` through AcqStore's extension API.

## Definition of done

This workflow succeeds:

```python
save_path = "/tmp/my-image.nwb"

acq_image.save_as_nwb(save_path, overwrite=True)

loaded = AcqImage(
    save_path,
    load_images=False,
    load_analysis_csv=False,
)

assert loaded.images_loaded is False
assert loaded.analysis_csv_loaded is False

loaded.load_images()
loaded.load_analysis_csv()
```

This validates:

```bash
uv run --extra nwb pynwb-validate /tmp/my-image.nwb
uv run pytest tests/
```

And multi-member NWB is intentionally opened through:

```python
images = AcqImageList.from_nwb("collection.nwb")
```

or:

```python
images = load_nwb_collection("collection.nwb")
```

never silently reduced to one AcqImage.
