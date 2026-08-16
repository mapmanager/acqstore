# AcqStore NWB Support — Final Developer Roadmap

## Status

This roadmap replaces all prior NWB implementation plans.

The implementation target is:

> **AcqStore must load a stock NWB file that has never been written by AcqStore.**
>
> `AcqImage("file.nwb")` must work when the file contains exactly one supported image acquisition.
>
> AcqStore-specific JSON and analysis tables are optional round-trip metadata layered on top of standard NWB, not prerequisites for reading NWB.

The implementation must preserve AcqStore's existing lazy-loading architecture for both pixel data and analysis result tables.

Do not invent a parallel "container import" taxonomy. NWB is an ordinary supported acquisition file format and should participate in the normal file-loader registry, as OME-Zarr does.

---

# 1. Scope

## 1.1 Supported in this implementation

### Stock NWB import

Support standard NWB files containing static images represented with:

```text
Images
└── GrayscaleImage
```

Mapping:

```text
one GrayscaleImage
    -> AcqImage YX

multiple same-shape GrayscaleImage objects in one Images container
    -> AcqImage CYX
```

The C dimension is AcqStore's channel dimension. All channels within one AcqImage must have identical YX shape.

A stock NWB file does **not** need any AcqStore scratch JSON or DynamicTables.

### AcqStore NWB import

If the NWB also contains AcqStore metadata written by `save_nwb()` or `save_nwb_collection()`:

```text
AcqStore JSON
analysis DynamicTables
```

restore them through `NwbPersistence`.

These are optional enhancements to the standard NWB image loader.

### Single and multiple acquisitions

A physical NWB file may contain:

```text
one supported Images container
```

or:

```text
multiple independent supported Images containers
```

Each independent `Images` container is one logical AcqImage.

Different logical AcqImages in the same NWB may have different:

```text
YX shape
number of channels
axis calibration
AcqStore metadata
analysis tables
```

Do not stack, resize, pad, or normalize independent AcqImages.

---

## 1.2 Explicitly out of scope

Do not implement these in this pass:

```text
ImageSeries
TwoPhotonSeries
OnePhotonSeries
time dimensions
Z dimensions
segmentation/PlaneSegmentation
RGB/RGBA semantic mapping
BIDS
DANDI upload/download
in-place NWB mutation
atomic update of an existing NWB
renaming load_analysis_csv()
large unrelated persistence refactors
CloudScope source changes
AcqStore Server source changes
```

Unsupported standard NWB image types must fail clearly, not be guessed.

---

# 2. Architectural principles

## 2.1 NWB is a real BaseFileLoader format

Target hierarchy:

```text
BaseFileLoader
├── TiffFileLoader
├── OirFileLoader
├── CziFileLoader
├── Nd2FileLoader
├── OmeZarrFileLoader
└── NwbFileLoader
```

`NwbFileLoader` must be registered in the normal file-loader registry.

There is no `CONTAINER_IMPORT_EXTENSIONS` category.

There is no special supported-extension list for NWB.

After registration:

```python
assert "nwb" in get_registered_import_extensions()
assert "nwb" in get_supported_import_extensions()
assert "nwb" in get_allowed_import_extensions()
```

This allows existing AcqStore consumers, including CloudScope file dialogs and folder discovery, to see `.nwb` through the existing extension APIs.

---

## 2.2 Path alone is enough to construct the loader

The earlier concern that a path is insufficient was incorrect.

A path is sufficient because `NwbFileLoader(path)` can open the NWB metadata and enumerate supported static image acquisitions without loading image pixels.

Constructor contract:

```python
NwbFileLoader(
    path: str,
    *,
    member_id: str | None = None,
)
```

Behavior:

```text
member_id is None
    enumerate supported NWB image acquisitions

    0 supported acquisitions
        -> clear ValueError

    1 supported acquisition
        -> select it

    >1 supported acquisitions
        -> clear ambiguity ValueError directing caller to
           load_nwb_collection()/AcqImageList.from_nwb()

member_id provided
    select that exact supported acquisition
    -> used by load_nwb_collection()
```

The loader constructor may read small HDF5/NWB metadata.

It must not materialize pixel arrays.

It must not materialize analysis DynamicTables.

---

## 2.3 Standard NWB image data and AcqStore persistence are separate concerns

Pixel loading:

```text
NwbFileLoader
```

AcqStore-specific persisted state:

```text
NwbPersistence
```

For a stock NWB:

```text
NwbFileLoader
    -> standard NWB image pixels

NwbPersistence
    -> no AcqStore JSON
    -> no AcqStore analysis tables
    -> this is valid
```

For an AcqStore-exported NWB:

```text
NwbFileLoader
    -> standard NWB image pixels

NwbPersistence
    -> optional AcqStore JSON
    -> optional AcqStore analysis DynamicTables
```

Missing AcqStore metadata must never make an otherwise supported stock NWB fail to load.

---

# 3. Public API

## 3.1 Ordinary constructor

These must work:

```python
img = AcqImage("stock-single.nwb")
```

and:

```python
img = AcqImage(
    "stock-single.nwb",
    load_images=False,
    load_analysis_csv=False,
)
```

`AcqImage(...)` keeps its existing default semantics.

If its current defaults are eager for pixels, NWB must behave exactly the same:

```python
AcqImage("single.nwb")
```

must be eager unless the caller explicitly requests lazy loading.

Do not silently change constructor defaults for NWB.

---

## 3.2 Explicit NWB helpers

Keep:

```python
from acqstore.nwb_io import load_nwb, save_nwb

img = load_nwb("single.nwb")
save_nwb(img, "export.nwb")
```

and convenience APIs:

```python
img = AcqImage.from_nwb("single.nwb")
img.save_as_nwb("export.nwb")
```

`load_nwb()` / `AcqImage.from_nwb()` may keep their existing lazy defaults if those are already the implemented/documented defaults.

That means these two APIs may intentionally differ:

```python
AcqImage("file.nwb")
    -> normal AcqImage constructor defaults

load_nwb("file.nwb")
    -> NWB helper defaults
```

Test this explicitly.

---

## 3.3 Collection helpers

Keep:

```python
from acqstore.nwb_io import load_nwb_collection, save_nwb_collection

images = load_nwb_collection("collection.nwb")
save_nwb_collection(images, "export.nwb")
```

and:

```python
images = AcqImageList.from_nwb("collection.nwb")
images.save_as_nwb("export.nwb")
```

Do not silently reduce a multi-acquisition NWB to its first acquisition.

---

# 4. Standard NWB discovery

Create one canonical standard-NWB discovery helper.

Recommended internal model:

```python
@dataclass(frozen=True)
class NwbImageMember:
    """Describe one supported static image acquisition in an NWB file."""

    member_id: str
    container_name: str
    channel_names: tuple[str, ...]
    shape_yx: tuple[int, int]
    dtype: str
    display_name: str
```

Additional fields may be added only when the source demonstrates they are required.

Do not place pixel NumPy arrays in this structure.

Recommended function:

```python
def inspect_nwb_image_members(path: str) -> tuple[NwbImageMember, ...]:
    """Inspect supported static image acquisitions without loading pixels.

    Args:
        path: Path to an NWB file.

    Returns:
        Supported image members discovered in the file.

    Raises:
        ImportError: If the optional NWB dependency is unavailable.
        ValueError: If the file cannot be opened as NWB.
    """
```

This helper must understand **standard NWB**, not AcqStore-specific manifests.

AcqStore manifests may later enrich a discovered member, but they must not define whether the image is visible.

---

# 5. `NwbFileLoader`

File:

```text
src/acqstore/acq_image/file_loaders/nwb_file_loader.py
```

Implement as a normal `BaseFileLoader` subclass.

Recommended constructor:

```python
class NwbFileLoader(BaseFileLoader):
    """Lazy loader for one static image acquisition in an NWB file."""

    def __init__(
        self,
        path: str,
        *,
        member_id: str | None = None,
    ) -> None:
        ...
```

Constructor sequence:

```text
1. require PyNWB lazily
2. inspect supported standard NWB image members
3. resolve member
4. store path/member metadata only
5. construct ImageHeader from available standard metadata
6. leave pixels unloaded
```

### Resolution logic

```python
if member_id is None:
    if len(members) == 0:
        raise ValueError(
            "NWB file contains no AcqStore-supported static image acquisitions. "
            "Supported generic NWB image type in this version: "
            "Images containing GrayscaleImage objects."
        )

    if len(members) > 1:
        raise ValueError(
            "NWB file contains multiple supported image acquisitions; "
            "use load_nwb_collection() or AcqImageList.from_nwb()."
        )

    selected = members[0]
else:
    selected = find exact member
    if not found:
        raise ValueError(...)
```

### Pixel load

The existing `BaseFileLoader` interface is source of truth.

Implement the exact required method(s) used by other file loaders.

When pixels are requested:

```text
open NWB
locate selected Images container
read selected GrayscaleImage channel data
convert NWB XY -> AcqStore YX if required by current static-image storage
stack channels on axis 0 for CYX
close NWB
return AcqStore-compatible pixel array
```

Mappings:

```text
one channel:
    YX

multiple channels:
    CYX
```

Validate that all channel images in one `Images` container have identical shape.

If not:

```python
raise ValueError(
    "NWB Images container contains channels with different image shapes; "
    "AcqStore CYX requires equal YX shape for all channels."
)
```

Do not retain a permanently open NWB/HDF5 handle unless the existing BaseFileLoader architecture already establishes a safe lifetime mechanism.

Prefer open/read/close per lazy load for v1.

---

# 6. Loader registration

File:

```text
src/acqstore/acq_image/file_loaders/loader_registry.py
```

Register NWB normally:

```python
_FILE_LOADER_FACTORIES = {
    ...
    "nwb": NwbFileLoader,
}
```

Use the same callable/class convention the existing registry already uses.

Do not add:

```text
CONTAINER_IMPORT_EXTENSIONS
get_container_import_extensions()
special NWB supported lists
```

Do not add an early `.nwb` branch in `AcqImage.__init__()` merely to bypass the loader factory.

Normal flow should be:

```text
AcqImage("file.nwb")
    -> create_file_loader(path)
    -> loader registry
    -> NwbFileLoader(path)
```

This is the desired architecture.

---

# 7. Supported extension APIs

Inspect and preserve current source behavior in:

```text
src/acqstore/acq_image/supported_import_extensions.py
```

The expected result after normal registration is:

```python
"nwb" in get_registered_import_extensions()
"nwb" in get_supported_import_extensions()
"nwb" in get_allowed_import_extensions()
```

Do not create a separate extension pathway unless the actual current source proves one is required.

CloudScope and AcqStore Server should pick `.nwb` up through their existing AcqStore calls.

No consumer-specific `.nwb` hard-coding in this work.

---

# 8. Persistence backend

Source:

```text
src/acqstore/acq_image/persistence.py
```

NWB must not use `FileSidecarPersistence`.

A stock NWB must never cause AcqStore to look for:

```text
file.nwb.json
file.nwb.velocity.csv
...
```

Implement/retain:

```python
class NwbPersistence(...):
    ...
```

The persistence backend needs to know the selected logical NWB member.

Recommended source of member identity:

```text
NwbFileLoader.member_id
```

Do not re-discover the member independently if the loader already selected it.

The exact way `AcqImage._initialize()` passes loader information into persistence must be based on the current source.

Preferred design:

```python
create_persistence_backend(
    path,
    *,
    file_loader=None,
)
```

or equivalent minimal change so the backend can inspect:

```python
isinstance(file_loader, NwbFileLoader)
file_loader.member_id
```

Do not add suffix-only member guessing.

### Stock NWB semantics

If no AcqStore JSON exists:

```text
load persisted state
    -> no-op

load analysis tables
    -> no-op
```

This is not an error.

ROIs remain empty/default according to normal AcqImage initialization.

AcqStore analyses remain absent unless there is standard-to-AcqStore analysis mapping explicitly implemented later.

### AcqStore NWB semantics

If our embedded JSON exists for the selected member:

```text
restore:
    image metadata
    ROIs
    analysis summaries/configuration
    peaks stored in JSON
    any other current sidecar payload
```

If DynamicTables exist:

```text
restore analysis result DataFrames lazily
```

Do not duplicate ROIs into NWB tables.

---

# 9. Lazy analysis tables

Existing public API remains:

```python
img.load_analysis_csv()
img.unload_analysis_csv()
img.analysis_csv_loaded
```

Do not rename it in this pass.

For NwbPersistence:

```python
load_analysis_tables()
```

must:

```text
open NWB
locate selected member's AcqStore analysis DynamicTables
convert only those tables to DataFrames
close NWB
return tables keyed exactly as AcqAnalysisSet expects
```

For stock NWB with no AcqStore tables:

```text
return no tables
```

Then preserve the existing `AcqAnalysisSet` loaded-state contract.

Do not casually change `results_csv_loaded()` semantics.

Relevant files/tests:

```text
src/acqstore/acq_image/acq_analysis_set.py
tests/acqstore/test_results_csv_loaded.py
```

If a test fails, classify it:

```text
real API bug
stale test assumption
incomplete test fixture
intentional API change
```

Do not change production semantics only to preserve a stale test.

---

# 10. Logical identity

One NWB path may contain many logical AcqImages.

Therefore:

```text
path
    physical NWB path

file_id
    unique logical AcqImage identity
```

Recommended:

```python
file_id = f"{path}#{member_id}"
```

For ordinary formats:

```python
file_id == path
```

remains unchanged.

Each NWB member should also have a useful display name.

Recommended default when no AcqStore display name exists:

```python
display_name = member.container_name
```

or another stable standard-NWB acquisition name.

Do not show 500 identical physical NWB filenames in UI/tree rows.

AcqStore-exported NWB may preserve the original AcqImage display name in optional AcqStore metadata.

---

# 11. Multi-acquisition NWB

`NwbFileLoader(path)` without member ID is intentionally strict.

For:

```text
one supported acquisition
```

it works.

For:

```text
multiple supported acquisitions
```

it raises ambiguity.

That means:

```python
AcqImage("multi.nwb")
```

raises clearly.

`load_nwb_collection()` must:

```text
1. inspect standard NWB members
2. create one AcqImage per member
3. construct NwbFileLoader(path, member_id=...)
4. construct NwbPersistence(path, member_id=...)
5. preserve unique file_id
6. preserve display name
7. leave pixels lazy by default
8. leave analysis tables lazy by default
```

Do not require an AcqStore collection manifest to enumerate stock NWB acquisitions.

If an AcqStore collection manifest exists, use it only to enrich ordering/names/round-trip metadata, not to determine whether stock NWB can be opened.

---

# 12. `load_nwb()`

`load_nwb(path)` should use the same standard discovery logic as `NwbFileLoader`.

Do not have a separate AcqStore-manifest-only read path.

Recommended:

```python
def load_nwb(
    path: str,
    *,
    load_images: bool = False,
    load_analysis_csv: bool = False,
) -> AcqImage:
    """Load exactly one supported image acquisition from an NWB file."""
```

Implementation should construct an AcqImage using:

```text
NwbFileLoader(path)
NwbPersistence(...)
```

through the same internal initialization path used by ordinary AcqImage construction.

Avoid duplicating AcqImage hydration logic.

If the current architecture makes reuse difficult, extract a small internal helper rather than maintaining two nearly identical initialization paths.

---

# 13. `load_nwb_collection()`

Recommended:

```python
def load_nwb_collection(
    path: str,
    *,
    load_images: bool = False,
    load_analysis_csv: bool = False,
) -> AcqImageList:
    """Load all supported static image acquisitions from an NWB file."""
```

Open once for metadata discovery if practical.

Do not read pixel datasets during collection creation unless `load_images=True`.

Do not materialize DynamicTables unless `load_analysis_csv=True`.

With 500 members and lazy defaults, loading the collection must remain lightweight.

---

# 14. NWB export

Export remains explicit:

```python
save_nwb(...)
save_nwb_collection(...)

AcqImage.save_as_nwb(...)
AcqImageList.save_as_nwb(...)
```

The NWB written by AcqStore should be:

```text
valid standard NWB image representation
+
optional AcqStore round-trip JSON
+
optional AcqStore analysis DynamicTables
```

A third-party NWB reader should still see ordinary standard static images.

AcqStore should gain richer round-trip behavior from its optional metadata.

---

# 15. `AcqImage.save()` boundary

Existing AcqStore `save()` is explicit, not automatic.

Do not describe it as implicit.

For an NWB-backed AcqImage:

```python
img.save()
```

must be rejected.

Recommended error:

```text
NWB-backed AcqImages do not support in-place persistence.
Use save_nwb() or save_nwb_collection() to export a new NWB file.
```

Reason:

```text
AcqImage.save()
```

is designed around existing native/sidecar persistence and must not create:

```text
file.nwb.json
file.nwb.velocity.csv
```

or attempt structural HDF5 mutation.

README must state:

> NWB support provides load/import into mutable in-memory AcqImage objects and explicit export to a new NWB file. In-place mutation of an existing NWB file is not implemented.

---

# 16. Export completeness with lazy source data

When exporting an AcqImage, the export must contain complete data even if the source object is currently lazy.

For each source AcqImage:

```text
remember:
    images_loaded before export
    analysis_csv_loaded before export

if pixels needed:
    load images

if persisted analysis tables needed:
    load analysis tables

write complete NWB representation

restore original lazy state:
    unload pixels if they were initially unloaded
    unload analysis tables if they were initially unloaded
```

This is a hard correctness requirement.

Do not export an incomplete NWB merely because result DataFrames were not resident at the time `save_nwb()` was called.

---

# 17. Bounded-memory collection export

A 500-member AcqImageList must not accumulate every image and table in RAM.

Use the already agreed first strategy:

```text
create NWB file

for each AcqImage:
    materialize that member only
    append/write member to NWB
    release PyNWB graph for that member
    restore member's previous lazy state

finalize collection metadata
```

Use PyNWB/HDF5 `r+` append/write semantics first.

Memory target:

```text
approximately one AcqImage
+ its analysis DataFrames
+ normal HDF5/PyNWB overhead
```

Do not introduce `DataChunkIterator`/`H5DataIO` until profiling shows repeated `r+` writes are too slow or memory behavior is insufficient.

This optimization decision is deferred, not forgotten.

---

# 18. Optional PyNWB dependency

PyNWB must remain optional.

`pyproject.toml`:

```toml
[project.optional-dependencies]
nwb = [
    "pynwb>=<version-range selected from current project compatibility>",
]
```

Do not guess a version range if the current local project already has one. Inspect the source and retain/adjust intentionally.

Install:

```bash
uv sync --extra nwb
```

Core AcqStore import must work without PyNWB.

Opening `.nwb` or calling NWB APIs without the extra must raise a clear error such as:

```text
NWB support requires the optional 'nwb' dependencies.
Install with: uv sync --extra nwb
```

Do not import PyNWB from a module imported unconditionally by core AcqStore if that would make the optional dependency mandatory.

Lazy-import PyNWB inside NWB-specific execution paths as required.

---

# 19. Static image axis mapping

AcqStore uses:

```text
YX
CYX
```

For the existing AcqStore NWB export, verify the actual PyNWB `GrayscaleImage` axis semantics against the installed PyNWB version and the synthetic validated prototype.

Do not infer or silently transpose without a test.

The current prototype used:

```python
nwb_xy = acqstore_yx.T
```

and reversed on load.

Before final implementation, make this behavior explicit with a non-square synthetic image:

```text
YX = 30000 x 100
```

so an accidental transpose cannot pass unnoticed.

Required round-trip assertion:

```python
assert restored.shape == original.shape
assert np.array_equal(restored, original)
```

Use different values along Y/X if needed to prove orientation, not merely shape.

---

# 20. Standard NWB metadata mapping

Do not guess arbitrary standard NWB fields into AcqStore metadata.

For v1:

- populate `ImageHeader` from standard fields only when mapping is unambiguous;
- otherwise leave the corresponding AcqStore field at its normal unknown/default value.

Document each mapping in code.

Kymographs remain ordinary YX/CYX AcqImages in AcqStore. If AcqStore-exported metadata says:

```text
Y unit = s
X unit = um
```

restore that through optional AcqStore JSON.

For a generic stock NWB, do not infer "kymograph" from shape.

---

# 21. Subject/session metadata for export

Use the existing structured metadata API:

```python
NwbMetadata
NwbSubjectMetadata
```

`NwbMetadata` should be optional for local export.

Safe defaults:

```text
identifier
    generated unique value

session_start_time
    current save-time datetime in America/New_York

session_description
    non-biological generic AcqStore export description
```

Do not fabricate:

```text
subject_id
species
sex
age
```

If `subject` is omitted, local NWB export remains valid if PyNWB allows it.

DANDI-required biological metadata will be addressed later at DANDI packaging time or supplied explicitly by the caller.

---

# 22. Files to inspect/edit

Use the current repository as source of truth.

Likely production files:

```text
pyproject.toml

src/acqstore/nwb_io.py
src/acqstore/nwb/README.md

src/acqstore/acq_image/acq_image.py
src/acqstore/acq_image/acq_analysis_set.py
src/acqstore/acq_image/persistence.py

src/acqstore/acq_image/file_loaders/nwb_file_loader.py
src/acqstore/acq_image/file_loaders/loader_registry.py
src/acqstore/acq_image/file_loaders/file_loader_factory.py
src/acqstore/acq_image/supported_import_extensions.py

src/acqstore/acq_image_list/acq_image_list.py
```

The exact AcqImageList path must be verified from the current source before editing.

Do not edit every listed file automatically.

Inspect first and edit only what the implementation actually requires.

Do not edit `__init__.py` files unless a concrete technical requirement is found.

The project intentionally keeps them thin/empty.

---

# 23. Test fixtures and `__new__`

Existing tests may create AcqImage objects with:

```python
AcqImage.__new__(AcqImage)
```

and manually assign private fields.

If production initialization introduces required internal fields such as:

```text
_file_id
_display_name
_persistence
```

test doubles that bypass initialization must initialize them explicitly.

Do not add `getattr()` fallbacks to production properties solely to make incomplete `__new__` fixtures pass.

Classify such failures as stale/incomplete test fixtures and repair the fixtures.

---

# 24. Tests — stock NWB is acceptance test #1

Create a stock NWB test fixture using only PyNWB standard structures.

It must contain **no AcqStore scratch JSON and no AcqStore DynamicTables**.

Example:

```text
NWBFile
└── acquisition/
    └── Images("images")
        └── GrayscaleImage("channel_0")
```

Then verify:

```python
img = AcqImage(stock_nwb_path)
```

works.

This test must exist before AcqStore-round-trip NWB tests are considered sufficient.

---

# 25. Required loader tests

## 25.1 Registration

```python
assert "nwb" in get_registered_import_extensions()
assert "nwb" in get_supported_import_extensions()
assert "nwb" in get_allowed_import_extensions()
```

## 25.2 `create_file_loader()`

Single static acquisition:

```python
loader = create_file_loader(single_stock_nwb)
assert isinstance(loader, NwbFileLoader)
```

Multiple static acquisitions:

```python
with pytest.raises(ValueError, match="multiple supported image acquisitions"):
    create_file_loader(multi_stock_nwb)
```

## 25.3 Missing optional dependency

Test the project's established optional-dependency error pattern if feasible without making the test environment brittle.

At minimum unit-test the dependency guard.

---

# 26. Required stock NWB tests

## One YX image

```python
img = AcqImage(stock_yx_nwb)
assert img.shape == expected_shape
assert pixel equality/orientation
```

Test both constructor defaults and explicit lazy flags.

## CYX

Create:

```text
Images
├── channel_0
└── channel_1
```

same YX shape.

Verify:

```text
AcqImage shape = CYX
channel order stable
pixel values exact
```

## Different-size channels in one Images container

Must raise a clear AcqStore compatibility error.

## Unsupported NWB image type

Create or use a minimal valid unsupported standard image acquisition.

Verify error clearly states:

```text
valid NWB
but no currently supported static image acquisition
```

and identifies supported v1 type(s).

Do not say "not an AcqStore NWB".

---

# 27. Required lazy-loading tests

For:

```python
img = AcqImage(
    stock_nwb,
    load_images=False,
    load_analysis_csv=False,
)
```

verify:

```python
assert img.images_loaded is False
```

Then:

```python
img.load_images()
assert img.images_loaded is True

img.unload_images()
assert img.images_loaded is False
```

For an AcqStore-exported NWB containing analysis tables:

```python
assert img.analysis_csv_loaded is False

img.load_analysis_csv()
assert img.analysis_csv_loaded is True

img.unload_analysis_csv()
assert img.analysis_csv_loaded is False
```

Loading one collection member must leave all other members' pixels/tables unloaded.

---

# 28. Required persistence tests

## Stock NWB

Verify no AcqStore JSON is required.

Verify loading does not search for or require:

```text
file.nwb.json
*.csv sidecars
```

## AcqStore NWB

Verify existing JSON round-trip restores:

```text
ROIs
header/acquisition metadata represented by current sidecar payload
analysis summary/configuration
peak information
```

Verify analysis DynamicTables round-trip to DataFrames.

Do not add an NWB ROI table.

---

# 29. Required identity tests

For multiple acquisitions in one physical NWB:

```python
images = load_nwb_collection(path)
```

verify:

```python
assert images[0].path == images[1].path
assert images[0].file_id != images[1].file_id
assert images[0].name != images[1].name
```

Verify:

```text
AcqImageList._files_by_id
```

contains every member.

---

# 30. Required collection tests

Create a stock NWB with:

```text
member A:
    CYX = 2 x 1024 x 1024

member B:
    YX = 30000 x 100
```

This structure has already been manually validated with `pynwb-validate`.

Test:

```python
AcqImage("multi.nwb")
```

raises ambiguity.

Test:

```python
images = load_nwb_collection("multi.nwb")
```

returns two AcqImages with correct independent shapes.

Lazy defaults:

```python
all(not img.images_loaded for img in images)
```

unless explicit eager flags are passed.

Load one member:

```python
images[0].load_images()
```

and verify:

```python
images[0].images_loaded is True
images[1].images_loaded is False
```

---

# 31. Required save/export tests

## Single AcqImage

Start with source AcqImage lazy.

Export:

```python
save_nwb(img, path)
```

Verify:

```text
complete pixels present
complete analysis DynamicTables present when applicable
source object's original loaded/unloaded state restored
```

Validate:

```bash
uv run --extra nwb pynwb-validate file.nwb
```

Reload with:

```python
AcqImage(path)
```

and verify round trip.

## Collection

Start with multiple lazy AcqImages.

Export:

```python
save_nwb_collection(images, path)
```

Verify all members are present and independently shaped.

Verify original lazy states restored.

Add instrumentation/unit hooks if necessary to prove the exporter does not intentionally retain all source arrays in a Python collection before final write.

Do not claim bounded memory solely from code inspection; perform at least a practical test with multiple large synthetic arrays.

---

# 32. `results_csv_loaded()` tests

Do not redesign this API without evidence.

Relevant:

```text
tests/acqstore/test_results_csv_loaded.py
```

Preserve expected semantics unless the implementation exposes a genuine mismatch.

If tests encode filesystem existence rather than runtime loaded state, inspect callers before changing anything.

Tests may be updated when the API contract is intentionally corrected.

Do not preserve a known-bad API merely for test compatibility.

---

# 33. Full regression tests

After focused tests:

```bash
uv sync --extra nwb
```

Run focused:

```bash
uv run pytest tests/acqstore/nwb/ -q
```

Then relevant loader/persistence tests.

Then mandatory full suite:

```bash
uv run pytest tests/
```

Every failure must be classified:

```text
production regression
stale fixture
intentional API change
unrelated/environmental failure
```

Do not stop after NWB tests pass.

---

# 34. README

Keep self-contained NWB documentation at:

```text
src/acqstore/nwb/README.md
```

Do not edit the repository root README for this work.

README must include:

## Install

```bash
uv sync --extra nwb
```

## Load stock NWB

```python
from acqstore import AcqImage

img = AcqImage("stock.nwb")
```

Use the project's actual public import path for `AcqImage`; verify before documenting.

## Lazy load

```python
img = AcqImage(
    "stock.nwb",
    load_images=False,
    load_analysis_csv=False,
)

img.load_images()
img.unload_images()
```

## Explicit helper

```python
from acqstore.nwb_io import load_nwb

img = load_nwb("stock.nwb")
```

## Collection

```python
from acqstore.nwb_io import load_nwb_collection

images = load_nwb_collection("collection.nwb")
```

## Export

```python
img.save_as_nwb("export.nwb")
```

and module form.

## Validation

```bash
uv run --extra nwb pynwb-validate export.nwb
```

## Boundary

Clearly state:

```text
Stock NWB loading is supported for the documented standard image types.
AcqStore metadata is optional.
NWB-backed AcqImage objects are mutable in memory.
In-place persistence back into an existing NWB is not implemented.
Use explicit NWB export to create a new file.
```

## Current supported generic NWB structures

Document exactly:

```text
Images / GrayscaleImage
YX
CYX
```

No claims beyond tested support.

---

# 35. Code style

All new/edited Python code must have:

```text
Google-style module docstrings
Google-style class docstrings
Google-style function/method docstrings
typed parameters
typed return values
Raises sections where applicable
inline # comments for non-obvious logic
```

Keep code:

```text
KISS
DRY
small
testable
lazy by design
```

Do not add abstractions merely to anticipate future NWB types.

---

# 36. Implementation sequence

Follow this exact order.

## Phase A — generic stock NWB loader

1. Inspect BaseFileLoader API.
2. Implement standard NWB member discovery.
3. Implement `NwbFileLoader(path, member_id=None)`.
4. Register `"nwb"` normally.
5. Make `create_file_loader(single_stock_nwb)` work.
6. Make `AcqImage(single_stock_nwb)` work.
7. Add YX/CYX stock tests.
8. Add ambiguity/unsupported-type tests.
9. Verify lazy pixels.

Do not proceed until stock NWB passes.

## Phase B — NWB persistence

1. Ensure `.nwb` selects `NwbPersistence`.
2. Stock NWB without AcqStore metadata must remain valid.
3. Restore optional AcqStore JSON when present.
4. Restore optional analysis DynamicTables lazily.
5. Verify no `.nwb.json`/CSV sidecar behavior leaks in.

## Phase C — collections

1. Enumerate standard NWB members.
2. Build one `NwbFileLoader(path, member_id=...)` per member.
3. Build matching NwbPersistence.
4. Unique file IDs.
5. Display names.
6. Lazy member isolation.
7. Different-shape collection tests.

## Phase D — export

1. Single standard-NWB export.
2. Optional AcqStore JSON.
3. DynamicTables.
4. Lazy source materialization/restoration.
5. Multi-member bounded-memory export.
6. `pynwb-validate`.

## Phase E — regression/docs

1. Full tests.
2. Repair stale test fixtures only when justified.
3. Complete NWB README.
4. Verify supported extensions expose `nwb`.
5. Verify CloudScope picks it up without source changes.

---

# 37. Definition of done

The feature is not done until all of these work.

## Stock NWB

```python
img = AcqImage("stock-never-touched-by-acqstore.nwb")
```

No AcqStore manifest required.

## Lazy stock NWB

```python
img = AcqImage(
    "stock.nwb",
    load_images=False,
    load_analysis_csv=False,
)

assert img.images_loaded is False

img.load_images()
assert img.images_loaded is True
```

## AcqStore round trip

```python
img.save_as_nwb("acqstore-export.nwb")

restored = AcqImage("acqstore-export.nwb")
```

Pixels, JSON state, ROIs, analysis metadata and result tables round-trip according to current AcqStore APIs.

## Collection

```python
images = load_nwb_collection("multi.nwb")
```

Different independent shapes/channels work and remain lazy.

## Validation

```bash
uv run --extra nwb pynwb-validate acqstore-export.nwb
```

reports no schema errors.

## Regression

```bash
uv run pytest tests/
```

passes, except for explicitly documented unrelated/environmental failures.

---

# 38. Final non-negotiable review checklist

Before merging:

- [ ] Stock NWB with no AcqStore metadata loads.
- [ ] `NwbFileLoader` is a real registered `BaseFileLoader`.
- [ ] No container-extension taxonomy exists.
- [ ] `.nwb` appears through normal registered/supported/allowed APIs.
- [ ] Pixel loading is lazy when requested.
- [ ] Analysis tables are lazy.
- [ ] Stock NWB does not look for sidecars.
- [ ] AcqStore metadata is optional.
- [ ] Multiple acquisitions are not silently collapsed.
- [ ] Collection members have unique logical IDs.
- [ ] Collection members retain useful display names.
- [ ] Different independent shapes work.
- [ ] `AcqImage.save()` rejects NWB-backed in-place persistence.
- [ ] Explicit NWB export is complete even from lazy source objects.
- [ ] Collection export does not intentionally accumulate all source arrays/tables.
- [ ] No `__init__.py` edits without a demonstrated requirement.
- [ ] `results_csv_loaded()` was not changed casually.
- [ ] `__new__` test fixtures are repaired rather than weakening production invariants.
- [ ] All code has Google-style docstrings and type annotations.
- [ ] NWB README is self-contained.
- [ ] `pynwb-validate` passes.
- [ ] Full pytest suite was run.
