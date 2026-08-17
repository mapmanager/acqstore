# AcqStore Multi-Image OME-Zarr Roadmap

## Goal

Add an AcqStore writer that creates one OME-Zarr store containing multiple unrelated `AcqImage` images, each with its own shape, dimensionality, metadata, multiscale pyramid, and AcqStore analysis files.

This roadmap covers **AcqStore creation only**. DANDI/BIL upload and browser verification are later validation steps.

---

## Target Dataset Contract

One dataset:

```text
dataset.ome.zarr/
├── acqimagelist.json
├── velocity.csv
├── sum_intensity.csv
├── <other_top_level_table>.csv
│
├── image_000/
│   ├── acqimage.json
│   ├── analysis/
│   │   ├── <analysis_1>.csv
│   │   └── <analysis_2>.csv
│   ├── 0/
│   ├── 1/
│   └── ...
│
├── image_001/
│   ├── acqimage.json
│   ├── analysis/
│   │   └── ...
│   ├── 0/
│   ├── 1/
│   └── ...
│
└── image_002/
    ├── acqimage.json
    ├── analysis/
    │   └── ...
    ├── 0/
    ├── 1/
    └── ...
```

### Required properties

- One top-level `.ome.zarr` store.
- Zero or more unrelated `AcqImage` entries.
- Each `AcqImage` is stored as an independent internal OME-Zarr image.
- Images may have different shapes and dimensionalities, e.g. `YX`, `CYX`, `ZYX`, `ZCYX`.
- Each image has its own multiscale pyramid.
- Each internal image must remain directly addressable as an OME-Zarr image for stock viewers such as Neuroglancer.
- Each image has a literal `acqimage.json`.
- Each image may have zero or more analysis CSV files.
- The root has a literal `acqimagelist.json`.
- The root may have AcqImageList-level CSV tables, including `velocity.csv` and `sum_intensity.csv`.

---

## Phase 1 — Freeze the Writer Contract

Define one explicit internal schema before implementation.

### Stable image identity

Every `AcqImage` must have a stable ID used as its group name:

```text
image_000
image_001
image_002
```

`acqimagelist.json` must map AcqStore identity to the internal OME-Zarr image path.

Example:

```json
{
  "format": "acqstore-multi-image-ome-zarr",
  "version": "1",
  "images": [
    {
      "id": "image_000",
      "path": "image_000",
      "metadata": "image_000/acqimage.json"
    }
  ],
  "tables": {
    "velocity": "velocity.csv",
    "sum_intensity": "sum_intensity.csv"
  }
}
```

### Per-image JSON

`image_NNN/acqimage.json` should contain the normal serialized AcqImage metadata needed by AcqStore, plus references to any analysis CSV files.

Do not duplicate pixel data or pyramid metadata in this JSON unless AcqStore already requires those fields.

### Analysis CSVs

Use literal CSV files for now.

`acqimage.json` is authoritative for which per-image analysis CSVs exist and what they mean.

---

## Phase 2 — Add a New Public Export API

Do not change existing `save()`, `save_native_zarr()`, or single-image `save_as_ome_zarr()` behavior.

Add an additive collection API, for example:

```python
AcqImageList.save_as_ome_zarr(
    path,
    *,
    overwrite=False,
    multiscale=True,
)
```

or, if separation is preferred:

```python
from acqstore.ome_zarr_io import save_ome_zarr_collection

save_ome_zarr_collection(
    acq_image_list,
    path,
    overwrite=False,
)
```

The public API must accept one `AcqImageList` and produce the complete store atomically enough that a failed export does not silently appear valid.

---

## Phase 3 — Reuse the Existing Single-Image OME-Zarr Writer

Do not implement image serialization twice.

Refactor only as necessary so the existing OME-Zarr writer can target an internal group:

```text
dataset.ome.zarr/image_000/
dataset.ome.zarr/image_001/
dataset.ome.zarr/image_002/
```

For each `AcqImage`:

1. determine source axes and shape;
2. create its image group;
3. write full-resolution pixels;
4. generate its multiscale pyramid;
5. write valid OME-Zarr metadata for that image;
6. verify the internal group can be opened independently.

Different images must not be forced into one common shape, axis set, chunk layout, or pyramid depth.

---

## Phase 4 — Write AcqImage Sidecars

After each image is successfully written:

```text
image_NNN/acqimage.json
```

Then write:

```text
image_NNN/analysis/*.csv
```

Rules:

- JSON must serialize deterministically.
- CSV filenames must be stable.
- `acqimage.json` must reference relative CSV paths.
- No analysis CSV is required when an image has no analysis.
- Failure to write required AcqStore metadata should fail the export rather than leave a partially valid dataset.

---

## Phase 5 — Write AcqImageList Sidecars

At the root write:

```text
acqimagelist.json
velocity.csv
sum_intensity.csv
```

and any other agreed AcqImageList-level CSVs.

`acqimagelist.json` must provide enough information for AcqStore to enumerate the internal images without directory crawling.

It should contain:

- format/version;
- ordered image IDs;
- relative internal image paths;
- relative `acqimage.json` paths;
- known top-level table paths;
- only dataset-level metadata that belongs to the `AcqImageList`.

---

## Phase 6 — Implement Read-Back Tests Before DANDI Testing

Create one synthetic fixture containing at least three deliberately heterogeneous images, for example:

```text
image_000: YX
image_001: CYX
image_002: ZCYX
```

Use different dimensions and pyramid depths.

Include:

- `acqimagelist.json`;
- one `acqimage.json` per image;
- 2–3 per-image analysis CSVs across the dataset;
- 2–3 root CSVs including `velocity.csv` and `sum_intensity.csv`.

### Required tests

1. Export completes.
2. Store layout exactly matches the contract.
3. Each internal image opens independently with the OME-Zarr reader.
4. Shapes, axes, dtype, scale, and pyramid levels round-trip correctly.
5. `acqimagelist.json` round-trips exactly.
6. Every `acqimage.json` round-trips exactly.
7. Every CSV round-trips exactly.
8. AcqStore can reconstruct the `AcqImageList` using `acqimagelist.json`, not directory discovery.
9. AcqStore can reconstruct each `AcqImage` using its own `acqimage.json`.
10. Per-image analyses are found from `acqimage.json`.
11. Root tables are found from their agreed conventional names or root manifest.
12. Export with heterogeneous image shapes never attempts to concatenate images.
13. Existing single-image OME-Zarr tests remain unchanged and pass.

---

## Phase 7 — Add Validation Utilities

Add one validation helper for AcqStore's own contract:

```python
validate_acqstore_ome_zarr(path)
```

It should verify:

- root `acqimagelist.json` exists and parses;
- every listed image path exists;
- every listed image is a valid readable OME-Zarr image;
- every listed `acqimage.json` exists and parses;
- referenced analysis CSVs exist;
- expected root CSVs exist when declared;
- IDs and paths are unique;
- no manifest reference escapes the dataset root.

OME-Zarr validation and AcqStore-contract validation should remain separate concerns.

---

## Phase 8 — Produce the Synthetic Artifact

Add a small developer script, for example:

```text
scripts/ome_zarr/make_synthetic_multi_image.py
```

It should reproducibly generate:

```text
synthetic-acqstore.ome.zarr/
```

with the three heterogeneous images plus JSON and CSV sidecars.

This exact artifact becomes the test object for the later DANDI Sandbox experiment.

---

## Acceptance Criteria

Implementation is complete when:

- AcqStore creates one `.ome.zarr` store containing at least three unrelated images with different n-dimensional shapes.
- Every internal image has its own valid multiscale OME-Zarr representation.
- Every internal image can be addressed and opened independently.
- `acqimagelist.json` and every `acqimage.json` are present and correct.
- Per-image analysis CSVs and root CSVs are present and correct.
- AcqStore can read the exported dataset back using its own manifests/conventions.
- Existing AcqStore save/export behavior is not broken.
- The synthetic store is ready, without modification, for the subsequent DANDI Sandbox upload/preservation test.

## Immediate Next Action

Implement the **dataset contract and synthetic writer test first**, then adapt the existing single-image OME-Zarr writer to write into named internal groups. Do not begin DANDI integration until the local artifact passes all read-back and OME-Zarr validation tests.
