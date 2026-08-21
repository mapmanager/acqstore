# AcqStore OME-Zarr export contract

AcqStore writes native OME-NGFF image data plus a small set of AcqStore-owned JSON resources. The OME-NGFF specification remains authoritative for image groups, multiscales, arrays, chunks, codecs, and `zarr.json`. AcqStore's JSON Schema covers only the additive resources described here.

## Saved collection layout

```text
collection.ome.zarr/
├── acqstore/
│   ├── acq_image_collection.json
│   └── analysis_tables/
└── acq_images/
    └── acq_image_000/
        ├── zarr.json
        ├── 0/
        ├── reference/                  # optional independent OME-Zarr image
        └── acqstore/
            ├── manifest.json
            ├── acq_image.json
            └── analysis/
```

The schema is a development and validation resource in the AcqStore package. It is not copied into exported OME-Zarr stores.

## Contract definitions

The canonical schema is `acqstore_ome_zarr_contract.schema.json` in `acqstore.acq_image.io.export_schema`. It uses JSON Schema Draft 2020-12 and currently defines:

- `collectionManifestV1` for `acqstore/acq_image_collection.json`
- `nativeImageManifestV2` for each `acq_images/<id>/acqstore/manifest.json`
- `acqImageSidecarV2` for each `acq_images/<id>/acqstore/acq_image.json`
- shared definitions for paths, image summaries, ROIs, analysis resources, reference metadata, and scan paths

This first schema intentionally leaves experiment metadata, image-header metadata, image contrast entries, analysis summaries, and detection parameters open. Their containing fields and JSON value kinds are checked, but their format-specific keys are not yet frozen.

## Reference images and scan paths

A collection entry reports reference availability through `summary.has_reference_image`. When true, it also declares `reference_image_path`. The corresponding native image manifest declares its member-relative `reference_image` group, and the sidecar contains `reference_image_metadata`.

Reference metadata describes channel count and physical calibration. Scan geometry uses:

- `has_scan_path`
- `scan_path_num_points`
- `scan_path_x_pixels`
- `scan_path_y_pixels`

The X and Y arrays use full-resolution reference-image pixel coordinates. A scan path may contain two points for a line segment or more points for a scanner trajectory. When no path exists, the point count is zero and both coordinate arrays are empty.

## Consumer lookup sequence

A static consumer such as CloudScope Web resolves one selected image by joining three AcqStore-owned documents:

1. Read `acqstore/acq_image_collection.json` and select an `acq_images` entry.
2. Resolve that entry's `manifest_path` and `sidecar_path` relative to the collection root.
3. Read analysis resource paths from the native manifest and descriptive analysis values from the matching sidecar entry. The join key is `(analysis_name, channel, roi_id)`.

For a reference image, `summary.has_reference_image` is the availability flag. When it is true, the collection entry's `reference_image_path` and the native manifest's `reference_image` must resolve to the same OME-Zarr image group. Reference calibration and scan-path coordinates come from the sidecar's `reference_image_metadata`.

Consumers should not infer reference-image paths or scan geometry from filenames, native OME-NGFF metadata, or ROI records. A two-point scan path can be displayed as a line segment. The producer also permits trajectories with more than two points, so a consumer limited to line segments must reject those explicitly rather than silently truncating them.

## Validation boundary

JSON Schema checks the structure and local values of one JSON document. Python validation remains responsible for relationships that span resources, including:

- path containment beneath the collection root
- agreement between collection and native reference declarations
- uniqueness of image and analysis identities
- coordinate-array lengths agreeing with `scan_path_num_points`
- OME-NGFF validity and successful pyramid decoding

Exporter contract tests generate real collections and validate their JSON resources against the schema. A deliberate serialized-format change must update the writer, schema, tests, and compatibility documentation together.
