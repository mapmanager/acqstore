# AcqStore OME-Zarr Collection v1

Status: initial normative specification draft.

The canonical machine-readable schema is published at
[`https://mapmanager.github.io/acqstore/schemas/acqstore-ome-zarr-collection-v1.schema.json`](https://mapmanager.github.io/acqstore/schemas/acqstore-ome-zarr-collection-v1.schema.json).

## Python export

```python
from acqstore.acq_image.acq_image_list import AcqImageList
from acqstore.acq_image.io.ome_zarr_collection_v1.exporter import (
    AcqStoreOmeZarrCollectionExporter,
)

images = AcqImageList(
    "/path/to/acquisitions",
    load_images=True,
    load_analysis_csv=True,
)
output = AcqStoreOmeZarrCollectionExporter(
    "/path/to/collection.ome.zarr",
    overwrite=False,
).export(images)
```

The exporter is intentionally separate from `AcqImage` and `AcqImageList`.
It consumes their existing public Python APIs without adding format-specific
methods to those domain classes.

The key words **MUST**, **MUST NOT**, **SHOULD**, **SHOULD NOT**, and **MAY** are to be interpreted as described by RFC 2119 and RFC 8174 when shown in capitals.

## 1. Scope and authority boundary

AcqStore OME-Zarr Collection v1 is an additive discovery and scientific-metadata layer for a collection of independently valid OME-Zarr images. It defines four JSON documents and links to CSV resources.

This specification **does not define, replace, modify, or override OME-Zarr/OME-NGFF image saving semantics**. OME-NGFF remains the sole authority for pixels, arrays, pyramid levels, axes, coordinate transformations, chunks, codecs, and all other OME-NGFF metadata. Every primary and reference image linked by this specification MUST remain a valid, independently openable OME-Zarr image. A generic OME-NGFF reader MAY ignore every AcqStore resource.

The machine-readable schema is `acqstore-ome-zarr-collection-v1.schema.json` and uses JSON Schema Draft 2020-12.

## 2. Discovery and paths

The chosen v1 collection entry point is `acqstore/collection.json`, relative to the root of the collection container. This is the only magic path in v1.

All paths stored in the four documents:

- MUST be non-empty, slash-separated relative paths;
- MUST be resolved relative to the collection root, defined as the parent directory of the `acqstore/` directory containing the entry point;
- MUST NOT be absolute paths or URLs;
- MUST NOT contain `.` or `..` path segments, backslashes, query strings, or fragments; and
- MUST point directly to the named resource. Clients MUST NOT infer an AcqStore resource from an OME-Zarr path, an ID, a directory name, or another resource path.

The path base is the same for every AcqStore document, including nested `acqimage.json`, `analyses.json`, and `reference-image.json` documents. Relocation of a complete collection preserves meaning when its relative layout is preserved.

## 3. Identity and consistency

Collection, image/member, ROI, and analysis IDs are non-empty stable opaque strings. IDs carry no ordering, path, type, or display-name semantics. Implementations MUST NOT derive identity from array position, a human-readable name, or a directory name. UUIDs and ULIDs are suitable but not required.

Within their respective scopes, member IDs, ROI IDs, analysis IDs, and resource IDs MUST be unique. A collection member's `id` MUST equal `image_id` in every linked `acqimage.json`, `analyses.json`, and `reference-image.json`. An analysis `roi_id`, when present, MUST identify a ROI in the linked `acqimage.json`. These cross-document requirements require collection-level validation in addition to JSON Schema validation.

`format` and `version` determine compatibility. Producer package versions are provenance only.

## 4. `collection.json`

`collection.json` is the authoritative membership and discovery document.

| Field | Requirement | Meaning |
|---|---|---|
| `format` | required | Constant `acqstore-ome-zarr-collection`. |
| `version` | required | Integer constant `1`. |
| `id` | required | Stable opaque collection ID. |
| `name` | required | Human-readable collection name; not identity. |
| `members` | required | Ordered list of member descriptors; MAY be empty. Order is presentation only. |
| `created` | optional | RFC 3339 date-time for this collection representation. |
| `producer` | optional | Object with required `name` and optional `version`; provenance only. |
| `resources` | optional | Collection resources. In v1 it may contain `tables`. |
| `metadata` | optional | Untyped JSON object. |

Each `members[]` object has:

| Field | Requirement | Meaning |
|---|---|---|
| `id` | required | Stable opaque image ID. |
| `name` | required | Human-readable member name. |
| `ome_zarr` | required | Explicit path to an independently valid primary OME-Zarr image. |
| `resources.acqimage` | required | Explicit path to `acqimage.json`. |
| `resources.analyses` | optional | Explicit path to `analyses.json`. Absence means no analysis document is declared. |
| `reference_image` | optional | First-class reference-image link; see section 8. |
| `summary` | optional | Untyped, denormalized discovery object. Never authoritative for OME-NGFF interpretation. |

Collection-level `resources.tables[]` are generic CSV links. Each has a stable `id`, `media_type` equal to `text/csv`, and an explicit `path`. V1 assigns no domain semantics or column schema to tables such as velocity or sum-intensity pools. Consumers use application knowledge to interpret them.

The AcqStore exporter writes a pool table only when that pool contains at least
one completed analysis result. Its exported pool tables include
`acq_image_id`, `roi_id`, and `channel` columns that explicitly link a row to
the opaque Collection v1 identities. These columns are an AcqStore application
convention, not a generic Collection v1 requirement.

## 5. `acqimage.json`

| Field | Requirement | Meaning |
|---|---|---|
| `format` | required | Constant `acqstore-acqimage`. |
| `version` | required | Integer constant `1`. |
| `image_id` | required | ID of the owning collection member. |
| `accepted` | required | AcqStore acceptance state. |
| `rois` | required | ROI list; MAY be empty. |
| `experiment_metadata` | optional | Untyped JSON object. |
| `image_metadata` | optional | Untyped JSON object. |
| `axis_display` | optional | Sparse AcqStore display interpretations; see section 7. |

The JSON shape is a public interchange contract. It is not required to equal any Python class serialization such as `AcqImage.to_dict()`.

## 6. ROI semantics

All v1 ROI coordinates are `[x, y]` in `primary-image-full-resolution-pixels`. Coordinates are zero-based non-negative integer pixel indices. They are independent of pyramid level and MUST be interpreted against the primary image's full-resolution OME-NGFF array.

Every ROI requires stable opaque `id`, `type`, and `coordinate_space`. `name` and an untyped `metadata` object are optional.

- A `point` ROI has one `position` pixel index.
- A `line` ROI has `start` and `stop` endpoint pixel indices. Both endpoints identify pixels; this draft does not specify subpixel geometry.
- A `rectangle` ROI has inclusive `start` and exclusive `stop` coordinates. It covers `start.x <= x < stop.x` and `start.y <= y < stop.y`. Therefore each stop component MUST be greater than its corresponding start component; this relational constraint is enforced by collection-level validation.

V1 defines only point, line, and axis-aligned rectangle ROIs.

## 7. `axis_display`

`axis_display` is optional and sparse. Each property name MUST match an existing OME-NGFF axis name. Only an axis whose AcqStore scientific display interpretation differs from its stored OME-NGFF interpretation SHOULD appear.

Each override requires `type`, `unit`, and positive `scale`. `scale` is expressed in the declared display unit per full-resolution pixel along that axis. For example:

```json
"axis_display": {
  "y": {
    "type": "time",
    "unit": "second",
    "scale": 0.002
  }
}
```

This tells an AcqStore-aware viewer to display the stored `y` axis as time. It does not edit, supersede, or reinterpret OME-NGFF metadata for generic clients. Ordinary images SHOULD omit `axis_display`; unchanged axes MUST NOT be repeated merely for completeness.

## 8. Reference images and `reference-image.json`

A reference image is optional for a collection member. When present, it is first-class and MUST be stored as a fully formed, independently valid OME-Zarr image. It has no AcqStore analysis document or experimental metadata.

The member's `reference_image` object requires:

- `ome_zarr`: explicit path to the reference OME-Zarr image; and
- `metadata`: explicit path to its separate `reference-image.json`.

`reference-image.json` requires `format` = `acqstore-reference-image`, `version` = `1`, and the owning primary member's `image_id`. It MAY contain an untyped descriptive `metadata` object and a `scan_path`.

`scan_path.coordinate_space` is the constant `reference-image-full-resolution-pixels`. `scan_path.points` contains at least two `[x, y]` zero-based non-negative integer pixel indices in traversal order. The points refer to the full-resolution reference image, never a pyramid level. Absence of `scan_path` means no scan path is declared; no separate boolean or point count is used.

## 9. `analyses.json`

| Field | Requirement | Meaning |
|---|---|---|
| `format` | required | Constant `acqstore-analyses`. |
| `version` | required | Integer constant `1`. |
| `image_id` | required | ID of the owning collection member. |
| `analyses` | required | Analysis-instance list; MAY be empty. |

Each analysis requires a stable opaque `id` and a non-empty application-defined `type`. It MAY include `roi_id`, a zero-based integer `channel`, untyped JSON objects named `parameters` and `summary`, and one or more CSV `resources`. Summary-only analyses omit `resources`; this directly reflects AcqStore's public analysis model, in which summary and table outputs are independent.

Each analysis resource requires a stable `id`, `media_type` equal to `text/csv`, and an explicit relative `path`. JSON holds the typed analysis envelope and links; CSV holds tabular results. V1 does not standardize analysis type names, CSV columns, parameters, or summary contents.

## 10. Conformance

A conforming collection MUST satisfy all of the following:

1. Each JSON document validates against the Draft 2020-12 schema.
2. Every declared relative path resolves and has the declared resource kind.
3. Every primary and reference `ome_zarr` path resolves to an independently conforming OME-NGFF image.
4. IDs are unique in scope and all cross-document ID references match as defined above.
5. ROI relational constraints, including exclusive rectangle stops, hold.

Unknown properties are rejected in typed v1 structures. Extensible scientific or producer-specific data belongs only in the explicitly untyped objects.

## 11. Deferred from v1

The following are intentionally out of scope:

- changes to OME-NGFF pixels, pyramids, axes, transformations, chunks, codecs, or saving behavior;
- standardized schemas for experiment metadata, image metadata, analysis parameters, analysis summaries, collection/member summaries, or generic metadata objects;
- standardized analysis type names and CSV column schemas, including velocity and sum-intensity pools;
- remote resource URLs, inferred resource paths, directory crawling, and content-addressed identity;
- ROI types beyond point, line, and axis-aligned rectangle, subpixel coordinates, and physical-coordinate ROIs;
- an OME-NGFF Collections/RFC-8 dependency or compatibility claim; and
- exporter and client implementation details.

These may be added only by a future compatible extension or specification version. They do not block v1 export and consumption.
