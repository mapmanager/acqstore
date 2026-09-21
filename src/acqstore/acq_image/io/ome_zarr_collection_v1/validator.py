"""Validate AcqStore OME-Zarr Collection v1 additive resources.

JSON Schema validates each document while this module enforces resource
existence, identity, uniqueness, and relationships spanning documents. It does
not replace independent OME-NGFF image validation.
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib.resources import files
from pathlib import Path
from typing import Any

try:
    from jsonschema import Draft202012Validator, FormatChecker
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install the 'jsonschema' package to run this validator.") from exc


class ConformanceError(ValueError):
    """Raised when an AcqStore v1 collection violates its public contract."""


def get_schema_path() -> Path:
    """Return the canonical packaged v1 schema path.

    Returns:
        Filesystem path to the packaged Draft 2020-12 schema.
    """
    resource = files('acqstore.acq_image.io.ome_zarr_collection_v1.schema').joinpath(
        'acqstore-ome-zarr-collection-v1.schema.json'
    )
    return Path(str(resource))


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object.

    Args:
        path: JSON file path.

    Returns:
        Parsed JSON object.

    Raises:
        ConformanceError: If the file cannot be read as a JSON object.
    """
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConformanceError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConformanceError(f"Expected a JSON object: {path}")
    return value


def require_unique(values: list[str], label: str) -> None:
    """Require unique identifiers within one contract scope.

    Args:
        values: Identifiers from one scope.
        label: Identifier category used in error messages.

    Returns:
        None.

    Raises:
        ConformanceError: If any identifier is duplicated.
    """
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        raise ConformanceError(f"Duplicate {label}: {', '.join(sorted(duplicates))}")


def resolve_existing(root: Path, relative: str, label: str) -> Path:
    """Resolve and require one collection-root-relative resource.

    Args:
        root: Collection root.
        relative: Schema-validated relative path.
        label: Resource category used in error messages.

    Returns:
        Existing resource path.

    Raises:
        ConformanceError: If the resource does not exist.
    """
    path = root / relative
    if not path.exists():
        raise ConformanceError(f"Missing {label}: {relative}")
    return path


def schema_validate(validator: Draft202012Validator, value: Any, label: str) -> None:
    """Validate one JSON value with useful resource diagnostics.

    Args:
        validator: Configured Draft 2020-12 validator.
        value: JSON value to validate.
        label: Resource label used in error messages.

    Returns:
        None.

    Raises:
        ConformanceError: If schema validation fails.
    """
    errors = sorted(validator.iter_errors(value), key=lambda error: list(error.absolute_path))
    if errors:
        error = errors[0]
        location = "/".join(str(part) for part in error.absolute_path) or "<root>"
        raise ConformanceError(f"Schema error in {label} at {location}: {error.message}")


def validate_collection(root: Path, schema_path: Path) -> None:
    """Validate one complete AcqStore v1 additive resource graph.

    Args:
        root: Collection root containing ``acqstore/collection.json``.
        schema_path: Canonical bundled schema path.

    Returns:
        None.

    Raises:
        ConformanceError: If any schema or cross-document rule fails.
    """
    schema = load_json(schema_path)
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())

    manifest_path = root / "acqstore" / "collection.json"
    collection = load_json(manifest_path)
    schema_validate(validator, collection, "acqstore/collection.json")

    members = collection["members"]
    require_unique([member["id"] for member in members], "member IDs")

    collection_tables = collection.get("resources", {}).get("tables", [])
    require_unique([resource["id"] for resource in collection_tables], "collection table resource IDs")
    for resource in collection_tables:
        resolve_existing(root, resource["path"], "collection CSV resource")

    for member in members:
        image_id = member["id"]
        resolve_existing(root, member["ome_zarr"], f"primary OME-Zarr path for {image_id}")

        acqimage_path = resolve_existing(root, member["resources"]["acqimage"], "acqimage.json")
        acqimage = load_json(acqimage_path)
        schema_validate(validator, acqimage, str(acqimage_path.relative_to(root)))
        if acqimage["format"] != "acqstore-acqimage":
            raise ConformanceError(f"Expected acqimage document: {acqimage_path.relative_to(root)}")
        if acqimage["image_id"] != image_id:
            raise ConformanceError(f"acqimage image_id does not match member {image_id}")

        rois = acqimage["rois"]
        roi_ids = [roi["id"] for roi in rois]
        require_unique(roi_ids, f"ROI IDs for {image_id}")
        roi_id_set = set(roi_ids)
        for roi in rois:
            if roi["type"] == "rectangle":
                start, stop = roi["start"], roi["stop"]
                if not (stop[0] > start[0] and stop[1] > start[1]):
                    raise ConformanceError(
                        f"Rectangle ROI {roi['id']} stop must be greater than start on both axes"
                    )

        analyses_relative = member["resources"].get("analyses")
        if analyses_relative is not None:
            analyses_path = resolve_existing(root, analyses_relative, "analyses.json")
            analyses_document = load_json(analyses_path)
            schema_validate(validator, analyses_document, str(analyses_path.relative_to(root)))
            if analyses_document["format"] != "acqstore-analyses":
                raise ConformanceError(f"Expected analyses document: {analyses_path.relative_to(root)}")
            if analyses_document["image_id"] != image_id:
                raise ConformanceError(f"analyses image_id does not match member {image_id}")
            analyses = analyses_document["analyses"]
            require_unique([analysis["id"] for analysis in analyses], f"analysis IDs for {image_id}")
            for analysis in analyses:
                roi_id = analysis.get("roi_id")
                if roi_id is not None and roi_id not in roi_id_set:
                    raise ConformanceError(
                        f"Analysis {analysis['id']} refers to unknown ROI {roi_id}"
                    )
                resources = analysis.get("resources", [])
                require_unique(
                    [resource["id"] for resource in resources],
                    f"resource IDs for analysis {analysis['id']}",
                )
                for resource in resources:
                    resolve_existing(root, resource["path"], "analysis CSV resource")

        reference_link = member.get("reference_image")
        if reference_link is not None:
            resolve_existing(root, reference_link["ome_zarr"], f"reference OME-Zarr path for {image_id}")
            reference_path = resolve_existing(root, reference_link["metadata"], "reference-image.json")
            reference = load_json(reference_path)
            schema_validate(validator, reference, str(reference_path.relative_to(root)))
            if reference["format"] != "acqstore-reference-image":
                raise ConformanceError(f"Expected reference-image document: {reference_path.relative_to(root)}")
            if reference["image_id"] != image_id:
                raise ConformanceError(f"reference-image image_id does not match member {image_id}")


def main() -> int:
    """Run the validator command-line interface.

    Returns:
        Process exit status: zero for valid and one for invalid.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection_root", type=Path)
    parser.add_argument(
        "--schema",
        type=Path,
        default=get_schema_path(),
    )
    args = parser.parse_args()
    try:
        validate_collection(args.collection_root.resolve(), args.schema.resolve())
    except ConformanceError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    print("VALID: AcqStore OME-Zarr Collection v1 additive resources")
    print("NOTE: OME-NGFF image conformance must be validated separately.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
