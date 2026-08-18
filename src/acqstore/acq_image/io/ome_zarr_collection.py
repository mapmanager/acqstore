"""Export an :class:`AcqImageList` as a multi-image native OME-Zarr store.

This module owns export orchestration only. Each collection member remains an
independent AcqStore-native OME-Zarr image and can be opened directly at its
manifest-declared child path. Collection loading is intentionally out of scope.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from acqstore.acq_image.acq_image import AcqImage
    from acqstore.acq_image.acq_image_list import AcqImageList


COLLECTION_FORMAT = "acqstore-multi-image-ome-zarr"
COLLECTION_FORMAT_VERSION = 1
COLLECTION_ZARR_FORMAT = 3


def write_acq_image_native_ome_zarr(
    acq_image: AcqImage,
    destination: str | Path,
) -> Path:
    """Write one native AcqStore OME-Zarr image without changing dirty state.

    This delegates to the established native directory writer but deliberately
    avoids :meth:`AcqImage.save_native_zarr`, whose save-like public behavior
    marks the source object clean after writing.

    Args:
        acq_image: Loaded acquisition image to export.
        destination: New local directory-store path.

    Returns:
        Resolved output path.

    Raises:
        FileExistsError: If the destination already exists.
        ValueError: If the image is not fully loaded or cannot be represented
            as OME-Zarr.
    """
    if not acq_image.images_loaded:
        raise ValueError(f"AcqImage pixels are not loaded: {acq_image.name}")
    if not acq_image.analysis_csv_loaded:
        raise ValueError(f"AcqImage analysis tables are not loaded: {acq_image.name}")

    dest = Path(destination).expanduser().resolve(strict=False)
    acq_image._save_native_zarr_directory(
        dest,
        overwrite=False,
        zarr_format=COLLECTION_ZARR_FORMAT,
    )
    return dest


def export_acq_image_list_ome_zarr(
    acq_image_list: AcqImageList,
    destination: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Export a loaded acquisition list as one multi-image OME-Zarr store.

    The destination is built in a sibling staging directory and installed only
    after its root manifest, collection tables, and independently readable
    child images have been verified. V1 supports local directory-backed Zarr v3
    stores only.

    Args:
        acq_image_list: Non-empty collection with resident pixels and analysis
            result tables for every member.
        destination: Local output directory ending in ``.ome.zarr``.
        overwrite: Whether an existing destination may be replaced.

    Returns:
        Resolved destination path.

    Raises:
        TypeError: If ``acq_image_list`` is not an ``AcqImageList``.
        ValueError: If an input or destination precondition is not satisfied.
        FileExistsError: If the destination exists and overwrite is false.
    """
    from acqstore.acq_image.acq_image_list import AcqImageList

    if not isinstance(acq_image_list, AcqImageList):
        raise TypeError("acq_image_list must be an AcqImageList")
    dest = _validated_destination(destination)
    members = tuple(acq_image_list)
    if not members:
        raise ValueError("Cannot export an empty AcqImageList")
    if dest.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists: {dest}")
    _validate_loaded_members(members)

    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{dest.name}.staging-",
        dir=dest.parent,
    ) as temp_dir:
        staged = Path(temp_dir) / dest.name
        _build_collection(acq_image_list, members, staged)
        _verify_collection(staged, expected_count=len(members))
        _install_staged_directory(staged, dest, overwrite=overwrite)
    return dest


def _validated_destination(destination: str | Path) -> Path:
    """Return a normalized local ``.ome.zarr`` destination.

    Args:
        destination: Requested collection output path.

    Returns:
        Resolved local destination.

    Raises:
        ValueError: If the destination is remote or lacks the required suffix.
    """
    raw = str(destination)
    if "://" in raw:
        raise ValueError("V1 collection export supports local paths only")
    dest = Path(destination).expanduser().resolve(strict=False)
    if not dest.name.lower().endswith(".ome.zarr"):
        raise ValueError("Collection destination must end in '.ome.zarr'")
    return dest


def _validate_loaded_members(members: tuple[AcqImage, ...]) -> None:
    """Require resident primary pixels and analysis tables for every member.

    Args:
        members: Ordered acquisition images selected for export.

    Raises:
        ValueError: If any required runtime data is not loaded.
    """
    for acq_image in members:
        if not acq_image.images_loaded:
            raise ValueError(f"AcqImage pixels are not loaded: {acq_image.name}")
        if not acq_image.analysis_csv_loaded:
            raise ValueError(f"AcqImage analysis tables are not loaded: {acq_image.name}")


def _build_collection(
    acq_image_list: AcqImageList,
    members: tuple[AcqImage, ...],
    staged: Path,
) -> None:
    """Build the complete unverified collection in a staging directory.

    Args:
        acq_image_list: Source collection that owns the root analysis pools.
        members: Ordered acquisition images to export.
        staged: New staging destination.
    """
    zarr = _import_zarr()
    root = zarr.open_group(str(staged), mode="w", zarr_format=COLLECTION_ZARR_FORMAT)
    root.create_group("images")

    image_entries: list[dict[str, str]] = []
    for index, acq_image in enumerate(members):
        image_id = f"image_{index:03d}"
        relative_path = PurePosixPath("images", image_id)
        child = staged.joinpath(*relative_path.parts)
        reference_path = _write_collection_member(acq_image, child, relative_path)
        entry = {
            "id": image_id,
            "path": relative_path.as_posix(),
            "sidecar": (relative_path / "acqstore" / "acq_image.json").as_posix(),
            "native_manifest": (
                relative_path / "acqstore" / "manifest.json"
            ).as_posix(),
        }
        if reference_path is not None:
            entry["reference_image"] = reference_path
        image_entries.append(entry)

    tables_dir = staged / "acqstore" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=False)
    _write_dataframe(acq_image_list.velocity_analysis_pool.get_dataframe(), tables_dir / "velocity.csv")
    _write_dataframe(
        acq_image_list.sum_intensity_analysis_pool.get_dataframe(),
        tables_dir / "sum_intensity.csv",
    )
    _write_json(
        staged / "acqstore" / "manifest.json",
        {
            "format": COLLECTION_FORMAT,
            "version": COLLECTION_FORMAT_VERSION,
            "zarr_format": COLLECTION_ZARR_FORMAT,
            "images": image_entries,
            "tables": {
                "velocity": "acqstore/tables/velocity.csv",
                "sum_intensity": "acqstore/tables/sum_intensity.csv",
            },
        },
    )


def _write_collection_member(
    acq_image: AcqImage,
    child: Path,
    relative_path: PurePosixPath,
) -> str | None:
    """Write one primary image and its optional reference image.

    Reference pixels are loaded through the existing ``AcqImage`` public API.
    If export caused lazy reference data to become resident, that data is
    released after both the native sidecar and reference OME-Zarr are written.

    Args:
        acq_image: Source acquisition image.
        child: Absolute staging path for the primary native image.
        relative_path: Manifest-relative primary image path.

    Returns:
        Relative reference-image path, or ``None`` when the source has no
        reference image.
    """
    reference_was_loaded = acq_image.images.reference_data_loaded
    try:
        has_reference = acq_image.images.has_reference_image
        write_acq_image_native_ome_zarr(acq_image, child)
        if not has_reference:
            return None
        reference_path = relative_path / "reference"
        acq_image.save_reference_as_ome_zarr(
            child / "reference",
            overwrite=False,
            zarr_format=COLLECTION_ZARR_FORMAT,
        )
        return reference_path.as_posix()
    finally:
        if not reference_was_loaded:
            acq_image.images.unload_reference_data()


def _verify_collection(staged: Path, *, expected_count: int) -> None:
    """Verify structural resources and independently open every child image.

    Args:
        staged: Staged collection root.
        expected_count: Required number of manifest image entries.

    Raises:
        ValueError: If the staged collection contract is incomplete or invalid.
    """
    from acqstore.acq_image.io.ome_zarr import read_acq_pixels_ome_zarr

    zarr = _import_zarr()
    zarr.open_group(str(staged), mode="r")
    manifest_path = staged / "acqstore" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != COLLECTION_FORMAT:
        raise ValueError("Staged collection manifest has an invalid format")
    images = manifest.get("images")
    if not isinstance(images, list) or len(images) != expected_count:
        raise ValueError("Staged collection manifest has an invalid image count")

    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for entry in images:
        if not isinstance(entry, dict):
            raise ValueError("Staged collection manifest image entries must be objects")
        image_id = str(entry.get("id", ""))
        image_path = _safe_manifest_path(staged, entry.get("path"))
        sidecar_path = _safe_manifest_path(staged, entry.get("sidecar"))
        native_manifest_path = _safe_manifest_path(staged, entry.get("native_manifest"))
        if not image_id or image_id in seen_ids:
            raise ValueError(f"Duplicate or empty collection image id: {image_id!r}")
        relative_image_path = image_path.relative_to(staged).as_posix()
        if relative_image_path in seen_paths:
            raise ValueError(f"Duplicate collection image path: {relative_image_path}")
        seen_ids.add(image_id)
        seen_paths.add(relative_image_path)
        if not sidecar_path.is_file() or not native_manifest_path.is_file():
            raise ValueError(f"Native AcqStore metadata is missing for {image_id}")
        read_acq_pixels_ome_zarr(image_path, lazy=True)
        reference_raw = entry.get("reference_image")
        if reference_raw is not None:
            reference_path = _safe_manifest_path(staged, reference_raw)
            read_acq_pixels_ome_zarr(reference_path, lazy=True)

    tables = manifest.get("tables")
    if not isinstance(tables, dict):
        raise ValueError("Staged collection manifest tables must be an object")
    for table_name in ("velocity", "sum_intensity"):
        table_path = _safe_manifest_path(staged, tables.get(table_name))
        if not table_path.is_file():
            raise ValueError(f"Collection table is missing: {table_name}")


def _safe_manifest_path(root: Path, raw: Any) -> Path:
    """Resolve one manifest path without allowing root escape.

    Args:
        root: Collection root directory.
        raw: Untrusted manifest path value.

    Returns:
        Resolved path contained by ``root``.

    Raises:
        ValueError: If the value is empty, absolute, or escapes ``root``.
    """
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"Manifest path must be a non-empty string, got {raw!r}")
    relative = PurePosixPath(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Manifest path escapes the collection root: {raw!r}")
    resolved = root.joinpath(*relative.parts).resolve(strict=False)
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Manifest path escapes the collection root: {raw!r}")
    return resolved


def _install_staged_directory(staged: Path, destination: Path, *, overwrite: bool) -> None:
    """Install a verified staging directory with rollback on replacement failure.

    Args:
        staged: Verified staging directory.
        destination: Final collection path.
        overwrite: Whether an existing destination may be replaced.

    Raises:
        FileExistsError: If the destination exists and overwrite is false.
    """
    if not destination.exists():
        os.replace(staged, destination)
        return
    if not overwrite:
        raise FileExistsError(f"Destination already exists: {destination}")

    backup = staged.parent / f"{destination.name}.backup-{uuid.uuid4().hex}"
    os.replace(destination, backup)
    try:
        os.replace(staged, destination)
    except BaseException:
        os.replace(backup, destination)
        raise
    _remove_path(backup)


def _remove_path(path: Path) -> None:
    """Remove a known staging backup after successful installation.

    Args:
        path: Explicit backup file or directory path.
    """
    import shutil

    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _write_dataframe(dataframe: pd.DataFrame, destination: Path) -> None:
    """Write one collection table with deterministic column order.

    Args:
        dataframe: Runtime analysis-pool table.
        destination: CSV output path whose parent already exists.
    """
    dataframe.to_csv(destination, index=False)


def _write_json(destination: Path, payload: dict[str, Any]) -> None:
    """Write one deterministic, newline-terminated JSON object.

    Args:
        destination: JSON output path.
        payload: JSON-serializable object.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _import_zarr() -> Any:
    """Import the required Zarr runtime.

    Returns:
        Imported :mod:`zarr` module.

    Raises:
        ImportError: If Zarr is unavailable.
    """
    try:
        import zarr
    except ImportError as exc:  # pragma: no cover - required project dependency
        raise ImportError("OME-Zarr collection export requires 'zarr'") from exc
    return zarr
