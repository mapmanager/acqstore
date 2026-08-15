"""Static web-dataset export for :class:`AcqImage` and :class:`AcqImageList`.

The exported format is intentionally read-only and frontend-agnostic.  It
normalizes all source image formats to OME-Zarr and writes explicit JSON/CSV
metadata so a thin browser client does not need AcqStore-specific scientific
logic or source-file loaders.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .acq_pixels import AcqPixels
from .io.ome_zarr import _write_acq_pixels_ome_zarr_for_web
from .roi import LineROI, RectROI

if TYPE_CHECKING:
    from .acq_image import AcqImage
    from .acq_image_list import AcqImageList, LoadWarning
    from .analysis.model import BaseAnalysis

WEB_DATASET_FORMAT = "acqstore-web-dataset"
WEB_ACQIMAGE_FORMAT = "acqstore-web-acqimage"
WEB_FORMAT_VERSION = 1


@dataclass(frozen=True, slots=True)
class WebDatasetExportReport:
    """Result from :func:`load_and_export_web_dataset`."""

    destination: str
    exported_images: int
    discovered_images: int
    warnings: tuple[LoadWarning, ...]


def export_acq_image(
    acq_image: AcqImage,
    destination: str | Path,
    *,
    overwrite: bool = False,
    image_id: str | None = None,
) -> Path:
    """Export one loaded ``AcqImage`` as one web-viewable image package.

    The destination contains ``acqimage.json``, ``image.ome.zarr``, optional
    ``reference.ome.zarr``, and per-analysis CSV resources.
    """
    dest = Path(destination).expanduser().resolve(strict=False)
    resolved_id = image_id or _standalone_image_id(acq_image)

    def build(staging: Path) -> None:
        _write_acq_image_package(acq_image, staging, image_id=resolved_id)

    _replace_directory_atomically(dest, overwrite=overwrite, build=build)
    return dest


def export_acq_image_list(
    acq_image_list: AcqImageList,
    destination: str | Path,
    *,
    name: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Export a loaded ``AcqImageList`` as an AcqStore Web Dataset v1.

    Existing destinations are rejected unless ``overwrite=True``.  Replacement
    is whole-dataset and staged before the old destination is removed, avoiding
    stale files from prior exports.  A valid existing dataset UUID is preserved
    across replacement exports.
    """
    dest = Path(destination).expanduser().resolve(strict=False)
    existing_dataset_id = _existing_dataset_id(dest) if overwrite else None
    dataset_id = existing_dataset_id or str(uuid.uuid4())
    source_root = _source_root(acq_image_list)
    dataset_name = _dataset_name(acq_image_list, dest, explicit_name=name)

    def build(staging: Path) -> None:
        images_dir = staging / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        index_images: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        for acq_image in acq_image_list:
            image_id = _list_image_id(acq_image, source_root=source_root)
            if image_id in seen_ids:
                raise ValueError(f"Duplicate exported AcqImage id: {image_id}")
            seen_ids.add(image_id)
            image_dir = images_dir / image_id
            detail = _write_acq_image_package(acq_image, image_dir, image_id=image_id)
            index_images.append(_build_dataset_image_index(detail, image_id=image_id))

        payload = build_dataset_document(
            dataset_id=dataset_id,
            name=dataset_name,
            images=index_images,
        )
        _write_json(staging / "dataset.json", payload)

    _replace_directory_atomically(dest, overwrite=overwrite, build=build)
    return dest


def load_and_export_web_dataset(
    source_path: str | Path,
    destination: str | Path,
    *,
    name: str | None = None,
    overwrite: bool = False,
    folder_depth: int = 4,
) -> WebDatasetExportReport:
    """Safely load a local source path and export every successfully loaded image.

    This is the reusable end-to-end entry point used by the development script
    and is suitable for notebooks or other Python clients.  File discovery/load
    warnings are retained, while any failure exporting a successfully loaded
    image raises and aborts the staged export.
    """
    from .acq_image_list import AcqImageList, PathKind

    source = Path(source_path).expanduser()
    if source.suffix.lower() == ".csv":
        kind = PathKind.CSV
    elif source.is_dir():
        kind = PathKind.FOLDER
    else:
        kind = PathKind.FILE

    result = AcqImageList.load_safe(
        str(source),
        kind=kind,
        folder_depth=folder_depth,
        load_images=True,
        load_analysis_csv=True,
    )
    export_acq_image_list(
        result.acq_image_list,
        destination,
        name=name,
        overwrite=overwrite,
    )
    return WebDatasetExportReport(
        destination=str(Path(destination).expanduser().resolve(strict=False)),
        exported_images=len(result.acq_image_list),
        discovered_images=int(result.discovered_count),
        warnings=tuple(result.warnings),
    )


def build_dataset_document(
    *,
    dataset_id: str,
    name: str,
    images: list[dict[str, Any]],
    created_utc: str | None = None,
) -> dict[str, Any]:
    """Build an AcqStore Web Dataset v1 manifest without writing files."""
    return {
        "format": WEB_DATASET_FORMAT,
        "format_version": WEB_FORMAT_VERSION,
        "id": dataset_id,
        "name": name,
        "acqstore_version": _acqstore_version(),
        "created_utc": created_utc or _utc_now_iso(),
        "images": images,
    }


def build_dataset_image_index(
    detail: dict[str, Any],
    *,
    image_id: str,
    href: str | None = None,
) -> dict[str, Any]:
    """Build one dataset manifest row from an AcqImage web document."""
    row = _build_dataset_image_index(detail, image_id=image_id)
    if href is not None:
        row["href"] = href
    return row


def build_dataset_image_index_from_acq_image(
    acq_image: AcqImage,
    *,
    image_id: str,
    href: str,
) -> dict[str, Any]:
    """Build a manifest row from header and sidecar metadata only.

    This function deliberately avoids :attr:`AcqImage.pixels` and analysis CSV
    tables so collection discovery remains memory-bounded.
    """
    image = _build_image_payload(acq_image)
    return {
        "id": image_id,
        "name": acq_image.name,
        "href": href,
        "shape": list(image["shape"]),
        "dims": list(image["dims"]),
        "sizes": dict(image["sizes"]),
        "dtype": image["dtype"],
        "axes": list(image["axes"]),
        "acquisition": dict(image["acquisition"]),
        "num_channels": int(image["num_channels"]),
        "num_rois": int(acq_image.rois.num_rois),
        "analysis_types": sorted(
            {str(analysis.key.analysis_name) for analysis in acq_image.analysis_set.as_list()}
        ),
        "accepted": bool(acq_image.get_schema_row()["accept"]),
        "has_reference_image": bool(acq_image.images.has_reference_image),
    }


def web_image_id(acq_image: AcqImage, *, source_root: str | Path | None = None) -> str:
    """Return the stable web identifier for an AcqImage."""
    root = None if source_root is None else Path(source_root).expanduser().resolve(strict=False)
    return _list_image_id(acq_image, source_root=root)


def _write_acq_image_package(acq_image: AcqImage, image_dir: Path, *, image_id: str) -> dict[str, Any]:
    image_dir.mkdir(parents=True, exist_ok=False)

    _write_acq_pixels_ome_zarr_for_web(
        acq_image.pixels,
        image_dir / "image.ome.zarr",
        overwrite=False,
    )

    reference = acq_image.images.reference_image
    if reference is not None:
        _write_acq_pixels_ome_zarr_for_web(
            acq_image._reference_acq_pixels(),
            image_dir / "reference.ome.zarr",
            overwrite=False,
        )

    analyses_payload = _export_completed_analyses(acq_image, image_dir)
    payload = build_acq_image_document(
        acq_image,
        image_id=image_id,
        analyses=analyses_payload,
    )
    _write_json(image_dir / "acqimage.json", payload)
    return payload


def build_acq_image_document(
    acq_image: AcqImage,
    *,
    image_id: str,
    image_href: str = "image.ome.zarr",
    reference_href: str = "reference.ome.zarr",
    analyses: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the frontend-neutral AcqImage web document without writing files.

    Args:
        acq_image: Loaded acquisition domain object.
        image_id: Stable image identifier used by the containing dataset.
        image_href: Resource link for primary pixels.
        reference_href: Resource link for optional reference pixels.
        analyses: Optional prebuilt analysis documents. When omitted, completed
            analyses are described using relative static-export links.

    Returns:
        JSON-safe AcqStore Web AcqImage v1 document.
    """
    reference = acq_image.images.reference_image
    resolved_analyses = analyses
    if resolved_analyses is None:
        resolved_analyses = [
            build_analysis_document(analysis)
            for analysis in completed_analyses(acq_image)
        ]
    return {
        "format": WEB_ACQIMAGE_FORMAT,
        "format_version": WEB_FORMAT_VERSION,
        "id": image_id,
        "name": acq_image.name,
        "accepted": bool(acq_image.get_schema_row()["accept"]),
        "image": _build_image_payload(acq_image, href=image_href),
        "rois": [_export_roi(roi) for roi in acq_image.rois],
        "analyses": resolved_analyses,
        "metadata": {
            "experiment": _jsonable(acq_image.get_metadata_section("experiment_metadata").get_values()),
            "image_header": _jsonable(acq_image.get_metadata_section("acq_image_header").get_values()),
        },
        "reference_image": (
            None
            if reference is None
            else _build_reference_payload(acq_image, reference, href=reference_href)
        ),
    }


def _build_image_payload(acq_image: AcqImage, *, href: str = "image.ome.zarr") -> dict[str, Any]:
    header = acq_image.images.header.with_coerced_physical_calibration()
    channels: list[dict[str, Any]] = []
    for channel in range(int(header.num_channels)):
        contrast = acq_image.get_image_contrast(channel)
        channels.append(
            {
                "index": int(channel),
                "contrast": None if contrast is None else _jsonable(asdict(contrast)),
            }
        )
    return {
        "href": href,
        **_header_pixel_descriptor(header),
        "default_channel": 0 if header.num_channels > 0 else None,
        "channels": channels,
        "acquisition": {
            "date": str(header.date or ""),
            "time": str(header.time or ""),
        },
    }


def _build_reference_payload(
    acq_image: AcqImage,
    reference: Any,
    *,
    href: str = "reference.ome.zarr",
) -> dict[str, Any]:
    data = np.asarray(reference.array)
    dims = [str(dim).lower() for dim in reference.dims]
    shape = [int(v) for v in data.shape]
    scale_map = {str(k).lower(): v for k, v in dict(reference.coord_scales).items()}
    unit_map = {str(k).lower(): v for k, v in dict(reference.coord_units).items()}
    axes: list[dict[str, Any]] = []
    for i, dim in enumerate(dims):
        try:
            spacing = float(scale_map.get(dim, 1.0))
        except (TypeError, ValueError):
            spacing = 1.0
        if not np.isfinite(spacing) or spacing <= 0.0:
            spacing = 1.0
        axes.append(
            {
                "name": dim,
                "size": shape[i],
                "spacing": spacing,
                "unit": str(unit_map.get(dim, "Pixels") or "Pixels"),
            }
        )
    metadata = _jsonable(acq_image.get_metadata_section("reference_image_metadata").get_values())
    return {
        "href": href,
        "shape": shape,
        "dims": dims,
        "sizes": {dim: shape[i] for i, dim in enumerate(dims)},
        "dtype": str(data.dtype),
        "axes": axes,
        "num_channels": int(reference.num_channels),
        "metadata": metadata,
    }


def _pixel_descriptor(pixels: AcqPixels) -> dict[str, Any]:
    return _header_pixel_descriptor(pixels.header)


def _header_pixel_descriptor(header: Any) -> dict[str, Any]:
    """Build a pixel descriptor from an ImageHeader without loading pixels."""
    header = header.with_coerced_physical_calibration()
    dims = [str(dim).lower() for dim in header.dims]
    shape = [int(v) for v in header.shape]
    sizes = {dim: shape[i] for i, dim in enumerate(dims)}
    axes = []
    for i, dim in enumerate(dims):
        axes.append(
            {
                "name": dim,
                "size": shape[i],
                "spacing": float(header.physical_units[i]),
                "unit": str(header.physical_units_labels[i]),
            }
        )
    return {
        "shape": shape,
        "dims": dims,
        "sizes": sizes,
        "dtype": str(header.dtype),
        "axes": axes,
        "num_channels": int(header.num_channels),
    }


def _export_roi(roi: Any) -> dict[str, Any]:
    if isinstance(roi, RectROI):
        return {
            "id": int(roi.roi_id),
            "type": "rect",
            "name": str(roi.name),
            "note": str(roi.note),
            "x_start": int(roi.bounds.dim1_start),
            "x_stop": int(roi.bounds.dim1_stop),
            "y_start": int(roi.bounds.dim0_start),
            "y_stop": int(roi.bounds.dim0_stop),
        }
    if isinstance(roi, LineROI):
        return {
            "id": int(roi.roi_id),
            "type": "line",
            "name": str(roi.name),
            "note": str(roi.note),
            "x0": int(roi.endpoints.col0),
            "y0": int(roi.endpoints.row0),
            "x1": int(roi.endpoints.col1),
            "y1": int(roi.endpoints.row1),
        }
    raise TypeError(f"Unsupported ROI type for web export: {type(roi).__name__}")


def _export_completed_analyses(acq_image: AcqImage, image_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for analysis in completed_analyses(acq_image):
        out.append(_export_analysis(analysis, image_dir))
    return out


def completed_analyses(acq_image: AcqImage) -> tuple[BaseAnalysis, ...]:
    """Return completed analyses in stable web-document order."""
    completed = [a for a in acq_image.analysis_set.as_list() if a.result.table is not None]
    completed.sort(key=lambda a: (a.key.analysis_name, a.key.channel, a.key.roi_id))
    return tuple(completed)


def _export_analysis(analysis: BaseAnalysis, image_dir: Path) -> dict[str, Any]:
    analysis_id = _analysis_id(
        analysis.key.analysis_name,
        channel=int(analysis.key.channel),
        roi_id=int(analysis.key.roi_id),
    )
    rel_dir = Path("analysis") / analysis_id
    dest_dir = image_dir / rel_dir
    dest_dir.mkdir(parents=True, exist_ok=False)

    table = analysis.result.table
    assert table is not None
    table.to_csv(dest_dir / "table.csv", index=False)

    plot_payload: dict[str, Any] | None = None
    plot_data = analysis.get_plot_data()
    if plot_data is not None:
        pd.DataFrame({"x": list(plot_data.x), "y": list(plot_data.y)}).to_csv(
            dest_dir / "plot.csv", index=False
        )
        plot_payload = {
            "href": (rel_dir / "plot.csv").as_posix(),
            "x_column": "x",
            "y_column": "y",
            "x_label": str(plot_data.x_label),
            "y_label": str(plot_data.y_label),
            "series_name": str(plot_data.series_name),
        }

    peaks_payload: dict[str, Any] | None = None
    peak_rows_getter = getattr(analysis, "get_pool_peak_rows", None)
    peak_columns_getter = getattr(analysis, "get_pool_peak_columns", None)
    if callable(peak_rows_getter):
        peak_rows = tuple(peak_rows_getter())
        columns = tuple(peak_columns_getter()) if callable(peak_columns_getter) else ()
        pd.DataFrame(list(peak_rows), columns=list(columns) or None).to_csv(
            dest_dir / "peaks.csv", index=False
        )
        peaks_payload = {
            "href": (rel_dir / "peaks.csv").as_posix(),
            "count": len(peak_rows),
        }

    return build_analysis_document(
        analysis,
        table_href=(rel_dir / "table.csv").as_posix(),
        plot_href=None if plot_payload is None else plot_payload["href"],
        peaks_href=None if peaks_payload is None else peaks_payload["href"],
    )


def build_analysis_document(
    analysis: BaseAnalysis,
    *,
    table_href: str | None = None,
    plot_href: str | None = None,
    peaks_href: str | None = None,
) -> dict[str, Any]:
    """Build one analysis web document without writing its tabular resources."""
    analysis_id = _analysis_id(
        analysis.key.analysis_name,
        channel=int(analysis.key.channel),
        roi_id=int(analysis.key.roi_id),
    )
    rel_dir = Path("analysis") / analysis_id
    plot_data = analysis.get_plot_data()
    display_name = (
        str(plot_data.series_name)
        if plot_data is not None and str(plot_data.series_name).strip()
        else analysis.key.analysis_name.replace("_", " ").title()
    )
    plot_payload = None
    if plot_data is not None:
        plot_payload = {
            "href": plot_href or (rel_dir / "plot.csv").as_posix(),
            "x_column": "x",
            "y_column": "y",
            "x_label": str(plot_data.x_label),
            "y_label": str(plot_data.y_label),
            "series_name": str(plot_data.series_name),
        }
    peaks_payload = None
    peak_rows_getter = getattr(analysis, "get_pool_peak_rows", None)
    if callable(peak_rows_getter):
        peaks_payload = {
            "href": peaks_href or (rel_dir / "peaks.csv").as_posix(),
            "count": len(tuple(peak_rows_getter())),
        }
    payload: dict[str, Any] = {
        "id": analysis_id,
        "analysis_type": str(analysis.key.analysis_name),
        "display_name": display_name,
        "channel": int(analysis.key.channel),
        "roi_id": int(analysis.key.roi_id),
        "summary": _jsonable(analysis.result.summary),
        "table": {"href": table_href or (rel_dir / "table.csv").as_posix()},
        "plot": plot_payload,
    }
    if peaks_payload is not None:
        payload["peaks"] = peaks_payload
    return payload


def analysis_plot_csv(analysis: BaseAnalysis) -> str:
    """Return one completed analysis plot as CSV text for HTTP or file delivery."""
    plot_data = analysis.get_plot_data()
    if plot_data is None:
        raise ValueError(f"Analysis {analysis.key.analysis_name!r} has no plot data")
    return pd.DataFrame({"x": list(plot_data.x), "y": list(plot_data.y)}).to_csv(index=False)


def analysis_table_csv(analysis: BaseAnalysis) -> str:
    """Return one completed analysis result table as CSV text."""
    table = analysis.result.table
    if table is None:
        raise ValueError(f"Analysis {analysis.key.analysis_name!r} has no result table")
    return table.to_csv(index=False)


def analysis_peaks_csv(analysis: BaseAnalysis) -> str:
    """Return one analysis peak table as CSV text."""
    rows_getter = getattr(analysis, "get_pool_peak_rows", None)
    columns_getter = getattr(analysis, "get_pool_peak_columns", None)
    if not callable(rows_getter):
        raise ValueError(f"Analysis {analysis.key.analysis_name!r} has no peak data")
    rows = tuple(rows_getter())
    columns = tuple(columns_getter()) if callable(columns_getter) else ()
    return pd.DataFrame(list(rows), columns=list(columns) or None).to_csv(index=False)


def _build_dataset_image_index(detail: dict[str, Any], *, image_id: str) -> dict[str, Any]:
    image = detail["image"]
    analyses = detail["analyses"]
    return {
        "id": image_id,
        "name": detail["name"],
        "href": f"images/{image_id}/acqimage.json",
        "shape": list(image["shape"]),
        "dims": list(image["dims"]),
        "sizes": dict(image["sizes"]),
        "dtype": image["dtype"],
        "axes": list(image["axes"]),
        "acquisition": dict(image["acquisition"]),
        "num_channels": int(image["num_channels"]),
        "num_rois": len(detail["rois"]),
        "analysis_types": sorted({str(a["analysis_type"]) for a in analyses}),
        "accepted": bool(detail["accepted"]),
        "has_reference_image": detail["reference_image"] is not None,
    }


def _analysis_id(analysis_name: str, *, channel: int, roi_id: int) -> str:
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(analysis_name)).strip("-") or "analysis"
    return f"{safe_name}__c{channel}__r{roi_id}"


def _standalone_image_id(acq_image: AcqImage) -> str:
    return _stable_id("image", acq_image.name)


def _list_image_id(acq_image: AcqImage, *, source_root: Path | None) -> str:
    path = Path(acq_image.path).expanduser().resolve(strict=False)
    if source_root is not None:
        try:
            identity = path.relative_to(source_root).as_posix()
        except ValueError:
            identity = path.name
    else:
        identity = path.name
    return _stable_id("image", identity)


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def _source_root(acq_image_list: AcqImageList) -> Path | None:
    raw = getattr(acq_image_list, "source_root_path", None)
    if not raw:
        return None
    return Path(str(raw)).expanduser().resolve(strict=False)


def _dataset_name(
    acq_image_list: AcqImageList,
    destination: Path,
    *,
    explicit_name: str | None,
) -> str:
    if explicit_name is not None and explicit_name.strip():
        return explicit_name.strip()
    source_root = _source_root(acq_image_list)
    if source_root is not None and source_root.name:
        return source_root.name
    return destination.name


def _existing_dataset_id(destination: Path) -> str | None:
    manifest = destination / "dataset.json"
    if not manifest.is_file():
        return None
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("format") != WEB_DATASET_FORMAT or payload.get("format_version") != WEB_FORMAT_VERSION:
        return None
    raw = payload.get("id")
    return str(raw) if raw else None


def _replace_directory_atomically(
    destination: Path,
    *,
    overwrite: bool,
    build: Callable[[Path], None],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists: {destination}")

    staging_parent = Path(tempfile.mkdtemp(prefix=f".{destination.name}.web-export-", dir=destination.parent))
    staging = staging_parent / destination.name
    try:
        build(staging)
        if destination.exists():
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        os.replace(staging, destination)
    finally:
        shutil.rmtree(staging_parent, ignore_errors=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if value is pd.NA:
        return None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return value.as_posix()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _acqstore_version() -> str:
    try:
        return package_version("acqstore")
    except PackageNotFoundError:
        return "unknown"


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
