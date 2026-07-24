"""TIFF export helpers for :class:`acqstore.acq_image.acq_image.AcqImage`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from acqstore.acq_image.acq_pixels import AcqPixels


@dataclass(frozen=True, slots=True)
class TiffWriteOptions:
    """Tifffile metadata and resolution options for one export."""

    metadata: dict[str, Any]
    resolution: tuple[float, float] | None


def save_pixels_as_tif(
    pixels: AcqPixels,
    path: str | Path,
    *,
    imagej_metadata: bool = True,
    overwrite: bool = False,
) -> None:
    """Write full pixel data to a standard TIFF file with optional ImageJ metadata.

    Args:
        pixels: Pixel data to export. The full array is materialized before
            writing because TIFF export is an eager file export path.
        path: Destination TIFF filename. The caller must provide the complete
            filename.
        imagej_metadata: When true, request ImageJ-compatible metadata from
            :mod:`tifffile`. Real ImageJ X/Y calibration is written only when
            both axes share the same physical unit. An ImageJ ``Info`` note is
            always written with the acqstore per-axis calibration.
        overwrite: Whether to replace an existing TIFF file.

    Raises:
        FileExistsError: If ``path`` exists and ``overwrite`` is false.
        ValueError: If ``path`` is empty or names a directory.
        ImportError: If :mod:`tifffile` is not installed.
    """
    try:
        import tifffile
    except ImportError as exc:  # pragma: no cover - project dependency
        raise ImportError('TIFF export requires tifffile') from exc

    dest = Path(path)
    if not str(dest):
        raise ValueError('TIFF export requires an explicit destination filename')
    if dest.exists():
        if dest.is_dir():
            raise ValueError(f'TIFF destination is a directory: {dest}')
        if not overwrite:
            raise FileExistsError(f'TIFF destination already exists: {dest}')
    dest.parent.mkdir(parents=True, exist_ok=True)

    data = np.asarray(pixels.get_array(0))
    options = _build_tifffile_options(pixels, imagej_metadata=imagej_metadata)
    kwargs: dict[str, Any] = {
        'imagej': bool(imagej_metadata),
        'metadata': options.metadata,
    }
    if options.resolution is not None:
        kwargs['resolution'] = options.resolution
    tifffile.imwrite(dest, data, **kwargs)


def _build_tifffile_options(
    pixels: AcqPixels,
    *,
    imagej_metadata: bool,
) -> TiffWriteOptions:
    """Return tifffile metadata and resolution for the exported array.

    Args:
        pixels: Pixel object being exported.
        imagej_metadata: Whether to include ImageJ-specific keys.

    Returns:
        Metadata and optional ImageJ X/Y resolution for :func:`tifffile.imwrite`.
    """
    metadata: dict[str, Any] = {'axes': ''.join(pixels.axes)}
    if not imagej_metadata:
        return TiffWriteOptions(metadata=metadata, resolution=None)

    metadata['Info'] = _build_imagej_info(pixels)
    resolution = _imagej_xy_resolution(pixels)
    unit = _common_xy_unit(pixels)
    if resolution is not None:
        if unit is None:
            raise ValueError('Internal error: ImageJ resolution requires a common X/Y unit')
        metadata['unit'] = unit

    spacing = _spacing_for_dim(pixels, 'Z')
    if spacing is not None:
        metadata['spacing'] = spacing
    frame_interval = _spacing_for_dim(pixels, 'T')
    if frame_interval is not None:
        metadata['finterval'] = frame_interval

    return TiffWriteOptions(metadata=metadata, resolution=resolution)


def _build_imagej_info(pixels: AcqPixels) -> str:
    """Return an ImageJ ``Info`` note with exact acqstore axis calibration.

    Args:
        pixels: Pixel object being exported.

    Returns:
        Multiline note for the ImageJ/Fiji ``Info`` metadata field.
    """
    lines = ['Saved by acqstore.', 'Axis calibration:']
    for axis, value, label in zip(
        pixels.axes,
        pixels.header.physical_units,
        pixels.header.physical_units_labels,
        strict=True,
    ):
        lines.append(f'{axis} = {value} {label}')
    return '\n'.join(lines)


def _spacing_for_dim(pixels: AcqPixels, dim: str) -> float | None:
    if dim not in pixels.axes:
        return None
    index = pixels.axes.index(dim)
    value = float(pixels.header.physical_units[index])
    if value <= 0.0 or not np.isfinite(value):
        raise ValueError(f'Physical spacing for {dim!r} must be finite and > 0, got {value!r}')
    return value


def _imagej_xy_resolution(pixels: AcqPixels) -> tuple[float, float] | None:
    """Return ImageJ resolution when X and Y share one physical unit.

    ImageJ's basic TIFF calibration uses one unit for both X and Y. For mixed
    kymograph axes such as Y=seconds and X=micrometer, acqstore writes the exact
    calibration to ``Info`` but deliberately does not write misleading ImageJ
    X/Y calibration.
    """
    if 'X' not in pixels.axes or 'Y' not in pixels.axes:
        return None
    unit = _common_xy_unit(pixels)
    if unit is None:
        return None
    x_step = _positive_step_for_axis(pixels, 'X')
    y_step = _positive_step_for_axis(pixels, 'Y')
    return (1.0 / x_step, 1.0 / y_step)


def _positive_step_for_axis(pixels: AcqPixels, axis: str) -> float:
    index = pixels.axes.index(axis)
    value = float(pixels.header.physical_units[index])
    if value <= 0.0 or not np.isfinite(value):
        raise ValueError(f'Physical spacing for {axis!r} must be finite and > 0, got {value!r}')
    return value


def _common_xy_unit(pixels: AcqPixels) -> str | None:
    """Return the shared ImageJ unit for X/Y, or ``None`` for mixed/unknown units."""
    if 'X' not in pixels.axes or 'Y' not in pixels.axes:
        return None
    labels: list[str] = []
    for dim in ('Y', 'X'):
        index = pixels.axes.index(dim)
        label = str(pixels.header.physical_units_labels[index]).strip()
        if not label or label.lower() == 'pixels':
            return None
        labels.append(_imagej_unit_label(label))
    first = labels[0]
    if all(label == first for label in labels):
        return first
    return None


def _imagej_unit_label(label: str) -> str:
    normalized = label.strip()
    low = normalized.lower()
    if low in {'micrometer', 'micrometers', 'micron', 'microns', 'µm'}:
        return 'um'
    if low in {'second', 'seconds'}:
        return 'sec'
    return normalized
