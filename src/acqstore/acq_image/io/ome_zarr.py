"""OME-Zarr/NGFF IO helpers for :class:`~acqstore.acq_image.acq_pixels.AcqPixels`.

The public acqstore API is intentionally independent of the concrete OME-Zarr
implementation library. Writing uses ``bioio-ome-zarr`` so new exports default
to OME-NGFF 0.5 / Zarr v3 while still allowing Zarr v2 / NGFF 0.4 on request.
Reading validates OME-NGFF metadata strictly instead of inventing missing axes
or physical calibration. Native ``.cs.ome.zarr`` stores also embed an acqstore
header snapshot, which is treated as required and strict when present.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from acqstore.acq_image.acq_pixels import AcqPixels
from acqstore.acq_image.file_loaders.base_file_loader import ImageHeader
from acqstore.acq_image.io.store_utils import (
    ensure_store_absent,
    is_s3_path,
    is_zip_store_path,
    zip_directory_store,
)

_OME_NGFF_VERSION_BY_FORMAT = {2: '0.4', 3: '0.5'}
_DEFAULT_ZARR_FORMAT = 3
_MIN_PYRAMID_SPATIAL_SIZE = 16


def _import_zarr() -> Any:
    """Import :mod:`zarr` or raise an actionable error.

    Returns:
        Imported :mod:`zarr` module.

    Raises:
        ImportError: If :mod:`zarr` is not installed.
    """
    try:
        import zarr
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError(
            "OME-Zarr reads require optional dependency 'zarr'. Install with: uv add zarr"
        ) from exc
    return zarr


def _import_bioio_writer() -> type[Any]:
    """Import the public BioIO OME-Zarr writer.

    Returns:
        ``bioio_ome_zarr.writers.OMEZarrWriter``.

    Raises:
        ImportError: If :mod:`bioio_ome_zarr` is not installed.
    """
    try:
        from bioio_ome_zarr.writers import OMEZarrWriter
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise ImportError(
            "OME-Zarr exports require optional dependency 'bioio-ome-zarr'. "
            'Install with: uv add bioio-ome-zarr s3fs'
        ) from exc
    return OMEZarrWriter


def _open_group(path: str | Path, *, mode: str) -> Any:
    """Open a Zarr group using the public Zarr API.

    Args:
        path: Local directory, local ZIP store, or fsspec-compatible store path.
        mode: Zarr open mode.

    Returns:
        Open Zarr group.

    Raises:
        ImportError: If required Zarr support is unavailable.
    """
    zarr = _import_zarr()
    path_text = str(path)
    if is_zip_store_path(path_text):
        zip_store_cls = getattr(getattr(zarr, 'storage', object()), 'ZipStore', None)
        if zip_store_cls is None:
            raise ImportError('Reading ZIP-backed Zarr stores requires zarr.storage.ZipStore')
        store = zip_store_cls(path_text, mode='r')
        try:
            return zarr.open_group(store=store, mode=mode)
        except TypeError:
            return zarr.open_group(store, mode=mode)
    try:
        return zarr.open_group(store=path_text, mode=mode)
    except TypeError:
        return zarr.open_group(path_text, mode=mode)


def _axis_type(axis: str) -> str:
    """Return the OME-NGFF axis type for an acqstore axis label."""
    if axis in {'X', 'Y', 'Z'}:
        return 'space'
    if axis == 'T':
        return 'time'
    if axis == 'C':
        return 'channel'
    raise ValueError(f'Unsupported OME-Zarr axis {axis!r}; expected T/C/Z/Y/X')


def _axis_unit(axis: str, header: ImageHeader) -> str | None:
    """Return the OME-NGFF unit for ``axis`` from ``header``.

    Pixel units are represented by omitting the unit, per common NGFF practice.
    Malformed header calibration raises instead of silently defaulting.
    """
    if axis not in header.dims:
        raise ValueError(f'Axis {axis!r} not present in header dims {header.dims!r}')
    index = header.dims.index(axis)
    if index >= len(header.physical_units_labels):
        raise ValueError(
            f'Missing physical unit label for axis {axis!r}; '
            f'labels={header.physical_units_labels!r}'
        )
    raw = str(header.physical_units_labels[index]).strip()
    if not raw:
        raise ValueError(f'Empty physical unit label for axis {axis!r}')
    if raw.lower() == 'pixels':
        return None
    return raw


def _scale_vector(header: ImageHeader) -> list[float]:
    """Return strict per-axis physical scale values for ``header``.

    Raises:
        ValueError: If any scale is missing, non-numeric, non-finite, or <= 0.
    """
    if len(header.physical_units) != len(header.dims):
        raise ValueError(
            f'Physical unit count {len(header.physical_units)} does not match '
            f'dims count {len(header.dims)} for dims={header.dims!r}'
        )
    scale: list[float] = []
    for axis, raw in zip(header.dims, header.physical_units, strict=True):
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'Physical scale for axis {axis!r} is not numeric: {raw!r}') from exc
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f'Physical scale for axis {axis!r} must be finite and > 0, got {value!r}')
        scale.append(value)
    return scale


def build_ome_ngff_metadata(
    pixels: AcqPixels,
    *,
    zarr_format: int = _DEFAULT_ZARR_FORMAT,
) -> dict[str, Any]:
    """Return expected OME-NGFF multiscales metadata for ``pixels``.

    Args:
        pixels: Source pixels and acquisition metadata.
        zarr_format: Target Zarr format, ``3`` for NGFF 0.5 or ``2`` for NGFF
            0.4.

    Returns:
        OME-NGFF ``multiscales`` entry for one image.

    Raises:
        ValueError: If ``zarr_format`` is not ``2`` or ``3`` or the pixel
            metadata is malformed.
    """
    version = _ngff_version_for_zarr_format(zarr_format)
    axes = []
    for axis in pixels.axes:
        entry: dict[str, Any] = {'name': axis.lower(), 'type': _axis_type(axis)}
        unit = _axis_unit(axis, pixels.header)
        if unit is not None:
            entry['unit'] = unit
        axes.append(entry)
    datasets = []
    for level, shape in enumerate(build_multiscale_level_shapes(pixels.shape, pixels.axes)):
        scale = _level_scale_vector(pixels.header, pixels.shape, shape)
        datasets.append(
            {
                'path': str(level),
                'coordinateTransformations': [{'type': 'scale', 'scale': scale}],
            }
        )
    return {
        'version': version,
        'axes': axes,
        'datasets': datasets,
        'metadata': {'method': 'acqstore'},
    }


def build_multiscale_level_shapes(
    shape: tuple[int, ...],
    axes: tuple[str, ...],
    *,
    min_spatial_size: int = _MIN_PYRAMID_SPATIAL_SIZE,
) -> list[tuple[int, ...]]:
    """Return pyramid level shapes with Y/X downsampled by powers of two.

    Args:
        shape: Full-resolution shape.
        axes: Axis labels aligned with ``shape``.
        min_spatial_size: Stop once either spatial axis would fall below this
            size.

    Returns:
        Level-0-first list of shapes. Small images may return only level 0.

    Raises:
        ValueError: If ``axes`` does not contain Y and X.
    """
    if len(shape) != len(axes):
        raise ValueError(f'Shape rank {len(shape)} does not match axes {axes!r}')
    if 'Y' not in axes or 'X' not in axes:
        raise ValueError(f'OME-Zarr export requires Y and X axes; got {axes!r}')
    levels = [tuple(int(x) for x in shape)]
    y_axis = axes.index('Y')
    x_axis = axes.index('X')
    current = list(levels[0])
    while current[y_axis] // 2 >= min_spatial_size and current[x_axis] // 2 >= min_spatial_size:
        current = list(current)
        current[y_axis] = max(1, current[y_axis] // 2)
        current[x_axis] = max(1, current[x_axis] // 2)
        levels.append(tuple(current))
    return levels


def write_acq_pixels_ome_zarr(
    pixels: AcqPixels,
    path: str | Path,
    *,
    overwrite: bool = False,
    zarr_format: int = _DEFAULT_ZARR_FORMAT,
    include_acqstore_pixels: bool = True,
) -> None:
    """Write one acquisition's pixels as an OME-Zarr-compatible store.

    Args:
        pixels: Pixel object to write.
        path: Destination store. Local directory stores, local ``.zip`` stores,
            and ``s3://`` fsspec paths are supported.
        overwrite: Whether to replace an existing local destination. S3 overwrite
            cleanup must be performed by the caller or AWS CLI.
        zarr_format: Target Zarr format. ``3`` writes NGFF 0.5; ``2`` writes
            NGFF 0.4.
        include_acqstore_pixels: When true, embed a strict acqstore header
            snapshot in root attrs for native round-tripping.

    Raises:
        ValueError: If ``zarr_format`` is not ``2`` or ``3`` or metadata is
            malformed.
        ImportError: If ``bioio-ome-zarr`` is missing.
    """
    _ngff_version_for_zarr_format(zarr_format)
    _validate_pixels_for_ome_zarr(pixels)
    if is_zip_store_path(path):
        if is_s3_path(path):
            raise ValueError('ZIP-backed OME-Zarr writes are only supported for local paths')
        with tempfile.TemporaryDirectory(prefix='acqstore_ome_zarr_') as tmpdir:
            tmp_store = Path(tmpdir) / Path(str(path)[:-4]).name
            _write_acq_pixels_ome_zarr_directory(
                pixels,
                tmp_store,
                overwrite=True,
                zarr_format=zarr_format,
                include_acqstore_pixels=include_acqstore_pixels,
            )
            zip_directory_store(tmp_store, path, overwrite=overwrite)
        return
    _write_acq_pixels_ome_zarr_directory(
        pixels,
        path,
        overwrite=overwrite,
        zarr_format=zarr_format,
        include_acqstore_pixels=include_acqstore_pixels,
    )


def _validate_pixels_for_ome_zarr(pixels: AcqPixels) -> None:
    """Validate pixel/header consistency before writing OME-Zarr."""
    data_shape = tuple(int(x) for x in np.asarray(pixels.get_array(0)).shape)
    if data_shape != pixels.shape:
        raise ValueError(f'Pixel array shape {data_shape!r} does not match header shape {pixels.shape!r}')
    if len(pixels.axes) != len(pixels.shape):
        raise ValueError(f'Axes {pixels.axes!r} do not match shape {pixels.shape!r}')
    _scale_vector(pixels.header)
    for axis in pixels.axes:
        _axis_type(axis)
        _axis_unit(axis, pixels.header)


def _write_acq_pixels_ome_zarr_directory(
    pixels: AcqPixels,
    path: str | Path,
    *,
    overwrite: bool,
    zarr_format: int,
    include_acqstore_pixels: bool,
) -> None:
    if not is_s3_path(path):
        ensure_store_absent(path, overwrite=overwrite)
    writer_cls = _import_bioio_writer()
    arr = np.asarray(pixels.get_array(0))
    level_shapes = build_multiscale_level_shapes(tuple(int(x) for x in arr.shape), pixels.axes)
    writer = writer_cls(
        store=str(path),
        level_shapes=level_shapes,
        dtype=arr.dtype,
        zarr_format=zarr_format,
        axes_names=[axis.lower() for axis in pixels.axes],
        axes_types=[_axis_type(axis) for axis in pixels.axes],
        axes_units=[_axis_unit(axis, pixels.header) for axis in pixels.axes],
        physical_pixel_size=_scale_vector(pixels.header),
        image_name=Path(str(path).rstrip('/')).name,
        creator_info={'name': 'acqstore', 'version': '0.1'} if zarr_format == 3 else None,
    )
    writer.write_full_volume(arr)
    if include_acqstore_pixels:
        _write_root_acqstore_pixels_attrs(path, pixels)


def _write_root_acqstore_pixels_attrs(path: str | Path, pixels: AcqPixels) -> None:
    """Embed strict acqstore pixel metadata for native stores."""
    group = _open_group(path, mode='a')
    group.attrs['acqstore_pixels'] = pixels.as_acqstore_metadata()


def _ngff_version_for_zarr_format(zarr_format: int) -> str:
    """Return OME-NGFF version for a Zarr format number."""
    try:
        return _OME_NGFF_VERSION_BY_FORMAT[int(zarr_format)]
    except (KeyError, ValueError) as exc:
        raise ValueError(f'zarr_format must be 2 or 3, got {zarr_format!r}') from exc


def _level_scale_vector(
    header: ImageHeader,
    base_shape: tuple[int, ...],
    level_shape: tuple[int, ...],
) -> list[float]:
    """Return the scale vector for one pyramid level."""
    base_scale = _scale_vector(header)
    if len(base_shape) != len(level_shape) or len(base_shape) != len(base_scale):
        raise ValueError(
            f'Level shape {level_shape!r}, base shape {base_shape!r}, and '
            f'scale {base_scale!r} must have equal rank'
        )
    out: list[float] = []
    for i, scale in enumerate(base_scale):
        if int(level_shape[i]) <= 0:
            raise ValueError(f'Invalid level shape {level_shape!r}')
        out.append(float(scale) * (float(base_shape[i]) / float(level_shape[i])))
    return out


def read_acq_pixels_ome_zarr(path: str | Path, *, lazy: bool = True) -> AcqPixels:
    """Read one OME-Zarr-compatible image store into :class:`AcqPixels`.

    Args:
        path: Local, local-ZIP, or S3-backed OME-Zarr store.
        lazy: When true, keep the full-resolution backing array lazy when the
            underlying Zarr implementation supports it.

    Returns:
        Loaded :class:`AcqPixels` wrapper.

    Raises:
        ValueError: If required OME-NGFF or native acqstore metadata is missing
            or malformed.
    """
    group = _open_group(path, mode='r')
    attrs = dict(group.attrs)
    multiscale = _first_multiscale(attrs)
    dataset_path = _dataset_path_from_multiscale(multiscale)
    arr = group[dataset_path]
    data: Any = arr if lazy else np.asarray(arr)
    levels = _read_pyramid_levels(group, multiscale, lazy=lazy)
    header = _header_from_attrs(path, attrs, arr, multiscale)
    return AcqPixels(
        data=data,
        header=header,
        source_path=str(path),
        acquisition_metadata={'ome_ngff': _ome_metadata(attrs)},
        source_metadata={'zarr_attrs': attrs},
        levels=levels,
    )


def _read_pyramid_levels(group: Any, multiscale: dict[str, Any], *, lazy: bool) -> tuple[Any, ...]:
    datasets = _datasets_from_multiscale(multiscale)
    if len(datasets) <= 1:
        return ()
    out: list[Any] = []
    for ds in datasets[1:]:
        path = _dataset_path_from_dataset(ds)
        arr = group[path]
        out.append(arr if lazy else np.asarray(arr))
    return tuple(out)


def _ome_metadata(attrs: dict[str, Any]) -> dict[str, Any] | list[Any]:
    """Return the OME metadata object from root attrs."""
    if 'ome' in attrs:
        ome = attrs['ome']
        if not isinstance(ome, dict):
            raise ValueError("OME-Zarr attr 'ome' must be an object")
        return ome
    multiscales = attrs['multiscales']
    return {'multiscales': multiscales}


def _first_multiscale(attrs: dict[str, Any]) -> dict[str, Any]:
    """Return the first multiscales entry from NGFF 0.4 or 0.5 attrs."""
    if 'ome' in attrs:
        ome = attrs['ome']
        if not isinstance(ome, dict):
            raise ValueError("OME-Zarr attr 'ome' must be an object")
        if 'multiscales' not in ome:
            raise ValueError("OME-Zarr attr 'ome' is missing required key 'multiscales'")
        multiscales = ome['multiscales']
    else:
        if 'multiscales' not in attrs:
            raise ValueError("OME-Zarr root attrs are missing required key 'multiscales'")
        multiscales = attrs['multiscales']
    if not isinstance(multiscales, list) or not multiscales:
        raise ValueError("OME-Zarr 'multiscales' must be a non-empty list")
    first = multiscales[0]
    if not isinstance(first, dict):
        raise ValueError("OME-Zarr first 'multiscales' entry must be an object")
    return first


def _datasets_from_multiscale(multiscale: dict[str, Any]) -> list[dict[str, Any]]:
    if 'datasets' not in multiscale:
        raise ValueError("OME-Zarr multiscales entry is missing required key 'datasets'")
    datasets = multiscale['datasets']
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("OME-Zarr 'datasets' must be a non-empty list")
    out: list[dict[str, Any]] = []
    for i, ds in enumerate(datasets):
        if not isinstance(ds, dict):
            raise ValueError(f"OME-Zarr dataset entry {i} must be an object")
        out.append(ds)
    return out


def _dataset_path_from_dataset(dataset: dict[str, Any]) -> str:
    if 'path' not in dataset:
        raise ValueError("OME-Zarr dataset entry is missing required key 'path'")
    path = dataset['path']
    if not isinstance(path, str) or not path:
        raise ValueError(f"OME-Zarr dataset path must be a non-empty string, got {path!r}")
    return path


def _dataset_path_from_multiscale(multiscale: dict[str, Any]) -> str:
    """Return level-0 dataset path from a validated multiscales entry."""
    return _dataset_path_from_dataset(_datasets_from_multiscale(multiscale)[0])


def _dataset_path_from_attrs(attrs: dict[str, Any]) -> str:
    """Return level-0 dataset path from root attrs.

    Args:
        attrs: Zarr root attributes.

    Returns:
        Dataset path for level 0.

    Raises:
        ValueError: If required OME-NGFF metadata is missing or malformed.
    """
    return _dataset_path_from_multiscale(_first_multiscale(attrs))


def _header_from_attrs(
    path: str | Path,
    attrs: dict[str, Any],
    arr: Any,
    multiscale: dict[str, Any],
) -> ImageHeader:
    acqstore_pixels = attrs.get('acqstore_pixels')
    if acqstore_pixels is not None:
        if not isinstance(acqstore_pixels, dict):
            raise ValueError("Native CS OME-Zarr attr 'acqstore_pixels' must be an object")
        if 'header' not in acqstore_pixels:
            raise ValueError("Native CS OME-Zarr attr 'acqstore_pixels' is missing key 'header'")
        raw_header = acqstore_pixels['header']
        if not isinstance(raw_header, dict):
            raise ValueError("Native CS OME-Zarr 'acqstore_pixels.header' must be an object")
        return _header_from_json_dict(path, raw_header)

    dims = _dims_from_multiscale(multiscale, int(arr.ndim))
    shape = tuple(int(x) for x in arr.shape)
    if len(dims) != len(shape):
        raise ValueError(f'OME-Zarr dims {dims!r} do not match array shape {shape!r}')
    sizes = {dims[i]: shape[i] for i in range(len(dims))}
    dtype = np.dtype(arr.dtype)
    num_channels = int(sizes['C']) if 'C' in sizes else 1
    physical_units, physical_units_labels = _physical_from_multiscale(multiscale, dims)
    return ImageHeader(
        path=str(path),
        shape=shape,
        dims=dims,
        sizes=sizes,
        dtype=dtype,
        num_channels=num_channels,
        num_scenes=1,
        physical_units=physical_units,
        physical_units_labels=physical_units_labels,
        date='',
        time='',
    )


def _require_key(raw: dict[str, Any], key: str, context: str) -> Any:
    if key not in raw:
        raise ValueError(f'{context} is missing required key {key!r}')
    return raw[key]


def _header_from_json_dict(path: str | Path, raw: dict[str, Any]) -> ImageHeader:
    """Build an :class:`ImageHeader` from strict native CS metadata."""
    context = 'Native CS OME-Zarr acqstore_pixels.header'
    shape_raw = _require_key(raw, 'shape', context)
    dims_raw = _require_key(raw, 'dims', context)
    sizes_raw = _require_key(raw, 'sizes', context)
    dtype_raw = _require_key(raw, 'dtype', context)
    num_channels_raw = _require_key(raw, 'num_channels', context)
    physical_units_raw = _require_key(raw, 'physical_units', context)
    physical_units_labels_raw = _require_key(raw, 'physical_units_labels', context)
    date_raw = _require_key(raw, 'date', context)
    time_raw = _require_key(raw, 'time', context)
    file_size_raw = _require_key(raw, 'file_size', context)

    if not isinstance(shape_raw, list) or not shape_raw:
        raise ValueError(f'{context} shape must be a non-empty list')
    shape = tuple(int(x) for x in shape_raw)
    if not isinstance(dims_raw, list) or len(dims_raw) != len(shape):
        raise ValueError(f'{context} dims must be a list matching shape rank')
    dims = tuple(str(x).upper() for x in dims_raw)
    if not isinstance(sizes_raw, dict):
        raise ValueError(f'{context} sizes must be an object')
    sizes = {str(k).upper(): int(v) for k, v in sizes_raw.items()}
    expected_sizes = {dims[i]: shape[i] for i in range(len(dims))}
    if sizes != expected_sizes:
        raise ValueError(f'{context} sizes {sizes!r} do not match dims/shape {expected_sizes!r}')
    if not isinstance(physical_units_raw, list) or len(physical_units_raw) != len(dims):
        raise ValueError(f'{context} physical_units must be a list matching dims')
    physical_units = tuple(_strict_float_unit(value, dims[i]) for i, value in enumerate(physical_units_raw))
    if not isinstance(physical_units_labels_raw, list) or len(physical_units_labels_raw) != len(dims):
        raise ValueError(f'{context} physical_units_labels must be a list matching dims')
    physical_units_labels = tuple(_strict_unit_label(value, dims[i]) for i, value in enumerate(physical_units_labels_raw))
    return ImageHeader(
        path=str(path),
        shape=shape,
        dims=dims,
        sizes=sizes,
        dtype=np.dtype(str(dtype_raw)),
        num_channels=int(num_channels_raw),
        num_scenes=1,
        physical_units=physical_units,
        physical_units_labels=physical_units_labels,
        date=str(date_raw),
        time=str(time_raw),
        file_size=str(file_size_raw),
    )


def _strict_float_unit(value: Any, axis: str) -> float:
    if value is None:
        raise ValueError(f'Physical scale for axis {axis!r} is missing')
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Physical scale for axis {axis!r} is not numeric: {value!r}') from exc
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError(f'Physical scale for axis {axis!r} must be finite and > 0, got {out!r}')
    return out


def _strict_unit_label(value: Any, axis: str) -> str:
    label = str(value).strip()
    if not label:
        raise ValueError(f'Physical unit label for axis {axis!r} is empty')
    return label


def _dims_from_multiscale(multiscale: dict[str, Any], ndim: int) -> tuple[str, ...]:
    if 'axes' not in multiscale:
        raise ValueError("OME-Zarr multiscales entry is missing required key 'axes'")
    axes = multiscale['axes']
    if not isinstance(axes, list) or len(axes) != ndim:
        raise ValueError(f"OME-Zarr 'axes' must be a list of length {ndim}")
    names: list[str] = []
    for i, axis in enumerate(axes):
        if not isinstance(axis, dict):
            raise ValueError(f'OME-Zarr axis entry {i} must be an object')
        if 'name' not in axis:
            raise ValueError(f"OME-Zarr axis entry {i} is missing required key 'name'")
        name = axis['name']
        if not isinstance(name, str) or not name:
            raise ValueError(f'OME-Zarr axis name at index {i} must be a non-empty string')
        names.append(name.upper())
    return tuple(names)


def _default_dims_for_ndim(ndim: int) -> tuple[str, ...]:
    """Return canonical dims for non-OME legacy callers.

    This helper is retained for older tests/imports, but OME-Zarr reading no
    longer calls it as a fallback.
    """
    if ndim == 2:
        return ('Y', 'X')
    if ndim == 3:
        return ('Z', 'Y', 'X')
    if ndim == 4:
        return ('C', 'Z', 'Y', 'X')
    if ndim == 5:
        return ('T', 'C', 'Z', 'Y', 'X')
    raise ValueError(f'Unsupported OME-Zarr array rank {ndim}; expected 2-5')


def _physical_from_multiscale(
    multiscale: dict[str, Any],
    dims: tuple[str, ...],
) -> tuple[tuple[float, ...], tuple[str, ...]]:
    axes = multiscale['axes']
    labels: list[str] = []
    for i, axis in enumerate(axes):
        if not isinstance(axis, dict):
            raise ValueError(f'OME-Zarr axis entry {i} must be an object')
        unit = axis.get('unit')
        labels.append('Pixels' if unit is None else _strict_unit_label(unit, dims[i]))

    datasets = _datasets_from_multiscale(multiscale)
    txs = datasets[0].get('coordinateTransformations')
    if not isinstance(txs, list) or not txs:
        raise ValueError("OME-Zarr level-0 dataset is missing coordinateTransformations")
    scale_values: list[Any] | None = None
    for tx in txs:
        if not isinstance(tx, dict):
            raise ValueError('OME-Zarr coordinate transformation entries must be objects')
        if tx.get('type') == 'scale':
            scale = tx.get('scale')
            if not isinstance(scale, list):
                raise ValueError("OME-Zarr scale transformation must contain list key 'scale'")
            scale_values = scale
            break
    if scale_values is None:
        raise ValueError("OME-Zarr level-0 dataset is missing a scale coordinate transformation")
    if len(scale_values) != len(dims):
        raise ValueError(
            f'OME-Zarr scale length {len(scale_values)} does not match dims {dims!r}'
        )
    units = tuple(_strict_float_unit(value, dims[i]) for i, value in enumerate(scale_values))
    return units, tuple(labels)
