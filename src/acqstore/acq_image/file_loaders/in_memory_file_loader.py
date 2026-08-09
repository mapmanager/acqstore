"""In-memory NumPy image loader for programmatically constructed acquisitions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .base_file_loader import BaseFileLoader, ImageHeader

SUPPORTED_IN_MEMORY_AXES = (
    ('Y', 'X'),
    ('C', 'Y', 'X'),
    ('Z', 'Y', 'X'),
    ('C', 'Z', 'Y', 'X'),
)


class InMemoryFileLoader(BaseFileLoader):
    """File-loader-compatible access to an in-memory real-valued NumPy array.

    Args:
        data: Nonempty NumPy array with a real integer or floating dtype.
        axes: Explicit dimensions, exactly YX, CYX, ZYX, or CZYX.
        source_id: Nonempty logical identity for the in-memory acquisition.
        axis_spacing: Optional finite positive spacing keyed by declared axis.
        axis_units: Optional nonempty unit labels keyed by declared axis.

    Raises:
        TypeError: If data is not a NumPy array.
        ValueError: If dtype, axes, shape, source identity, or metadata is invalid.
    """

    def __init__(
        self,
        data: np.ndarray,
        axes: Sequence[str],
        *,
        source_id: str,
        axis_spacing: Mapping[str, float] | None = None,
        axis_units: Mapping[str, str] | None = None,
    ) -> None:
        if not isinstance(data, np.ndarray):
            raise TypeError('data must be a NumPy array')
        if not (np.issubdtype(data.dtype, np.integer) or np.issubdtype(data.dtype, np.floating)):
            raise ValueError('data dtype must be a real NumPy integer or floating dtype')
        axes_tuple = tuple(axes)
        if axes_tuple not in SUPPORTED_IN_MEMORY_AXES:
            raise ValueError('axes must be exactly YX, CYX, ZYX, or CZYX')
        if data.ndim != len(axes_tuple):
            raise ValueError('data dimensions must match the explicitly supplied axes')
        if any(int(size) <= 0 for size in data.shape):
            raise ValueError('all data dimensions must be greater than zero')
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError('source_id must be a nonempty string')

        spacing = _validate_axis_spacing(axis_spacing, axes_tuple)
        units = _validate_axis_units(axis_units, axes_tuple)
        physical_units = tuple(spacing.get(axis, 1.0) for axis in axes_tuple)
        physical_labels = tuple(units.get(axis, 'Pixels') for axis in axes_tuple)
        shape = tuple(int(size) for size in data.shape)
        sizes = dict(zip(axes_tuple, shape, strict=True))
        path = f'memory://{source_id}'
        header = ImageHeader(
            path=path,
            shape=shape,
            dims=axes_tuple,
            sizes=sizes,
            dtype=np.dtype(data.dtype),
            num_channels=int(sizes.get('C', 1)),
            num_scenes=1,
            physical_units=physical_units,
            physical_units_labels=physical_labels,
        )
        self._source_data = data
        super().__init__(path, header)

    def read_header(self) -> ImageHeader:
        """Reject file-oriented header loading for an in-memory source.

        Raises:
            RuntimeError: Always; the header is injected during construction.
        """
        raise RuntimeError('InMemoryFileLoader receives its header during construction')

    def _load_full_image_array(self) -> np.ndarray:
        """Return the original in-memory array without copying."""
        return self._source_data


def _validate_axis_spacing(
    values: Mapping[str, float] | None,
    axes: tuple[str, ...],
) -> dict[str, float]:
    result = dict(values or {})
    undeclared = result.keys() - set(axes)
    if undeclared:
        raise ValueError(f'axis_spacing refers to undeclared axes: {sorted(undeclared)}')
    validated: dict[str, float] = {}
    for axis, raw_value in result.items():
        if isinstance(raw_value, bool):
            raise ValueError(f'axis_spacing[{axis!r}] must be finite and greater than zero')
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as error:
            raise ValueError(f'axis_spacing[{axis!r}] must be finite and greater than zero') from error
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'axis_spacing[{axis!r}] must be finite and greater than zero')
        validated[axis] = value
    return validated


def _validate_axis_units(
    values: Mapping[str, str] | None,
    axes: tuple[str, ...],
) -> dict[str, str]:
    result = dict(values or {})
    undeclared = result.keys() - set(axes)
    if undeclared:
        raise ValueError(f'axis_units refers to undeclared axes: {sorted(undeclared)}')
    for axis, value in result.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'axis_units[{axis!r}] must be a nonempty string')
    return result
