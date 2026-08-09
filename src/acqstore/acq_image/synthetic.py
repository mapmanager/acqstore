"""Generic deterministic synthetic acquisition pixels."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .file_loaders.in_memory_file_loader import SUPPORTED_IN_MEMORY_AXES


def synthetic_pixels(
    axes: Sequence[str],
    shape: Sequence[int],
    *,
    dtype: np.dtype | str | type = np.uint16,
) -> np.ndarray:
    """Create deterministic coordinate-coded pixels for a generic acquisition.

    The generator is intended to prove axis selection, spatial addressing, dtype,
    and metadata behavior. It is not a scientific simulation. Analysis-specific
    synthetic generators remain separate from this generic acquisition utility.

    Integer outputs use deterministic modular conversion from a 16-bit coordinate
    code. Floating outputs map that code into the inclusive range ``0..1``.

    Args:
        axes: Explicit dimensions, exactly YX, CYX, ZYX, or CZYX.
        shape: Positive sizes corresponding exactly to axes.
        dtype: Real NumPy integer or floating output dtype.

    Returns:
        A deterministic NumPy array with the requested shape and dtype.

    Raises:
        ValueError: If axes, shape, or dtype is unsupported.
    """
    axes_tuple = tuple(axes)
    shape_tuple = tuple(shape)
    if axes_tuple not in SUPPORTED_IN_MEMORY_AXES:
        raise ValueError('axes must be exactly YX, CYX, ZYX, or CZYX')
    if len(shape_tuple) != len(axes_tuple):
        raise ValueError('shape must contain one dimension for each axis')
    if any(not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in shape_tuple):
        raise ValueError('synthetic dimensions must be positive integers')
    resolved_dtype = np.dtype(dtype)
    is_integer = np.issubdtype(resolved_dtype, np.integer)
    is_floating = np.issubdtype(resolved_dtype, np.floating)
    if not (is_integer or is_floating):
        raise ValueError('dtype must be a real NumPy integer or floating dtype')

    values = np.full(shape_tuple, 9973, dtype=np.uint16)
    # Odd, pairwise-distinct weights remain distinguishable after conversion to
    # narrow integer dtypes such as uint8 while providing broad uint16 coverage.
    weights = {'C': 8191, 'Z': 4093, 'Y': 257, 'X': 17}
    for dimension, axis in enumerate(axes_tuple):
        coordinate_shape = [1] * len(shape_tuple)
        coordinate_shape[dimension] = shape_tuple[dimension]
        coordinate = np.arange(shape_tuple[dimension], dtype=np.uint64).reshape(coordinate_shape)
        contribution = ((coordinate * weights[axis]) % 65536).astype(np.uint16)
        np.add(values, contribution, out=values)
    if is_floating:
        return (values.astype(np.float64) / 65535.0).astype(resolved_dtype)
    return values.astype(resolved_dtype)
