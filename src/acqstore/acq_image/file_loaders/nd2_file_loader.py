"""ND2 file loader for CloudScope acquisition images.

This module contains the CloudScope/acqstore adapter for Nikon ND2 files using
:mod:`nd2`. Header metadata are read at construction time without materializing
pixels. Pixel arrays are loaded lazily through :meth:`BaseFileLoader.load_image_data`.

Version 1 supports one acquisition position. When the ND2 file contains a
position axis, pixels are loaded from position ``0`` and the position axis is
excluded from the exposed image header.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, BinaryIO

import nd2
import numpy as np

from .base_file_loader import BaseFileLoader, ImageHeader


_ND2_POSITION_DIM = "P"
_SPATIAL_DIMS = {"X", "Y", "Z"}
_SPATIAL_UNIT_LABEL = "um"


class Nd2FileLoader(BaseFileLoader):
    """Lazy-loading Nikon ND2 reader using :mod:`nd2`.

    Args:
        path: Filesystem path to the ND2 file.
        header: Optional pre-built header.
    """

    @contextmanager
    def _open_nd2(self) -> Iterator[Any]:
        """Yield an open ``nd2.ND2File`` from :attr:`path` or :attr:`_stream`.

        Yields:
            Open ``nd2.ND2File`` instance.
        """
        if self._stream is not None:
            self._stream.seek(0)
            with nd2.ND2File(self._stream) as nd2_file:
                yield nd2_file
        else:
            with nd2.ND2File(self.path) as nd2_file:
                yield nd2_file

    @classmethod
    def read_header_from_stream(cls, stream: BinaryIO, filename: str) -> ImageHeader:
        """Read ND2 header from a stream without retaining the loader.

        Args:
            stream: Binary stream positioned at an ND2 file.
            filename: Logical filename to store on the returned header.

        Returns:
            Header for the ND2 file.
        """
        return cls.from_stream(stream, filename).header

    def read_header(self) -> ImageHeader:
        """Read ND2 image header metadata.

        Returns:
            Header for the default acquisition position.
        """
        with self._open_nd2() as nd2_file:
            return _image_header_from_nd2_file(self.path, nd2_file)

    def _load_full_image_array(self) -> np.ndarray:
        """Load full pixel data from the ND2 default position.

        Returns:
            NumPy array containing ND2 pixels for position ``0`` when the file
            has multiple positions, otherwise the file's full pixel array.

        Raises:
            ValueError: If loaded pixel shape does not match the parsed header.
        """
        with self._open_nd2() as nd2_file:
            if self._header.num_scenes > 1:
                pixels = np.asarray(nd2_file.asarray(position=0))
            else:
                pixels = np.asarray(nd2_file.asarray())

        loaded_shape = tuple(int(x) for x in pixels.shape)
        if loaded_shape != self._header.shape:
            raise ValueError(
                "ND2 loaded pixel shape does not match header: "
                f"loaded {loaded_shape}, expected {self._header.shape} for {self.path!r}"
            )
        return pixels


def _image_header_from_nd2_file(path: str, nd2_file: Any) -> ImageHeader:
    """Build an :class:`ImageHeader` from an open ``nd2.ND2File``.

    Args:
        path: Filesystem path or logical source name.
        nd2_file: Open ``nd2.ND2File``-like object.

    Returns:
        Header describing the default acquisition position.

    Raises:
        ValueError: If ND2 axis metadata is missing or inconsistent.
    """
    raw_sizes = {str(dim): int(size) for dim, size in dict(nd2_file.sizes).items()}
    if not raw_sizes:
        raise ValueError(f"ND2 file {path!r} did not report image axis sizes")

    num_scenes = int(raw_sizes.get(_ND2_POSITION_DIM, 1))
    sizes = {dim: size for dim, size in raw_sizes.items() if dim != _ND2_POSITION_DIM}
    dims = tuple(sizes.keys())
    shape = tuple(int(sizes[dim]) for dim in dims)

    raw_shape = tuple(int(x) for x in nd2_file.shape)
    expected_raw_shape = tuple(int(raw_sizes[dim]) for dim in raw_sizes)
    if raw_shape != expected_raw_shape:
        raise ValueError(
            "ND2 shape does not match sizes metadata: "
            f"shape {raw_shape}, sizes-derived shape {expected_raw_shape} for {path!r}"
        )

    dtype = np.dtype(nd2_file.dtype)
    physical_units, physical_units_labels = _physical_calibration_for_dims(dims, nd2_file)

    return ImageHeader(
        path=path,
        shape=shape,
        dims=dims,
        sizes=sizes,
        dtype=dtype,
        num_channels=int(sizes.get("C", 1)),
        num_scenes=num_scenes,
        physical_units=physical_units,
        physical_units_labels=physical_units_labels,
    )


def _physical_calibration_for_dims(
    dims: tuple[str, ...],
    nd2_file: Any,
) -> tuple[tuple[float, ...], tuple[str, ...]]:
    """Return per-axis calibration aligned to ND2 dimensions.

    Args:
        dims: ND2 axis labels excluding the position axis.
        nd2_file: Open ``nd2.ND2File``-like object.

    Returns:
        ``(physical_units, physical_units_labels)`` aligned to ``dims``.
    """
    defaults, default_labels = ImageHeader.default_physical_for_dims(dims)
    voxel_size = nd2_file.voxel_size()
    spatial_steps = {
        "X": getattr(voxel_size, "x", None),
        "Y": getattr(voxel_size, "y", None),
        "Z": getattr(voxel_size, "z", None),
    }

    units: list[float] = []
    labels: list[str] = []
    for index, dim in enumerate(dims):
        if dim in _SPATIAL_DIMS:
            value = spatial_steps.get(dim)
            try:
                step = float(value)
            except (TypeError, ValueError):
                step = float(defaults[index])
            units.append(step)
            labels.append(_SPATIAL_UNIT_LABEL)
            continue
        units.append(float(defaults[index]))
        labels.append(default_labels[index])

    return tuple(units), tuple(labels)
