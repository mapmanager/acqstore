"""Pixel data and OME/NGFF-style acquisition metadata for one acquisition."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .file_loaders.base_file_loader import ImageHeader
from .roi import ImageBounds, RectRoiBounds


@dataclass(slots=True)
class AcqPixels:
    """Pixels plus acquisition metadata for one independent image acquisition.

    ``AcqPixels`` is the lowest-level acqstore image object. It intentionally
    excludes CloudScope/acqstore-specific ROIs, analysis results, view state, and
    user annotations. The metadata fields are shaped to map cleanly to OME-NGFF
    image metadata where the standard has a representation, while preserving
    source/vendor metadata in structured dictionaries when it does not.

    The backing arrays are intentionally typed as ``Any`` so instances can wrap
    NumPy arrays today and Zarr/Dask-backed arrays later without changing the
    public API.
    """

    data: Any
    header: ImageHeader
    source_path: str | None = None
    acquisition_metadata: Mapping[str, Any] = field(default_factory=dict)
    source_metadata: Mapping[str, Any] = field(default_factory=dict)
    levels: tuple[Any, ...] = ()

    @property
    def axes(self) -> tuple[str, ...]:
        """Return axis labels in the order of the stored full-resolution array."""
        return self.header.dims

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the full-resolution pixel shape."""
        return tuple(int(x) for x in self.header.shape)

    @property
    def dtype(self) -> np.dtype:
        """Return the full-resolution pixel dtype."""
        return np.dtype(self.header.dtype)

    @property
    def num_channels(self) -> int:
        """Return the number of channels exposed by this acquisition."""
        return int(self.header.num_channels)

    @property
    def channel_indices(self) -> list[int]:
        """Return zero-based channel indices."""
        return list(range(self.num_channels))

    @property
    def default_channel(self) -> int | None:
        """Return channel ``0`` when channels exist, otherwise ``None``."""
        return 0 if self.num_channels > 0 else None

    def get_array(self, level: int = 0) -> Any:
        """Return the array-like object for one pyramid level.

        Args:
            level: Pyramid level. Level ``0`` is full resolution.

        Returns:
            The backing array-like object. This may be NumPy, Zarr, Dask, or any
            object implementing shape/dtype and NumPy-compatible slicing.

        Raises:
            IndexError: If ``level`` is invalid.
        """
        if level == 0:
            return self.data
        index = level - 1
        if index < 0 or index >= len(self.levels):
            raise IndexError(f"Pyramid level {level} out of range")
        return self.levels[index]

    def get_plane(
        self,
        *,
        t: int = 0,
        c: int = 0,
        z: int = 0,
        level: int = 0,
        as_numpy: bool = True,
    ) -> Any:
        """Return one ``(Y, X)`` plane after optional ``T``/``Z``/``C`` selection.

        Args:
            t: Time index when a ``T`` axis exists.
            c: Channel index when a ``C`` axis exists.
            z: Z index when a ``Z`` axis exists.
            level: Pyramid level, where ``0`` is full resolution.
            as_numpy: When true, coerce the result to ``numpy.ndarray``. When
                false, return the sliced array-like object when possible.

        Returns:
            Two-dimensional plane in ``(Y, X)`` order.
        """
        arr = self.get_array(level)
        dims_list = list(self.axes)
        if "Y" not in dims_list or "X" not in dims_list:
            raise ValueError(f"Expected axes to include Y and X; got axes={self.axes}")
        if c < 0 or c >= self.num_channels:
            raise IndexError(f"Channel index {c} out of range for {self.num_channels} channels")
        if "C" not in dims_list and c != 0:
            raise IndexError(f"No C axis; only channel 0 is valid, got {c}")

        arr_work = arr

        def _take_and_drop(dim_label: str, index: int) -> None:
            nonlocal arr_work, dims_list
            if dim_label not in dims_list:
                return
            axis = dims_list.index(dim_label)
            size = int(arr_work.shape[axis])
            if index < 0 or index >= size:
                raise IndexError(
                    f"{dim_label} index {index} out of range for size {size} "
                    f"(axes={tuple(dims_list)})"
                )
            slicer: list[object] = [slice(None)] * len(dims_list)
            slicer[axis] = index
            arr_work = arr_work[tuple(slicer)]
            dims_list.pop(axis)

        _take_and_drop("T", t)
        _take_and_drop("Z", z)
        _take_and_drop("C", c)

        if tuple(dims_list) != ("Y", "X"):
            raise ValueError(
                f"After selecting T/Z/C, expected remaining axes (Y, X); "
                f"got {tuple(dims_list)} with shape {arr_work.shape}"
            )
        return np.asarray(arr_work) if as_numpy else arr_work

    def get_crop(
        self,
        bounds: RectRoiBounds,
        *,
        t: int = 0,
        c: int = 0,
        z: int = 0,
        level: int = 0,
        as_numpy: bool = True,
    ) -> Any:
        """Return a ``(Y, X)`` crop for one plane using rectangular bounds."""
        plane = self.get_plane(t=t, c=c, z=z, level=level, as_numpy=as_numpy)
        img_bounds = ImageBounds(
            width=int(plane.shape[1]),
            height=int(plane.shape[0]),
            num_slices=1,
        )
        b = bounds.clamped_to(img_bounds)
        cropped = plane[b.dim0_start : b.dim0_stop, b.dim1_start : b.dim1_stop]
        return np.asarray(cropped) if as_numpy else cropped

    def get_channel_data(self, channel: int, *, as_numpy: bool = True) -> Any:
        """Return the full array for one channel, with the ``C`` axis removed."""
        if channel < 0 or channel >= self.num_channels:
            raise IndexError(f"Channel index {channel} out of range for {self.num_channels} channels")
        arr = self.get_array(0)
        if "C" not in self.axes:
            if self.num_channels != 1:
                raise ValueError(f"No C axis in axes but num_channels={self.num_channels}")
            if channel != 0:
                raise IndexError(f"No C axis; only channel 0 is valid, got {channel}")
            return np.asarray(arr) if as_numpy else arr
        c_axis = self.axes.index("C")
        slicer: list[object] = [slice(None)] * len(self.axes)
        slicer[c_axis] = channel
        out = arr[tuple(slicer)]
        return np.asarray(out) if as_numpy else out

    def get_image_physical_units(self) -> tuple[float, float]:
        """Return physical step along ``Y`` then ``X`` for image planes."""
        dims = self.axes
        if "Y" not in dims or "X" not in dims:
            raise ValueError(f"Expected axes to include Y and X; got axes={dims!r}")
        i_y = dims.index("Y")
        i_x = dims.index("X")
        return (float(self.header.physical_units[i_y]), float(self.header.physical_units[i_x]))

    def to_ome_zarr(
        self,
        path: str | Path,
        *,
        overwrite: bool = False,
        zarr_format: int = 3,
        include_acqstore_pixels: bool = True,
    ) -> None:
        """Write this acquisition as a single-image OME-Zarr-compatible store.

        Args:
            path: Destination OME-Zarr store path. Local directories, local ZIP
                stores, and ``s3://`` stores are supported by the writer backend.
            overwrite: Whether to replace an existing local destination.
            zarr_format: Target Zarr format. ``3`` writes NGFF 0.5; ``2``
                writes NGFF 0.4.
            include_acqstore_pixels: When true, embed a lightweight acqstore
                header snapshot for round-tripping.

        Returns:
            None.
        """
        from .io.ome_zarr import write_acq_pixels_ome_zarr

        write_acq_pixels_ome_zarr(
            self,
            path,
            overwrite=overwrite,
            zarr_format=zarr_format,
            include_acqstore_pixels=include_acqstore_pixels,
        )

    @classmethod
    def from_ome_zarr(cls, path: str | Path, *, lazy: bool = True) -> "AcqPixels":
        """Load pixels and metadata from one OME-Zarr-compatible image store."""
        from .io.ome_zarr import read_acq_pixels_ome_zarr

        return read_acq_pixels_ome_zarr(path, lazy=lazy)

    def as_acqstore_metadata(self) -> dict[str, Any]:
        """Return acqstore-owned metadata for embedding in native stores."""
        return {
            "header": self.header.as_json_dict(),
            "source_path": self.source_path,
            "acquisition_metadata": dict(self.acquisition_metadata),
            "source_metadata": dict(self.source_metadata),
        }
