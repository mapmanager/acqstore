"""Tests for the AcqPixels core image object."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from acqstore.acq_image.acq_pixels import AcqPixels
from acqstore.acq_image.file_loaders.base_file_loader import ImageHeader
from acqstore.acq_image.file_loaders.tiff_file_loader import TiffFileLoader


def _header(path: Path, shape: tuple[int, ...], dims: tuple[str, ...]) -> ImageHeader:
    physical_units, physical_units_labels = ImageHeader.default_physical_for_dims(dims)
    sizes = {dims[i]: int(shape[i]) for i in range(len(dims))}
    return ImageHeader(
        path=str(path),
        shape=shape,
        dims=dims,
        sizes=sizes,
        dtype=np.dtype("uint16"),
        num_channels=int(sizes.get("C", 1)),
        num_scenes=1,
        physical_units=physical_units,
        physical_units_labels=physical_units_labels,
    )


def test_acq_pixels_get_plane_selects_tzc_to_yx(tmp_path: Path) -> None:
    data = np.arange(2 * 3 * 4 * 5 * 6, dtype=np.uint16).reshape(2, 3, 4, 5, 6)
    pixels = AcqPixels(data=data, header=_header(tmp_path / "x.tif", data.shape, ("T", "C", "Z", "Y", "X")))

    plane = pixels.get_plane(t=1, c=2, z=3)

    np.testing.assert_array_equal(plane, data[1, 2, 3, :, :])


def test_acq_pixels_rejects_invalid_channel(tmp_path: Path) -> None:
    data = np.zeros((2, 5, 6), dtype=np.uint16)
    pixels = AcqPixels(data=data, header=_header(tmp_path / "x.tif", data.shape, ("C", "Y", "X")))

    with pytest.raises(IndexError, match="Channel index"):
        pixels.get_plane(c=3)


def test_base_file_loader_load_pixels_wraps_existing_loader(tmp_path: Path) -> None:
    path = tmp_path / "x.tif"
    import tifffile

    tifffile.imwrite(path, np.zeros((4, 5), dtype=np.uint8))
    loader = TiffFileLoader(str(path), load_olympus_header=False)

    pixels = loader.load_pixels()

    assert isinstance(pixels, AcqPixels)
    assert pixels.axes == ("Y", "X")
    assert pixels.shape == (4, 5)
    np.testing.assert_array_equal(pixels.get_plane(), np.zeros((4, 5), dtype=np.uint8))
