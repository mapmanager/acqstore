"""Tests for explicit pixel-load APIs on AcqImage and BaseFileLoader."""

from __future__ import annotations

import numpy as np
import pytest

from acqstore.acq_image.acq_image import AcqImage
from acqstore.acq_image.file_loaders.base_file_loader import BaseFileLoader, ImageHeader


class _ArrayLoader(BaseFileLoader):
    """Minimal loader backed by an in-memory array for unit tests."""

    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr
        dims = ('Y', 'X')
        shape = tuple(int(x) for x in arr.shape)
        sizes = {'Y': shape[0], 'X': shape[1]}
        pu, pl = ImageHeader.default_physical_for_dims(dims)
        header = ImageHeader(
            path='/tmp/test_unit.tif',
            shape=shape,
            dims=dims,
            sizes=sizes,
            dtype=np.dtype(arr.dtype),
            num_channels=1,
            num_scenes=1,
            physical_units=pu,
            physical_units_labels=pl,
        )
        super().__init__(header.path, header)
        self.load_calls = 0

    def read_header(self) -> ImageHeader:
        raise NotImplementedError

    def _load_full_image_array(self) -> np.ndarray:
        self.load_calls += 1
        return self._arr


def test_pixels_loaded_false_until_load() -> None:
    loader = _ArrayLoader(np.arange(24, dtype=np.uint8).reshape(4, 6))
    assert loader.pixels_loaded() is False
    loader.load_image_data()
    assert loader.pixels_loaded() is True


def test_get_slice_data_loaded_fails_before_load() -> None:
    loader = _ArrayLoader(np.arange(24, dtype=np.uint8).reshape(4, 6))
    with pytest.raises(RuntimeError, match='not loaded'):
        loader.get_slice_data_loaded(0)


def test_get_slice_data_loaded_returns_plane_without_disk_read() -> None:
    volume = np.arange(24, dtype=np.uint8).reshape(4, 6)
    loader = _ArrayLoader(volume)
    loader._img_data = volume
    plane = loader.get_slice_data_loaded(0)
    assert plane.shape == (4, 6)
    assert loader.load_calls == 0


def test_acq_image_pixels_loaded_and_load_delegate() -> None:
    acq = AcqImage.__new__(AcqImage)
    acq._pixels = None

    class _Images:
        def __init__(self) -> None:
            self.load_calls = 0

        def load_pixels(self):
            self.load_calls += 1
            return object()

    images = _Images()
    acq._images = images  # type: ignore[assignment]
    assert acq.images_loaded is False
    acq.load_images()
    assert acq.images_loaded is True
    assert images.load_calls == 1
    assert acq.pixels_loaded() is True
