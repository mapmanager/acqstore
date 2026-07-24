"""Tests for :class:`BaseFileLoader` ROI crops and physical calibration."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import tifffile

from acqstore.acq_image.acq_image import AcqImage
from acqstore.acq_image.file_loaders.base_file_loader import BaseFileLoader, ImageHeader
from acqstore.acq_image.roi import LineEndpoints, RectRoiBounds


class _ArrayLoader(BaseFileLoader):
    """Minimal loader backed by an in-memory array for unit tests."""

    def __init__(self, arr: np.ndarray, dims: tuple[str, ...], *, physical_units: tuple[float, ...] | None = None) -> None:
        self._arr = arr
        shape = tuple(int(x) for x in arr.shape)
        sizes = {dims[i]: shape[i] for i in range(len(dims))}
        num_channels = int(sizes['C']) if 'C' in sizes else 1
        if physical_units is None:
            pu, pl = ImageHeader.default_physical_for_dims(dims)
        else:
            pu = physical_units
            pl = tuple('µm' for _ in physical_units)
        header = ImageHeader(
            path='/tmp/test_unit.tif',
            shape=shape,
            dims=dims,
            sizes=sizes,
            dtype=np.dtype(arr.dtype),
            num_channels=num_channels,
            num_scenes=1,
            physical_units=pu,
            physical_units_labels=pl,
        )
        super().__init__(header.path, header)

    def read_header(self) -> ImageHeader:
        raise NotImplementedError

    def _load_full_image_array(self) -> np.ndarray:
        return self._arr


def test_get_image_physical_units_yx_order() -> None:
    """Steps follow Y then X axis order."""
    arr = np.zeros((4, 6), dtype=np.uint16)
    loader = _ArrayLoader(
        arr,
        ('Y', 'X'),
        physical_units=(0.5, 2.0),
    )
    assert loader.get_image_physical_units() == (0.5, 2.0)


def test_get_roi_rect_image_crop_matches_bounds() -> None:
    """Rectangular crop returns the expected sub-array."""
    arr = np.arange(20, dtype=np.uint16).reshape(4, 5)
    loader = _ArrayLoader(arr, ('Y', 'X'))
    bounds = RectRoiBounds(dim0_start=1, dim0_stop=3, dim1_start=2, dim1_stop=5)
    crop = loader.get_roi_rect_image(0, bounds)
    np.testing.assert_array_equal(crop, arr[1:3, 2:5])


def test_get_roi_rect_image_clamps_to_slice() -> None:
    """Out-of-slice bounds are clamped."""
    arr = np.ones((3, 3), dtype=np.uint8)
    loader = _ArrayLoader(arr, ('Y', 'X'))
    bounds = RectRoiBounds(dim0_start=-1, dim0_stop=99, dim1_start=0, dim1_stop=2)
    crop = loader.get_roi_rect_image(0, bounds)
    assert crop.shape == (3, 2)


def test_get_roi_rect_image_z_slice() -> None:
    """z selects the plane before row/col crop."""
    arr = np.stack(
        [
            np.zeros((2, 2), dtype=np.uint8),
            np.ones((2, 2), dtype=np.uint8) * 7,
        ],
        axis=0,
    )
    loader = _ArrayLoader(arr, ('Z', 'Y', 'X'))
    bounds = RectRoiBounds(0, 2, 0, 2)
    crop0 = loader.get_roi_rect_image(0, bounds, z=0)
    crop1 = loader.get_roi_rect_image(0, bounds, z=1)
    assert np.all(crop0 == 0)
    assert np.all(crop1 == 7)


def test_get_image_physical_units_raises_without_yx() -> None:
    """Missing Y or X in dims raises."""
    arr = np.zeros((2, 3), dtype=np.uint8)
    loader = _ArrayLoader(arr, ('A', 'B'))
    with pytest.raises(ValueError, match='Y and X'):
        loader.get_image_physical_units()


def test_acq_image_get_roi_image_uses_z0_slice(tmp_path: Path) -> None:
    """AcqImage ROI crop uses z=0 only for stacks."""
    z0 = np.arange(12, dtype=np.uint16).reshape(3, 4)
    z1 = np.zeros((3, 4), dtype=np.uint16)
    vol = np.stack([z0, z1], axis=0)
    path = tmp_path / 'stack.tif'
    tifffile.imwrite(path, vol, metadata={'axes': 'ZYX'}, photometric='minisblack')

    acq = AcqImage(str(path))
    rect = acq.rois.create_rect_roi(RectRoiBounds(1, 3, 0, 2))
    out = acq.get_roi_image(channel=0, roi_id=rect.roi_id)
    np.testing.assert_array_equal(out, z0[1:3, 0:2])


def test_acq_image_get_roi_image_line_roi_raises(tmp_path: Path) -> None:
    """Line ROI must not be passed to get_roi_image."""
    arr = np.zeros((4, 5), dtype=np.uint8)
    path = tmp_path / 'plane.tif'
    tifffile.imwrite(path, arr)

    acq = AcqImage(str(path))
    line = acq.rois.create_line_roi(LineEndpoints(0, 0, 1, 1))
    with pytest.raises(TypeError, match='rectangular'):
        acq.get_roi_image(channel=0, roi_id=line.roi_id)


def test_acq_image_get_roi_image_missing_roi_raises(tmp_path: Path) -> None:
    """Unknown roi_id raises ValueError."""
    arr = np.zeros((2, 2), dtype=np.uint8)
    path = tmp_path / 'tiny.tif'
    tifffile.imwrite(path, arr)

    acq = AcqImage(str(path))
    with pytest.raises(ValueError, match='not found'):
        acq.get_roi_image(channel=0, roi_id=999)


def test_acq_image_get_image_physical_units_matches_loader(tmp_path: Path) -> None:
    """AcqImage delegates physical units to the backing loader."""
    arr = np.zeros((2, 2), dtype=np.uint8)
    path = tmp_path / 'cal.tif'
    tifffile.imwrite(path, arr)

    acq = AcqImage(str(path))
    assert acq.get_image_physical_units() == acq.images.get_image_physical_units()


def test_sidecar_json_not_required_for_roi_api(tmp_path: Path) -> None:
    """AcqImage works when no sidecar exists (temp tif only)."""
    path = tmp_path / 'only.tif'
    tifffile.imwrite(path, np.ones((2, 3), dtype=np.uint16))
    acq = AcqImage(str(path))
    assert not Path(f'{path}.json').is_file()
    r = acq.rois.create_rect_roi(RectRoiBounds(0, 1, 0, 2))
    crop = acq.get_roi_image(0, r.roi_id)
    assert crop.shape == (1, 2)


def test_get_slice_data_loaded_missing_channel_axis_includes_context() -> None:
    """Multi-channel header without C axis should raise a descriptive ValueError."""
    arr = np.zeros((10, 8), dtype=np.uint16)
    loader = _ArrayLoader(arr, ('Y', 'X'))
    loader._header = replace(loader._header, num_channels=2)
    loader.load_image_data()

    with pytest.raises(ValueError, match="num_channels=2") as exc_info:
        loader.get_slice_data_loaded(0)

    message = str(exc_info.value)
    assert "test_unit.tif" in message
    assert "dims=('Y', 'X')" in message
    assert "loaded shape (10, 8)" in message
    assert "no C axis" in message
