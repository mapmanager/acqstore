"""Round-trip tests for pure OME-Zarr AcqPixels persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from acqstore.acq_image.acq_pixels import AcqPixels
from acqstore.acq_image.io.ome_zarr import (
    _dataset_path_from_attrs,
    read_acq_pixels_ome_zarr,
    write_acq_pixels_ome_zarr,
)


def test_pure_ome_zarr_round_trips_mixed_axis_physical_units(
    tmp_path: Path,
    make_pixels: Callable[..., AcqPixels],
) -> None:
    """Pure OME-Zarr must preserve Y=time and X=distance calibration."""
    path = tmp_path / 'sample.ome.zarr'
    pixels = make_pixels(path)

    write_acq_pixels_ome_zarr(
        pixels,
        path,
        include_acqstore_pixels=False,
        zarr_format=3,
    )
    loaded = read_acq_pixels_ome_zarr(path, lazy=False)

    assert loaded.axes == ('Y', 'X')
    assert loaded.shape == pixels.shape
    assert loaded.dtype == pixels.dtype
    assert loaded.header.physical_units == (0.0005, 0.01)
    assert loaded.header.physical_units_labels == ('seconds', 'micrometer')
    np.testing.assert_array_equal(loaded.get_array(0), pixels.get_array(0))


def test_pure_ome_zarr_round_trips_spatial_physical_units(
    tmp_path: Path,
    make_pixels: Callable[..., AcqPixels],
) -> None:
    """Pure OME-Zarr must preserve normal micrometer/micrometer calibration."""
    path = tmp_path / 'spatial.ome.zarr'
    pixels = make_pixels(
        path,
        physical_units=(0.5, 0.25),
        physical_units_labels=('micrometer', 'micrometer'),
    )

    write_acq_pixels_ome_zarr(
        pixels,
        path,
        include_acqstore_pixels=False,
        zarr_format=3,
    )
    loaded = read_acq_pixels_ome_zarr(path, lazy=False)

    assert loaded.header.physical_units == (0.5, 0.25)
    assert loaded.header.physical_units_labels == ('micrometer', 'micrometer')


def test_pure_ome_zarr_v2_round_trips_calibration(
    tmp_path: Path,
    make_pixels: Callable[..., AcqPixels],
) -> None:
    """Optional Zarr v2 / NGFF 0.4 export must preserve calibration too."""
    path = tmp_path / 'sample_v2.ome.zarr'
    pixels = make_pixels(path)

    write_acq_pixels_ome_zarr(
        pixels,
        path,
        include_acqstore_pixels=False,
        zarr_format=2,
    )
    loaded = read_acq_pixels_ome_zarr(path, lazy=False)

    assert loaded.header.physical_units == (0.0005, 0.01)
    assert loaded.header.physical_units_labels == ('seconds', 'micrometer')


def test_ome_zarr_missing_multiscales_fails_fast() -> None:
    """Reader must not invent dataset path or axes when NGFF metadata is absent."""
    with pytest.raises(ValueError, match='multiscales'):
        _dataset_path_from_attrs({})


def test_ome_zarr_invalid_scale_fails_fast(tmp_path: Path, make_pixels: Callable[..., AcqPixels]) -> None:
    """Malformed physical scale should fail instead of becoming 1.0 Pixels."""
    import zarr

    path = tmp_path / 'bad_scale.ome.zarr'
    pixels = make_pixels(path)
    write_acq_pixels_ome_zarr(pixels, path, include_acqstore_pixels=False)
    group = zarr.open_group(str(path), mode='a')
    attrs = dict(group.attrs)
    attrs['ome']['multiscales'][0]['datasets'][0]['coordinateTransformations'][0]['scale'] = [1.0, 'bad']
    group.attrs.update(attrs)

    with pytest.raises(ValueError, match='Physical scale'):
        read_acq_pixels_ome_zarr(path, lazy=False)
