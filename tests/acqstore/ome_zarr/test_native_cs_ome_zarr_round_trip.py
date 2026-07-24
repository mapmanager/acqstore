"""Round-trip tests for native CloudScope/acqstore OME-Zarr stores."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from acqstore.acq_image.acq_pixels import AcqPixels
from acqstore.acq_image.io.ome_zarr import read_acq_pixels_ome_zarr, write_acq_pixels_ome_zarr


def test_native_cs_ome_zarr_header_round_trips_strictly(
    tmp_path: Path,
    make_pixels: Callable[..., AcqPixels],
) -> None:
    """Native header metadata must round-trip without default fallbacks."""
    path = tmp_path / 'sample.cs.ome.zarr'
    pixels = make_pixels(path)

    write_acq_pixels_ome_zarr(pixels, path, include_acqstore_pixels=True)
    loaded = read_acq_pixels_ome_zarr(path, lazy=False)

    assert loaded.header.dims == ('Y', 'X')
    assert loaded.header.shape == (6, 4)
    assert loaded.header.dtype == np.dtype('uint16')
    assert loaded.header.num_channels == 1
    assert loaded.header.date == '20260709'
    assert loaded.header.time == '12:34:56'
    assert loaded.header.file_size == '24 bytes'
    assert loaded.header.physical_units == (0.0005, 0.01)
    assert loaded.header.physical_units_labels == ('seconds', 'micrometer')


def test_native_cs_ome_zarr_missing_header_key_fails_fast(
    tmp_path: Path,
    make_pixels: Callable[..., AcqPixels],
) -> None:
    """Native CS metadata is owned by acqstore and must be complete."""
    import zarr

    path = tmp_path / 'missing_header_key.cs.ome.zarr'
    pixels = make_pixels(path)
    write_acq_pixels_ome_zarr(pixels, path, include_acqstore_pixels=True)
    group = zarr.open_group(str(path), mode='a')
    acqstore_pixels = dict(group.attrs['acqstore_pixels'])
    header = dict(acqstore_pixels['header'])
    del header['dtype']
    acqstore_pixels['header'] = header
    group.attrs['acqstore_pixels'] = acqstore_pixels

    with pytest.raises(ValueError, match="'dtype'"):
        read_acq_pixels_ome_zarr(path, lazy=False)
