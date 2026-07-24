"""Tests for TIFF ImageJ metadata export behavior."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import tifffile

from acqstore.acq_image.acq_pixels import AcqPixels
from acqstore.acq_image.io.tiff import save_pixels_as_tif


def _imagej_metadata(path: Path) -> dict[str, object]:
    with tifffile.TiffFile(path) as tif:
        return dict(tif.imagej_metadata or {})


def test_tiff_export_writes_info_note_for_mixed_kymograph_units(
    tmp_path: Path,
    make_pixels: Callable[..., AcqPixels],
) -> None:
    """Mixed Y seconds / X micrometer data should not get false XY calibration."""
    path = tmp_path / 'mixed_units.tif'
    pixels = make_pixels(path)

    save_pixels_as_tif(pixels, path)

    metadata = _imagej_metadata(path)
    assert 'unit' not in metadata
    info = str(metadata['Info'])
    assert 'Saved by acqstore.' in info
    assert 'Y = 0.0005 seconds' in info
    assert 'X = 0.01 micrometer' in info
    np.testing.assert_array_equal(tifffile.imread(path), pixels.get_array(0))


def test_tiff_export_writes_imagej_resolution_for_shared_spatial_units(
    tmp_path: Path,
    make_pixels: Callable[..., AcqPixels],
) -> None:
    """Shared X/Y spatial units should be written as ImageJ calibration."""
    path = tmp_path / 'spatial_units.tif'
    pixels = make_pixels(
        path,
        physical_units=(0.5, 0.25),
        physical_units_labels=('micrometer', 'micrometer'),
    )

    save_pixels_as_tif(pixels, path)

    metadata = _imagej_metadata(path)
    assert metadata['unit'] == 'um'
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        x_res = page.tags['XResolution'].value
        y_res = page.tags['YResolution'].value
    assert x_res[0] / x_res[1] == 4.0
    assert y_res[0] / y_res[1] == 2.0
