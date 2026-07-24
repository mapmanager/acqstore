"""Tests for AcqImage TIFF export helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from acqstore.acq_image.acq_image import AcqImage
from acqstore.acq_image.io.tiff import save_pixels_as_tif
from acqstore.acq_image.acq_pixels import AcqPixels
from acqstore.acq_image.file_loaders.base_file_loader import ImageHeader, ReferenceImage


def _pixels(path: Path) -> AcqPixels:
    data = np.arange(3 * 4, dtype=np.uint16).reshape(3, 4)
    header = ImageHeader(
        path=str(path),
        shape=data.shape,
        dims=('Y', 'X'),
        sizes={'Y': 3, 'X': 4},
        dtype=data.dtype,
        num_channels=1,
        num_scenes=1,
        physical_units=(0.5, 0.25),
        physical_units_labels=('micrometer', 'micrometer'),
    )
    return AcqPixels(data=data, header=header, source_path=str(path))


def test_save_pixels_as_tif_writes_full_data(tmp_path: Path) -> None:
    """TIFF export should write the full array, not a display plane subset."""
    dest = tmp_path / 'export.tif'
    pixels = _pixels(dest)

    save_pixels_as_tif(pixels, dest)

    np.testing.assert_array_equal(tifffile.imread(dest), pixels.get_array())


def test_save_pixels_as_tif_refuses_existing_file_without_overwrite(tmp_path: Path) -> None:
    """TIFF export should fail fast rather than silently replacing files."""
    dest = tmp_path / 'export.tif'
    pixels = _pixels(dest)
    save_pixels_as_tif(pixels, dest)

    with pytest.raises(FileExistsError, match='already exists'):
        save_pixels_as_tif(pixels, dest)


def test_acq_image_save_as_tif_requires_explicit_filename(tmp_path: Path) -> None:
    """AcqImage TIFF export should use exactly the caller-provided filename."""
    src = tmp_path / 'source.tif'
    data = np.arange(12, dtype=np.uint16).reshape(3, 4)
    tifffile.imwrite(src, data)
    acq = AcqImage(str(src))
    dest = tmp_path / 'chosen_name.tif'

    acq.save_as_tif(dest)

    assert dest.is_file()
    np.testing.assert_array_equal(tifffile.imread(dest), data)


class _ReferenceLoader:
    """Minimal loader exposing an optional reference image."""

    def __init__(self, reference_image: ReferenceImage | None) -> None:
        self.reference_image = reference_image


def _acq_with_reference(
    source_path: Path,
    reference_image: ReferenceImage | None,
) -> AcqImage:
    """Build an AcqImage test instance around a reference-image snapshot."""
    acq = AcqImage.__new__(AcqImage)
    acq.path = str(source_path)
    acq._images = _ReferenceLoader(reference_image)
    return acq


def test_acq_image_save_reference_as_tif_writes_all_channels_and_calibration(
    tmp_path: Path,
) -> None:
    """Reference export preserves the complete CYX array and ImageJ XY scale."""
    data = np.arange(2 * 3 * 4, dtype=np.uint16).reshape(2, 3, 4)
    reference = ReferenceImage(
        array=data,
        dims=('C', 'Y', 'X'),
        num_channels=2,
        line_roi=None,
        coord_units=(('Y', 'um'), ('X', 'um')),
        coord_scales=(('Y', 0.5), ('X', 0.25)),
        coords=(),
    )
    acq = _acq_with_reference(tmp_path / 'source.czi', reference)
    dest = tmp_path / 'source-reference.tif'

    acq.save_reference_as_tif(dest)

    np.testing.assert_array_equal(tifffile.imread(dest), data)
    with tifffile.TiffFile(dest) as tif:
        metadata = dict(tif.imagej_metadata or {})
        x_res = tif.pages[0].tags['XResolution'].value
        y_res = tif.pages[0].tags['YResolution'].value
    assert metadata['unit'] == 'um'
    assert x_res[0] / x_res[1] == 4.0
    assert y_res[0] / y_res[1] == 2.0


def test_acq_image_save_reference_as_tif_does_not_render_scan_path(
    tmp_path: Path,
) -> None:
    """Reference export writes source pixels without burning in scan-path data."""
    data = np.zeros((4, 5), dtype=np.uint8)
    reference = ReferenceImage(
        array=data,
        dims=('Y', 'X'),
        num_channels=1,
        line_roi=(0.0, 0.0, 4.0, 3.0),
        coord_units=(('Y', 'um'), ('X', 'um')),
        coord_scales=(('Y', 1.0), ('X', 1.0)),
        coords=(),
        scan_path=np.asarray([[0.0, 4.0], [0.0, 3.0]]),
    )
    acq = _acq_with_reference(tmp_path / 'source.oir', reference)
    dest = tmp_path / 'source-reference.tif'

    acq.save_reference_as_tif(dest)

    np.testing.assert_array_equal(tifffile.imread(dest), data)


def test_acq_image_save_reference_as_tif_requires_reference(tmp_path: Path) -> None:
    """Reference export fails clearly when the acquisition has no reference."""
    acq = _acq_with_reference(tmp_path / 'source.nd2', None)

    with pytest.raises(ValueError, match='Acquisition has no reference image'):
        acq.save_reference_as_tif(tmp_path / 'source-reference.tif')


def test_acq_image_save_reference_as_tif_honors_overwrite(tmp_path: Path) -> None:
    """Reference export uses the standard TIFF overwrite contract."""
    reference = ReferenceImage(
        array=np.zeros((2, 3), dtype=np.uint8),
        dims=('Y', 'X'),
        num_channels=1,
        line_roi=None,
        coord_units=(('Y', 'um'), ('X', 'um')),
        coord_scales=(('Y', 1.0), ('X', 1.0)),
        coords=(),
    )
    acq = _acq_with_reference(tmp_path / 'source.oir', reference)
    dest = tmp_path / 'source-reference.tif'
    acq.save_reference_as_tif(dest)

    with pytest.raises(FileExistsError, match='already exists'):
        acq.save_reference_as_tif(dest)

    acq.save_reference_as_tif(dest, overwrite=True)
