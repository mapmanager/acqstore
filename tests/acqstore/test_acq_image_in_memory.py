"""Tests for additive in-memory and generic synthetic AcqImage construction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from acqstore.acq_image import AcqImage
from acqstore.acq_image.synthetic import synthetic_pixels


@pytest.mark.parametrize(
    ('axes', 'shape', 'plane_kwargs'),
    [
        (('Y', 'X'), (5, 7), {}),
        (('C', 'Y', 'X'), (2, 5, 7), {'c': 1}),
        (('Z', 'Y', 'X'), (3, 5, 7), {'z': 2}),
        (('C', 'Z', 'Y', 'X'), (2, 3, 5, 7), {'c': 1, 'z': 2}),
    ],
)
def test_from_array_supports_explicit_plane_organizations(
    axes: tuple[str, ...],
    shape: tuple[int, ...],
    plane_kwargs: dict[str, int],
) -> None:
    data = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    acq = AcqImage.from_array(data, axes=axes, source_id='array-source')

    assert acq.is_memory_backed is True
    assert acq.file_id == 'memory://array-source'
    assert acq.name == 'array-source'
    assert acq.pixels.data is data
    assert acq.pixels.axes == axes
    assert acq.pixels.shape == shape
    plane = acq.pixels.get_plane(**plane_kwargs)
    assert plane.shape == shape[-2:]


@pytest.mark.parametrize('dtype', [np.uint8, np.uint16, np.int16, np.int32, np.float32, np.float64])
def test_from_synthetic_preserves_supported_dtype(dtype: type[np.generic]) -> None:
    acq = AcqImage.from_synthetic(
        (2, 3, 5, 7),
        axes=('C', 'Z', 'Y', 'X'),
        source_id=f'synthetic-{np.dtype(dtype).name}',
        dtype=dtype,
    )

    assert acq.pixels.data.dtype == np.dtype(dtype)
    assert acq.pixels.get_plane(c=1, z=2).shape == (5, 7)
    if np.issubdtype(np.dtype(dtype), np.floating):
        assert float(acq.pixels.data.min()) >= 0.0
        assert float(acq.pixels.data.max()) <= 1.0


def test_synthetic_pixels_are_deterministic_and_encode_selectors() -> None:
    first = synthetic_pixels(('C', 'Z', 'Y', 'X'), (2, 3, 5, 7), dtype=np.uint8)
    second = synthetic_pixels(('C', 'Z', 'Y', 'X'), (2, 3, 5, 7), dtype=np.uint8)

    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first[0, 0], first[1, 0])
    assert not np.array_equal(first[0, 0], first[0, 1])
    assert first[0, 0, 0, 0] != first[0, 0, 0, 1]
    assert first[0, 0, 0, 0] != first[0, 0, 1, 0]


def test_from_array_preserves_noncontiguous_data_without_copying() -> None:
    data = np.arange(48, dtype=np.uint8).reshape(6, 8)[:, ::2]
    assert not data.flags.c_contiguous

    acq = AcqImage.from_array(data, axes=('Y', 'X'), source_id='noncontiguous')

    assert acq.pixels.data is data
    assert not acq.pixels.data.flags.c_contiguous


def test_from_array_can_defer_normalized_pixels_wrapper() -> None:
    data = np.zeros((4, 6), dtype=np.float32)
    acq = AcqImage.from_array(
        data,
        axes=('Y', 'X'),
        source_id='lazy-wrapper',
        load_images=False,
    )

    assert acq.images_loaded is False
    acq.load_images()
    assert acq.images_loaded is True
    assert acq.pixels.data is data


def test_axis_spacing_and_units_populate_image_header() -> None:
    acq = AcqImage.from_synthetic(
        (20, 10),
        axes=('Y', 'X'),
        source_id='line-scan',
        dtype=np.uint8,
        axis_spacing={'Y': 0.001, 'X': 0.2},
        axis_units={'Y': 's', 'X': 'um'},
    )

    assert acq.get_image_physical_units() == (0.001, 0.2)
    assert acq.images.header.physical_units_labels == ('s', 'um')


def test_in_memory_implicit_persistence_is_rejected(tmp_path: Path) -> None:
    acq = AcqImage.from_synthetic(
        (5, 7),
        axes=('Y', 'X'),
        source_id='no-sidecar',
    )

    with pytest.raises(RuntimeError, match='in-memory acquisition'):
        acq.save()
    with pytest.raises(RuntimeError, match='in-memory acquisition'):
        acq.get_sidecar_json_path()
    with pytest.raises(RuntimeError, match='in-memory acquisition'):
        acq.load_analysis_csv()

    destination = tmp_path / 'explicit.tif'
    acq.save_as_tif(destination)
    assert destination.is_file()
    assert tifffile.imread(destination).shape == (5, 7)


@pytest.mark.parametrize('dtype', [np.bool_, np.complex64, object, 'U2', 'datetime64[ns]'])
def test_synthetic_rejects_non_real_image_dtype(dtype) -> None:
    with pytest.raises(ValueError, match='real NumPy integer or floating'):
        AcqImage.from_synthetic(
            (2, 3),
            axes=('Y', 'X'),
            source_id='bad-dtype',
            dtype=dtype,
        )


@pytest.mark.parametrize(
    ('axes', 'shape'),
    [
        (('X', 'Y'), (2, 3)),
        (('T', 'Y', 'X'), (2, 3, 4)),
        (('Y', 'X'), (2,)),
        (('Y', 'X'), (2, 0)),
    ],
)
def test_synthetic_rejects_unsupported_axes_and_shape(axes, shape) -> None:
    with pytest.raises(ValueError):
        AcqImage.from_synthetic(shape, axes=axes, source_id='bad-shape')


def test_from_array_rejects_metadata_for_undeclared_axes() -> None:
    with pytest.raises(ValueError, match='undeclared'):
        AcqImage.from_array(
            np.zeros((2, 3), dtype=np.uint8),
            axes=('Y', 'X'),
            source_id='bad-metadata',
            axis_spacing={'Z': 1.0},
        )
