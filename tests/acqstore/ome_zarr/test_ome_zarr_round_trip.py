"""Round-trip tests for pure OME-Zarr AcqPixels persistence."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from acqstore.acq_image.acq_pixels import AcqPixels
from acqstore.acq_image.acq_image import AcqImage
from acqstore.acq_image.io.ome_zarr import (
    _dataset_path_from_attrs,
    build_ome_zarr_chunk_shapes,
    build_ome_ngff_metadata,
    read_acq_pixels_ome_zarr,
    write_acq_pixels_ome_zarr,
)


@pytest.mark.parametrize(
    ("axes", "level_shapes", "expected"),
    [
        (
            ("Y", "X"),
            [(2500, 281), (1250, 140), (625, 70)],
            [(256, 256), (256, 140), (256, 70)],
        ),
        (
            ("C", "Y", "X"),
            [(2, 1024, 1024), (2, 512, 512)],
            [(1, 256, 256), (1, 256, 256)],
        ),
        (
            ("Z", "C", "Y", "X"),
            [(100, 2, 1024, 1024), (100, 2, 128, 128)],
            [(1, 1, 256, 256), (1, 1, 128, 128)],
        ),
    ],
)
def test_axis_aware_chunk_shapes_are_spatially_tiled(
    axes: tuple[str, ...],
    level_shapes: list[tuple[int, ...]],
    expected: list[tuple[int, ...]],
) -> None:
    """Chunks tile Y/X while isolating channel, depth, and time axes."""
    assert build_ome_zarr_chunk_shapes(level_shapes, axes) == expected


def test_established_ome_zarr_public_signatures_are_unchanged() -> None:
    """Web export policy must not appear on established public APIs."""
    def shape(callable_object: object) -> list[tuple[str, inspect._ParameterKind, object]]:
        return [
            (parameter.name, parameter.kind, parameter.default)
            for parameter in inspect.signature(callable_object).parameters.values()
        ]

    positional = inspect.Parameter.POSITIONAL_OR_KEYWORD
    keyword = inspect.Parameter.KEYWORD_ONLY
    required = inspect.Parameter.empty
    assert shape(AcqPixels.to_ome_zarr) == [
        ('self', positional, required),
        ('path', positional, required),
        ('overwrite', keyword, False),
        ('zarr_format', keyword, 3),
        ('include_acqstore_pixels', keyword, True),
    ]
    expected_acq_image = [
        ('self', positional, required),
        ('path', positional, required),
        ('overwrite', keyword, False),
        ('zarr_format', keyword, 3),
    ]
    assert shape(AcqImage.save_as_ome_zarr) == expected_acq_image
    assert shape(AcqImage.save_reference_as_ome_zarr) == expected_acq_image
    assert shape(write_acq_pixels_ome_zarr) == [
        ('pixels', positional, required),
        ('path', positional, required),
        ('overwrite', keyword, False),
        ('zarr_format', keyword, 3),
        ('include_acqstore_pixels', keyword, True),
    ]
    assert shape(build_ome_ngff_metadata) == [
        ('pixels', positional, required),
        ('zarr_format', keyword, 3),
    ]


def test_acq_image_temporal_y_export_preserves_existing_behavior(tmp_path: Path) -> None:
    """AcqImage temporal-Y pure export continues to retain its calibration."""
    acq = AcqImage.from_array(
        np.arange(6 * 4, dtype=np.uint16).reshape(6, 4),
        axes=('Y', 'X'),
        source_id='kymograph.tif',
        axis_spacing={'Y': 0.0005, 'X': 0.01},
        axis_units={'Y': 'seconds', 'X': 'micrometer'},
    )
    path = tmp_path / 'acq-image.ome.zarr'

    acq.save_as_ome_zarr(path)
    loaded = read_acq_pixels_ome_zarr(path, lazy=False)

    assert loaded.header.physical_units == (0.0005, 0.01)
    assert loaded.header.physical_units_labels == ('seconds', 'micrometer')


def test_pure_ome_zarr_round_trips_mixed_axis_physical_units(
    tmp_path: Path,
    make_pixels: Callable[..., AcqPixels],
) -> None:
    """Existing pure OME-Zarr behavior preserves Y=time calibration."""
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


@pytest.mark.parametrize(
    ('dims', 'shape', 'units', 'labels', 'expected_types'),
    [
        (('Y', 'X'), (6, 4), (0.5, 0.25), ('micrometer', 'micrometer'), ['space', 'space']),
        (
            ('C', 'Y', 'X'),
            (2, 6, 4),
            (1.0, 0.5, 0.25),
            ('Pixels', 'micrometer', 'micrometer'),
            ['channel', 'space', 'space'],
        ),
        (
            ('Z', 'C', 'Y', 'X'),
            (3, 2, 6, 4),
            (1.0, 1.0, 0.5, 0.25),
            ('micrometer', 'Pixels', 'micrometer', 'micrometer'),
            ['space', 'channel', 'space', 'space'],
        ),
    ],
)
def test_conventional_spatial_metadata_is_unchanged(
    tmp_path: Path,
    make_pixels: Callable[..., AcqPixels],
    dims: tuple[str, ...],
    shape: tuple[int, ...],
    units: tuple[float, ...],
    labels: tuple[str, ...],
    expected_types: list[str],
) -> None:
    """Conventional YX/CYX/ZCYX axes retain calibrated NGFF metadata."""
    pixels = make_pixels(
        tmp_path / 'spatial.ome.zarr',
        data=np.zeros(shape, dtype=np.uint16),
        dims=dims,
        physical_units=units,
        physical_units_labels=labels,
    )

    metadata = build_ome_ngff_metadata(pixels)

    assert [axis['type'] for axis in metadata['axes']] == expected_types
    assert metadata['datasets'][0]['coordinateTransformations'][0]['scale'] == list(units)
    for axis, label in zip(metadata['axes'], labels, strict=True):
        assert axis.get('unit') == (None if label == 'Pixels' else label)


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
    """Optional Zarr v2 / NGFF 0.4 export preserves existing calibration."""
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
