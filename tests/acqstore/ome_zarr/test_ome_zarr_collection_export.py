"""Tests for multi-image AcqStore OME-Zarr collection export."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from acqstore.acq_image.acq_image import AcqImage
from acqstore.acq_image.acq_image_list import AcqImageList
from acqstore.acq_image.analysis.velocity_analysis.radon_velocity_analysis import (
    RadonVelocityAnalysis,
)
from acqstore.acq_image.analysis.model import AnalysisKey
from acqstore.acq_image.file_loaders.base_file_loader import ReferenceImage
from acqstore.acq_image.io.ome_zarr import read_acq_pixels_ome_zarr
from acqstore.acq_image.io.ome_zarr_collection import (
    COLLECTION_FORMAT,
    export_acq_image_list_ome_zarr,
    write_acq_image_native_ome_zarr,
)


def _collection(*images: AcqImage) -> AcqImageList:
    """Build an in-memory collection around explicit acquisition images."""
    collection = AcqImageList.__new__(AcqImageList)
    collection.path = 'memory://collection'
    collection.source_root_path = None
    collection.file_list = [image.path for image in images]
    collection._files = list(images)
    collection._files_by_id = {image.file_id: image for image in images}
    collection._attach_analysis_pools()
    return collection


def _image(
    name: str,
    shape: tuple[int, ...],
    axes: tuple[str, ...],
    *,
    dtype: str | np.dtype[np.generic] = 'uint16',
) -> AcqImage:
    """Build one calibrated in-memory acquisition fixture."""
    data = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    spacing = {axis: 1.0 for axis in axes}
    units = {axis: 'Pixels' for axis in axes}
    for axis in ('Z', 'Y', 'X'):
        if axis in axes:
            units[axis] = 'micrometer'
    return AcqImage.from_array(
        data,
        axes=axes,
        source_id=name,
        axis_spacing=spacing,
        axis_units=units,
    )


def _attach_reference_image(image: AcqImage) -> ReferenceImage:
    """Attach one calibrated reference snapshot to an in-memory fixture."""
    scan_path = np.asarray([[273.0, 261.0], [222.0, 228.0]])
    reference = ReferenceImage(
        array=np.arange(512 * 512, dtype=np.uint16).reshape(512, 512),
        dims=('Y', 'X'),
        num_channels=1,
        line_roi=(273.0, 222.0, 261.0, 228.0),
        coord_units=(('Y', 'um'), ('X', 'um')),
        coord_scales=(('Y', 0.38098425710482), ('X', 0.38098425710482)),
        coords=(),
        scan_path=scan_path,
    )
    image.images._referenceImage = reference
    return reference


def _tree_bytes(root: Path) -> dict[str, bytes]:
    """Return every file in a directory tree keyed by relative POSIX path."""
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in sorted(root.rglob('*')) if path.is_file()}


def test_exports_heterogeneous_native_images_in_one_zarr_hierarchy(tmp_path: Path) -> None:
    """Heterogeneous members remain independent images in one hierarchy."""
    images = (
        _image('yx.tif', (64, 64), ('Y', 'X')),
        _image('cyx.tif', (2, 48, 32), ('C', 'Y', 'X'), dtype=np.dtype('uint8')),
        _image('czyx.tif', (2, 3, 20, 18), ('C', 'Z', 'Y', 'X')),
    )
    velocity_analysis = RadonVelocityAnalysis(channel=0, roi_id=1)
    velocity_analysis.result.table = pd.DataFrame({'time': [0.0, 1.0], 'velocity': [2.5, 3.5]})
    velocity_analysis.result.summary = {'velocity_mean': 3.0}
    images[0].analysis_set.add(velocity_analysis)
    images[0].analysis_set._results_csv_loaded = True
    collection = _collection(*images)
    destination = tmp_path / 'dataset.ome.zarr'

    result = export_acq_image_list_ome_zarr(collection, destination)

    assert result == destination.resolve()
    manifest = json.loads((destination / 'acqstore' / 'acq_image_collection.json').read_text())
    assert manifest['format'] == COLLECTION_FORMAT
    assert manifest['version'] == 1
    assert manifest['zarr_format'] == 3
    assert manifest['name'] == 'dataset.ome.zarr'
    assert manifest['created_utc'].endswith('Z')
    assert isinstance(manifest['acqstore_version'], str)
    assert [entry['id'] for entry in manifest['acq_images']] == [
        'acq_image_000',
        'acq_image_001',
        'acq_image_002',
    ]
    assert manifest['analysis_tables'] == {
        'sum_intensity': 'acqstore/analysis_tables/sum_intensity.csv',
        'velocity': 'acqstore/analysis_tables/velocity.csv',
    }
    first_entry = manifest['acq_images'][0]
    assert first_entry['name'] == 'yx.tif'
    assert first_entry['source'] == {'filename': 'yx.tif', 'relative_path': None}
    assert first_entry['summary'] == {
        'accepted': True,
        'acquisition': {'date': '', 'time': ''},
        'analysis_types': ['radon_velocity'],
        'dims': ['y', 'x'],
        'dtype': 'uint16',
        'has_reference_image': False,
        'num_channels': 1,
        'num_rois': 0,
        'shape': [64, 64],
        'sizes': {'x': 64, 'y': 64},
    }

    for index, source in enumerate(images):
        child = destination / 'acq_images' / f'acq_image_{index:03d}'
        loaded = read_acq_pixels_ome_zarr(child, lazy=False)
        assert loaded.shape == source.pixels.shape
        assert loaded.axes == source.pixels.axes
        assert loaded.dtype == source.pixels.dtype
        np.testing.assert_array_equal(loaded.get_array(0), source.pixels.get_array(0))
        assert (child / 'acqstore' / 'acq_image.json').is_file()
        assert (child / 'acqstore' / 'manifest.json').is_file()
    assert not (destination / 'acq_images' / 'acq_image_000' / 'acqstore' / 'analysis' / 'radon_velocity.csv').exists()
    instance_table = destination / 'acq_images' / 'acq_image_000' / 'acqstore' / 'analysis' / 'radon_velocity__c0__r1.table.csv'
    assert instance_table.is_file()
    child_manifest = json.loads((destination / 'acq_images' / 'acq_image_000' / 'acqstore' / 'manifest.json').read_text(encoding='utf-8'))
    assert child_manifest['version'] == 2
    assert child_manifest['analyses'] == [
        {
            'analysis_name': 'radon_velocity',
            'channel': 0,
            'id': 'radon_velocity__c0__r1',
            'resources': {
                'peaks': None,
                'table': 'acqstore/analysis/radon_velocity__c0__r1.table.csv',
            },
            'roi_id': 1,
        }
    ]

    import zarr

    root = zarr.open_group(str(destination), mode='r')
    assert 'acq_images' in root
    first = zarr.open_group(str(destination / 'acq_images' / 'acq_image_000'), mode='r')
    assert len(first.attrs['ome']['multiscales'][0]['datasets']) >= 2
    assert first['0'].chunks == (64, 64)
    assert first['1'].chunks == (32, 32)
    third = zarr.open_group(str(destination / 'acq_images' / 'acq_image_002'), mode='r')
    assert len(third.attrs['ome']['multiscales'][0]['datasets']) == 1

    velocity = pd.read_csv(destination / 'acqstore' / 'analysis_tables' / 'velocity.csv')
    sum_intensity = pd.read_csv(destination / 'acqstore' / 'analysis_tables' / 'sum_intensity.csv')
    assert list(velocity.columns) == list(collection.velocity_analysis_pool.columns)
    assert list(sum_intensity.columns) == list(collection.sum_intensity_analysis_pool.columns)


def test_export_preserves_source_dirty_state(tmp_path: Path) -> None:
    """Collection export must not claim source-side changes were saved."""
    image = _image('dirty.tif', (32, 32), ('Y', 'X'))
    image.analysis_set.set_dirty()
    collection = _collection(image)

    export_acq_image_list_ome_zarr(collection, tmp_path / 'dirty.ome.zarr')

    assert image.is_dirty


def test_collection_child_is_byte_identical_to_native_export(tmp_path: Path) -> None:
    """The additive collection wrapper must not alter native child content."""
    direct_image = _image('native.tif', (32, 24), ('Y', 'X'))
    collection_image = _image('native.tif', (32, 24), ('Y', 'X'))
    direct = write_acq_image_native_ome_zarr(
        direct_image,
        tmp_path / 'direct' / 'image_000',
    )
    collection = tmp_path / 'collection.ome.zarr'

    export_acq_image_list_ome_zarr(_collection(collection_image), collection)

    child = collection / 'acq_images' / 'acq_image_000'
    assert _tree_bytes(child) == _tree_bytes(direct)


def test_native_v2_reader_loads_exact_per_instance_table(tmp_path: Path) -> None:
    """Native round-trip follows manifest identity instead of combined CSVs."""
    image = _image('native.tif', (32, 16), ('Y', 'X'))
    analysis = RadonVelocityAnalysis(channel=0, roi_id=7)
    analysis.result.table = pd.DataFrame({'time_s': [0.0, 1.0], 'velocity': [2.5, 3.5]})
    image.analysis_set.add(analysis)
    image.analysis_set._results_csv_loaded = True
    destination = tmp_path / 'native.cs.ome.zarr'

    image.save_native_zarr(destination)
    loaded = AcqImage(
        str(destination),
        load_images=False,
        load_analysis_csv=True,
    )

    loaded_analysis = loaded.analysis_set.get(AnalysisKey('radon_velocity', 0, 7))
    assert loaded_analysis is not None
    pd.testing.assert_frame_equal(loaded_analysis.result.table, analysis.result.table)
    resources = sorted((destination / 'acqstore' / 'analysis').glob('*.csv'))
    assert [path.name for path in resources] == ['radon_velocity__c0__r7.table.csv']


def test_exports_reference_as_independent_ome_zarr_with_sidecar_geometry(
    tmp_path: Path,
) -> None:
    """Reference pixels are OME-Zarr while scan geometry remains in the sidecar."""
    image = _image('reference.oir', (30000, 14), ('Y', 'X'))
    reference = _attach_reference_image(image)
    destination = tmp_path / 'reference.ome.zarr'

    export_acq_image_list_ome_zarr(_collection(image), destination)

    manifest = json.loads((destination / 'acqstore' / 'acq_image_collection.json').read_text(encoding='utf-8'))
    assert manifest['acq_images'][0]['reference_image_path'] == 'acq_images/acq_image_000/reference'
    reference_path = destination / 'acq_images' / 'acq_image_000' / 'reference'
    loaded = read_acq_pixels_ome_zarr(reference_path, lazy=False)
    assert loaded.shape == (512, 512)
    assert loaded.axes == ('Y', 'X')
    assert loaded.header.physical_units == (0.38098425710482, 0.38098425710482)
    np.testing.assert_array_equal(loaded.get_array(0), reference.array)

    import zarr

    reference_group = zarr.open_group(str(reference_path), mode='r')
    assert reference_group['0'].chunks == (256, 256)
    assert len(reference_group.attrs['ome']['multiscales'][0]['datasets']) == 6
    sidecar = json.loads((destination / 'acq_images' / 'acq_image_000' / 'acqstore' / 'acq_image.json').read_text(encoding='utf-8'))
    metadata = sidecar['reference_image_metadata']
    assert metadata['line_roi'] == '(273.0, 222.0, 261.0, 228.0)'
    assert metadata['scan_path_x_pixels'] == [273.0, 261.0]
    assert metadata['scan_path_y_pixels'] == [222.0, 228.0]


def test_rejects_unloaded_pixels_during_preflight(tmp_path: Path) -> None:
    """Preflight must reject missing resident pixels before staging output."""
    image = _image('unloaded.tif', (16, 16), ('Y', 'X'))
    image.unload_images()

    with pytest.raises(ValueError, match='pixels are not loaded'):
        export_acq_image_list_ome_zarr(
            _collection(image),
            tmp_path / 'unloaded.ome.zarr',
        )
    assert not (tmp_path / 'unloaded.ome.zarr').exists()


def test_rejects_empty_collection_and_non_ome_destination(tmp_path: Path) -> None:
    """V1 requires a non-empty list and an explicit OME-Zarr destination."""
    with pytest.raises(ValueError, match='empty'):
        export_acq_image_list_ome_zarr(_collection(), tmp_path / 'empty.ome.zarr')
    with pytest.raises(ValueError, match='must end'):
        export_acq_image_list_ome_zarr(
            _collection(_image('one.tif', (16, 16), ('Y', 'X'))),
            tmp_path / 'dataset.zarr',
        )


def test_existing_destination_requires_overwrite_and_is_replaced(tmp_path: Path) -> None:
    """Overwrite replaces the complete prior collection without stale files."""
    destination = tmp_path / 'replace.ome.zarr'
    first = _collection(_image('first.tif', (16, 16), ('Y', 'X')))
    export_acq_image_list_ome_zarr(first, destination)
    stale = destination / 'stale.txt'
    stale.write_text('old', encoding='utf-8')

    with pytest.raises(FileExistsError):
        export_acq_image_list_ome_zarr(first, destination)

    second = _collection(
        _image('a.tif', (16, 16), ('Y', 'X')),
        _image('b.tif', (16, 16), ('Y', 'X')),
    )
    export_acq_image_list_ome_zarr(second, destination, overwrite=True)

    assert not stale.exists()
    manifest = json.loads((destination / 'acqstore' / 'acq_image_collection.json').read_text())
    assert len(manifest['acq_images']) == 2


def test_failed_child_export_does_not_install_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child failure must not expose a partial final collection."""
    from acqstore.acq_image.io import ome_zarr_collection

    destination = tmp_path / 'failed.ome.zarr'

    def fail(*args: object, **kwargs: object) -> Path:
        """Inject a deterministic child-export failure."""
        raise RuntimeError('injected export failure')

    monkeypatch.setattr(ome_zarr_collection, 'write_acq_image_native_ome_zarr', fail)
    with pytest.raises(RuntimeError, match='injected'):
        export_acq_image_list_ome_zarr(
            _collection(_image('one.tif', (16, 16), ('Y', 'X'))),
            destination,
        )
    assert not destination.exists()
