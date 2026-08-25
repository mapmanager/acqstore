"""Tests for object-oriented AcqStore OME-Zarr Collection v1 export."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from acqstore.acq_image.acq_image import AcqImage
from acqstore.acq_image.acq_image_list import AcqImageList
from acqstore.acq_image.analysis.velocity_analysis.radon_velocity_analysis import (
    RadonVelocityAnalysis,
)
from acqstore.acq_image.file_loaders.base_file_loader import ReferenceImage
from acqstore.acq_image.io.ome_zarr import read_acq_pixels_ome_zarr
from acqstore.acq_image.io.ome_zarr_collection_v1.exporter import (
    AcqStoreOmeZarrCollectionExporter,
)


def test_exports_real_independent_ome_zarr_image(
    tmp_path: Path,
    make_acq_image: Callable[..., AcqImage],
    make_acq_image_list: Callable[..., AcqImageList],
) -> None:
    """Export a real image plus schema-valid additive v1 resources."""
    image = make_acq_image()
    collection = make_acq_image_list(image)
    destination = tmp_path / 'collection.ome.zarr'

    result = AcqStoreOmeZarrCollectionExporter(destination).export(collection)

    assert result == destination.resolve()
    manifest = json.loads((destination / 'acqstore' / 'collection.json').read_text())
    assert manifest['format'] == 'acqstore-ome-zarr-collection'
    assert uuid.UUID(manifest['id']).version == 4
    assert len(manifest['members']) == 1
    member = manifest['members'][0]
    assert uuid.UUID(member['id']).version == 4
    assert member['id'] != image.file_id
    assert member['ome_zarr'] == f"images/{member['id']}"
    loaded = read_acq_pixels_ome_zarr(destination / member['ome_zarr'], lazy=False)
    assert loaded.shape == image.pixels.shape


def test_writes_sparse_temporal_y_axis_display(
    tmp_path: Path,
    make_acq_image: Callable[..., AcqImage],
    make_acq_image_list: Callable[..., AcqImageList],
) -> None:
    """Write only the exceptional temporal interpretation of raster Y."""
    image = make_acq_image(
        'kymograph.tif',
        axis_units={'Y': 'seconds', 'X': 'micrometer'},
        axis_spacing={'Y': 0.002, 'X': 0.25},
    )
    destination = tmp_path / 'kymograph.ome.zarr'

    AcqStoreOmeZarrCollectionExporter(destination).export(make_acq_image_list(image))

    manifest = json.loads((destination / 'acqstore' / 'collection.json').read_text())
    acqimage_path = destination / manifest['members'][0]['resources']['acqimage']
    document = json.loads(acqimage_path.read_text())
    assert document['axis_display'] == {
        'y': {'type': 'time', 'unit': 'second', 'scale': 0.002}
    }


def test_exports_roi_analysis_and_csv_resource(
    tmp_path: Path,
    make_acq_image: Callable[..., AcqImage],
    make_acq_image_list: Callable[..., AcqImageList],
) -> None:
    """Map source ROI/analysis identities to opaque v1 resource identities."""
    image = make_acq_image()
    roi = image.rois.create_rect_roi(name='analysis ROI')
    analysis = RadonVelocityAnalysis(channel=0, roi_id=roi.roi_id)
    analysis.result.summary = {'velocity_mean': 2.5}
    analysis.result.table = pd.DataFrame({'time': [0.0, 1.0], 'velocity': [2.0, 3.0]})
    image.analysis_set.add(analysis)
    image.analysis_set._results_csv_loaded = True
    destination = tmp_path / 'analysis.ome.zarr'

    AcqStoreOmeZarrCollectionExporter(destination).export(make_acq_image_list(image))

    manifest = json.loads((destination / 'acqstore' / 'collection.json').read_text())
    member = manifest['members'][0]
    acqimage = json.loads((destination / member['resources']['acqimage']).read_text())
    analyses = json.loads((destination / member['resources']['analyses']).read_text())
    roi_id = acqimage['rois'][0]['id']
    analysis_document = analyses['analyses'][0]
    assert uuid.UUID(roi_id).version == 4
    assert uuid.UUID(analysis_document['id']).version == 4
    assert analysis_document['roi_id'] == roi_id
    assert analysis_document['parameters'] == analysis.detection_params
    assert analysis_document['summary'] == {'velocity_mean': 2.5}
    assert (destination / analysis_document['resources'][0]['path']).is_file()
    collection_tables = manifest.get('resources', {}).get('tables', [])
    assert {resource['id'] for resource in collection_tables} == {
        'velocity',
        'sum_intensity',
    }
    assert all((destination / resource['path']).is_file() for resource in collection_tables)


def test_exports_reference_image_and_integer_scan_path(
    tmp_path: Path,
    make_acq_image: Callable[..., AcqImage],
    make_acq_image_list: Callable[..., AcqImageList],
) -> None:
    """Export reference pixels independently and scan points in full-resolution pixels."""
    image = make_acq_image('reference.oir')
    reference = ReferenceImage(
        array=np.arange(32 * 24, dtype=np.uint16).reshape(32, 24),
        dims=('Y', 'X'),
        num_channels=1,
        line_roi=(4.0, 5.0, 20.0, 18.0),
        coord_units=(('Y', 'um'), ('X', 'um')),
        coord_scales=(('Y', 0.5), ('X', 0.5)),
        coords=(),
        scan_path=np.asarray([[4.0, 20.0], [5.0, 18.0]]),
    )
    image.images._referenceImage = reference
    destination = tmp_path / 'reference.ome.zarr'

    AcqStoreOmeZarrCollectionExporter(destination).export(make_acq_image_list(image))

    manifest = json.loads((destination / 'acqstore' / 'collection.json').read_text())
    link = manifest['members'][0]['reference_image']
    metadata = json.loads((destination / link['metadata']).read_text())
    assert metadata['scan_path'] == {
        'coordinate_space': 'reference-image-full-resolution-pixels',
        'points': [[4, 5], [20, 18]],
    }
    loaded = read_acq_pixels_ome_zarr(destination / link['ome_zarr'], lazy=False)
    assert loaded.shape == reference.array.shape


def test_rejects_existing_destination_without_overwrite(
    tmp_path: Path,
    make_acq_image: Callable[..., AcqImage],
    make_acq_image_list: Callable[..., AcqImageList],
) -> None:
    """Protect an existing collection unless replacement is explicit."""
    collection = make_acq_image_list(make_acq_image())
    destination = tmp_path / 'existing.ome.zarr'
    AcqStoreOmeZarrCollectionExporter(destination).export(collection)

    with pytest.raises(FileExistsError, match='Destination already exists'):
        AcqStoreOmeZarrCollectionExporter(destination).export(collection)


def test_overwrite_creates_a_fresh_collection_identity(
    tmp_path: Path,
    make_acq_image: Callable[..., AcqImage],
    make_acq_image_list: Callable[..., AcqImageList],
) -> None:
    """Treat an explicit overwrite as creation of a new persistent artifact."""
    collection = make_acq_image_list(make_acq_image())
    destination = tmp_path / 'overwrite.ome.zarr'
    AcqStoreOmeZarrCollectionExporter(destination).export(collection)
    first = json.loads((destination / 'acqstore' / 'collection.json').read_text())

    AcqStoreOmeZarrCollectionExporter(destination, overwrite=True).export(collection)

    second = json.loads((destination / 'acqstore' / 'collection.json').read_text())
    assert second['id'] != first['id']
    assert second['members'][0]['id'] != first['members'][0]['id']


def test_rejects_fractional_reference_scan_coordinates(
    tmp_path: Path,
    make_acq_image: Callable[..., AcqImage],
    make_acq_image_list: Callable[..., AcqImageList],
) -> None:
    """Reject rather than round a scan path that violates integer v1 semantics."""
    image = make_acq_image('fractional-reference.oir')
    image.images._referenceImage = ReferenceImage(
        array=np.zeros((16, 16), dtype=np.uint16),
        dims=('Y', 'X'),
        num_channels=1,
        line_roi=None,
        coord_units=(('Y', 'um'), ('X', 'um')),
        coord_scales=(('Y', 1.0), ('X', 1.0)),
        coords=(),
        scan_path=np.asarray([[1.5, 10.0], [2.0, 12.0]]),
    )

    with pytest.raises(ValueError, match='non-negative integer'):
        AcqStoreOmeZarrCollectionExporter(tmp_path / 'fractional.ome.zarr').export(
            make_acq_image_list(image)
        )
    assert not (tmp_path / 'fractional.ome.zarr').exists()


def test_rejects_completed_analysis_without_required_csv_resource(
    tmp_path: Path,
    make_acq_image: Callable[..., AcqImage],
    make_acq_image_list: Callable[..., AcqImageList],
) -> None:
    """Prevent silent loss of a completed summary-only analysis."""
    image = make_acq_image()
    roi = image.rois.create_rect_roi()
    analysis = RadonVelocityAnalysis(channel=0, roi_id=roi.roi_id)
    analysis.result.summary = {'velocity_mean': 2.5}
    image.analysis_set.add(analysis)
    image.analysis_set._results_csv_loaded = True
    destination = tmp_path / 'summary-only.ome.zarr'

    with pytest.raises(ValueError, match='summary but no CSV table resource'):
        AcqStoreOmeZarrCollectionExporter(destination).export(make_acq_image_list(image))
    assert not destination.exists()
