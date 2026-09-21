"""Negative conformance tests for AcqStore Collection v1 validation."""

from __future__ import annotations

import copy
import json
import shutil
from collections.abc import Callable
from importlib.resources import files
from pathlib import Path
from typing import Any

import pytest

from acqstore.acq_image.io.ome_zarr_collection_v1.validator import (
    ConformanceError,
    validate_collection,
)


FIXTURE_ROOT = Path(__file__).parent / 'fixtures' / 'contract'


def _schema_path() -> Path:
    """Return the packaged canonical schema path.

    Returns:
        Filesystem path to the v1 schema.
    """
    resource = files('acqstore.acq_image.io.ome_zarr_collection_v1.schema').joinpath(
        'acqstore-ome-zarr-collection-v1.schema.json'
    )
    return Path(str(resource))


@pytest.fixture
def collection(tmp_path: Path) -> Path:
    """Copy the valid contract fixture for isolated mutation.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Mutable collection root.
    """
    target = tmp_path / 'collection'
    shutil.copytree(FIXTURE_ROOT, target)
    return target


def _mutate(path: Path, callback: Callable[[dict[str, Any]], None]) -> None:
    """Apply one JSON mutation and rewrite the fixture document.

    Args:
        path: JSON document path.
        callback: Mutation applied to the parsed JSON object.

    Returns:
        None.
    """
    value = json.loads(path.read_text(encoding='utf-8'))
    callback(value)
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


@pytest.mark.parametrize(
    ('mutation', 'message'),
    [
        (
            lambda value: value['members'].append(copy.deepcopy(value['members'][0])),
            'Duplicate member IDs',
        ),
        (
            lambda value: value['members'][0].update(ome_zarr='../outside'),
            'Schema error',
        ),
        (
            lambda value: value['members'][0]['resources'].update(acqimage='missing.json'),
            'Missing acqimage.json',
        ),
    ],
)
def test_invalid_collection_cases(
    collection: Path,
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    """Reject duplicate IDs, invalid paths, and missing documents.

    Args:
        collection: Mutable valid fixture copy.
        mutation: Invalid collection mutation.
        message: Expected validation error fragment.
    """
    _mutate(collection / 'acqstore' / 'collection.json', mutation)
    with pytest.raises(ConformanceError, match=message):
        validate_collection(collection, _schema_path())


def test_rejects_mismatched_image_id(collection: Path) -> None:
    """Reject an AcqImage document owned by another member."""
    _mutate(
        collection / 'metadata' / 'kymograph-001' / 'acqimage.json',
        lambda value: value.update(image_id='wrong-image'),
    )
    with pytest.raises(ConformanceError, match='acqimage image_id does not match'):
        validate_collection(collection, _schema_path())


def test_rejects_duplicate_roi_id(collection: Path) -> None:
    """Reject duplicate ROI identity within one image."""

    def duplicate(value: dict[str, Any]) -> None:
        """Duplicate the first ROI identifier.

        Args:
            value: Parsed AcqImage document.
        """
        value['rois'][1]['id'] = value['rois'][0]['id']

    _mutate(collection / 'metadata' / 'kymograph-001' / 'acqimage.json', duplicate)
    with pytest.raises(ConformanceError, match='Duplicate ROI IDs'):
        validate_collection(collection, _schema_path())


def test_rejects_invalid_rectangle_stop(collection: Path) -> None:
    """Reject rectangle stops that are not greater than starts."""

    def collapse(value: dict[str, Any]) -> None:
        """Collapse the rectangle to an empty extent.

        Args:
            value: Parsed AcqImage document.
        """
        value['rois'][2]['stop'] = value['rois'][2]['start']

    _mutate(collection / 'metadata' / 'kymograph-001' / 'acqimage.json', collapse)
    with pytest.raises(ConformanceError, match='stop must be greater'):
        validate_collection(collection, _schema_path())


def test_rejects_analysis_unknown_roi(collection: Path) -> None:
    """Reject an analysis referencing an undeclared ROI."""
    _mutate(
        collection / 'metadata' / 'kymograph-001' / 'analyses.json',
        lambda value: value['analyses'][0].update(roi_id=999),
    )
    with pytest.raises(ConformanceError, match='unknown ROI'):
        validate_collection(collection, _schema_path())


def test_rejects_mismatched_reference_owner(collection: Path) -> None:
    """Reject reference metadata owned by another primary image."""
    _mutate(
        collection / 'metadata' / 'kymograph-001' / 'reference-image.json',
        lambda value: value.update(image_id='wrong-image'),
    )
    with pytest.raises(ConformanceError, match='reference-image image_id does not match'):
        validate_collection(collection, _schema_path())


def test_rejects_missing_analysis_csv(collection: Path) -> None:
    """Reject a declared analysis resource that does not exist."""
    (collection / 'analysis' / 'kymograph-001' / 'velocity.csv').unlink()
    with pytest.raises(ConformanceError, match='Missing analysis CSV resource'):
        validate_collection(collection, _schema_path())
