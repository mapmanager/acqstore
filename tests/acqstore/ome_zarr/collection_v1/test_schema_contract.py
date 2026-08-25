"""Conformance tests for the frozen AcqStore OME-Zarr Collection v1 schema."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

from jsonschema import Draft202012Validator

from acqstore.acq_image.io.ome_zarr_collection_v1.validator import validate_collection


FIXTURE_ROOT = Path(__file__).parent / 'fixtures' / 'contract'


def _schema_path() -> Path:
    """Return the filesystem path to the packaged canonical schema.

    Returns:
        Path to the packaged Draft 2020-12 schema.
    """
    resource = files('acqstore.acq_image.io.ome_zarr_collection_v1.schema').joinpath(
        'acqstore-ome-zarr-collection-v1.schema.json'
    )
    return Path(str(resource))


def test_packaged_schema_is_valid_draft_2020_12() -> None:
    """Require the canonical packaged schema to satisfy its metaschema."""
    schema = json.loads(_schema_path().read_text(encoding='utf-8'))
    Draft202012Validator.check_schema(schema)


def test_complete_additive_contract_fixture_is_valid() -> None:
    """Require the complete JSON/CSV contract fixture to validate."""
    validate_collection(FIXTURE_ROOT, _schema_path())
