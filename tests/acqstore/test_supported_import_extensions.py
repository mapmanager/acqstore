"""Tests for runtime supported import extension configuration."""

from __future__ import annotations

import pytest

from acqstore.acq_image.supported_import_extensions import (
    DEFAULT_IMPORT_EXTENSIONS,
    add_allowed_import_extension,
    get_allowed_import_extensions,
    get_supported_import_extensions,
    remove_allowed_import_extension,
    reset_allowed_import_extensions,
    set_allowed_import_extensions,
)


@pytest.fixture(autouse=True)
def reset_extensions() -> None:
    reset_allowed_import_extensions()


def test_defaults_exposed_in_sorted_runtime_snapshot() -> None:
    assert get_allowed_import_extensions() == tuple(sorted(DEFAULT_IMPORT_EXTENSIONS))


def test_supported_import_extensions_are_loader_backed() -> None:
    assert get_supported_import_extensions() == (
        'cs.ome.zarr',
        'cs.ome.zarr.zip',
        'czi',
        'nd2',
        'oir',
        'ome.zarr',
        'ome.zarr.zip',
        'tif',
    )


def test_set_allowed_import_extensions_normalizes_values() -> None:
    set_allowed_import_extensions(['.TIF', ' OIR ', 'czi'])
    assert get_allowed_import_extensions() == ('czi', 'oir', 'tif')


def test_set_allowed_import_extensions_rejects_empty_set() -> None:
    with pytest.raises(ValueError, match='must not be empty'):
        set_allowed_import_extensions(['', '   '])


def test_set_allowed_import_extensions_rejects_unregistered_loader_extension() -> None:
    with pytest.raises(ValueError, match='Unsupported acquisition import extension'):
        set_allowed_import_extensions(['.tif', '.new'])


def test_add_and_remove_allowed_import_extension() -> None:
    remove_allowed_import_extension('oir')
    assert 'oir' not in get_allowed_import_extensions()
    add_allowed_import_extension('.OIR')
    assert 'oir' in get_allowed_import_extensions()


def test_add_allowed_import_extension_rejects_unregistered_loader_extension() -> None:
    with pytest.raises(ValueError, match='Unsupported acquisition import extension'):
        add_allowed_import_extension('.new')
