"""Tests for discovering directory-backed native Zarr stores."""

from __future__ import annotations

from pathlib import Path

from acqstore.acq_image.acq_image_list import _build_file_list
from acqstore.acq_image.supported_import_extensions import (
    normalize_import_extension_for_path,
    path_has_allowed_import_extension,
)


def test_compound_zarr_suffix_normalization() -> None:
    assert normalize_import_extension_for_path("a.ome.zarr") == "ome.zarr"
    assert normalize_import_extension_for_path("a.cs.ome.zarr") == "cs.ome.zarr"
    assert path_has_allowed_import_extension("a.cs.ome.zarr") is True


def test_build_file_list_discovers_zarr_store_directories(tmp_path: Path) -> None:
    store = tmp_path / "sample.cs.ome.zarr"
    store.mkdir()
    nested = store / "0"
    nested.mkdir()
    (nested / ".zarray").write_text("{}")

    found = _build_file_list(tmp_path, ("tif", "cs.ome.zarr"), folder_depth=2)

    assert found == [str(store.resolve())]
