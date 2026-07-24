"""Tests for :func:`acqstore.acq_image.acq_image.parent_grandparent_folder_names`."""

from __future__ import annotations

from pathlib import Path

from acqstore.acq_image.acq_image import parent_grandparent_folder_names


def test_parent_grandparent_stream_returns_empty() -> None:
    assert parent_grandparent_folder_names('/any/path.tif', loaded_from_stream=True) == ('', '')


def test_parent_grandparent_empty_path() -> None:
    assert parent_grandparent_folder_names('', loaded_from_stream=False) == ('', '')


def test_parent_grandparent_nested_directories(tmp_path: Path) -> None:
    leaf = tmp_path / 'a' / 'b' / 'c.tif'
    leaf.parent.mkdir(parents=True)
    leaf.write_text('x', encoding='utf-8')
    parent, grandparent = parent_grandparent_folder_names(
        str(leaf.resolve()),
        loaded_from_stream=False,
    )
    assert parent == 'b'
    assert grandparent == 'a'


def test_parent_grandparent_when_parent_dir_missing(tmp_path: Path) -> None:
    missing_parent = tmp_path / 'no_such_dir' / 'f.tif'
    assert parent_grandparent_folder_names(
        str(missing_parent),
        loaded_from_stream=False,
    ) == ('', '')
