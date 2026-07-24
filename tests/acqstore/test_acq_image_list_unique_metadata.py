"""Tests for AcqImageList.get_unique_metadata_values."""

from __future__ import annotations

from pathlib import Path

import pytest

from acqstore.acq_image.acq_image_list import AcqImageList


_OIR_FIXTURE = Path(__file__).resolve().parents[2] / 'tests/acqstore/data/oir-samples/20251030_A106_0004.oir'


def test_get_unique_metadata_values_returns_sorted_unique_strings() -> None:
    if not _OIR_FIXTURE.is_file():
        pytest.skip(f'Missing OIR fixture: {_OIR_FIXTURE}')
    file_list = AcqImageList(str(_OIR_FIXTURE), folder_depth=1)
    acq = file_list.get_file_by_index(0)
    acq.apply_metadata_patch('experiment_metadata', {'species': 'mouse', 'condition': 'control'})
    values = file_list.get_unique_metadata_values('species')
    assert values == ['mouse']


def test_get_unique_metadata_values_ignores_blank_values() -> None:
    if not _OIR_FIXTURE.is_file():
        pytest.skip(f'Missing OIR fixture: {_OIR_FIXTURE}')
    file_list = AcqImageList(str(_OIR_FIXTURE), folder_depth=1)
    acq = file_list.get_file_by_index(0)
    acq.apply_metadata_patch('experiment_metadata', {'species': '', 'condition': '  '})
    assert file_list.get_unique_metadata_values('species') == []


def test_get_unique_metadata_values_unknown_field_raises() -> None:
    if not _OIR_FIXTURE.is_file():
        pytest.skip(f'Missing OIR fixture: {_OIR_FIXTURE}')
    file_list = AcqImageList(str(_OIR_FIXTURE), folder_depth=1)
    with pytest.raises(ValueError, match='Unknown experiment_metadata field'):
        file_list.get_unique_metadata_values('not_a_field')


def test_get_unique_metadata_values_non_string_field_raises() -> None:
    if not _OIR_FIXTURE.is_file():
        pytest.skip(f'Missing OIR fixture: {_OIR_FIXTURE}')
    file_list = AcqImageList(str(_OIR_FIXTURE), folder_depth=1)
    with pytest.raises(ValueError, match='not a string field'):
        file_list.get_unique_metadata_values('depth')
