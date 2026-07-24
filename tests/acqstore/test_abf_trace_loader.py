"""Tests for the pyABF-backed ABF trace loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from acqstore.acq_trace.file_loaders.abf_trace_loader import AbfTraceLoader

ABF_DATA_DIR = Path(__file__).parent / 'data' / 'abf'
ABF_0016 = ABF_DATA_DIR / '2021_07_20_0016.abf'
ABF_0008 = ABF_DATA_DIR / '2021_07_20_0008.abf'

requires_abf = pytest.mark.skipif(
    not ABF_0016.is_file(),
    reason='ABF fixture missing (tests/acqstore/data/abf)',
)


@requires_abf
def test_abf_trace_loader_reads_header_from_uploaded_file() -> None:
    """ABF loader should expose header metadata without caller sweep access."""
    loader = AbfTraceLoader(ABF_0016)

    header = loader.header

    assert header.path == str(ABF_0016)
    assert header.file_size.endswith('KB')
    assert header.num_sweeps == 17
    assert header.num_channels == 2
    assert header.samples_per_second == 10_000
    assert header.samples_per_sweep == 1_600
    assert header.channel_names == ('Vm_scaled', 'I_Output')
    assert header.channel_units == ('mV', 'pA')
    assert header.command_names[:2] == ('I_Clamp', 'Cmd 1')
    assert header.command_units[:2] == ('nA', 'mV')
    assert header.protocol == 'I-Clamp MedDRG'
    assert header.acquisition_datetime.startswith('2021-07-20')
    assert header.x_units == 's'
    assert header.format_dims_display() == 'sweeps:17 channels:2 samples:1600'


@requires_abf
@pytest.mark.parametrize(
    ('filename', 'expected_sweeps'),
    [
        ('2021_07_20_0008.abf', 1),
        ('2021_07_20_0002.abf', 2),
        ('2021_07_20_0013.abf', 5),
        ('2021_07_20_0016.abf', 17),
    ],
)
def test_abf_trace_loader_reads_sweep_count_for_uploaded_examples(
    filename: str,
    expected_sweeps: int,
) -> None:
    """ABF loader should preserve sweep counts from each example file."""
    loader = AbfTraceLoader(ABF_DATA_DIR / filename)

    assert loader.header.num_sweeps == expected_sweeps
    assert loader.header.num_channels == 2
    assert loader.header.samples_per_sweep == 1_600


@requires_abf
def test_abf_trace_loader_get_sweep_returns_recording_command_and_epochs() -> None:
    """ABF sweep access should return time, values, command, and epoch labels."""
    loader = AbfTraceLoader(ABF_0016)

    sweep = loader.get_sweep(channel_index=0, sweep_index=0)

    assert sweep.sweep_index == 0
    assert sweep.channel_index == 0
    assert sweep.num_samples == 1_600
    assert sweep.value_units == 'mV'
    assert sweep.command_units == 'nA'
    assert sweep.time_sec.shape == (1_600,)
    assert sweep.values.shape == (1_600,)
    assert sweep.command_values is not None
    assert sweep.command_values.shape == (1_600,)
    assert sweep.epoch_index_values.shape == (1_600,)
    assert np.isclose(sweep.time_sec[0], 0.0)
    assert np.all(np.diff(sweep.time_sec) > 0)
    assert np.isclose(sweep.time_sec[1] - sweep.time_sec[0], 0.0001)
    assert len(sweep.epoch_table) == 5
    assert sweep.epoch_index_values[0] == 0
    assert sweep.epoch_index_values[25] == 1
    assert sweep.epoch_index_values[275] == 2
    assert sweep.epoch_index_values[1_275] == 3
    assert sweep.epoch_index_values[1_525] == 4
    assert set(np.unique(sweep.epoch_index_values)) == {0, 1, 2, 3, 4}


@requires_abf
def test_abf_trace_loader_get_sweep_reads_second_channel_units() -> None:
    """ABF sweep access should respect the selected input channel."""
    loader = AbfTraceLoader(ABF_0016)

    sweep = loader.get_sweep(channel_index=1, sweep_index=3)

    assert sweep.sweep_index == 3
    assert sweep.channel_index == 1
    assert sweep.value_units == 'pA'
    assert sweep.command_units == 'mV'
    assert sweep.time_sec.shape == (1_600,)
    assert sweep.values.shape == (1_600,)


@requires_abf
@pytest.mark.parametrize(
    ('channel_index', 'sweep_index', 'match'),
    [
        (-1, 0, 'channel_index out of range'),
        (2, 0, 'channel_index out of range'),
        (0, -1, 'sweep_index out of range'),
        (0, 17, 'sweep_index out of range'),
    ],
)
def test_abf_trace_loader_rejects_invalid_indices(
    channel_index: int,
    sweep_index: int,
    match: str,
) -> None:
    """ABF loader should fail fast for invalid sweep/channel indices."""
    loader = AbfTraceLoader(ABF_0016)

    with pytest.raises(ValueError, match=match):
        loader.get_sweep(channel_index=channel_index, sweep_index=sweep_index)


def test_abf_trace_loader_missing_file_raises_file_not_found() -> None:
    """ABF loader should reject missing paths with a clear exception."""
    with pytest.raises(FileNotFoundError, match='ABF file does not exist'):
        AbfTraceLoader(ABF_DATA_DIR / 'missing.abf')


def test_abf_trace_loader_directory_path_raises_value_error(tmp_path: Path) -> None:
    """ABF loader should reject directory paths."""
    with pytest.raises(ValueError, match='ABF path is not a file'):
        AbfTraceLoader(tmp_path)


@requires_abf
def test_abf_trace_loader_info_contains_useful_summary() -> None:
    """Info text should summarize the file for scripts and debugging."""
    loader = AbfTraceLoader(ABF_0008)

    info = loader.info()

    assert 'ABF Trace File' in info
    assert '2021_07_20_0008.abf' in info
    assert 'sweeps: 1' in info
    assert 'channels: 2' in info
    assert '[0] Vm_scaled (mV)' in info
    assert '[1] I_Output (pA)' in info
    assert '[0] I_Clamp (nA)' in info
