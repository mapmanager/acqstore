"""Tests for the AcqTrace public API."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from acqstore.acq_image.metadata import ExperimentMetadata
from acqstore.acq_trace.acq_trace import AcqTrace
from acqstore.acq_trace.epoch_data import EpochInterval, EpochTable
from acqstore.acq_trace.sweep_data import SweepData
from acqstore.acq_trace.trace_header import TraceHeader
from acqstore.acq_types import AcqModality

ABF_DATA_DIR = Path(__file__).parent / 'data' / 'abf'
ABF_0016 = ABF_DATA_DIR / '2021_07_20_0016.abf'

requires_abf = pytest.mark.skipif(
    not ABF_0016.is_file(),
    reason='ABF fixture missing (tests/acqstore/data/abf)',
)


@requires_abf
def test_acq_trace_constructs_from_abf_and_exposes_header() -> None:
    """AcqTrace should expose basic trace identity and header metadata."""
    trace = AcqTrace(ABF_0016)

    assert trace.modality == AcqModality.TRACE
    assert trace.path == str(ABF_0016)
    assert trace.name == '2021_07_20_0016.abf'
    assert trace.file_id == str(ABF_0016)
    assert trace.accepted is True
    assert trace.is_dirty is False
    assert isinstance(trace.experiment_metadata, ExperimentMetadata)
    assert trace.get_trace_header().num_sweeps == 17
    assert trace.trace_header.num_channels == 2


@requires_abf
def test_acq_trace_preserves_constructor_state() -> None:
    """AcqTrace should preserve accepted state and provided metadata object."""
    metadata = ExperimentMetadata(species='mouse', condition='test')

    trace = AcqTrace(ABF_0016, accepted=False, experiment_metadata=metadata)

    assert trace.accepted is False
    assert trace.experiment_metadata is metadata
    assert trace.experiment_metadata.species == 'mouse'
    assert trace.experiment_metadata.condition == 'test'


@requires_abf
def test_acq_trace_get_sweep_delegates_to_loader() -> None:
    """AcqTrace should return SweepData from the underlying ABF loader."""
    trace = AcqTrace(ABF_0016)

    sweep = trace.get_sweep(channel_index=0, sweep_index=0)

    assert isinstance(sweep, SweepData)
    assert sweep.num_samples == 1_600
    assert sweep.value_units == 'mV'
    assert sweep.command_values is not None


@requires_abf
def test_acq_trace_iter_sweeps_returns_all_sweeps_for_channel() -> None:
    """AcqTrace should iterate over every sweep for a selected channel."""
    trace = AcqTrace(ABF_0016)

    sweeps = list(trace.iter_sweeps(channel_index=0))

    assert len(sweeps) == 17
    assert [sweep.sweep_index for sweep in sweeps] == list(range(17))
    assert {sweep.channel_index for sweep in sweeps} == {0}


@requires_abf
def test_acq_trace_get_sweep_trace_table_returns_per_sample_dataframe() -> None:
    """AcqTrace should return one per-sample table for a selected sweep."""
    trace = AcqTrace(ABF_0016)

    table = trace.get_sweep_trace_table(channel_index=0, sweep_index=0)

    assert list(table.columns) == ['time_sec', 'value', 'command', 'epoch']
    assert len(table) == 1_600
    assert table.loc[0, 'time_sec'] == 0
    assert table.loc[0, 'epoch'] == 0
    assert table.loc[275, 'epoch'] == 2
    assert pd.api.types.is_float_dtype(table['value'])


@requires_abf
def test_acq_trace_get_channel_trace_table_returns_wide_dataframe() -> None:
    """AcqTrace should return one wide table with all sweeps for a channel."""
    trace = AcqTrace(ABF_0016)

    table = trace.get_channel_trace_table(channel_index=0)

    assert len(table) == 1_600
    assert table.columns[:4].tolist() == [
        'time_sec',
        'sweep_0',
        'sweep_0_command',
        'sweep_0_epoch',
    ]
    assert 'sweep_16' in table.columns
    assert 'sweep_16_command' in table.columns
    assert 'sweep_16_epoch' in table.columns
    assert table.loc[0, 'sweep_0_epoch'] == 0
    assert table.loc[1_525, 'sweep_16_epoch'] == 4


@requires_abf
def test_acq_trace_get_epoch_table_returns_all_sweeps() -> None:
    """AcqTrace should expose compact epoch interval rows for all sweeps."""
    trace = AcqTrace(ABF_0016)

    table = trace.get_epoch_table(channel_index=0)

    assert len(table) == 85
    assert table['sweep_index'].nunique() == 17
    assert table['epoch_index'].nunique() == 5
    assert table.loc[0, 'start_sample'] == 0
    assert table.loc[0, 'end_sample'] == 25
    assert table.loc[0, 'duration_samples'] == 25
    assert table.loc[0, 'start_sec'] == 0
    assert table.loc[0, 'end_sec'] == 0.0025
    assert table.loc[0, 'duration_sec'] == 0.0025


@requires_abf
def test_acq_trace_get_epoch_table_returns_one_sweep() -> None:
    """AcqTrace should expose compact epoch rows for one selected sweep."""
    trace = AcqTrace(ABF_0016)

    table = trace.get_epoch_table(channel_index=0, sweep_index=3)

    assert len(table) == 5
    assert table['sweep_index'].tolist() == [3, 3, 3, 3, 3]
    assert table['epoch_index'].tolist() == [0, 1, 2, 3, 4]


@requires_abf
def test_acq_trace_to_summary_dict_returns_structured_summary() -> None:
    """AcqTrace should return a summary dictionary for callers."""
    trace = AcqTrace(ABF_0016)

    summary = trace.to_summary_dict()

    assert summary['modality'] == 'trace'
    assert summary['accepted'] is True
    assert summary['name'] == '2021_07_20_0016.abf'
    assert isinstance(summary['trace_header'], dict)
    assert summary['trace_header']['num_sweeps'] == 17


@requires_abf
def test_acq_trace_info_returns_loader_summary() -> None:
    """AcqTrace info should provide a script-friendly file overview."""
    trace = AcqTrace(ABF_0016)

    info = trace.info()

    assert 'ABF Trace File' in info
    assert 'sweeps: 17' in info
    assert 'Vm_scaled' in info
    assert 'I_Clamp' in info


def test_acq_trace_rejects_unsupported_extension(tmp_path: Path) -> None:
    """AcqTrace should fail fast for unsupported trace file types."""
    text_path = tmp_path / 'not_abf.txt'
    text_path.write_text('not an abf file')

    with pytest.raises(ValueError, match='Unsupported trace file extension'):
        AcqTrace(text_path)


@requires_abf
def test_acq_trace_sidecar_round_trips_metadata_and_acceptance(tmp_path: Path) -> None:
    """AcqTrace should save and reload trace-specific sidecar JSON."""
    abf_path = tmp_path / ABF_0016.name
    abf_path.write_bytes(ABF_0016.read_bytes())
    metadata = ExperimentMetadata(species='mouse', condition='baseline')
    trace = AcqTrace(abf_path, accepted=False, experiment_metadata=metadata)

    trace.save()
    reloaded = AcqTrace(abf_path)

    assert reloaded.accepted is False
    assert reloaded.experiment_metadata.species == 'mouse'
    assert reloaded.experiment_metadata.condition == 'baseline'
    assert reloaded.is_dirty is False


@requires_abf
def test_acq_trace_sidecar_json_contains_trace_fields_only(tmp_path: Path) -> None:
    """AcqTrace sidecar JSON should not persist image-only fields."""
    abf_path = tmp_path / ABF_0016.name
    abf_path.write_bytes(ABF_0016.read_bytes())
    trace = AcqTrace(abf_path)

    trace.save()
    payload = json.loads(Path(trace.get_sidecar_json_path()).read_text())

    assert payload['version'] == 1
    assert payload['modality'] == 'trace'
    assert payload['accepted'] is True
    assert 'experiment_metadata' in payload
    assert 'trace_header_metadata' in payload
    assert 'rois' not in payload
    assert 'image_header_metadata' not in payload
    assert 'image_contrast' not in payload
    assert 'analysis' not in payload


@requires_abf
def test_acq_trace_ignores_invalid_sidecar_with_warning(tmp_path: Path) -> None:
    """AcqTrace should ignore invalid sidecar payloads during construction."""
    abf_path = tmp_path / ABF_0016.name
    abf_path.write_bytes(ABF_0016.read_bytes())
    Path(f'{abf_path}.json').write_text('{"version": 999}', encoding='utf-8')

    trace = AcqTrace(abf_path)

    assert trace.accepted is True
    assert trace.experiment_metadata.species == ''


@requires_abf
def test_acq_trace_load_lazy_data_and_unload_lazy_data_are_noops() -> None:
    """AcqTrace lazy lifecycle placeholders should be callable."""
    trace = AcqTrace(ABF_0016)

    trace.load_lazy_data()
    trace.unload_lazy_data()

    assert trace.trace_header.num_sweeps == 17


def test_trace_header_validates_channel_metadata_lengths() -> None:
    """TraceHeader should reject inconsistent channel metadata."""
    with pytest.raises(ValueError, match='channel_names length'):
        TraceHeader(
            path='x.abf',
            file_size='1 KB',
            num_sweeps=1,
            num_channels=2,
            samples_per_second=10_000,
            samples_per_sweep=100,
            channel_names=('Vm',),
            channel_units=('mV', 'pA'),
            command_names=('Cmd',),
            command_units=('nA',),
        )


def test_epoch_table_generates_per_sample_labels() -> None:
    """EpochTable should convert intervals to one label per sample."""
    table = EpochTable(
        (
            EpochInterval(0, 0, 2, -0.2, 'Step'),
            EpochInterval(1, 2, 5, -0.7, 'Step'),
        )
    )

    labels = table.to_sample_labels(6)

    assert labels.tolist() == [0, 0, 1, 1, 1, -1]
    assert table.to_dicts()[1]['epoch_type'] == 'Step'
    assert table.to_dicts()[1]['duration_samples'] == 3


def test_epoch_table_to_dataframe_adds_seconds_and_identifiers() -> None:
    """EpochTable should produce a compact interval DataFrame."""
    table = EpochTable((EpochInterval(0, 10, 20, -0.2, 'Step'),))

    df = table.to_dataframe(
        samples_per_second=10_000,
        channel_index=1,
        sweep_index=2,
    )

    assert df.loc[0, 'channel_index'] == 1
    assert df.loc[0, 'sweep_index'] == 2
    assert df.loc[0, 'epoch_index'] == 0
    assert df.loc[0, 'start_sec'] == 0.001
    assert df.loc[0, 'end_sec'] == 0.002
    assert df.loc[0, 'duration_sec'] == 0.001


def test_epoch_interval_rejects_empty_interval() -> None:
    """EpochInterval should reject intervals without positive length."""
    with pytest.raises(ValueError, match='end_sample must be greater'):
        EpochInterval(0, 10, 10, 0.0, 'Step')


def test_epoch_table_rejects_interval_beyond_num_samples() -> None:
    """EpochTable labels should reject intervals beyond the target sweep."""
    table = EpochTable((EpochInterval(0, 0, 10, 0.0, 'Step'),))

    with pytest.raises(ValueError, match='extends beyond sweep samples'):
        table.to_sample_labels(9)


def test_sweep_data_as_dataframe_fills_missing_command_with_nan() -> None:
    """SweepData should keep table shape when command data are absent."""
    sweep = SweepData(
        sweep_index=0,
        channel_index=0,
        time_sec=pd.Series([0.0, 0.1]).to_numpy(),
        values=pd.Series([1.0, 2.0]).to_numpy(),
        value_units='mV',
        command_values=None,
        command_units='',
        epoch_index_values=pd.Series([0, 1]).to_numpy(),
        epoch_table=EpochTable((EpochInterval(0, 0, 1, 0.0, 'Step'),)),
    )

    df = sweep.as_dataframe()

    assert df.columns.tolist() == ['time_sec', 'value', 'command', 'epoch']
    assert df['command'].isna().all()
