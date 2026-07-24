"""Tests for AcqTrace peak detection and shared one-dimensional peak core."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from acqstore.acq_trace.acq_trace import AcqTrace
from acqstore.acq_trace.analysis.trace_peak_detection import run_trace_peak_detection
from acqstore.acq_trace.analysis.trace_peak_params import TracePeakDetectionParams
from acqstore.common_analysis.peak_detection_core import PeakDetectionCoreParams, detect_peaks_1d

ABF_DATA_DIR = Path(__file__).parent / 'data' / 'abf'
ABF_0016 = ABF_DATA_DIR / '2021_07_20_0016.abf'

requires_abf = pytest.mark.skipif(
    not ABF_0016.is_file(),
    reason='ABF fixture missing (tests/acqstore/data/abf)',
)


def test_detect_peaks_1d_finds_positive_peaks() -> None:
    """Shared peak core should find positive peaks in a synthetic trace."""
    time_sec = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4], dtype=float)
    values = np.asarray([0.0, 1.0, 0.0, 2.0, 0.0], dtype=float)

    result = detect_peaks_1d(time_sec=time_sec, values=values)

    assert result.num_peaks == 2
    assert result.peak_table['peak_index'].tolist() == [1, 3]
    assert result.peak_table['peak_value'].tolist() == [1.0, 2.0]
    assert result.trace_table['is_peak'].tolist() == [False, True, False, True, False]


def test_detect_peaks_1d_finds_negative_peaks() -> None:
    """Shared peak core should find negative peaks by inverting the signal."""
    time_sec = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4], dtype=float)
    values = np.asarray([0.0, -1.0, 0.0, -2.0, 0.0], dtype=float)
    params = PeakDetectionCoreParams(polarity='negative')

    result = detect_peaks_1d(time_sec=time_sec, values=values, params=params)

    assert result.num_peaks == 2
    assert result.peak_table['peak_index'].tolist() == [1, 3]
    assert result.peak_table['peak_value'].tolist() == [-1.0, -2.0]
    assert result.peak_table['detection_value'].tolist() == [1.0, 2.0]


def test_detect_peaks_1d_uses_prominence_and_distance_filters() -> None:
    """Shared peak core should pass prominence and distance filters to SciPy."""
    time_sec = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4], dtype=float)
    values = np.asarray([0.0, 1.0, 0.0, 2.0, 0.0], dtype=float)
    params = PeakDetectionCoreParams(prominence=1.5, min_distance_sec=0.1)

    result = detect_peaks_1d(time_sec=time_sec, values=values, params=params)

    assert result.num_peaks == 1
    assert result.peak_table.loc[0, 'peak_index'] == 3
    assert result.peak_table.loc[0, 'prominence'] == 2.0


def test_detect_peaks_1d_rejects_invalid_trace_arrays() -> None:
    """Shared peak core should fail fast for invalid trace inputs."""
    time_sec = np.asarray([0.0, 0.2, 0.1], dtype=float)
    values = np.asarray([0.0, 1.0, 0.0], dtype=float)

    with pytest.raises(ValueError, match='strictly increasing'):
        detect_peaks_1d(time_sec=time_sec, values=values)


def test_trace_peak_params_convert_to_core_params() -> None:
    """Trace peak params should produce modality-neutral core params."""
    params = TracePeakDetectionParams(
        polarity='negative',
        height=1.0,
        prominence=0.5,
        min_distance_sec=0.01,
    )

    core_params = params.to_core_params()

    assert core_params.polarity == 'negative'
    assert core_params.height == 1.0
    assert core_params.prominence == 0.5
    assert core_params.min_distance_sec == 0.01


@requires_abf
def test_run_trace_peak_detection_analyzes_one_sweep() -> None:
    """Trace peak detection should analyze one selected ABF sweep."""
    trace = AcqTrace(ABF_0016)
    params = TracePeakDetectionParams(polarity='positive', prominence=5.0)

    result = run_trace_peak_detection(
        trace,
        channel_index=0,
        sweep_index=0,
        params=params,
    )

    assert result.channel_index == 0
    assert result.sweep_index == 0
    assert result.num_sweeps_analyzed == 1
    assert isinstance(result.trace_table, pd.DataFrame)
    assert set(result.trace_table['sweep_index']) == {0}
    assert set(result.trace_table['channel_index']) == {0}
    assert 'epoch_index' in result.trace_table.columns
    assert 'command' in result.trace_table.columns
    assert 'global_peak_id' in result.peak_table.columns


@requires_abf
def test_run_trace_peak_detection_analyzes_all_sweeps_by_default() -> None:
    """Trace peak detection should analyze all sweeps when sweep_index is None."""
    trace = AcqTrace(ABF_0016)
    params = TracePeakDetectionParams(polarity='positive', prominence=5.0)

    result = run_trace_peak_detection(
        trace,
        channel_index=0,
        params=params,
    )

    assert result.sweep_index is None
    assert result.num_sweeps_analyzed == trace.trace_header.num_sweeps
    assert result.trace_table['sweep_index'].nunique() == trace.trace_header.num_sweeps
    assert set(result.peak_table['channel_index'].unique()).issubset({0})
    if not result.peak_table.empty:
        assert result.peak_table['sweep_index'].min() >= 0
        assert result.peak_table['sweep_index'].max() < trace.trace_header.num_sweeps
        assert 'epoch_index' in result.peak_table.columns
        assert 'epoch_start_sec' in result.peak_table.columns
        assert result.peak_table['global_peak_id'].tolist() == list(range(1, len(result.peak_table) + 1))


@requires_abf
def test_acq_trace_run_peak_detection_method_matches_function() -> None:
    """AcqTrace convenience method should delegate to trace peak detection."""
    trace = AcqTrace(ABF_0016)
    params = TracePeakDetectionParams(polarity='positive', prominence=5.0)

    via_method = trace.run_peak_detection(
        channel_index=0,
        sweep_index=0,
        params=params,
    )
    via_function = run_trace_peak_detection(
        trace,
        channel_index=0,
        sweep_index=0,
        params=params,
    )

    assert via_method.summary_dict() == via_function.summary_dict()
    assert via_method.trace_table.equals(via_function.trace_table)
    assert via_method.peak_table.equals(via_function.peak_table)
