"""Peak detection for AcqTrace sweep data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from acqstore.acq_trace.acq_trace import AcqTrace
from acqstore.acq_trace.analysis.trace_peak_params import TracePeakDetectionParams
from acqstore.acq_trace.analysis.trace_peak_result import TracePeakDetectionResult
from acqstore.acq_trace.sweep_data import SweepData
from acqstore.common_analysis.peak_detection_core import detect_peaks_1d


def run_trace_peak_detection(
    trace: AcqTrace,
    *,
    channel_index: int,
    sweep_index: int | None = None,
    params: TracePeakDetectionParams | None = None,
) -> TracePeakDetectionResult:
    """Run peak detection on one trace channel.

    Args:
        trace: Trace acquisition to analyze.
        channel_index: Zero-based input channel index.
        sweep_index: Optional zero-based sweep index. When None, all sweeps for
            ``channel_index`` are analyzed.
        params: Optional trace peak-detection parameters. Defaults to positive
            peak detection with no height, prominence, or distance filter.

    Returns:
        Detection result containing per-sample trace rows and one peak row per
        detected peak. When ``sweep_index`` is None, result tables contain rows
        from all sweeps.

    Raises:
        ValueError: If ``channel_index`` or ``sweep_index`` is out of range, or
            detection parameters are invalid.
    """
    normalized_params = params or TracePeakDetectionParams()
    sweeps = _select_sweeps(
        trace=trace,
        channel_index=channel_index,
        sweep_index=sweep_index,
    )
    trace_frames: list[pd.DataFrame] = []
    peak_frames: list[pd.DataFrame] = []
    next_global_peak_id = 1
    for sweep in sweeps:
        core_result = detect_peaks_1d(
            time_sec=sweep.time_sec,
            values=sweep.values,
            params=normalized_params.to_core_params(),
        )
        sweep_trace_table = _annotate_trace_table(
            core_table=core_result.trace_table,
            sweep=sweep,
        )
        sweep_peak_table = _annotate_peak_table(
            core_peak_table=core_result.peak_table,
            sweep=sweep,
            trace=trace,
            next_global_peak_id=next_global_peak_id,
        )
        if not sweep_peak_table.empty:
            next_global_peak_id += len(sweep_peak_table)
        trace_frames.append(sweep_trace_table)
        peak_frames.append(sweep_peak_table)
    trace_table = pd.concat(trace_frames, ignore_index=True) if trace_frames else pd.DataFrame()
    peak_table = pd.concat(peak_frames, ignore_index=True) if peak_frames else pd.DataFrame()
    return TracePeakDetectionResult(
        trace_table=trace_table,
        peak_table=peak_table,
        params=normalized_params,
        channel_index=channel_index,
        sweep_index=sweep_index,
    )


def _select_sweeps(
    *,
    trace: AcqTrace,
    channel_index: int,
    sweep_index: int | None,
) -> tuple[SweepData, ...]:
    """Return sweeps selected for analysis.

    Args:
        trace: Trace acquisition.
        channel_index: Zero-based channel index.
        sweep_index: Optional zero-based sweep index.

    Returns:
        Tuple of selected sweeps.

    Raises:
        ValueError: If indices are out of range.
    """
    if sweep_index is None:
        return tuple(trace.iter_sweeps(channel_index=channel_index))
    return (trace.get_sweep(channel_index=channel_index, sweep_index=sweep_index),)


def _annotate_trace_table(*, core_table: pd.DataFrame, sweep: SweepData) -> pd.DataFrame:
    """Add trace acquisition identifiers to one core trace table.

    Args:
        core_table: Modality-neutral per-sample result table.
        sweep: Sweep that produced the core result.

    Returns:
        Per-sample table with channel, sweep, command, and epoch columns.
    """
    table = core_table.copy()
    table.insert(0, 'channel_index', sweep.channel_index)
    table.insert(1, 'sweep_index', sweep.sweep_index)
    table['command'] = _command_column(sweep)
    table['epoch_index'] = sweep.epoch_index_values
    return table


def _annotate_peak_table(
    *,
    core_peak_table: pd.DataFrame,
    sweep: SweepData,
    trace: AcqTrace,
    next_global_peak_id: int,
) -> pd.DataFrame:
    """Add trace acquisition and epoch identifiers to one peak table.

    Args:
        core_peak_table: Modality-neutral peak table for one sweep.
        sweep: Sweep that produced the core result.
        trace: Source trace acquisition.
        next_global_peak_id: First global peak identifier for this table.

    Returns:
        Peak table with channel, sweep, units, and epoch columns.
    """
    if core_peak_table.empty:
        return _empty_trace_peak_table()
    header = trace.trace_header
    epoch_table = sweep.get_epoch_table(samples_per_second=header.samples_per_second)
    rows: list[dict[str, object]] = []
    for zero_based, record in enumerate(core_peak_table.to_dict(orient='records')):
        peak_index = int(record['peak_index'])
        epoch_index = int(sweep.epoch_index_values[peak_index])
        epoch_row = _find_epoch_row(epoch_table=epoch_table, epoch_index=epoch_index)
        row: dict[str, object] = {
            'global_peak_id': next_global_peak_id + zero_based,
            'sweep_peak_id': int(record['peak_id']),
            'channel_index': sweep.channel_index,
            'channel_name': header.channel_names[sweep.channel_index],
            'value_units': sweep.value_units,
            'sweep_index': sweep.sweep_index,
            'epoch_index': epoch_index,
            'epoch_type': epoch_row.get('epoch_type', ''),
            'epoch_level': epoch_row.get('level', np.nan),
            'epoch_start_sec': epoch_row.get('start_sec', np.nan),
            'epoch_end_sec': epoch_row.get('end_sec', np.nan),
            'epoch_duration_sec': epoch_row.get('duration_sec', np.nan),
            'accepted': True,
        }
        row.update(record)
        rows.append(row)
    return pd.DataFrame(rows)


def _command_column(sweep: SweepData) -> np.ndarray:
    """Return command values or NaN values for one sweep.

    Args:
        sweep: Source sweep.

    Returns:
        Command array with one value per sample.
    """
    if sweep.command_values is None:
        return np.full(sweep.num_samples, np.nan, dtype=float)
    return sweep.command_values


def _find_epoch_row(*, epoch_table: pd.DataFrame, epoch_index: int) -> dict[str, object]:
    """Return epoch metadata for one epoch label.

    Args:
        epoch_table: Epoch interval table for one sweep.
        epoch_index: Epoch index assigned to the peak sample.

    Returns:
        Epoch row as a dictionary. Empty dictionary when no interval row exists.
    """
    if epoch_table.empty or epoch_index < 0:
        return {}
    matched = epoch_table.loc[epoch_table['epoch_index'] == epoch_index]
    if matched.empty:
        return {}
    return dict(matched.iloc[0])


def _empty_trace_peak_table() -> pd.DataFrame:
    """Return an empty trace peak table with stable columns.

    Returns:
        Empty DataFrame using the trace peak result schema.
    """
    return pd.DataFrame(
        columns=[
            'global_peak_id',
            'sweep_peak_id',
            'channel_index',
            'channel_name',
            'value_units',
            'sweep_index',
            'epoch_index',
            'epoch_type',
            'epoch_level',
            'epoch_start_sec',
            'epoch_end_sec',
            'epoch_duration_sec',
            'accepted',
            'peak_id',
            'peak_index',
            'peak_time_sec',
            'peak_value',
            'detection_value',
            'polarity',
            'height',
            'prominence',
            'width_sec',
        ]
    )
