"""Generic event analysis anchored to caller-supplied seed indices."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .models import (
    EventDirection,
    TriggeredEvent,
    TriggeredEventParams,
    TriggeredEventStatus,
)
from .preprocessing import filter_signal


def analyze_triggered_events(
    time: np.ndarray,
    signal: np.ndarray,
    seed_indices: Sequence[int],
    params: TriggeredEventParams,
) -> list[TriggeredEvent]:
    """Analyze one signal event for each supplied seed index.

    The function measures both the pre-seed state and the post-seed response.
    Extremum search is bounded by ``post_search_window_points``, the next seed,
    and the end of the array. The broader event window can continue beyond the
    extremum search to measure recovery and area.

    Args:
        time: Regularly sampled time coordinates in seconds.
        signal: Signal values aligned one-to-one with ``time``.
        seed_indices: Ordered event anchors in sample-index coordinates.
        params: Event measurement and filtering parameters.

    Returns:
        One schema-backed result per seed, in input order.
    """
    time_values = np.asarray(time, dtype=float)
    raw = np.asarray(signal, dtype=float)
    _validate_inputs(time_values, raw, seed_indices, params)
    filtered = filter_signal(raw, params)
    dt = float(np.median(np.diff(time_values)))
    ordered = [int(seed) for seed in seed_indices]

    return [
        _analyze_one_event(
            time=time_values,
            signal=filtered,
            seed_id=seed_id,
            seed_index=seed,
            next_seed_index=ordered[seed_id + 1] if seed_id + 1 < len(ordered) else None,
            dt=dt,
            params=params,
        )
        for seed_id, seed in enumerate(ordered)
    ]


def events_to_dataframe(events: Sequence[TriggeredEvent]) -> pd.DataFrame:
    """Return one flat row per triggered event."""
    return pd.DataFrame([event.to_dict() for event in events])


def _validate_inputs(
    time: np.ndarray,
    signal: np.ndarray,
    seed_indices: Sequence[int],
    params: TriggeredEventParams,
) -> None:
    if time.ndim != 1 or signal.ndim != 1 or len(time) != len(signal):
        raise ValueError("time and signal must be aligned one-dimensional arrays")
    if len(time) < 2:
        raise ValueError("at least two samples are required")
    if not np.all(np.isfinite(time)):
        raise ValueError("time contains invalid values")
    intervals = np.diff(time)
    dt = float(np.median(intervals))
    if dt <= 0 or not np.allclose(intervals, dt, rtol=0.0, atol=max(1e-12, dt * 1e-9)):
        raise ValueError("time must be strictly increasing and regularly sampled")
    seeds = [int(seed) for seed in seed_indices]
    if seeds != sorted(seeds) or len(seeds) != len(set(seeds)):
        raise ValueError("seed_indices must be unique and increasing")
    if any(seed < 0 or seed >= len(signal) for seed in seeds):
        raise ValueError("seed index is outside the signal")
    if params.pre_points < 0 or params.post_points < 1:
        raise ValueError("pre_points must be >= 0 and post_points must be >= 1")
    if params.post_search_window_points < 1:
        raise ValueError("post_search_window_points must be >= 1")
    if not 0.0 < params.recovery_fraction <= 1.0:
        raise ValueError("recovery_fraction must be in (0, 1]")


def _analyze_one_event(
    *,
    time: np.ndarray,
    signal: np.ndarray,
    seed_id: int,
    seed_index: int,
    next_seed_index: int | None,
    dt: float,
    params: TriggeredEventParams,
) -> TriggeredEvent:
    warnings: list[str] = []
    n = len(signal)
    window_start = max(0, seed_index - params.pre_points)
    nominal_stop = seed_index + params.post_points + 1
    stop_by_signal = min(n, nominal_stop)
    window_stop = min(stop_by_signal, next_seed_index) if next_seed_index is not None else stop_by_signal
    truncated_next = next_seed_index is not None and next_seed_index < stop_by_signal
    truncated_end = nominal_stop > n

    baseline_start = max(window_start, seed_index + params.baseline_start_offset_points)
    baseline_stop = min(seed_index, seed_index + params.baseline_stop_offset_points)
    if baseline_stop <= baseline_start:
        return _failed_event(seed_id, seed_index, time, window_start, window_stop, next_seed_index,
                             truncated_next, truncated_end, "baseline interval is empty")

    baseline_values = signal[baseline_start:baseline_stop]
    if len(baseline_values) == 0 or not np.all(np.isfinite(baseline_values)):
        return _failed_event(seed_id, seed_index, time, window_start, window_stop, next_seed_index,
                             truncated_next, truncated_end, "baseline contains insufficient valid data")
    baseline = float(np.median(baseline_values))
    baseline_std = float(np.std(baseline_values))
    baseline_slope = _linear_slope(time[baseline_start:baseline_stop], baseline_values)
    pre_seed_value = float(signal[seed_index - 1]) if seed_index > 0 else None
    pre_seed_change = None if pre_seed_value is None else pre_seed_value - baseline

    search_stop = min(window_stop, seed_index + params.post_search_window_points + 1)
    if search_stop <= seed_index:
        return _failed_event(seed_id, seed_index, time, window_start, window_stop, next_seed_index,
                             truncated_next, truncated_end, "post-seed extremum search interval is empty")

    search = signal[seed_index:search_stop]
    local_extremum = int(np.argmin(search) if params.direction is EventDirection.NEGATIVE else np.argmax(search))
    extremum_index = seed_index + local_extremum
    extremum_value = float(signal[extremum_index])
    signed_amplitude = extremum_value - baseline
    amplitude = abs(signed_amplitude)
    to_ext_points = extremum_index - seed_index
    to_ext_sec = float(time[extremum_index] - time[seed_index])
    avg_slope = None if to_ext_sec <= 0 else signed_amplitude / to_ext_sec

    oriented = (baseline - signal) if params.direction is EventDirection.NEGATIVE else (signal - baseline)
    derivative = np.gradient(oriented, dt)
    max_oriented_slope = float(np.max(derivative[seed_index : extremum_index + 1]))

    recovery_index = _find_recovery(
        oriented=oriented,
        extremum_index=extremum_index,
        stop_index=window_stop,
        amplitude=amplitude,
        recovery_fraction=params.recovery_fraction,
    )
    recovery_detected = recovery_index is not None
    if not recovery_detected:
        warnings.append("recovery_not_detected_before_event_stop")
    if truncated_next:
        warnings.append("event_truncated_by_next_seed")
    if truncated_end:
        warnings.append("event_truncated_by_signal_end")

    auc = float(np.trapezoid(signal[seed_index:window_stop] - baseline, time[seed_index:window_stop])) if window_stop - seed_index >= 2 else None
    fractional = None if baseline == 0 else signed_amplitude / baseline
    status = TriggeredEventStatus.OK if not warnings else TriggeredEventStatus.PARTIAL

    return TriggeredEvent(
        schema_version=1, seed_id=seed_id, seed_index=seed_index,
        seed_time_sec=float(time[seed_index]), window_start_index=window_start,
        window_stop_index=window_stop, next_seed_index=next_seed_index,
        truncated_by_next_seed=truncated_next, truncated_by_signal_end=truncated_end,
        status=status, warnings=tuple(warnings), baseline_start_index=baseline_start,
        baseline_stop_index=baseline_stop, baseline_value=baseline,
        baseline_std=baseline_std, baseline_slope_per_sec=baseline_slope,
        pre_seed_value=pre_seed_value, pre_seed_change=pre_seed_change,
        extremum_index=extremum_index, extremum_time_sec=float(time[extremum_index]),
        extremum_value=extremum_value,
        time_to_extremum_from_seed_points=to_ext_points,
        time_to_extremum_from_seed_sec=to_ext_sec,
        signed_amplitude=signed_amplitude, amplitude=amplitude,
        fractional_amplitude=fractional,
        percent_amplitude=None if fractional is None else 100.0 * fractional,
        average_seed_to_extremum_slope_per_sec=avg_slope,
        maximum_oriented_slope_per_sec=max_oriented_slope,
        recovery_detected=recovery_detected, recovery_index=recovery_index,
        recovery_time_sec=None if recovery_index is None else float(time[recovery_index]),
        extremum_to_recovery_sec=None if recovery_index is None else float(time[recovery_index] - time[extremum_index]),
        seed_to_recovery_sec=None if recovery_index is None else float(time[recovery_index] - time[seed_index]),
        baseline_adjusted_auc_seed_to_stop=auc,
    )


def _find_recovery(*, oriented: np.ndarray, extremum_index: int, stop_index: int,
                   amplitude: float, recovery_fraction: float) -> int | None:
    if amplitude <= 0:
        return None
    remaining_threshold = amplitude * (1.0 - recovery_fraction)
    for index in range(extremum_index + 1, stop_index):
        if oriented[index] <= remaining_threshold:
            return index
    return None


def _linear_slope(time: np.ndarray, values: np.ndarray) -> float | None:
    if len(values) < 2:
        return None
    return float(np.polyfit(time, values, deg=1)[0])


def _failed_event(seed_id: int, seed_index: int, time: np.ndarray, window_start: int,
                  window_stop: int, next_seed_index: int | None,
                  truncated_next: bool, truncated_end: bool, reason: str) -> TriggeredEvent:
    return TriggeredEvent(
        schema_version=1, seed_id=seed_id, seed_index=seed_index,
        seed_time_sec=float(time[seed_index]), window_start_index=window_start,
        window_stop_index=window_stop, next_seed_index=next_seed_index,
        truncated_by_next_seed=truncated_next, truncated_by_signal_end=truncated_end,
        status=TriggeredEventStatus.FAILED, warnings=(reason,),
        baseline_start_index=None, baseline_stop_index=None, baseline_value=None,
        baseline_std=None, baseline_slope_per_sec=None, pre_seed_value=None,
        pre_seed_change=None, extremum_index=None, extremum_time_sec=None,
        extremum_value=None, time_to_extremum_from_seed_points=None,
        time_to_extremum_from_seed_sec=None, signed_amplitude=None, amplitude=None,
        fractional_amplitude=None, percent_amplitude=None,
        average_seed_to_extremum_slope_per_sec=None,
        maximum_oriented_slope_per_sec=None, recovery_detected=False,
        recovery_index=None, recovery_time_sec=None, extremum_to_recovery_sec=None,
        seed_to_recovery_sec=None, baseline_adjusted_auc_seed_to_stop=None,
    )
