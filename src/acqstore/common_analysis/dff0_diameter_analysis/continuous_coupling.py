"""Generic continuous lagged-correlation analysis for aligned signals."""

from __future__ import annotations

import numpy as np

from .models import (
    LaggedCorrelationParams,
    LaggedCorrelationResult,
    SignalFilterMethod,
    TriggeredEventParams,
)
from .preprocessing import filter_signal


def preprocess_continuous_signals(
    *,
    signal_a: np.ndarray,
    signal_b: np.ndarray,
    params: LaggedCorrelationParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter and optionally linearly detrend two continuous signals.

    Args:
        signal_a: First one-dimensional signal.
        signal_b: Second one-dimensional signal.
        params: Continuous preprocessing parameters.

    Returns:
        Processed copies of ``signal_a`` and ``signal_b``.
    """
    a_values = np.asarray(signal_a, dtype=float)
    b_values = np.asarray(signal_b, dtype=float)
    if a_values.ndim != 1 or b_values.ndim != 1:
        raise ValueError("signals must be one-dimensional")
    if len(a_values) != len(b_values):
        raise ValueError("signal lengths must match")

    filtered_a = _filter_for_continuous_analysis(
        a_values,
        method=params.reporter_filter_method,
        median_kernel_points=params.reporter_median_kernel_points,
        savgol_window_points=params.reporter_savgol_window_points,
        savgol_polyorder=params.reporter_savgol_polyorder,
    )
    filtered_b = _filter_for_continuous_analysis(
        b_values,
        method=params.diameter_filter_method,
        median_kernel_points=params.diameter_median_kernel_points,
        savgol_window_points=params.diameter_savgol_window_points,
        savgol_polyorder=params.diameter_savgol_polyorder,
    )
    if params.remove_linear_trend:
        filtered_a = _remove_linear_trend(filtered_a)
        filtered_b = _remove_linear_trend(filtered_b)
    return filtered_a, filtered_b


def analyze_lagged_correlation(
    *,
    time: np.ndarray,
    signal_a: np.ndarray,
    signal_b: np.ndarray,
    params: LaggedCorrelationParams,
) -> LaggedCorrelationResult:
    """Measure normalized Pearson correlation across integer lags.

    Positive lag means ``signal_a`` leads and ``signal_b`` follows. For the
    DFF0/diameter adapter, ``signal_a`` is the reporter and ``signal_b`` is
    diameter.

    Args:
        time: Shared regularly sampled time coordinate in seconds.
        signal_a: First aligned signal.
        signal_b: Second aligned signal.
        params: Filtering, detrending, lag-range, and overlap parameters.

    Returns:
        Schema-backed lagged-correlation result.

    Raises:
        ValueError: If arrays are not one-dimensional, aligned, regularly
            sampled, or compatible with the requested parameters.
    """
    time_values = np.asarray(time, dtype=float)
    a_values = np.asarray(signal_a, dtype=float)
    b_values = np.asarray(signal_b, dtype=float)
    _validate_inputs(time_values, a_values, b_values, params)

    seconds_per_point = float(np.median(np.diff(time_values)))
    filtered_a, filtered_b = preprocess_continuous_signals(
        signal_a=a_values,
        signal_b=b_values,
        params=params,
    )

    lag_points = tuple(range(-params.max_lag_points, params.max_lag_points + 1))
    correlations: list[float | None] = []
    overlaps: list[int] = []
    warnings: list[str] = []

    for lag in lag_points:
        a_segment, b_segment = _overlapping_segments(filtered_a, filtered_b, lag)
        valid = np.isfinite(a_segment) & np.isfinite(b_segment)
        valid_count = int(np.count_nonzero(valid))
        overlaps.append(valid_count)
        if valid_count < params.minimum_overlap_points:
            correlations.append(None)
            continue

        a_valid = a_segment[valid]
        b_valid = b_segment[valid]
        if np.isclose(np.std(a_valid), 0.0) or np.isclose(np.std(b_valid), 0.0):
            correlations.append(None)
            continue
        correlations.append(float(np.corrcoef(a_valid, b_valid)[0, 1]))

    finite_pairs = [
        (lag, correlation)
        for lag, correlation in zip(lag_points, correlations, strict=True)
        if correlation is not None and np.isfinite(correlation)
    ]
    if not finite_pairs:
        warnings.append("no_valid_lag_correlations")

    strongest_positive = _select_extreme(finite_pairs, mode="positive")
    strongest_negative = _select_extreme(finite_pairs, mode="negative")
    strongest_absolute = _select_extreme(finite_pairs, mode="absolute")
    zero_index = params.max_lag_points
    zero_lag = correlations[zero_index]

    return LaggedCorrelationResult(
        schema_version=1,
        seconds_per_point=seconds_per_point,
        lag_points=lag_points,
        lag_seconds=tuple(lag * seconds_per_point for lag in lag_points),
        correlation=tuple(correlations),
        overlap_points=tuple(overlaps),
        zero_lag_correlation=zero_lag,
        strongest_positive_correlation=_value_or_none(strongest_positive),
        strongest_positive_lag_points=_lag_or_none(strongest_positive),
        strongest_positive_lag_sec=_lag_seconds_or_none(
            strongest_positive, seconds_per_point
        ),
        strongest_negative_correlation=_value_or_none(strongest_negative),
        strongest_negative_lag_points=_lag_or_none(strongest_negative),
        strongest_negative_lag_sec=_lag_seconds_or_none(
            strongest_negative, seconds_per_point
        ),
        strongest_absolute_correlation=_value_or_none(strongest_absolute),
        strongest_absolute_lag_points=_lag_or_none(strongest_absolute),
        strongest_absolute_lag_sec=_lag_seconds_or_none(
            strongest_absolute, seconds_per_point
        ),
        warnings=tuple(warnings),
    )


def _validate_inputs(
    time: np.ndarray,
    signal_a: np.ndarray,
    signal_b: np.ndarray,
    params: LaggedCorrelationParams,
) -> None:
    """Validate aligned regularly sampled arrays and lag parameters."""
    if time.ndim != 1 or signal_a.ndim != 1 or signal_b.ndim != 1:
        raise ValueError("time and signals must be one-dimensional")
    if len(time) != len(signal_a) or len(time) != len(signal_b):
        raise ValueError("time and signal lengths must match")
    if len(time) < 3:
        raise ValueError("at least three samples are required")
    intervals = np.diff(time)
    seconds_per_point = float(np.median(intervals))
    if seconds_per_point <= 0:
        raise ValueError("time must increase monotonically")
    if not np.allclose(intervals, seconds_per_point, rtol=0.0, atol=1e-9):
        raise ValueError("time must be regularly sampled")
    if params.max_lag_points < 0:
        raise ValueError("max_lag_points must be non-negative")
    if params.max_lag_points >= len(time):
        raise ValueError("max_lag_points must be smaller than the signal length")
    if params.minimum_overlap_points < 2:
        raise ValueError("minimum_overlap_points must be at least 2")
    if params.minimum_overlap_points > len(time):
        raise ValueError("minimum_overlap_points cannot exceed signal length")


def _filter_for_continuous_analysis(
    signal: np.ndarray,
    *,
    method: SignalFilterMethod,
    median_kernel_points: int,
    savgol_window_points: int,
    savgol_polyorder: int,
) -> np.ndarray:
    """Apply one existing package filter with continuous-analysis settings."""
    filter_params = TriggeredEventParams(
        filter_method=method,
        median_kernel_points=median_kernel_points,
        savgol_window_points=savgol_window_points,
        savgol_polyorder=savgol_polyorder,
    )
    return filter_signal(signal, filter_params)


def _remove_linear_trend(signal: np.ndarray) -> np.ndarray:
    """Subtract a least-squares line from finite samples of one signal."""
    values = np.asarray(signal, dtype=float).copy()
    finite = np.isfinite(values)
    if np.count_nonzero(finite) < 2:
        return values
    x = np.arange(values.size, dtype=float)
    slope, intercept = np.polyfit(x[finite], values[finite], deg=1)
    values[finite] = values[finite] - (slope * x[finite] + intercept)
    return values


def _overlapping_segments(
    signal_a: np.ndarray,
    signal_b: np.ndarray,
    lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned overlap for one lag using the documented sign rule."""
    if lag > 0:
        return signal_a[:-lag], signal_b[lag:]
    if lag < 0:
        offset = -lag
        return signal_a[offset:], signal_b[:-offset]
    return signal_a, signal_b


def _select_extreme(
    pairs: list[tuple[int, float]],
    *,
    mode: str,
) -> tuple[int, float] | None:
    """Select one positive, negative, or absolute correlation extreme."""
    if mode == "positive":
        candidates = [pair for pair in pairs if pair[1] > 0]
        return max(candidates, key=lambda pair: pair[1], default=None)
    if mode == "negative":
        candidates = [pair for pair in pairs if pair[1] < 0]
        return min(candidates, key=lambda pair: pair[1], default=None)
    if mode == "absolute":
        return max(pairs, key=lambda pair: abs(pair[1]), default=None)
    raise ValueError(f"Unknown extreme mode: {mode}")


def _lag_or_none(pair: tuple[int, float] | None) -> int | None:
    """Return the lag component of an optional lag/correlation pair."""
    return None if pair is None else pair[0]


def _value_or_none(pair: tuple[int, float] | None) -> float | None:
    """Return the correlation component of an optional pair."""
    return None if pair is None else pair[1]


def _lag_seconds_or_none(
    pair: tuple[int, float] | None,
    seconds_per_point: float,
) -> float | None:
    """Convert an optional lag from points to seconds."""
    return None if pair is None else pair[0] * seconds_per_point
