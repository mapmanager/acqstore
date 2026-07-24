"""Modality-neutral one-dimensional peak detection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.signal import find_peaks, peak_widths

PeakPolarity = Literal['positive', 'negative']


@dataclass(frozen=True)
class PeakDetectionCoreParams:
    """Parameters for one-dimensional peak detection.

    Args:
        polarity: Whether to detect positive or negative peaks.
        height: Optional minimum peak height in transformed detection-signal
            units. For negative peaks, values are internally inverted before
            detection.
        prominence: Optional minimum prominence in transformed
            detection-signal units.
        min_distance_sec: Optional minimum distance between detected peaks in
            seconds.
        width_rel_height: Relative height used by SciPy when measuring peak
            widths. ``0.5`` measures full width at half prominence.
    """

    polarity: PeakPolarity = 'positive'
    height: float | None = None
    prominence: float | None = None
    min_distance_sec: float | None = None
    width_rel_height: float = 0.5

    def __post_init__(self) -> None:
        """Validate parameters.

        Raises:
            ValueError: If a parameter is outside the supported range.
        """
        if self.polarity not in {'positive', 'negative'}:
            raise ValueError(f'polarity must be "positive" or "negative", got {self.polarity!r}')
        if self.min_distance_sec is not None and self.min_distance_sec < 0:
            raise ValueError(f'min_distance_sec must be >= 0, got {self.min_distance_sec}')
        if self.width_rel_height <= 0 or self.width_rel_height >= 1:
            raise ValueError(f'width_rel_height must be between 0 and 1, got {self.width_rel_height}')


@dataclass(frozen=True)
class PeakDetectionCoreResult:
    """Result of modality-neutral one-dimensional peak detection.

    Args:
        trace_table: Per-sample table with detection-signal and peak-marker
            columns.
        peak_table: One row per detected peak.
    """

    trace_table: pd.DataFrame
    peak_table: pd.DataFrame

    @property
    def num_peaks(self) -> int:
        """Return the number of detected peaks.

        Returns:
            Number of rows in :attr:`peak_table`.
        """
        return int(len(self.peak_table))


def detect_peaks_1d(
    *,
    time_sec: npt.NDArray[np.floating],
    values: npt.NDArray[np.floating],
    params: PeakDetectionCoreParams | None = None,
) -> PeakDetectionCoreResult:
    """Detect peaks in one sampled one-dimensional trace.

    This function is deliberately modality-neutral. It does not know about ABF
    files, image ROIs, sweeps, epochs, CSV files, or GUI concepts. Callers are
    responsible for adding acquisition-specific identifiers to returned tables.

    Args:
        time_sec: One-dimensional time axis in seconds.
        values: One-dimensional signal values sampled at ``time_sec``.
        params: Optional detection parameters. Defaults to positive-peak
            detection with no height, prominence, or distance filter.

    Returns:
        Core detection result containing a per-sample trace table and one peak
        row per detected peak.

    Raises:
        ValueError: If arrays are not one-dimensional, lengths do not match,
            time is not strictly increasing, or parameters are invalid.
    """
    normalized_params = params or PeakDetectionCoreParams()
    time = np.asarray(time_sec, dtype=float)
    signal = np.asarray(values, dtype=float)
    _validate_trace_arrays(time, signal)
    detection_signal = signal if normalized_params.polarity == 'positive' else -signal
    distance_points = _distance_sec_to_points(
        time_sec=time,
        min_distance_sec=normalized_params.min_distance_sec,
    )
    peak_indices, properties = find_peaks(
        detection_signal,
        height=normalized_params.height,
        prominence=normalized_params.prominence,
        distance=distance_points,
    )
    widths_sec = _measure_widths_sec(
        detection_signal=detection_signal,
        peak_indices=peak_indices,
        time_sec=time,
        rel_height=normalized_params.width_rel_height,
    )
    trace_table = _build_trace_table(
        time_sec=time,
        values=signal,
        detection_signal=detection_signal,
        peak_indices=peak_indices,
    )
    peak_table = _build_peak_table(
        time_sec=time,
        values=signal,
        detection_signal=detection_signal,
        peak_indices=peak_indices,
        properties=properties,
        widths_sec=widths_sec,
        polarity=normalized_params.polarity,
    )
    return PeakDetectionCoreResult(trace_table=trace_table, peak_table=peak_table)


def _validate_trace_arrays(time_sec: npt.NDArray[np.floating], values: npt.NDArray[np.floating]) -> None:
    """Validate one-dimensional trace arrays.

    Args:
        time_sec: Time axis.
        values: Signal values.

    Raises:
        ValueError: If arrays are invalid.
    """
    if time_sec.ndim != 1:
        raise ValueError(f'time_sec must be one-dimensional, got shape {time_sec.shape}')
    if values.ndim != 1:
        raise ValueError(f'values must be one-dimensional, got shape {values.shape}')
    if len(time_sec) != len(values):
        raise ValueError(f'time_sec and values must have same length, got {len(time_sec)} and {len(values)}')
    if len(time_sec) == 0:
        raise ValueError('time_sec and values must not be empty')
    if np.isnan(time_sec).any() or np.isnan(values).any():
        raise ValueError('time_sec and values must not contain NaN')
    if np.any(np.diff(time_sec) <= 0):
        raise ValueError('time_sec must be strictly increasing')


def _distance_sec_to_points(
    *,
    time_sec: npt.NDArray[np.floating],
    min_distance_sec: float | None,
) -> int | None:
    """Convert minimum peak distance from seconds to sample points.

    Args:
        time_sec: Strictly increasing time axis.
        min_distance_sec: Minimum distance in seconds.

    Returns:
        Minimum distance in points, or None when no distance was requested.
    """
    if min_distance_sec is None:
        return None
    if min_distance_sec == 0:
        return 1
    median_dt = float(np.median(np.diff(time_sec)))
    return max(1, int(round(min_distance_sec / median_dt)))


def _measure_widths_sec(
    *,
    detection_signal: npt.NDArray[np.floating],
    peak_indices: npt.NDArray[np.integer],
    time_sec: npt.NDArray[np.floating],
    rel_height: float,
) -> npt.NDArray[np.floating]:
    """Measure peak widths in seconds.

    Args:
        detection_signal: Signal passed to ``find_peaks``.
        peak_indices: Detected peak indices.
        time_sec: Time axis.
        rel_height: SciPy relative-height argument.

    Returns:
        Widths in seconds, one per peak.
    """
    if len(peak_indices) == 0:
        return np.asarray([], dtype=float)
    widths_points = peak_widths(detection_signal, peak_indices, rel_height=rel_height)[0]
    median_dt = float(np.median(np.diff(time_sec)))
    return np.asarray(widths_points, dtype=float) * median_dt


def _build_trace_table(
    *,
    time_sec: npt.NDArray[np.floating],
    values: npt.NDArray[np.floating],
    detection_signal: npt.NDArray[np.floating],
    peak_indices: npt.NDArray[np.integer],
) -> pd.DataFrame:
    """Build a per-sample trace table.

    Args:
        time_sec: Time axis in seconds.
        values: Original signal values.
        detection_signal: Polarity-normalized signal used for detection.
        peak_indices: Detected peak indices.

    Returns:
        DataFrame with one row per sample.
    """
    is_peak = np.zeros(len(time_sec), dtype=bool)
    peak_id = np.full(len(time_sec), -1, dtype=int)
    for zero_based, peak_index in enumerate(peak_indices):
        index = int(peak_index)
        is_peak[index] = True
        peak_id[index] = zero_based + 1
    return pd.DataFrame(
        {
            'sample_index': np.arange(len(time_sec), dtype=int),
            'time_sec': time_sec,
            'value': values,
            'detection_signal': detection_signal,
            'is_peak': is_peak,
            'peak_id': peak_id,
        }
    )


def _build_peak_table(
    *,
    time_sec: npt.NDArray[np.floating],
    values: npt.NDArray[np.floating],
    detection_signal: npt.NDArray[np.floating],
    peak_indices: npt.NDArray[np.integer],
    properties: dict[str, npt.NDArray[np.floating]],
    widths_sec: npt.NDArray[np.floating],
    polarity: PeakPolarity,
) -> pd.DataFrame:
    """Build one row per detected peak.

    Args:
        time_sec: Time axis in seconds.
        values: Original signal values.
        detection_signal: Polarity-normalized signal used for detection.
        peak_indices: Detected peak indices.
        properties: Properties returned by SciPy ``find_peaks``.
        widths_sec: Width measurements in seconds.
        polarity: Detection polarity.

    Returns:
        DataFrame with one row per detected peak.
    """
    rows: list[dict[str, object]] = []
    prominences = properties.get('prominences')
    peak_heights = properties.get('peak_heights')
    for zero_based, raw_index in enumerate(peak_indices):
        peak_index = int(raw_index)
        rows.append(
            {
                'peak_id': zero_based + 1,
                'peak_index': peak_index,
                'peak_time_sec': float(time_sec[peak_index]),
                'peak_value': float(values[peak_index]),
                'detection_value': float(detection_signal[peak_index]),
                'polarity': polarity,
                'height': _optional_property_value(peak_heights, zero_based),
                'prominence': _optional_property_value(prominences, zero_based),
                'width_sec': float(widths_sec[zero_based]) if zero_based < len(widths_sec) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _optional_property_value(values: npt.NDArray[np.floating] | None, index: int) -> float:
    """Return a SciPy property value or NaN.

    Args:
        values: Optional array returned by SciPy.
        index: Index into ``values``.

    Returns:
        Float property value or NaN when unavailable.
    """
    if values is None or index >= len(values):
        return float('nan')
    return float(values[index])
