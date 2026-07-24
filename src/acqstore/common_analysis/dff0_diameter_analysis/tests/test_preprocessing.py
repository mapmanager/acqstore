"""Tests for NaN-aware signal preprocessing."""

from __future__ import annotations

import numpy as np
import pytest

from acqstore.common_analysis.dff0_diameter_analysis.models import (
    SignalFilterMethod,
    TriggeredEventParams,
)
from acqstore.common_analysis.dff0_diameter_analysis.preprocessing import filter_signal


def test_none_preserves_nan() -> None:
    """NONE should pass missing values through unchanged."""
    signal = np.array([1.0, np.nan, 3.0])
    result = filter_signal(signal, TriggeredEventParams(filter_method=SignalFilterMethod.NONE))
    assert np.isnan(result[1])
    assert result[0] == 1.0
    assert result[2] == 3.0


def test_median_interpolates_before_filtering() -> None:
    """MEDIAN should fill gaps before median filtering and return finite values."""
    signal = np.array([1.0, np.nan, 3.0, 4.0, 3.0, 2.0, 1.0])
    result = filter_signal(
        signal,
        TriggeredEventParams(
            filter_method=SignalFilterMethod.MEDIAN,
            median_kernel_points=3,
        ),
    )
    assert np.all(np.isfinite(result))
    assert result.shape == signal.shape
    assert result[1] == pytest.approx(2.0)


def test_savgol_interpolates_before_filtering() -> None:
    """SAVGOL should fill gaps before smoothing and return finite values."""
    signal = np.array([10.0, np.nan, 10.0, 9.0, 8.0, 9.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    result = filter_signal(
        signal,
        TriggeredEventParams(
            filter_method=SignalFilterMethod.SAVGOL,
            savgol_window_points=5,
            savgol_polyorder=2,
        ),
    )
    assert np.all(np.isfinite(result))
    assert result.shape == signal.shape


def test_all_nan_filtered_methods_return_zeros() -> None:
    """Filtered methods should not raise when every sample is missing."""
    signal = np.array([np.nan, np.nan, np.nan])
    median = filter_signal(
        signal,
        TriggeredEventParams(
            filter_method=SignalFilterMethod.MEDIAN,
            median_kernel_points=3,
        ),
    )
    savgol = filter_signal(
        signal,
        TriggeredEventParams(
            filter_method=SignalFilterMethod.SAVGOL,
            savgol_window_points=3,
            savgol_polyorder=1,
        ),
    )
    assert np.all(median == 0.0)
    assert np.all(savgol == 0.0)


def test_finite_input_matches_prior_median_behavior() -> None:
    """Finite-only signals should still median-filter normally."""
    from scipy.signal import medfilt

    signal = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    result = filter_signal(
        signal,
        TriggeredEventParams(
            filter_method=SignalFilterMethod.MEDIAN,
            median_kernel_points=3,
        ),
    )
    assert result == pytest.approx(medfilt(signal, kernel_size=3))
