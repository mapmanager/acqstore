"""Tests for generic continuous lagged-correlation analysis."""

from __future__ import annotations

import numpy as np

from acqstore.common_analysis.dff0_diameter_analysis.continuous_coupling import (
    analyze_lagged_correlation,
)
from acqstore.common_analysis.dff0_diameter_analysis.models import (
    LaggedCorrelationParams,
    LaggedCorrelationResult,
    SignalFilterMethod,
)


def test_negative_correlation_detects_positive_reporter_lead() -> None:
    """A delayed inverse signal should peak negatively at a positive lag."""
    rng = np.random.default_rng(42)
    reporter = rng.normal(size=500)
    delay_points = 7
    diameter = np.zeros_like(reporter)
    diameter[delay_points:] = -reporter[:-delay_points]
    time = np.arange(reporter.size, dtype=float) * 0.01
    params = LaggedCorrelationParams(
        max_lag_points=20,
        minimum_overlap_points=100,
        reporter_filter_method=SignalFilterMethod.NONE,
        diameter_filter_method=SignalFilterMethod.NONE,
    )

    result = analyze_lagged_correlation(
        time=time,
        signal_a=reporter,
        signal_b=diameter,
        params=params,
    )

    assert result.strongest_negative_lag_points == delay_points
    assert result.strongest_negative_lag_sec is not None
    assert np.isclose(result.strongest_negative_lag_sec, delay_points * 0.01)
    assert result.strongest_negative_correlation is not None
    assert result.strongest_negative_correlation < -0.99


def test_result_serialization_round_trip() -> None:
    """Lagged-correlation results should preserve their schema on round trip."""
    time = np.arange(100, dtype=float) * 0.1
    signal = np.sin(time)
    result = analyze_lagged_correlation(
        time=time,
        signal_a=signal,
        signal_b=signal,
        params=LaggedCorrelationParams(
            max_lag_points=5,
            minimum_overlap_points=50,
            reporter_filter_method=SignalFilterMethod.NONE,
            diameter_filter_method=SignalFilterMethod.NONE,
        ),
    )

    restored = LaggedCorrelationResult.from_dict(result.to_dict())

    assert restored == result
    assert list(restored.to_dataframe().columns) == [
        "lag_points",
        "lag_sec",
        "correlation",
        "overlap_points",
    ]


def test_default_filters_match_initial_continuous_analysis_choice() -> None:
    """Defaults should use light reporter median and Sav-Gol diameter filtering."""
    params = LaggedCorrelationParams()

    assert params.reporter_filter_method is SignalFilterMethod.MEDIAN
    assert params.reporter_median_kernel_points == 3
    assert params.diameter_filter_method is SignalFilterMethod.SAVGOL
    assert params.diameter_savgol_window_points == 15
    assert params.diameter_savgol_polyorder == 4
