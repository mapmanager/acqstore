"""Tests for sum-intensity phase-2 event features."""

from __future__ import annotations

import numpy as np

from acqstore.acq_image.analysis.sum_intensity_analysis.sum_intensity_analysis import (
    SumIntensityAnalysis,
)
from acqstore.acq_image.analysis.sum_intensity_analysis.sum_intensity_core import (
    EventFeature,
    PeakEvent,
    SumIntensityTraceKey,
    run_sum_intensity,
)
from acqstore.acq_image.analysis.sum_intensity_analysis.sum_intensity_features import (
    SumIntensityFeatureCategory,
)


def _params(**overrides: object) -> dict[str, object]:
    """Return complete detection params for feature tests.

    Args:
        overrides: Parameter values to override.

    Returns:
        Detection parameter mapping.
    """
    params: dict[str, object] = {
        "window_radius_points": 0,
        "filter_method": "none",
        "median_filter_kernel_points": 3,
        "detrend_method": "none",
        "baseline_method": "percentile",
        "baseline_percentile": 0.0,
        "manual_f0_baseline": 1.0,
        "baseline_min_value": 1e-12,
        "baseline_window_ms": 10.0,
        "detection_method": "derivative_threshold",
        "polarity": "positive",
        "detection_source": SumIntensityTraceKey.DF_F_SIGNAL.value,
        "absolute_threshold": 1.0,
        "derivative_threshold_per_sec": 500.0,
        "refractory_period_ms": 2.0,
        "peak_search_window_ms": 10.0,
        "width_search_window_ms": 20.0,
        "level_fractions": "0.1,0.2,0.5,0.8,0.9",
    }
    params.update(overrides)
    return params


def test_feature_schema_documents_phase2_event_features() -> None:
    """Feature schema should document every phase-2 event feature."""
    schema = SumIntensityAnalysis.get_feature_schema()
    names = {field.name for field in schema}

    expected = {
        "baseline_mean",
        "baseline_std",
        "rise_10_90_sec",
        "decay_90_10_sec",
        "decay_time_sec",
        "max_rise_slope",
        "max_decay_slope",
        "auc",
        "prominence",
    }

    assert expected.issubset(names)
    assert all(field.algorithm for field in schema)
    assert {field.category for field in schema}.issuperset(
        {
            SumIntensityFeatureCategory.BASELINE,
            SumIntensityFeatureCategory.KINETICS,
            SumIntensityFeatureCategory.SLOPE,
            SumIntensityFeatureCategory.AREA,
            SumIntensityFeatureCategory.AMPLITUDE,
        }
    )

    df = SumIntensityAnalysis.get_feature_schema_dataframe()
    assert expected.issubset(set(df["name"].tolist()))
    assert "algorithm" in df.columns
    assert "category" in df.columns


def test_phase2_features_are_calculated_and_serialized() -> None:
    """Successful peaks should include documented event-feature records."""
    trace = np.array([1, 1, 1, 1, 2, 3, 6, 4, 2, 1, 1], dtype=float)
    image = np.repeat(trace[:, np.newaxis], 4, axis=1)

    result = run_sum_intensity(
        image,
        detection_params=_params(),
        physical_units=(0.001, 0.25),
    )

    event = result.events[0]
    assert event.baseline_mean.status == "ok"
    assert event.baseline_std.status == "ok"
    assert event.prominence.status == "ok"
    assert event.rise_10_90_sec.status == "ok"
    assert event.decay_90_10_sec.status == "ok"
    assert event.decay_time_sec.value == event.decay_90_10_sec.value
    assert event.max_rise_slope.status == "ok"
    assert event.max_decay_slope.status == "ok"
    assert event.auc.status == "ok"
    assert event.auc.value is not None and event.auc.value > 0

    record = event.to_json_dict()
    assert "features" in record
    parsed = PeakEvent.from_json_dict(record)
    assert parsed.auc == event.auc
    assert isinstance(parsed.prominence, EventFeature)


def test_decay_dependent_features_fail_without_right_10_crossing() -> None:
    """Decay and AUC features should fail when right_10 is unavailable."""
    trace = np.array([1, 1, 2, 3, 4, 5, 6], dtype=float)
    image = np.repeat(trace[:, np.newaxis], 3, axis=1)

    result = run_sum_intensity(
        image,
        detection_params=_params(
            derivative_threshold_per_sec=500.0,
            peak_search_window_ms=10.0,
            width_search_window_ms=5.0,
        ),
        physical_units=(0.001, 0.25),
    )

    event = result.events[0]
    assert event.decay_90_10_sec.status == "failed"
    assert event.decay_time_sec.status == "failed"
    assert event.max_decay_slope.status == "failed"
    assert event.max_decay_slope.reason == "right_10_crossing_unavailable"
    assert event.auc.status == "failed"
