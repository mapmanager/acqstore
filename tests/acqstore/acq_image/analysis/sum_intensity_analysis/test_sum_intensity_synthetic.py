"""Tests for synthetic sum-intensity analysis utilities."""

from __future__ import annotations

import numpy as np

from acqstore.acq_image.analysis.sum_intensity_analysis.sum_intensity_core import (
    PeakWidthLevel,
    SumIntensityEventPointKey,
    SumIntensityTraceKey,
    run_sum_intensity,
)
from acqstore.acq_image.analysis.sum_intensity_analysis.synthetic.synthetic_config import (
    SyntheticSumIntensityConfig,
)
from acqstore.acq_image.analysis.sum_intensity_analysis.synthetic.synthetic_events import (
    generate_poisson_event_times,
)
from acqstore.acq_image.analysis.sum_intensity_analysis.synthetic.synthetic_generator import (
    make_synthetic_sum_intensity_image,
)


def _default_params() -> dict[str, object]:
    """Return deterministic synthetic-analysis params for tests."""
    return {
        "window_radius_points": 0,
        "filter_method": "median",
        "median_filter_kernel_points": 3,
        "detrend_method": "single_exponential",
        "baseline_method": "percentile",
        "baseline_percentile": 20.0,
        "baseline_min_value": 1e-12,
        "baseline_window_ms": 3.0,
        "detection_method": "derivative_threshold",
        "polarity": "positive",
        "detection_source": SumIntensityTraceKey.DF_F_SIGNAL.value,
        "absolute_threshold": 0.1,
        "derivative_threshold_per_sec": 3.0,
        "refractory_period_ms": 500.0,
        "peak_search_window_ms": 300.0,
        "width_search_window_ms": 900.0,
        "level_fractions": "0.1,0.2,0.5,0.8,0.9",
    }


def test_synthetic_image_has_expected_shape_and_ground_truth() -> None:
    """Synthetic generator returns a real image and event table."""
    data = make_synthetic_sum_intensity_image(
        SyntheticSumIntensityConfig(
            num_timepoints=1000,
            num_spacepoints=64,
            event_times_sec=(0.5, 1.5, 2.5),
            noise_sigma=0.0,
            spatial_noise_sigma=0.0,
            pop_probability=0.0,
            seed=1,
        )
    )

    assert data.image.shape == (1000, 64)
    assert data.time_sec.shape == (1000,)
    assert len(data.ground_truth_events) == 3
    assert list(data.ground_truth_events.columns) == [
        "event_id",
        "onset_time_sec",
        "peak_time_sec",
        "amplitude",
    ]


def test_core_analysis_detects_synthetic_events_and_accessors() -> None:
    """Core analysis detects deterministic synthetic events and exposes primitives."""
    data = make_synthetic_sum_intensity_image(
        SyntheticSumIntensityConfig(
            event_times_sec=(1.0, 2.4, 3.8),
            event_amplitude=350.0,
            noise_sigma=1.0,
            spatial_noise_sigma=1.0,
            pop_probability=0.0,
            seed=2,
        )
    )
    result = run_sum_intensity(
        data.image,
        detection_params=_default_params(),
        physical_units=(data.seconds_per_line, data.um_per_pixel),
    )

    df_f = result.get_trace(SumIntensityTraceKey.DF_F_SIGNAL)
    peaks = result.get_event_points(SumIntensityEventPointKey.PEAKS)
    widths = result.get_width_trace(PeakWidthLevel.WIDTH_50)

    assert df_f.x.shape == df_f.y.shape == (data.image.shape[0],)
    assert result.summary["num_peaks"] == 3
    assert peaks.x.size == 3
    assert widths.metadata["connectgaps"] is False
    assert np.any(np.isnan(widths.x))


def test_width_search_window_records_failure() -> None:
    """Too-short width search window records level-crossing failures."""
    data = make_synthetic_sum_intensity_image(
        SyntheticSumIntensityConfig(
            event_times_sec=(1.0,),
            tau_decay_sec=1.0,
            event_amplitude=400.0,
            noise_sigma=0.0,
            spatial_noise_sigma=0.0,
            pop_probability=0.0,
            seed=3,
        )
    )
    params = _default_params()
    params["width_search_window_ms"] = 20.0
    result = run_sum_intensity(
        data.image,
        detection_params=params,
        physical_units=(data.seconds_per_line, data.um_per_pixel),
    )

    events = result.get_peak_events()
    assert events
    statuses = {crossing.status for crossing in events[0].level_crossings}
    assert "right_not_found_within_width_search_window" in statuses


def test_poisson_event_generation_is_seeded_and_sorted() -> None:
    """Poisson helper returns deterministic sorted times for seeded RNG."""
    rng1 = np.random.default_rng(10)
    rng2 = np.random.default_rng(10)

    times1 = generate_poisson_event_times(duration_sec=10.0, rate_hz=1.5, rng=rng1)
    times2 = generate_poisson_event_times(duration_sec=10.0, rate_hz=1.5, rng=rng2)

    assert np.array_equal(times1, times2)
    assert np.all(np.diff(times1) > 0)
    assert np.all(times1 >= 0.0)
    assert np.all(times1 < 10.0)
