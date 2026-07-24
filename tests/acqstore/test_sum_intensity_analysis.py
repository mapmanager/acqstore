"""Tests for SumIntensityAnalysis BaseAnalysis wrapper."""

from __future__ import annotations

import numpy as np

from acqstore.acq_image.acq_analysis_set import AcqAnalysisSet
from acqstore.acq_image.analysis.model import AnalysisKey
from acqstore.acq_image.analysis.registry import get_analysis_registry
from acqstore.acq_image.analysis.sum_intensity_analysis.sum_intensity_analysis import (
    SumIntensityAnalysis,
)
from acqstore.acq_image.analysis.sum_intensity_analysis.sum_intensity_core import (
    PeakWidthLevel,
    SumIntensityEventPointKey,
    SumIntensitySummaryKey,
    SumIntensityTraceKey,
)


class FakeProvider:
    """Fake sum-intensity analysis data provider."""

    def get_roi_image(self, channel: int, roi_id: int) -> np.ndarray:
        """Return synthetic ROI kymograph data.

        Args:
            channel: Channel index.
            roi_id: ROI identifier.

        Returns:
            Synthetic two-dimensional ROI image.
        """
        _ = channel, roi_id
        trace = np.array([1, 1, 2, 3, 6, 3, 1, 1], dtype=float)
        return np.repeat(trace[:, np.newaxis], 4, axis=1)

    def get_image_physical_units(self) -> tuple[float, float]:
        """Return synthetic physical units.

        Returns:
            ``(seconds_per_line, um_per_pixel)``.
        """
        return (0.001, 0.25)


def _params() -> dict[str, object]:
    """Return deterministic wrapper-test detection params.

    Returns:
        Detection parameter mapping using the current public API names.
    """
    return {
        "window_radius_points": 0,
        "filter_method": "none",
        "median_filter_kernel_points": 3,
        "detrend_method": "none",
        "baseline_method": "percentile",
        "baseline_percentile": 0.0,
        "baseline_min_value": 1e-12,
        "baseline_window_ms": 3.0,
        "detection_method": "derivative_threshold",
        "polarity": "positive",
        "detection_source": SumIntensityTraceKey.DF_F_SIGNAL.value,
        "absolute_threshold": 1.0,
        "derivative_threshold_per_sec": 500.0,
        "refractory_period_ms": 2.0,
        "peak_search_window_ms": 5.0,
        "width_search_window_ms": 20.0,
        "level_fractions": "0.1,0.2,0.5,0.8,0.9",
    }


def test_sum_intensity_schema_defaults_and_detection_method() -> None:
    """SumIntensityAnalysis should expose documented detection defaults."""
    analysis = SumIntensityAnalysis(channel=0, roi_id=1)

    assert analysis.detection_params["detection_method"] == "derivative_threshold"
    assert analysis.detection_params["detrend_method"] == "single_exponential"
    assert analysis.detection_params["filter_method"] == "median"
    assert analysis.detection_params["baseline_method"] == "percentile"
    assert analysis.detection_params["detection_source"] == SumIntensityTraceKey.DF_F_SIGNAL.value
    assert "derivative_threshold" not in analysis.detection_params
    assert "derivative_threshold_per_sec" in analysis.detection_params
    assert SumIntensityAnalysis.exclusive_group is None


def test_sum_intensity_analysis_run_populates_result() -> None:
    """SumIntensityAnalysis should populate table, summary, plot, and events."""
    analysis = SumIntensityAnalysis(channel=0, roi_id=1, detection_params=_params())
    analysis.run(FakeProvider())

    assert analysis.is_dirty()
    assert "norm_sum_intensity" in analysis.get_table_columns()
    assert "df_f_signal" in analysis.get_table_columns()
    assert analysis.result.summary["num_peaks"] == 1
    assert analysis.get_summary_value(SumIntensitySummaryKey.NUM_PEAKS) == 1
    assert analysis.get_summary_value(SumIntensitySummaryKey.F0_BASELINE) == 1.0
    assert len(analysis.get_peak_events()) == 1

    df_f_trace = analysis.get_trace(SumIntensityTraceKey.DF_F_SIGNAL)
    assert df_f_trace.y_label == "df/f0"
    peak_points = analysis.get_event_points(SumIntensityEventPointKey.PEAKS)
    assert peak_points.x.size == 1
    width_trace = analysis.get_width_trace(PeakWidthLevel.WIDTH_50)
    assert width_trace.metadata["connectgaps"] is False

    plot = analysis.get_plot_data()
    assert plot is not None
    assert plot.x_label == "Time (s)"
    assert plot.series_name == "df/f0 signal"


def test_analysis_set_runs_sum_intensity() -> None:
    """AcqAnalysisSet should orchestrate SumIntensityAnalysis."""
    analysis_set = AcqAnalysisSet("fake.tif", data_provider=FakeProvider())
    analysis = analysis_set.create(
        "sum_intensity",
        channel=0,
        roi_id=1,
        detection_params=_params(),
    )
    assert isinstance(analysis, SumIntensityAnalysis)
    analysis_set.run_analysis(AnalysisKey("sum_intensity", 0, 1))
    assert analysis_set.is_dirty()
    assert analysis.get_column("time_sec")
    assert analysis.get_column("df_f_signal")


def test_sum_intensity_is_registered_builtin() -> None:
    """Analysis registry should include sum_intensity as a builtin."""
    registry = get_analysis_registry()
    assert registry["sum_intensity"] is SumIntensityAnalysis


def test_get_percentile_f0_baseline_matches_percentile_run_f0() -> None:
    """Percentile F0 accessor should match f0_baseline after a percentile run."""
    analysis = SumIntensityAnalysis(channel=0, roi_id=1, detection_params=_params())
    analysis.run(FakeProvider())

    auto_f0 = analysis.get_percentile_f0_baseline()

    assert auto_f0 == float(analysis.get_summary_value(SumIntensitySummaryKey.F0_BASELINE))


def test_get_percentile_f0_baseline_works_after_manual_run() -> None:
    """Percentile F0 accessor should recompute auto F0 even after a manual run."""
    params = _params()
    params["baseline_method"] = "manual"
    params["manual_f0_baseline"] = 4.0
    params["baseline_percentile"] = 0.0
    analysis = SumIntensityAnalysis(channel=0, roi_id=1, detection_params=params)
    analysis.run(FakeProvider())

    assert analysis.get_summary_value(SumIntensitySummaryKey.F0_BASELINE) == 4.0
    assert analysis.get_percentile_f0_baseline() == 1.0
    assert analysis.get_percentile_f0_baseline(percentile=100.0) == 6.0
