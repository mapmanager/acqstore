"""Tests for sum-intensity flat summary projection."""

from __future__ import annotations

from acqstore.acq_image.analysis.sum_intensity_analysis.sum_intensity_analysis import (
    SumIntensityAnalysis,
)


def test_sum_intensity_get_summary_values_excludes_peak_events() -> None:
    """Flat summary values should omit nested peak-event records."""
    analysis = SumIntensityAnalysis(channel=0, roi_id=1)
    analysis.result.summary = {
        "analysis_date": "260628",
        "analysis_time": "18:13:30.794",
        "analysis_version": 1,
        "num_peaks": 11,
        "detection_method": "derivative_threshold",
        "detrend_method": "single_exponential",
        "errors": [],
        "peak_events": [{"peak_id": 1, "status": "ok"}],
    }

    flat = analysis.get_summary_values()

    assert "peak_events" not in flat
    assert flat["num_peaks"] == 11
    assert flat["detection_method"] == "derivative_threshold"
    assert flat["detrend_method"] == "single_exponential"
    assert flat["errors"] == []
