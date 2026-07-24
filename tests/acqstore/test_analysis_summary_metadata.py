"""Tests for run metadata stamped into analysis summaries."""

from __future__ import annotations

import re

import numpy as np

from acqstore.acq_image.analysis.diameter_analysis.diameter_analysis import DiameterAnalysis
from acqstore.acq_image.analysis.event_analysis.event_analysis import (
    EVENT_SUMMARY_VERSION,
    EventAnalysis,
)
from acqstore.acq_image.analysis.heart_rate_analysis.heart_rate_analysis import (
    HEART_RATE_SUMMARY_VERSION,
    HeartRateAnalysis,
)
from acqstore.acq_image.analysis.model import RUN_SUMMARY_METADATA_KEYS
from acqstore.acq_image.analysis.velocity_analysis.radon_velocity_analysis import (
    RadonVelocityAnalysis,
)


class _FakeKymographProvider:
    """Minimal provider for velocity and diameter runs."""

    def get_roi_image(self, channel: int, roi_id: int) -> np.ndarray:
        _ = channel, roi_id
        image = np.zeros((96, 32), dtype=np.float32)
        y, x = np.indices(image.shape)
        image += np.sin((x + y) / 8.0) * 100.0 + 200.0
        image[:, 12:20] = 180.0
        return image

    def get_image_physical_units(self) -> tuple[float, float]:
        return (0.001, 0.2)


def _assert_metadata_first(summary: dict[str, object]) -> None:
    keys = list(summary.keys())
    assert keys[:3] == list(RUN_SUMMARY_METADATA_KEYS)
    assert re.fullmatch(r"\d{6}", str(summary["analysis_date"]))
    assert re.fullmatch(r"\d{2}:\d{2}:\d{2}\.\d{3}", str(summary["analysis_time"]))


def test_finalize_summary_orders_metadata_first() -> None:
    analysis = RadonVelocityAnalysis(channel=0, roi_id=1)
    summary = analysis.finalize_summary(
        {"num_windows": 3, "velocity_mean": 1.0, "velocity_median": 1.0, "velocity_cv": 0.1}
    )
    _assert_metadata_first(summary)
    assert summary["analysis_version"] == 1
    assert summary["num_windows"] == 3


def test_radon_velocity_run_stamps_summary_metadata() -> None:
    analysis = RadonVelocityAnalysis(channel=0, roi_id=1, detection_params={"window_width": 16})
    analysis.set_execution_options(use_multiprocessing=False)
    analysis.run(_FakeKymographProvider())
    _assert_metadata_first(analysis.result.summary)
    assert analysis.result.summary["analysis_version"] == 1


def test_diameter_run_stamps_summary_metadata() -> None:
    analysis = DiameterAnalysis(channel=0, roi_id=1)
    analysis.set_execution_options(use_threads=False)
    analysis.run(_FakeKymographProvider())
    _assert_metadata_first(analysis.result.summary)
    assert analysis.result.summary["analysis_version"] == 1
    assert "num_rows" in analysis.result.summary


def test_heart_rate_run_stamps_summary_metadata() -> None:
    velocity = RadonVelocityAnalysis(channel=0, roi_id=1, detection_params={"window_width": 16})
    velocity.set_execution_options(use_multiprocessing=False)
    velocity.run(_FakeKymographProvider())
    heart_rate = HeartRateAnalysis(channel=0, roi_id=1)
    heart_rate.run(
        _FakeKymographProvider(),
        dependencies={"radon_velocity": velocity},
    )
    _assert_metadata_first(heart_rate.result.summary)
    assert heart_rate.result.summary["analysis_version"] == HEART_RATE_SUMMARY_VERSION
    assert heart_rate.result.summary["version"] == HEART_RATE_SUMMARY_VERSION


def test_event_run_stamps_summary_metadata() -> None:
    velocity = RadonVelocityAnalysis(channel=0, roi_id=1, detection_params={"window_width": 16})
    velocity.set_execution_options(use_multiprocessing=False)
    velocity.run(_FakeKymographProvider())
    event = EventAnalysis(channel=0, roi_id=1)
    event.run(
        _FakeKymographProvider(),
        dependencies={"radon_velocity": velocity},
    )
    _assert_metadata_first(event.result.summary)
    assert event.result.summary["analysis_version"] == EVENT_SUMMARY_VERSION
    assert event.result.summary["version"] == EVENT_SUMMARY_VERSION


def test_summary_columns_include_run_metadata() -> None:
    assert RadonVelocityAnalysis.get_summary_columns()[:3] == RUN_SUMMARY_METADATA_KEYS
    assert DiameterAnalysis.get_summary_columns()[:3] == RUN_SUMMARY_METADATA_KEYS
    assert HeartRateAnalysis.get_summary_columns()[:3] == RUN_SUMMARY_METADATA_KEYS
    assert EventAnalysis.get_summary_columns()[:3] == RUN_SUMMARY_METADATA_KEYS


def test_finalize_summary_omits_analysis_version_when_unset() -> None:
    class _NoVersionAnalysis(RadonVelocityAnalysis):
        analysis_version = None

    analysis = _NoVersionAnalysis(channel=0, roi_id=1)
    summary = analysis.finalize_summary({"num_windows": 1})
    assert "analysis_version" not in summary
    assert list(summary.keys())[:2] == ["analysis_date", "analysis_time"]
