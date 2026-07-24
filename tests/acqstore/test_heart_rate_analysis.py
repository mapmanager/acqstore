"""Tests for heart-rate analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from acqstore.acq_image.acq_analysis_set import AcqAnalysisSet
from acqstore.acq_image.analysis.heart_rate_analysis.heart_rate_analysis import (
    HeartRateAnalysis,
)
from acqstore.acq_image.analysis.registry import get_analysis_class
from acqstore.acq_image.analysis.velocity_analysis.radon_velocity_analysis import (
    RadonVelocityAnalysis,
)

EXPECTED_BPM = 360.0
EXPECTED_FS_HZ = 100.0


def _make_velocity_analysis(
    *,
    channel: int = 0,
    roi_id: int = 1,
    n_samples: int = 1000,
    freq_hz: float = 6.0,
    fs_hz: float = EXPECTED_FS_HZ,
) -> RadonVelocityAnalysis:
    """Build a velocity analysis with a synthetic pulsatile velocity table.

    Args:
        channel: Channel index.
        roi_id: ROI identifier.
        n_samples: Number of velocity samples.
        freq_hz: Pulsation frequency in Hz (heart rate in bpm is ``freq * 60``).
        fs_hz: Sample rate in Hz.

    Returns:
        Velocity analysis whose ``get_plot_data`` returns the synthetic series.
    """
    t = np.arange(n_samples, dtype=float) / fs_hz
    rng = np.random.default_rng(0)
    velocity = 2.0 + np.sin(2.0 * np.pi * freq_hz * t) + 0.01 * rng.standard_normal(n_samples)
    analysis = RadonVelocityAnalysis(channel=channel, roi_id=roi_id)
    analysis.result.table = pd.DataFrame({"time_s": t, "velocity": velocity})
    return analysis


def test_heart_rate_registered() -> None:
    """Heart-rate analysis should be registered under 'heart_rate'."""
    assert get_analysis_class("heart_rate") is HeartRateAnalysis


def test_detection_schema_defaults() -> None:
    """Detection schema should split bpm bounds and use the edge-margin sentinel."""
    defaults = HeartRateAnalysis.get_default_detection_params()
    assert defaults["bpm_min"] == 240.0
    assert defaults["bpm_max"] == 600.0
    assert defaults["edge_margin_hz"] == -1.0
    assert defaults["do_segments"] is False
    assert defaults["agree_tol_bpm"] == 30.0


def test_run_estimates_expected_heart_rate() -> None:
    """Both methods should recover the synthetic heart rate via the analysis set."""
    analysis_set = AcqAnalysisSet("fake.tif", data_provider=object())
    velocity = _make_velocity_analysis()
    heart_rate = HeartRateAnalysis(channel=0, roi_id=1)
    analysis_set.add(velocity)
    analysis_set.add(heart_rate)

    result = analysis_set.run_analysis(heart_rate.key)
    summary = result.summary

    assert result.table is None
    assert summary["version"] == 1
    assert summary["status"] == "ok"
    assert summary["n_total"] == 1000
    assert summary["n_valid"] == 1000

    lomb = summary["lomb"]
    welch = summary["welch"]
    assert lomb["bpm"] == pytest.approx(EXPECTED_BPM, abs=10.0)
    assert welch["bpm"] == pytest.approx(EXPECTED_BPM, abs=30.0)
    assert lomb["status"] == "ok"
    assert welch["status"] == "ok"

    agreement = summary["agreement"]
    assert agreement is not None
    assert agreement["agree_ok"] is True
    assert agreement["agree_tol_bpm"] == 30.0


def test_run_marks_dirty_and_has_no_table() -> None:
    """Running heart-rate analysis should set dirty and produce no table."""
    velocity = _make_velocity_analysis()
    heart_rate = HeartRateAnalysis(channel=0, roi_id=1)
    heart_rate.set_clean()

    heart_rate.run(object(), dependencies={"radon_velocity": velocity})

    assert heart_rate.is_dirty()
    assert heart_rate.has_table() is False


def test_insufficient_samples_reports_status() -> None:
    """Too few samples should yield insufficient_valid status and null bpm."""
    velocity = _make_velocity_analysis(n_samples=100)
    heart_rate = HeartRateAnalysis(channel=0, roi_id=1)

    result = heart_rate.run(object(), dependencies={"radon_velocity": velocity})
    summary = result.summary

    assert summary["status"] == "insufficient_valid"
    assert summary["lomb"]["bpm"] is None
    assert summary["welch"]["bpm"] is None
    assert summary["lomb"]["status"] == "insufficient_valid"
    assert summary["agreement"] is None


def test_missing_dependency_raises() -> None:
    """Running without the velocity dependency should raise ValueError."""
    heart_rate = HeartRateAnalysis(channel=0, roi_id=1)

    with pytest.raises(ValueError):
        heart_rate.run(object(), dependencies={})


def test_segments_summary_present_when_enabled() -> None:
    """do_segments should add a compact segment summary without raw arrays."""
    velocity = _make_velocity_analysis()
    heart_rate = HeartRateAnalysis(
        channel=0,
        roi_id=1,
        detection_params={"do_segments": True},
    )

    result = heart_rate.run(object(), dependencies={"radon_velocity": velocity})
    seg = result.summary["segments_summary"]

    assert seg["method"] == "welch"
    assert "n_windows" in seg
    assert "median_bpm" in seg
    assert "t_center" not in seg
    assert "bpm" not in seg


def test_json_roundtrip_preserves_summary() -> None:
    """Serialized record should restore detection params and summary."""
    velocity = _make_velocity_analysis()
    heart_rate = HeartRateAnalysis(channel=2, roi_id=3)
    heart_rate.run(object(), dependencies={"radon_velocity": velocity})

    record = heart_rate.to_json_dict()
    restored = HeartRateAnalysis(channel=2, roi_id=3)
    restored.load_json_dict(record)

    assert restored.result.summary == heart_rate.result.summary
    assert restored.detection_params == heart_rate.detection_params
    assert restored.is_dirty() is False
