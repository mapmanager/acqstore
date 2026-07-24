"""Tests for AcqStore analysis plot data API."""

from __future__ import annotations

import pandas as pd
import pytest

from acqstore.acq_image.analysis.model import AnalysisPlotData
from acqstore.acq_image.analysis.velocity_analysis.radon_velocity_analysis import RadonVelocityAnalysis


def test_analysis_plot_data_validates_lengths() -> None:
    """AnalysisPlotData should reject mismatched x/y lengths."""
    with pytest.raises(ValueError):
        AnalysisPlotData(x=(0.0, 1.0), y=(1.0,), x_label="x", y_label="y")


def test_base_analysis_default_plot_data_is_none() -> None:
    """Unrun RadonVelocityAnalysis should have no plot data."""
    analysis = RadonVelocityAnalysis(channel=0, roi_id=1)

    assert analysis.get_plot_data() is None


def test_radon_velocity_analysis_plot_data_uses_time_and_velocity_columns() -> None:
    """RadonVelocityAnalysis should expose canonical velocity plot data."""
    analysis = RadonVelocityAnalysis(channel=0, roi_id=1)
    analysis.result.table = pd.DataFrame(
        {
            "time_s": [0.0, 1.0, 2.0],
            "velocity": [10.0, 11.0, 12.0],
        }
    )

    plot_data = analysis.get_plot_data()

    assert plot_data is not None
    assert plot_data.x == (0.0, 1.0, 2.0)
    assert plot_data.y == (10.0, 11.0, 12.0)
    assert plot_data.x_label == "Time (s)"
    assert plot_data.y_label == "Velocity"
    assert plot_data.series_name == "Radon velocity"
