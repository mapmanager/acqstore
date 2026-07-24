"""Tests for RadonVelocityAnalysis."""

from __future__ import annotations

import numpy as np

from acqstore.acq_image.acq_analysis_set import AcqAnalysisSet
from acqstore.acq_image.analysis.model import AnalysisKey
from acqstore.acq_image.analysis.velocity_analysis.radon_velocity_analysis import (
    RadonVelocityAnalysis,
)


class FakeProvider:
    """Fake analysis data provider."""

    def get_roi_image(self, channel: int, roi_id: int) -> np.ndarray:
        """Return synthetic ROI data.

        Args:
            channel: Channel index.
            roi_id: ROI identifier.

        Returns:
            Synthetic two-dimensional ROI image.
        """
        y, x = np.indices((96, 32))
        return (np.sin((x + y) / 8.0) * 100.0 + 200.0).astype(np.float32)

    def get_image_physical_units(self) -> tuple[float, float]:
        """Return synthetic physical units.

        Returns:
            ``(step_time, step_space)``.
        """
        return (0.001, 0.2)


def test_radon_velocity_schema() -> None:
    """RadonVelocityAnalysis should define window_width schema."""
    schema = RadonVelocityAnalysis.get_detection_schema()

    assert len(schema) == 1
    assert schema[0].name == "window_width"
    assert schema[0].choices == (16, 64, 128)


def test_radon_velocity_analysis_run_populates_result() -> None:
    """RadonVelocityAnalysis should populate summary/table and mark dirty."""
    analysis = RadonVelocityAnalysis(
        channel=0,
        roi_id=1,
        detection_params={"window_width": 16},
    )
    analysis.set_execution_options(use_multiprocessing=False)

    analysis.run(FakeProvider())

    assert analysis.is_dirty()
    # assert analysis.result.summary["window_width"] == 16
    assert "time_s" in analysis.get_table_columns()
    assert "velocity" in analysis.get_table_columns()


def test_analysis_set_runs_radon_velocity() -> None:
    """AcqAnalysisSet should orchestrate RadonVelocityAnalysis."""
    analysis_set = AcqAnalysisSet("fake.tif", data_provider=FakeProvider())
    analysis = analysis_set.create(
        "radon_velocity",
        channel=0,
        roi_id=1,
        detection_params={"window_width": 16},
    )
    assert isinstance(analysis, RadonVelocityAnalysis)
    analysis.set_execution_options(use_multiprocessing=False)

    analysis_set.run_analysis(AnalysisKey("radon_velocity", 0, 1))

    assert analysis_set.is_dirty()
    assert analysis.get_column("time_s")
