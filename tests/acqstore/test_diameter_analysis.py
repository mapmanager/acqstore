"""Tests for DiameterAnalysis BaseAnalysis wrapper."""

from __future__ import annotations

import numpy as np

from acqstore.acq_image.acq_analysis_set import AcqAnalysisSet
from acqstore.acq_image.analysis.model import AnalysisKey, AnalysisOverlayTraceData
from acqstore.acq_image.analysis.diameter_analysis.diameter_analysis import DiameterAnalysis


class FakeProvider:
    """Fake analysis data provider."""

    def get_roi_image(self, channel: int, roi_id: int) -> np.ndarray:
        """Return synthetic ROI kymograph data.

        Args:
            channel: Channel index.
            roi_id: ROI identifier.

        Returns:
            Synthetic two-dimensional ROI image.
        """
        _ = channel, roi_id
        image = np.zeros((48, 64), dtype=np.float32)
        image[:, 20:40] = 180.0
        return image

    def get_image_physical_units(self) -> tuple[float, float]:
        """Return synthetic physical units.

        Returns:
            ``(seconds_per_line, um_per_pixel)``.
        """
        return (0.001, 0.25)


def test_diameter_schema_has_methods_field() -> None:
    """Diameter schema should tag method-specific parameters."""
    schema = {field.name: field for field in DiameterAnalysis.get_detection_schema()}
    assert schema["threshold_mode"].methods == ("threshold_width",)
    assert schema["gradient_sigma"].methods == ("gradient_edges",)
    assert schema["post_filter_kernel_size"].methods is None


def test_diameter_analysis_run_populates_result() -> None:
    """DiameterAnalysis should populate summary/table and mark dirty."""
    analysis = DiameterAnalysis(channel=0, roi_id=1)
    analysis.set_execution_options(use_threads=False)
    analysis.run(FakeProvider())

    assert analysis.is_dirty()
    assert "time_s" in analysis.get_table_columns()
    assert "diameter_um_filt" in analysis.get_table_columns()


def test_diameter_overlay_traces_roi_local() -> None:
    """Overlay traces should use ROI-local time_s and edge_um columns."""
    analysis = DiameterAnalysis(channel=0, roi_id=1)
    analysis.set_execution_options(use_threads=False)
    analysis.run(FakeProvider())
    traces = analysis.get_overlay_traces()
    assert traces
    assert all(isinstance(trace, AnalysisOverlayTraceData) for trace in traces)
    assert {trace.trace_id for trace in traces} >= {"diameter_left_edge", "diameter_right_edge"}


def test_analysis_set_runs_diameter() -> None:
    """AcqAnalysisSet should orchestrate DiameterAnalysis."""
    analysis_set = AcqAnalysisSet("fake.tif", data_provider=FakeProvider())
    analysis = analysis_set.create("diameter", channel=0, roi_id=1)
    assert isinstance(analysis, DiameterAnalysis)
    analysis.set_execution_options(use_threads=False)
    analysis_set.run_analysis(AnalysisKey("diameter", 0, 1))
    assert analysis_set.is_dirty()
    assert analysis.get_column("time_s")
