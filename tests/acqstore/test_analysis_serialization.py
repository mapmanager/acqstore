"""Tests for analysis JSON serialization."""

from acqstore.acq_image.acq_analysis_set import AcqAnalysisSet
from acqstore.acq_image.analysis.velocity_analysis.radon_velocity_analysis import RadonVelocityAnalysis
from acqstore.acq_image.analysis.model import AnalysisKey


def test_serialize_json_analysis_returns_records() -> None:
    """serialize_json_analysis should return JSON-ready records."""
    analysis_set = AcqAnalysisSet("fake.tif")
    analysis = RadonVelocityAnalysis(channel=0, roi_id=1, detection_params={"window_width": 16})
    analysis.result.summary = {"mean_velocity": 2.5}
    analysis_set.add(analysis)

    records = analysis_set.serialize_json_analysis()

    assert records == [
        {
            "analysis_name": "radon_velocity",
            "channel": 0,
            "roi_id": 1,
            "detection_params": {"window_width": 16},
            "summary": {"mean_velocity": 2.5},
        }
    ]


def test_load_json_analysis_reconstructs_analysis_and_sets_clean() -> None:
    """load_json_analysis should reconstruct records and leave set clean."""
    records = [
        {
            "analysis_name": "radon_velocity",
            "channel": 0,
            "roi_id": 1,
            "detection_params": {"window_width": 16},
            "summary": {"mean_velocity": 2.5},
        }
    ]
    analysis_set = AcqAnalysisSet("fake.tif")

    analysis_set.load_json_analysis(records)

    analysis = analysis_set.get_required(AnalysisKey("radon_velocity", 0, 1))
    assert analysis.detection_params == {"window_width": 16}
    assert analysis.result.summary == {"mean_velocity": 2.5}
    assert not analysis_set.is_dirty()
