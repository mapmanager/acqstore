"""Tests for analysis key behavior."""

from acqstore.acq_image.analysis.model import AnalysisKey


def test_analysis_key_excludes_file_id() -> None:
    """AnalysisKey identity should be analysis name, channel, and ROI only."""
    key = AnalysisKey("radon_velocity", channel=1, roi_id=2)

    assert key.analysis_name == "radon_velocity"
    assert key.channel == 1
    assert key.roi_id == 2


def test_analysis_key_csv_filename_preserves_source_extension() -> None:
    """CSV filenames should preserve the source file extension."""
    key = AnalysisKey("radon_velocity", channel=0, roi_id=1)

    assert key.csv_filename("myfile.tif") == "myfile.tif.radon_velocity.csv"
