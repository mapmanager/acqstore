"""Tests for AcqAnalysisSet exclusive-group enforcement."""

from __future__ import annotations

import numpy as np
import pytest

from acqstore.acq_image.acq_analysis_set import AcqAnalysisSet
from acqstore.acq_image.analysis.diameter_analysis.diameter_analysis import DiameterAnalysis
from acqstore.acq_image.analysis.model import AnalysisExclusionError
from acqstore.acq_image.analysis.velocity_analysis.radon_velocity_analysis import (
    RadonVelocityAnalysis,
)


class FakeProvider:
    """Minimal analysis data provider."""

    def get_roi_image(self, channel: int, roi_id: int) -> np.ndarray:
        """Return synthetic ROI image.

        Args:
            channel: Channel index.
            roi_id: ROI identifier.

        Returns:
            Two-dimensional synthetic kymograph.
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


def test_diameter_and_radon_share_primary_kymograph_group() -> None:
    """Both primary-kymograph analyses should declare the same group."""
    assert DiameterAnalysis.exclusive_group == "primary_kymograph"
    assert RadonVelocityAnalysis.exclusive_group == "primary_kymograph"


def test_radon_blocks_diameter_for_same_roi() -> None:
    """Adding Diameter for the same (channel, roi_id) as Radon should fail."""
    analysis_set = AcqAnalysisSet("fake.tif", data_provider=FakeProvider())
    analysis_set.create("radon_velocity", channel=0, roi_id=1)

    with pytest.raises(AnalysisExclusionError):
        analysis_set.create("diameter", channel=0, roi_id=1)


def test_diameter_blocks_radon_for_same_roi() -> None:
    """Adding Radon for the same (channel, roi_id) as Diameter should fail."""
    analysis_set = AcqAnalysisSet("fake.tif", data_provider=FakeProvider())
    analysis_set.create("diameter", channel=0, roi_id=1)

    with pytest.raises(AnalysisExclusionError):
        analysis_set.create("radon_velocity", channel=0, roi_id=1)


def test_sum_intensity_coexists_with_diameter_for_same_roi() -> None:
    """Sum intensity is not in primary_kymograph and should coexist with diameter."""
    analysis_set = AcqAnalysisSet("fake.tif", data_provider=FakeProvider())
    analysis_set.create("diameter", channel=0, roi_id=1)
    analysis_set.create("sum_intensity", channel=0, roi_id=1)


def test_primary_group_does_not_block_other_channel_or_roi() -> None:
    """Different (channel, roi_id) should not conflict."""
    analysis_set = AcqAnalysisSet("fake.tif", data_provider=FakeProvider())
    analysis_set.create("radon_velocity", channel=0, roi_id=1)
    analysis_set.create("diameter", channel=0, roi_id=2)
    analysis_set.create("diameter", channel=1, roi_id=1)


def test_get_primary_kymograph_analysis_returns_active_one() -> None:
    """Helper should return the single primary-kymograph analysis."""
    analysis_set = AcqAnalysisSet("fake.tif", data_provider=FakeProvider())
    analysis_set.create("diameter", channel=0, roi_id=3)

    found = analysis_set.get_primary_kymograph_analysis(channel=0, roi_id=3)
    assert isinstance(found, DiameterAnalysis)
    assert analysis_set.get_primary_kymograph_analysis(channel=1, roi_id=3) is None


def test_load_json_analyses_rejects_conflicting_records() -> None:
    """Loading a sidecar with both Radon and Diameter for one ROI should fail."""
    analysis_set = AcqAnalysisSet("fake.tif", data_provider=FakeProvider())
    radon = analysis_set.create("radon_velocity", channel=0, roi_id=1)
    analysis_set.remove(radon.key)
    radon = RadonVelocityAnalysis(channel=0, roi_id=1)
    diameter = DiameterAnalysis(channel=0, roi_id=1)
    records = [radon.to_json_dict(), diameter.to_json_dict()]

    fresh_set = AcqAnalysisSet("fake.tif", data_provider=FakeProvider())
    with pytest.raises(AnalysisExclusionError):
        fresh_set.load_json_analysis(records)
