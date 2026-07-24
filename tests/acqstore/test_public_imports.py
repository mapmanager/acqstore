"""Tests for curated public import surfaces in acqstore."""

from __future__ import annotations


def test_public_acq_image_imports() -> None:
    """Public acq_image package imports should resolve to canonical classes."""
    from acqstore.acq_image import AcqImage, AcqImageList
    from acqstore.acq_image.acq_image import AcqImage as DeepAcqImage
    from acqstore.acq_image.acq_image_list import AcqImageList as DeepAcqImageList

    assert AcqImage is DeepAcqImage
    assert AcqImageList is DeepAcqImageList


def test_public_analysis_imports() -> None:
    """Public analysis package imports should resolve to canonical classes."""
    from acqstore.acq_image.analysis import (
        DiameterAnalysis,
        EventAnalysis,
        HeartRateAnalysis,
        RadonVelocityAnalysis,
    )
    from acqstore.acq_image.analysis.diameter_analysis.diameter_analysis import (
        DiameterAnalysis as DeepDiameterAnalysis,
    )
    from acqstore.acq_image.analysis.event_analysis.event_analysis import (
        EventAnalysis as DeepEventAnalysis,
    )
    from acqstore.acq_image.analysis.heart_rate_analysis.heart_rate_analysis import (
        HeartRateAnalysis as DeepHeartRateAnalysis,
    )
    from acqstore.acq_image.analysis.velocity_analysis.radon_velocity_analysis import (
        RadonVelocityAnalysis as DeepRadonVelocityAnalysis,
    )

    assert RadonVelocityAnalysis is DeepRadonVelocityAnalysis
    assert DiameterAnalysis is DeepDiameterAnalysis
    assert HeartRateAnalysis is DeepHeartRateAnalysis
    assert EventAnalysis is DeepEventAnalysis
