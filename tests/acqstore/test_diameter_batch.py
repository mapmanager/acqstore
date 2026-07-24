"""Tests for diameter batch helpers."""

from __future__ import annotations

import threading

import numpy as np

from acqstore.acq_image.acq_analysis_set import AcqAnalysisSet
from acqstore.acq_image.analysis.batch.acq_analysis_batch import AcqAnalysisBatch
from acqstore.acq_image.analysis.batch.diameter_batch_strategy import DiameterBatchStrategy
from acqstore.acq_image.analysis.batch.roi_mode import RoiBatchMode
from acqstore.acq_image.analysis.batch.types import AnalysisBatchKind, BatchFileOutcome
from acqstore.acq_image.analysis.data_provider import AcqImageAnalysisDataProvider
from acqstore.acq_image.roi import ImageBounds, RoiSet


class FakeAcqImage:
    """Small fake AcqImage for diameter batch tests."""

    def __init__(self, path: str) -> None:
        """Create fake acquisition image.

        Args:
            path: Fake source path.
        """
        self.path = path
        self.rois = RoiSet(ImageBounds(width=64, height=96, num_slices=1))
        self.analysis_set = AcqAnalysisSet(
            path,
            data_provider=AcqImageAnalysisDataProvider(self),
        )

    def get_roi_image(self, channel: int, roi_id: int) -> np.ndarray:
        """Return a synthetic vessel-like kymograph.

        Args:
            channel: Channel index.
            roi_id: ROI identifier.

        Returns:
            Synthetic ROI image.
        """
        _ = channel, roi_id
        image = np.zeros((96, 64), dtype=np.float32)
        image[:, 24:40] = 200.0
        return image

    def get_image_physical_units(self) -> tuple[float, float]:
        """Return synthetic physical units.

        Returns:
            ``(step_time, step_space)``.
        """
        return (0.001, 0.25)


def test_diameter_batch_existing_roi_skips_missing_roi() -> None:
    """Batch existing-ROI mode should skip files without the ROI."""
    acq = FakeAcqImage("fake.tif")
    strategy = DiameterBatchStrategy(
        channel=0,
        roi_mode=RoiBatchMode.ANALYZE_EXISTING_ROI,
        roi_id=1,
        detection_params={"diameter_method": "threshold_width"},
        use_threads=False,
    )

    result = strategy.process_file(acq, cancel_event=threading.Event())

    assert result.outcome is BatchFileOutcome.SKIPPED_MISSING_ROI
    assert strategy.kind is AnalysisBatchKind.DIAMETER


def test_diameter_batch_add_new_roi_runs_analysis() -> None:
    """Batch ADD_NEW_ROI mode should create ROI and run analysis."""
    acq = FakeAcqImage("fake.tif")
    strategy = DiameterBatchStrategy(
        channel=0,
        roi_mode=RoiBatchMode.ADD_NEW_ROI,
        roi_id=None,
        detection_params={"diameter_method": "threshold_width"},
        use_threads=False,
    )

    results = AcqAnalysisBatch([acq], strategy, max_parallel_files=1).run()

    assert results[0].outcome is BatchFileOutcome.OK
    assert results[0].analysis_name == "diameter"
    assert results[0].roi_id == 1
    assert acq.analysis_set.as_list()


def test_diameter_batch_existing_roi_replaces_analysis() -> None:
    """Batch existing-ROI mode should replace existing analysis."""
    acq = FakeAcqImage("fake.tif")
    roi = acq.rois.create_rect_roi()
    strategy = DiameterBatchStrategy(
        channel=0,
        roi_mode=RoiBatchMode.ANALYZE_EXISTING_ROI,
        roi_id=roi.roi_id,
        detection_params={"diameter_method": "threshold_width"},
        use_threads=False,
    )

    first = strategy.process_file(acq, cancel_event=threading.Event())
    second = strategy.process_file(acq, cancel_event=threading.Event())

    assert first.outcome is BatchFileOutcome.OK
    assert second.outcome is BatchFileOutcome.OK
    assert len(acq.analysis_set.as_list()) == 1


def test_diameter_batch_skips_existing_roi_primary_conflict() -> None:
    """Batch should skip a file with an opposing primary analysis on the ROI."""
    acq = FakeAcqImage("fake.tif")
    roi = acq.rois.create_rect_roi()
    acq.analysis_set.create(
        "radon_velocity",
        channel=0,
        roi_id=roi.roi_id,
        detection_params={"window_width": 16},
    )
    strategy = DiameterBatchStrategy(
        channel=0,
        roi_mode=RoiBatchMode.ANALYZE_EXISTING_ROI,
        roi_id=roi.roi_id,
        detection_params={"diameter_method": "threshold_width"},
        use_threads=False,
    )

    result = strategy.process_file(acq, cancel_event=threading.Event())

    assert result.outcome is BatchFileOutcome.SKIPPED_CONFLICT
    assert "radon_velocity" in result.message


def test_diameter_batch_returns_cancelled_when_cancel_requested_before_file() -> None:
    """Batch strategy should honor a pre-set cancellation event."""
    acq = FakeAcqImage("fake.tif")
    cancel_event = threading.Event()
    cancel_event.set()
    strategy = DiameterBatchStrategy(
        channel=0,
        roi_mode=RoiBatchMode.ADD_NEW_ROI,
        roi_id=None,
        detection_params={"diameter_method": "threshold_width"},
        use_threads=False,
    )

    result = strategy.process_file(acq, cancel_event=cancel_event)

    assert result.outcome is BatchFileOutcome.CANCELLED
