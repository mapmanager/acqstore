"""Tests for Radon velocity batch helpers."""

from __future__ import annotations

import numpy as np

from acqstore.acq_image.analysis.batch.acq_analysis_batch import AcqAnalysisBatch
from acqstore.acq_image.analysis.batch.radon_velocity_batch_strategy import (
    RadonVelocityBatchStrategy,
)
from acqstore.acq_image.analysis.batch.roi_mode import RoiBatchMode
from acqstore.acq_image.analysis.batch.types import BatchFileOutcome
from acqstore.acq_image.analysis.data_provider import AcqImageAnalysisDataProvider
from acqstore.acq_image.acq_analysis_set import AcqAnalysisSet
from acqstore.acq_image.roi import ImageBounds, RoiSet


class FakeAcqImage:
    """Small fake AcqImage for batch tests."""

    def __init__(self, path: str) -> None:
        """Create fake acquisition image.

        Args:
            path: Fake source path.
        """
        self.path = path
        self.rois = RoiSet(ImageBounds(width=32, height=96, num_slices=1))
        self.analysis_set = AcqAnalysisSet(
            path,
            data_provider=AcqImageAnalysisDataProvider(self),
        )

    def get_roi_image(self, channel: int, roi_id: int) -> np.ndarray:
        """Return synthetic ROI image.

        Args:
            channel: Channel index.
            roi_id: ROI identifier.

        Returns:
            Synthetic ROI image.
        """
        y, x = np.indices((96, 32))
        return (np.sin((x + y) / 8.0) * 100.0 + 200.0).astype(np.float32)

    def get_image_physical_units(self) -> tuple[float, float]:
        """Return synthetic physical units.

        Returns:
            ``(step_time, step_space)``.
        """
        return (0.001, 0.2)


def test_batch_existing_roi_skips_missing_roi() -> None:
    """Batch existing-ROI mode should skip files without the ROI."""
    acq = FakeAcqImage("fake.tif")
    strategy = RadonVelocityBatchStrategy(
        channel=0,
        roi_mode=RoiBatchMode.ANALYZE_EXISTING_ROI,
        roi_id=1,
        detection_params={"window_width": 16},
        use_multiprocessing=False,
    )

    result = strategy.process_file(acq, cancel_event=__import__("threading").Event())

    assert result.outcome is BatchFileOutcome.SKIPPED_MISSING_ROI


def test_batch_add_new_roi_runs_analysis() -> None:
    """Batch ADD_NEW_ROI mode should create ROI and run analysis."""
    acq = FakeAcqImage("fake.tif")
    strategy = RadonVelocityBatchStrategy(
        channel=0,
        roi_mode=RoiBatchMode.ADD_NEW_ROI,
        roi_id=None,
        detection_params={"window_width": 16},
        use_multiprocessing=False,
    )

    results = AcqAnalysisBatch([acq], strategy, max_parallel_files=1).run()

    assert results[0].outcome is BatchFileOutcome.OK
    assert results[0].roi_id == 1
    assert acq.analysis_set.as_list()


def test_batch_existing_roi_replaces_analysis() -> None:
    """Batch existing-ROI mode should replace existing analysis."""
    acq = FakeAcqImage("fake.tif")
    roi = acq.rois.create_rect_roi()
    strategy = RadonVelocityBatchStrategy(
        channel=0,
        roi_mode=RoiBatchMode.ANALYZE_EXISTING_ROI,
        roi_id=roi.roi_id,
        detection_params={"window_width": 16},
        use_multiprocessing=False,
    )

    first = strategy.process_file(acq, cancel_event=__import__("threading").Event())
    second = strategy.process_file(acq, cancel_event=__import__("threading").Event())

    assert first.outcome is BatchFileOutcome.OK
    assert second.outcome is BatchFileOutcome.OK
    assert len(acq.analysis_set.as_list()) == 1


def test_radon_batch_skips_existing_roi_primary_conflict() -> None:
    """Batch should skip a file with an opposing primary analysis on the ROI."""
    acq = FakeAcqImage("fake.tif")
    roi = acq.rois.create_rect_roi()
    acq.analysis_set.create(
        "diameter",
        channel=0,
        roi_id=roi.roi_id,
        detection_params={"diameter_method": "threshold_width"},
    )
    strategy = RadonVelocityBatchStrategy(
        channel=0,
        roi_mode=RoiBatchMode.ANALYZE_EXISTING_ROI,
        roi_id=roi.roi_id,
        detection_params={"window_width": 16},
        use_multiprocessing=False,
    )

    result = strategy.process_file(acq, cancel_event=__import__("threading").Event())

    assert result.outcome is BatchFileOutcome.SKIPPED_CONFLICT
    assert "diameter" in result.message


def test_radon_batch_returns_cancelled_when_cancel_requested_before_file() -> None:
    """Batch strategy should honor a pre-set cancellation event."""
    import threading

    acq = FakeAcqImage("fake.tif")
    cancel_event = threading.Event()
    cancel_event.set()
    strategy = RadonVelocityBatchStrategy(
        channel=0,
        roi_mode=RoiBatchMode.ADD_NEW_ROI,
        roi_id=None,
        detection_params={"window_width": 16},
        use_multiprocessing=False,
    )

    result = strategy.process_file(acq, cancel_event=cancel_event)

    assert result.outcome is BatchFileOutcome.CANCELLED
