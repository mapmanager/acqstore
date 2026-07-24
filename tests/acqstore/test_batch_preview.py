"""Tests for read-only AcqImage batch preview helpers."""

from __future__ import annotations

import numpy as np

from acqstore.acq_image.acq_analysis_set import AcqAnalysisSet
from acqstore.acq_image.analysis.batch.preview import (
    BatchPreviewOutcome,
    preview_batch_rows,
    roi_intersection_across_acq_images,
)
from acqstore.acq_image.analysis.batch.roi_mode import RoiBatchMode
from acqstore.acq_image.analysis.data_provider import AcqImageAnalysisDataProvider
from acqstore.acq_image.roi import ImageBounds, LineEndpoints, RoiSet


class FakeAcqImage:
    """Small fake AcqImage for preview tests."""

    def __init__(self, path: str) -> None:
        """Create fake acquisition image."""
        self.path = path
        self.rois = RoiSet(ImageBounds(width=64, height=96, num_slices=1))
        self.analysis_set = AcqAnalysisSet(
            path,
            data_provider=AcqImageAnalysisDataProvider(self),
        )

    @property
    def file_id(self) -> str:
        """Return stable file id."""
        return str(self.path)

    def get_roi_image(self, channel: int, roi_id: int) -> np.ndarray:
        """Return synthetic ROI image."""
        _ = channel, roi_id
        return np.zeros((96, 64), dtype=np.float32)

    def get_image_physical_units(self) -> tuple[float, float]:
        """Return synthetic physical units."""
        return (0.001, 0.25)


def test_roi_intersection_across_acq_images_uses_rect_rois_only() -> None:
    """Common ROI ids should include only rectangular ROIs by default."""
    first = FakeAcqImage("/tmp/a.oir")
    second = FakeAcqImage("/tmp/b.oir")
    first_rect = first.rois.create_rect_roi()
    second.rois.create_rect_roi()
    first.rois.create_line_roi(LineEndpoints(row0=0, col0=0, row1=5, col1=5))
    second.rois.create_line_roi(LineEndpoints(row0=0, col0=0, row1=5, col1=5))

    assert first_rect.roi_id == 1
    assert roi_intersection_across_acq_images([first, second]) == [1]


def test_preview_existing_roi_marks_missing_roi_as_skipped() -> None:
    """Existing-ROI preview should skip files without the requested ROI."""
    acq = FakeAcqImage("/tmp/a.oir")

    rows = preview_batch_rows(
        [acq],
        analysis_name="diameter",
        roi_mode=RoiBatchMode.ANALYZE_EXISTING_ROI,
        roi_id=1,
        channel=0,
    )

    assert rows[0].outcome is BatchPreviewOutcome.SKIPPED_MISSING_ROI
    assert rows[0].message == "ROI 1 not in file"


def test_preview_add_new_roi_is_pending_without_mutating_rois() -> None:
    """ADD_NEW_ROI preview should not create a ROI."""
    acq = FakeAcqImage("/tmp/a.oir")

    rows = preview_batch_rows(
        [acq],
        analysis_name="diameter",
        roi_mode=RoiBatchMode.ADD_NEW_ROI,
        roi_id=None,
        channel=0,
    )

    assert rows[0].outcome is BatchPreviewOutcome.PENDING
    assert rows[0].roi_id is None
    assert rows[0].message == "add new ROI per file"
    assert acq.rois.get_roi_ids() == []


def test_preview_existing_same_analysis_will_replace() -> None:
    """Preview should report same-analysis reruns as replacements."""
    acq = FakeAcqImage("/tmp/a.oir")
    roi = acq.rois.create_rect_roi()
    acq.analysis_set.create(
        "diameter",
        channel=0,
        roi_id=roi.roi_id,
        detection_params={"diameter_method": "threshold_width"},
    )

    rows = preview_batch_rows(
        [acq],
        analysis_name="diameter",
        roi_mode=RoiBatchMode.ANALYZE_EXISTING_ROI,
        roi_id=roi.roi_id,
        channel=0,
    )

    assert rows[0].outcome is BatchPreviewOutcome.PENDING
    assert rows[0].message == "will replace existing diameter"


def test_preview_existing_opposing_primary_analysis_is_conflict_skip() -> None:
    """Preview should skip files with an opposing primary analysis."""
    acq = FakeAcqImage("/tmp/a.oir")
    roi = acq.rois.create_rect_roi()
    acq.analysis_set.create(
        "radon_velocity",
        channel=0,
        roi_id=roi.roi_id,
        detection_params={"window_width": 16},
    )

    rows = preview_batch_rows(
        [acq],
        analysis_name="diameter",
        roi_mode=RoiBatchMode.ANALYZE_EXISTING_ROI,
        roi_id=roi.roi_id,
        channel=0,
    )

    assert rows[0].outcome is BatchPreviewOutcome.SKIPPED_CONFLICT
    assert rows[0].message == "conflict with radon_velocity"
