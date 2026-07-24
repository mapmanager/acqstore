"""Read-only preview helpers for AcqImage batch analysis."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from acqstore.acq_image.analysis.batch.roi_mode import RoiBatchMode
from acqstore.acq_image.analysis.batch.types import BatchFileOutcome
from acqstore.acq_image.roi import RectROI

if TYPE_CHECKING:
    from acqstore.acq_image.acq_image import AcqImage


class BatchPreviewOutcome(StrEnum):
    """Planned per-file outcome before a batch starts."""

    PENDING = "pending"
    SKIPPED_MISSING_ROI = BatchFileOutcome.SKIPPED_MISSING_ROI.value
    SKIPPED_CONFLICT = BatchFileOutcome.SKIPPED_CONFLICT.value


@dataclass(frozen=True, slots=True)
class BatchPreviewRow:
    """Read-only planned batch outcome for one file.

    Args:
        file_id: Stable acquisition file identifier.
        file_name: Human-readable file basename.
        analysis_name: Analysis name requested for the batch.
        channel: Channel index requested for the batch.
        roi_mode: ROI batch mode requested for the batch.
        roi_id: Shared ROI id for existing-ROI mode, otherwise None.
        outcome: Planned per-file outcome.
        message: Human-readable planned outcome message.
    """

    file_id: str
    file_name: str
    analysis_name: str
    channel: int
    roi_mode: RoiBatchMode
    roi_id: int | None
    outcome: BatchPreviewOutcome
    message: str


def roi_intersection_across_acq_images(
    acq_images: Sequence["AcqImage"],
    *,
    rect_only: bool = True,
) -> list[int]:
    """Return ROI ids present on every file.

    Args:
        acq_images: Acquisition images to inspect.
        rect_only: When True, include only rectangular ROIs. CloudScope batch
            velocity and diameter analysis require rectangular ROI images.

    Returns:
        Sorted ROI ids present on every image, or an empty list when no common
        ROI exists.
    """
    if not acq_images:
        return []
    roi_sets: list[set[int]] = []
    for acq_image in acq_images:
        roi_ids: set[int] = set()
        for roi in acq_image.rois:
            if rect_only and not isinstance(roi, RectROI):
                continue
            roi_ids.add(int(roi.roi_id))
        roi_sets.append(roi_ids)
    if not roi_sets:
        return []
    return sorted(set.intersection(*roi_sets))


def preview_batch_rows(
    acq_images: Sequence["AcqImage"],
    *,
    analysis_name: str,
    roi_mode: RoiBatchMode,
    roi_id: int | None,
    channel: int,
) -> list[BatchPreviewRow]:
    """Return read-only planned batch outcomes for a file sequence.

    This helper does not create ROIs, modify analysis sets, or run analysis.

    Args:
        acq_images: Acquisition images to preview in execution order.
        analysis_name: Analysis name requested for the batch.
        roi_mode: How each file's target ROI will be selected.
        roi_id: Shared ROI id for ``ANALYZE_EXISTING_ROI`` mode, otherwise None.
        channel: Channel index requested for the batch.

    Returns:
        Planned row for each acquisition image.
    """
    mode = RoiBatchMode(roi_mode)
    return [
        _preview_one(
            acq_image,
            analysis_name=analysis_name,
            roi_mode=mode,
            roi_id=roi_id,
            channel=int(channel),
        )
        for acq_image in acq_images
    ]


def _preview_one(
    acq_image: "AcqImage",
    *,
    analysis_name: str,
    roi_mode: RoiBatchMode,
    roi_id: int | None,
    channel: int,
) -> BatchPreviewRow:
    """Build one preview row.

    Args:
        acq_image: Acquisition image to inspect.
        analysis_name: Analysis name requested for the batch.
        roi_mode: How the target ROI will be selected.
        roi_id: Shared ROI id for existing-ROI mode, otherwise None.
        channel: Channel index requested for the batch.

    Returns:
        Planned row for the acquisition image.
    """
    file_id = _file_id(acq_image)
    file_name = Path(file_id).name
    if roi_mode is RoiBatchMode.ADD_NEW_ROI:
        return BatchPreviewRow(
            file_id=file_id,
            file_name=file_name,
            analysis_name=analysis_name,
            channel=channel,
            roi_mode=roi_mode,
            roi_id=None,
            outcome=BatchPreviewOutcome.PENDING,
            message="add new ROI per file",
        )

    if roi_id is None:
        return _skipped_missing_roi(
            acq_image,
            file_id=file_id,
            file_name=file_name,
            analysis_name=analysis_name,
            roi_mode=roi_mode,
            channel=channel,
            roi_id=None,
            message="no ROI id",
        )

    roi = acq_image.rois.get(int(roi_id))
    if roi is None:
        return _skipped_missing_roi(
            acq_image,
            file_id=file_id,
            file_name=file_name,
            analysis_name=analysis_name,
            roi_mode=roi_mode,
            channel=channel,
            roi_id=int(roi_id),
            message=f"ROI {roi_id} not in file",
        )
    if not isinstance(roi, RectROI):
        return _skipped_missing_roi(
            acq_image,
            file_id=file_id,
            file_name=file_name,
            analysis_name=analysis_name,
            roi_mode=roi_mode,
            channel=channel,
            roi_id=int(roi_id),
            message=f"ROI {roi_id} is not rectangular",
        )

    existing = acq_image.analysis_set.get_primary_kymograph_analysis(
        channel=int(channel),
        roi_id=int(roi_id),
    )
    if existing is not None and existing.key.analysis_name != analysis_name:
        return BatchPreviewRow(
            file_id=file_id,
            file_name=file_name,
            analysis_name=analysis_name,
            channel=channel,
            roi_mode=roi_mode,
            roi_id=int(roi_id),
            outcome=BatchPreviewOutcome.SKIPPED_CONFLICT,
            message=f"conflict with {existing.key.analysis_name}",
        )
    if existing is not None:
        message = f"will replace existing {analysis_name}"
    else:
        message = "will run"
    return BatchPreviewRow(
        file_id=file_id,
        file_name=file_name,
        analysis_name=analysis_name,
        channel=channel,
        roi_mode=roi_mode,
        roi_id=int(roi_id),
        outcome=BatchPreviewOutcome.PENDING,
        message=message,
    )


def _skipped_missing_roi(
    _acq_image: "AcqImage",
    *,
    file_id: str,
    file_name: str,
    analysis_name: str,
    roi_mode: RoiBatchMode,
    channel: int,
    roi_id: int | None,
    message: str,
) -> BatchPreviewRow:
    """Build a missing-ROI preview row."""
    return BatchPreviewRow(
        file_id=file_id,
        file_name=file_name,
        analysis_name=analysis_name,
        channel=channel,
        roi_mode=roi_mode,
        roi_id=roi_id,
        outcome=BatchPreviewOutcome.SKIPPED_MISSING_ROI,
        message=message,
    )


def _file_id(acq_image: "AcqImage") -> str:
    """Return a stable file id string from an AcqImage-like object."""
    value = getattr(acq_image, "file_id", None)
    if isinstance(value, str) and value:
        return value
    return str(acq_image.path)
