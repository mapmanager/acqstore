"""Batch strategy for diameter analysis.

The strategy chooses a target ROI for each file, creates a ``DiameterAnalysis``
through the file's ``AcqAnalysisSet``, configures runtime thread options, and
returns a structured per-file result. It does not save files; callers decide
when to persist changed analyses.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from acqstore.acq_image.analysis.batch.roi_mode import RoiBatchMode
from acqstore.acq_image.analysis.batch.types import (
    AnalysisBatchKind,
    BatchFileOutcome,
    BatchFileResult,
)
from acqstore.acq_image.analysis.diameter_analysis.diameter_analysis import DiameterAnalysis
from acqstore.acq_image.analysis.model import (
    AnalysisCancelled,
    AnalysisExclusionError,
    AnalysisKey,
    AnalysisRunContext,
)

if TYPE_CHECKING:
    from acqstore.acq_image.acq_image import AcqImage


class DiameterBatchStrategy:
    """Run diameter analysis over one file at a time for ``AcqAnalysisBatch``.

    The strategy intentionally uses the same ``AcqAnalysisSet.create`` and
    ``AcqAnalysisSet.run_analysis`` path as single-file GUI analysis. It only
    chooses the target ROI, configures runtime execution options, and converts
    exceptions/cancellation into per-file batch results.

    Args:
        channel: Explicit channel index used for every file.
        roi_mode: How each file's target ROI is selected.
        roi_id: Required ROI identifier for ``ANALYZE_EXISTING_ROI`` mode.
        detection_params: Detection parameters passed to each new analysis.
        use_threads: Whether each per-file diameter analysis may use threads.
        max_workers: Optional worker count for each per-file diameter analysis.
    """

    analysis_name = DiameterAnalysis.analysis_name
    kind = AnalysisBatchKind.DIAMETER

    def __init__(
        self,
        *,
        channel: int,
        roi_mode: RoiBatchMode,
        roi_id: int | None,
        detection_params: dict[str, object],
        use_threads: bool = True,
        max_workers: int | None = None,
    ) -> None:
        self._channel = int(channel)
        self._roi_mode = RoiBatchMode(roi_mode)
        self._roi_id = None if roi_id is None else int(roi_id)
        self._detection_params = dict(detection_params)
        self._use_threads = bool(use_threads)
        self._max_workers = max_workers
        DiameterAnalysis.validate_detection_params(self._detection_params)
        if self._roi_mode is RoiBatchMode.ANALYZE_EXISTING_ROI and self._roi_id is None:
            raise ValueError("roi_id is required for ANALYZE_EXISTING_ROI mode")

    def process_file(
        self,
        acq_image: AcqImage,
        *,
        cancel_event: threading.Event,
    ) -> BatchFileResult:
        """Run diameter analysis for one file.

        Args:
            acq_image: Acquisition image to analyze.
            cancel_event: Shared cancellation event.

        Returns:
            Per-file batch result. The ``AcqImage`` is mutated on success, but
            it is not saved.
        """
        if cancel_event.is_set():
            return self._result(acq_image, None, BatchFileOutcome.CANCELLED, "cancelled")

        target_roi_id = self._resolve_roi_id(acq_image)
        if target_roi_id is None:
            return self._result(
                acq_image,
                None,
                BatchFileOutcome.SKIPPED_MISSING_ROI,
                f"ROI {self._roi_id} not in file",
            )

        key = AnalysisKey(self.analysis_name, self._channel, target_roi_id)
        acq_image.analysis_set.remove(key)

        try:
            analysis = acq_image.analysis_set.create(
                self.analysis_name,
                channel=self._channel,
                roi_id=target_roi_id,
                detection_params=self._detection_params,
            )
            if isinstance(analysis, DiameterAnalysis):
                analysis.set_execution_options(
                    use_threads=self._use_threads,
                    max_workers=self._max_workers,
                )
            context = AnalysisRunContext(cancel_callback=cancel_event.is_set)
            acq_image.analysis_set.run_analysis(analysis.key, context=context)
        except AnalysisCancelled:
            return self._result(acq_image, target_roi_id, BatchFileOutcome.CANCELLED, "cancelled")
        except AnalysisExclusionError as exc:
            return self._result(acq_image, target_roi_id, BatchFileOutcome.SKIPPED_CONFLICT, str(exc))
        except Exception as exc:
            return self._result(acq_image, target_roi_id, BatchFileOutcome.FAILED, repr(exc))

        return self._result(acq_image, target_roi_id, BatchFileOutcome.OK, "ok")

    def _resolve_roi_id(self, acq_image: AcqImage) -> int | None:
        """Resolve the target ROI identifier for one file.

        Args:
            acq_image: Acquisition image to analyze.

        Returns:
            ROI identifier, or None when an existing ROI is required but absent.
        """
        if self._roi_mode is RoiBatchMode.ADD_NEW_ROI:
            roi = acq_image.rois.create_rect_roi()
            return roi.roi_id

        assert self._roi_id is not None
        if not acq_image.rois.has_roi(self._roi_id):
            return None
        return self._roi_id

    def _result(
        self,
        acq_image: AcqImage,
        roi_id: int | None,
        outcome: BatchFileOutcome,
        message: str,
    ) -> BatchFileResult:
        """Build a per-file batch result.

        Args:
            acq_image: Acquisition image that was processed.
            roi_id: ROI identifier used for analysis, if any.
            outcome: Batch outcome.
            message: Human-readable outcome message.

        Returns:
            Batch file result.
        """
        return BatchFileResult(
            file_path=str(acq_image.path),
            analysis_name=self.analysis_name,
            channel=self._channel,
            roi_id=roi_id,
            outcome=outcome,
            message=message,
        )
