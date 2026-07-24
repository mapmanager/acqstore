"""Batch strategy for Radon velocity analysis.

The strategy chooses a target ROI for each file, creates a ``RadonVelocityAnalysis``
through the file's ``AcqAnalysisSet``, configures runtime multiprocessing options,
and returns a structured per-file result. It does not save files; callers decide
when to persist changed analyses.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from acqstore.acq_image.acq_image import AcqImage
from acqstore.acq_image.analysis.batch.roi_mode import RoiBatchMode
from acqstore.acq_image.analysis.batch.types import (
    AnalysisBatchKind,
    BatchFileOutcome,
    BatchFileResult,
)
from acqstore.acq_image.analysis.model import (
    AnalysisCancelled,
    AnalysisExclusionError,
    AnalysisKey,
    AnalysisRunContext,
)
from acqstore.acq_image.analysis.velocity_analysis.radon_velocity_analysis import (
    RadonVelocityAnalysis,
)


class RadonVelocityBatchStrategy:
    """Run the same Radon velocity analysis over a list of files.

    Args:
        channel: Explicit channel index used for every file.
        roi_mode: How each file's target ROI is selected.
        roi_id: Required ROI identifier for ``ANALYZE_EXISTING_ROI`` mode.
        detection_params: Detection parameters passed to each new analysis.
        use_multiprocessing: Whether the per-file Radon analysis may use
            multiprocessing. This parameter is stored for API clarity; the
            current wrapper uses ``RadonVelocityAnalysis`` defaults until the
            analysis wrapper exposes runtime execution options.
        processes: Optional number of worker processes for the per-file Radon
            algorithm. ``None`` means the algorithm chooses a CPU-count based
            default. This parameter is stored for API clarity; the current
            wrapper uses ``RadonVelocityAnalysis`` defaults until execution
            options are exposed.
    """

    analysis_name = RadonVelocityAnalysis.analysis_name
    kind = AnalysisBatchKind.RADON_VELOCITY

    def __init__(
        self,
        *,
        channel: int,
        roi_mode: RoiBatchMode,
        roi_id: int | None,
        detection_params: dict[str, object],
        use_multiprocessing: bool = True,
        processes: int | None = None,
    ) -> None:
        self._channel = int(channel)
        self._roi_mode = RoiBatchMode(roi_mode)
        self._roi_id = None if roi_id is None else int(roi_id)
        self._detection_params = dict(detection_params)
        self._use_multiprocessing = bool(use_multiprocessing)
        self._processes = processes
        RadonVelocityAnalysis.validate_detection_params(self._detection_params)
        if self._roi_mode is RoiBatchMode.ANALYZE_EXISTING_ROI and self._roi_id is None:
            raise ValueError("roi_id is required for ANALYZE_EXISTING_ROI mode")

    def process_file(
        self,
        acq_image: 'AcqImage',
        *,
        cancel_event: threading.Event,
    ) -> BatchFileResult:
        """Run Radon velocity analysis for one file.

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
            if isinstance(analysis, RadonVelocityAnalysis):
                analysis.set_execution_options(
                    use_multiprocessing=self._use_multiprocessing,
                    processes=self._processes,
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

    def _resolve_roi_id(self, acq_image: 'AcqImage') -> int | None:
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
        acq_image: 'AcqImage',
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
