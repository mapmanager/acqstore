"""Generic batch runner for AcqImage analyses.

``AcqAnalysisBatch`` applies a one-file analysis strategy to many independent
``AcqImage`` objects. File-level work is scheduled with a thread pool so batch
runs can remain responsive and cancellable. Each strategy still uses the same
single-file analysis creation and execution APIs used by CloudScope and scripts.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Protocol
from concurrent.futures import ThreadPoolExecutor, as_completed

if TYPE_CHECKING:
    from acqstore.acq_image.acq_image import AcqImage
from acqstore.acq_image.analysis.batch.types import BatchFileOutcome, BatchFileResult


class BatchAnalysisStrategy(Protocol):
    """Protocol for one-file analysis strategies used by ``AcqAnalysisBatch``."""

    def process_file(
        self,
        acq_image: 'AcqImage',
        *,
        cancel_event: threading.Event,
    ) -> BatchFileResult:
        """Process one file.

        Args:
            acq_image: File to analyze.
            cancel_event: Shared cancellation event.

        Returns:
            Per-file batch result.
        """
        ...


class AcqAnalysisBatch:
    """Run one analysis strategy over many acquisition files.

    File-level work uses a ``ThreadPoolExecutor``. Each file is independent;
    inner analysis code may also use multiprocessing or threads as configured by
    the concrete strategy. This preserves the same scientific analysis path used
    by one-file GUI and scripting workflows while reducing wall-clock time for
    large file collections.

    Args:
        files: Acquisition images to process.
        strategy: Per-file analysis strategy.
        max_parallel_files: Maximum concurrent file workers. Values smaller
            than one are clamped to one.
    """

    def __init__(
        self,
        files: Sequence['AcqImage'],
        strategy: BatchAnalysisStrategy,
        *,
        max_parallel_files: int = 4,
    ) -> None:
        self._files = list(files)
        self._strategy = strategy
        self._max_parallel_files = max(1, int(max_parallel_files))

    def run(
        self,
        *,
        cancel_event: threading.Event | None = None,
        on_file_result: Callable[[BatchFileResult], None] | None = None,
    ) -> list[BatchFileResult]:
        """Run the batch and return results in input order.

        Args:
            cancel_event: Optional shared cancellation event. If omitted, a new
                unset event is used.
            on_file_result: Optional callback invoked as each file completes.

        Returns:
            Per-file results in the same order as input files.
        """
        if not self._files:
            return []

        cancel_event = cancel_event or threading.Event()
        workers = min(self._max_parallel_files, len(self._files))
        results: list[BatchFileResult | None] = [None] * len(self._files)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_index = {
                executor.submit(
                    self._strategy.process_file,
                    acq_image,
                    cancel_event=cancel_event,
                ): index
                for index, acq_image in enumerate(self._files)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover - defensive fallback
                    acq_image = self._files[index]
                    result = BatchFileResult(
                        file_path=str(acq_image.path),
                        analysis_name="unknown",
                        channel=-1,
                        roi_id=None,
                        outcome=BatchFileOutcome.FAILED,
                        message=repr(exc),
                    )
                results[index] = result
                if on_file_result is not None:
                    on_file_result(result)

        return [result for result in results if result is not None]
