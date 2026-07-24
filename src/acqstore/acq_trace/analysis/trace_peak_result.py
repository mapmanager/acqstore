"""Peak-detection result objects for trace acquisitions."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from acqstore.acq_trace.analysis.trace_peak_params import TracePeakDetectionParams


@dataclass(frozen=True)
class TracePeakDetectionResult:
    """Peak-detection result for one AcqTrace analysis run.

    Args:
        trace_table: Per-sample rows for all analyzed sweeps.
        peak_table: One row per detected peak.
        params: Parameters used for the detection run.
        channel_index: Zero-based analyzed channel index.
        sweep_index: Zero-based analyzed sweep index, or None when all sweeps
            were analyzed.
    """

    trace_table: pd.DataFrame
    peak_table: pd.DataFrame
    params: TracePeakDetectionParams
    channel_index: int
    sweep_index: int | None

    @property
    def num_peaks(self) -> int:
        """Return number of detected peaks.

        Returns:
            Number of rows in :attr:`peak_table`.
        """
        return int(len(self.peak_table))

    @property
    def num_sweeps_analyzed(self) -> int:
        """Return number of sweeps represented by the result.

        Returns:
            Number of unique sweep indices in :attr:`trace_table`.
        """
        if self.trace_table.empty:
            return 0
        return int(self.trace_table['sweep_index'].nunique())

    def summary_dict(self) -> dict[str, object]:
        """Return a compact result summary.

        Returns:
            Dictionary with analyzed target, parameter, and result counts.
        """
        return {
            'channel_index': self.channel_index,
            'sweep_index': self.sweep_index,
            'num_sweeps_analyzed': self.num_sweeps_analyzed,
            'num_peaks': self.num_peaks,
            'params': self.params.as_dict(),
        }
