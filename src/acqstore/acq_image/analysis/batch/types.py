"""Shared value types for batch analysis workflows.

These small enums and dataclasses describe what kind of batch analysis is being
run and what happened for each file. They are intentionally lightweight so GUI,
script, and test code can consume batch results without inspecting exceptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class AnalysisBatchKind(StrEnum):
    """Kind of analysis run in a batch."""

    RADON_VELOCITY = "radon_velocity"
    DIAMETER = "diameter"


class BatchFileOutcome(StrEnum):
    """Per-file batch result outcome."""

    OK = "ok"
    SKIPPED_MISSING_ROI = "skipped_missing_roi"
    SKIPPED_CONFLICT = "skipped_conflict"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class BatchFileResult:
    """Lightweight per-file batch result.

    Args:
        file_path: Source acquisition file path.
        analysis_name: Analysis type name.
        channel: Channel index used for analysis.
        roi_id: ROI identifier used for analysis, or None when unavailable.
        outcome: Per-file outcome.
        message: Human-readable outcome message.
    """

    file_path: str
    analysis_name: str
    channel: int
    roi_id: int | None
    outcome: BatchFileOutcome
    message: str
