"""ROI-selection modes for batch analysis strategies.

A batch strategy can either create a new ROI for each file or require an
existing ROI identifier. Keeping this as an enum makes GUI choices and scripted
batch configuration use the same vocabulary.
"""

from __future__ import annotations

from enum import StrEnum


class RoiBatchMode(StrEnum):
    """How a batch strategy chooses the ROI for each file."""

    ANALYZE_EXISTING_ROI = "analyze_existing_roi"
    ADD_NEW_ROI = "add_new_roi"
