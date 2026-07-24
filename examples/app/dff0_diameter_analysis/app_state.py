"""Mutable state for the standalone triggered-event analysis app."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from acqstore.acq_image.acq_image_list import AcqImageList
    from acqstore.common_analysis.dff0_diameter_analysis.analysis import (
        Dff0DiameterAnalysis,
    )
    from acqstore.common_analysis.dff0_diameter_analysis.continuous_analysis import (
        Dff0DiameterContinuousAnalysis,
    )

    from .analysis_hits import AnalysisHit


@dataclass(slots=True)
class AppState:
    """Hold per-page application state independently of NiceGUI widgets.

    Args:
        acq_image_list: Loaded acquisition collection.
        analysis_hits: Discoverable file/channel/ROI selections.
    """

    acq_image_list: AcqImageList
    analysis_hits: list[AnalysisHit]
    selected_hit: AnalysisHit | None = None
    analysis: Dff0DiameterAnalysis | None = None
    continuous_analysis: Dff0DiameterContinuousAnalysis | None = None
    _hits_by_id: dict[str, AnalysisHit] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Build the stable hit lookup used by grid callbacks."""
        self._hits_by_id = {hit.hit_id: hit for hit in self.analysis_hits}

    def select_hit(self, hit_id: str) -> AnalysisHit:
        """Select and return one hit by identifier.

        Args:
            hit_id: Stable identifier emitted by the AG Grid row.

        Returns:
            Selected analysis hit.

        Raises:
            KeyError: If the row identifier is unknown.
        """
        hit = self._hits_by_id[hit_id]
        self.selected_hit = hit
        self.analysis = None
        self.continuous_analysis = None
        return hit
