"""Discover AcqImage selections with paired sum-intensity and diameter analyses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from acqstore.acq_image.acq_image import AcqImage
    from acqstore.acq_image.acq_image_list import AcqImageList


@dataclass(frozen=True, slots=True)
class AnalysisHit:
    """One AcqImage channel/ROI selection with both required analyses.

    Args:
        hit_id: Stable application-local identifier for this hit.
        acq_image: Acquisition image that owns the analyses.
        file_name: Human-readable source file name.
        channel: Zero-based channel index.
        roi_id: ROI identifier.
        peak_count: Number of sum-intensity peak events.
    """

    hit_id: str
    acq_image: AcqImage
    file_name: str
    channel: int
    roi_id: int
    peak_count: int

    def to_grid_row(self) -> dict[str, object]:
        """Return a JSON-compatible row for the NiceGUI AG Grid.

        Returns:
            Flat row data containing only values needed by the grid and event
            callbacks.
        """
        return {
            "hit_id": self.hit_id,
            "file_name": self.file_name,
            "channel": self.channel,
            "roi_id": self.roi_id,
            "peak_count": self.peak_count,
        }


def find_analysis_hits(acq_image_list: AcqImageList) -> list[AnalysisHit]:
    """Find every channel/ROI pair with sum-intensity and diameter analyses.

    One acquisition may contribute multiple hits when more than one channel/ROI
    pair contains both required analyses.

    Args:
        acq_image_list: Loaded acquisition collection with analysis CSV tables.

    Returns:
        Hits in acquisition-list order and then channel/ROI order.
    """
    hits: list[AnalysisHit] = []

    for acq_image in acq_image_list:
        analysis_set = acq_image.analysis_set
        sum_keys = {
            (analysis.key.channel, analysis.key.roi_id)
            for analysis in analysis_set.as_list()
            if analysis.key.analysis_name == "sum_intensity"
        }
        diameter_keys = {
            (analysis.key.channel, analysis.key.roi_id)
            for analysis in analysis_set.as_list()
            if analysis.key.analysis_name == "diameter"
        }

        for channel, roi_id in sorted(sum_keys & diameter_keys):
            sum_analysis = analysis_set.get_analysis(
                "sum_intensity",
                channel=channel,
                roi_id=roi_id,
            )
            diameter_analysis = analysis_set.get_analysis(
                "diameter",
                channel=channel,
                roi_id=roi_id,
            )

            # The app requires loaded continuous tables for both analyses.
            if sum_analysis.result.table is None or diameter_analysis.result.table is None:
                continue
            if sum_analysis.result.table.empty or diameter_analysis.result.table.empty:
                continue

            peak_events = sum_analysis.get_peak_events()
            hit_id = f"{acq_image.file_id}|channel={channel}|roi={roi_id}"
            hits.append(
                AnalysisHit(
                    hit_id=hit_id,
                    acq_image=acq_image,
                    file_name=acq_image.name,
                    channel=channel,
                    roi_id=roi_id,
                    peak_count=len(peak_events),
                )
            )

    return hits
