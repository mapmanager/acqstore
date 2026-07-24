"""Exercise AcqStore event analysis from the command line.

Edit ``SOURCE_PATH`` before running.

Run:

    uv run python scripts/try_event_analysis.py
"""

from __future__ import annotations

from pathlib import Path

from acqstore.acq_image import AcqImage
from acqstore.acq_image.analysis import EventAnalysis
from acqstore.acq_image.analysis.event_analysis.event_analysis import EventType
from acqstore.acq_image.analysis.model import AnalysisKey
from acqstore.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)
setup_logging()

SOURCE_PATH = "/Users/cudmore/Sites/cloudscope/tests/acqstore/data/oir-samples/20251030_A106_0001.oir"


def get_target(acq_image: AcqImage) -> tuple[int, int]:
    """Return channel and ROI target for event analysis.

    Args:
        acq_image: Acquisition image.

    Returns:
        Tuple ``(channel, roi_id)``.
    """
    channel = acq_image.get_default_channel()
    roi_id = acq_image.get_default_roi()
    if channel is None:
        raise RuntimeError("No default channel available")
    if roi_id is None:
        roi = acq_image.rois.create_rect_roi(name="event_test", note="event test")
        roi_id = roi.roi_id
    return int(channel), int(roi_id)


def create_event_analysis(acq_image: AcqImage, channel: int, roi_id: int) -> EventAnalysis:
    """Create a fresh event analysis and add sample events.

    Args:
        acq_image: Acquisition image.
        channel: Channel index.
        roi_id: ROI identifier.

    Returns:
        Created event analysis.
    """
    key = AnalysisKey(EventAnalysis.analysis_name, channel, roi_id)
    acq_image.analysis_set.remove(key)
    analysis = acq_image.analysis_set.create(EventAnalysis.analysis_name, channel=channel, roi_id=roi_id)
    if not isinstance(analysis, EventAnalysis):
        raise TypeError(f"Expected EventAnalysis, got {type(analysis).__name__}")
    first = analysis.add_rect(1.0, 2.5, event_type=EventType.USER)
    second = analysis.add_rect(5.0, 7.0, event_type=EventType.TRANSIENT)
    analysis.update_rect(second.id, x1=8.0)
    logger.info("created events: %s", [event.to_json_dict() for event in analysis.get_rects()])
    analysis.delete_rect(first.id)
    return analysis


def main() -> None:
    """Run event analysis CRUD, save, and reload workflow."""
    path = str(Path(SOURCE_PATH).expanduser())
    acq_image = AcqImage(path)
    channel, roi_id = get_target(acq_image)
    analysis = create_event_analysis(acq_image, channel, roi_id)

    print("Analysis key:", analysis.key)
    print("Summary before save:", analysis.result.summary)
    acq_image.save()

    reloaded = AcqImage(path)
    loaded = reloaded.analysis_set.get_required(analysis.key)
    print("Summary after reload:", loaded.result.summary)
    if isinstance(loaded, EventAnalysis):
        print("Reloaded events:", [event.to_json_dict() for event in loaded.get_rects()])


if __name__ == "__main__":
    main()
