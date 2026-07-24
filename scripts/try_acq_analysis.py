"""Exercise the AcqStore analysis API with one hardcoded AcqImage path.

Edit ``SOURCE_PATH`` before running.

Run:

    uv run python scripts/try_acq_analysis.py
"""

from __future__ import annotations

from pathlib import Path

from acqstore.acq_image import AcqImage
from acqstore.acq_image.analysis import EventAnalysis
from acqstore.acq_image.analysis.model import AnalysisRunContext
from acqstore.acq_image.analysis import RadonVelocityAnalysis

SOURCE_PATH = "/Users/cudmore/Sites/cloudscope/tests/acqstore/data/oir-samples/20251030_A106_0001.oir"


def load_acq_image(path: str) -> AcqImage:
    """Load one acquisition image.

    Args:
        path: Acquisition file path.

    Returns:
        Loaded acquisition image.
    """
    return AcqImage(path)


def get_target(acq_image: AcqImage) -> tuple[int, int]:
    """Return channel and ROI target for test analysis.

    Args:
        acq_image: Acquisition image.

    Returns:
        Tuple ``(channel, roi_id)``.

    Raises:
        RuntimeError: If no channel or ROI is available.
    """
    channel_indices = acq_image.images.channel_indices
    roi_ids = acq_image.rois.get_roi_ids()
    if not channel_indices:
        raise RuntimeError("No channel available")
    if not roi_ids:
        raise RuntimeError("No ROI available")
    return channel_indices[0], roi_ids[0]


def run_velocity_analysis(acq_image: AcqImage, channel: int, roi_id: int) -> None:
    """Create and run velocity analysis.

    Args:
        acq_image: Acquisition image.
        channel: Channel index.
        roi_id: ROI identifier.

    Returns:
        None.
    """
    context = AnalysisRunContext(
        progress_callback=lambda fraction, message: print(f"velocity {fraction}: {message}")
    )
    acq_image.analysis_set.create_and_run(
        RadonVelocityAnalysis,
        channel=channel,
        roi_id=roi_id,
        detection_params={"window_width": 16},
        replace_existing=True,
        execution_options={"use_multiprocessing": False},
        context=context,
    )


def run_event_analysis(acq_image: AcqImage, channel: int, roi_id: int) -> None:
    """Create and run event analysis.

    Args:
        acq_image: Acquisition image.
        channel: Channel index.
        roi_id: ROI identifier.

    Returns:
        None.
    """
    acq_image.analysis_set.create_and_run(
        EventAnalysis,
        channel=channel,
        roi_id=roi_id,
        replace_existing=True,
    )


def save_acq_image(acq_image: AcqImage) -> None:
    """Save acquisition image sidecars.

    Args:
        acq_image: Acquisition image to save.

    Returns:
        None.
    """
    acq_image.save()


def reload_acq_image(path: str) -> AcqImage:
    """Reload an acquisition image from disk.

    Args:
        path: Acquisition file path.

    Returns:
        Reloaded acquisition image.
    """
    return AcqImage(path)


def print_analysis_summary(acq_image: AcqImage) -> None:
    """Print loaded analysis records and table columns.

    Args:
        acq_image: Acquisition image.

    Returns:
        None.
    """
    print("Analysis records:")
    for record in acq_image.analysis_set.serialize_json_analysis():
        print(record)

    for analysis in acq_image.analysis_set.as_list():
        print(
            analysis.key,
            "summary=",
            analysis.result.summary,
            "columns=",
            analysis.get_table_columns(),
        )


def main() -> None:
    """Run load -> analysis -> save -> reload workflow.

    Returns:
        None.
    """

    # load from folder path
    # path = '/Users/cudmore/Sites/cloudscope/example-data'
    # from acqstore.acq_image import AcqImageList
    # acq_image_list = AcqImageList(path)
    # path = acq_image_list.file_list[0]

    path = str(Path(SOURCE_PATH).expanduser())
    acq_image = load_acq_image(path)

    # add a default rect roi
    new_rect_roi = acq_image.rois.create_rect_roi(
        bounds=None,
        name="phase1_rect",
        note="created_by_try_create_roi",
    )

    channel, roi_id = get_target(acq_image)

    run_velocity_analysis(acq_image, channel, roi_id)
    run_event_analysis(acq_image, channel, roi_id)
    save_acq_image(acq_image)

    reloaded = reload_acq_image(path)
    print_analysis_summary(reloaded)


if __name__ == "__main__":
    main()
