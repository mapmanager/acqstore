"""Run one diameter analysis from example kymograph data.

Edit the constants near the top of the file before running.

Run:

    uv run python scripts/try_diameter_analysis.py
"""

from __future__ import annotations

from acqstore.acq_image import AcqImage
from acqstore.acq_image import AcqImageList
from acqstore.acq_image.analysis import DiameterAnalysis
from acqstore.acq_image.analysis.model import AnalysisRunContext, BaseAnalysis
from acqstore.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)
setup_logging()


def run_diameter_analysis(acq_image: AcqImage) -> DiameterAnalysis:
    """Create and run diameter analysis on one file.

    Args:
        acq_image: Acquisition image.

    Returns:
        Completed diameter analysis.
    """
    channel = 0

    new_roi = acq_image.rois.create_rect_roi(name="diameter_test", note="diameter test")
    roi_id = new_roi.roi_id

    detection_params = {
        "diameter_method": "threshold_width",
        "window_rows_odd": 5,
        "post_filter_kernel_size": 3,
    }

    context = AnalysisRunContext(
        progress_callback=lambda fraction, message: print(f"  === progress={fraction}: {message}")
    )
    logger.info("running analysis")
    analysis = acq_image.analysis_set.create_and_run(
        DiameterAnalysis,
        channel=channel,
        roi_id=roi_id,
        detection_params=detection_params,
        replace_existing=True,
        context=context,
    )
    return analysis


def plot_diameter_results(analysis: BaseAnalysis) -> None:
    """Plot diameter and edge traces with matplotlib.

    Args:
        analysis: Diameter analysis with result table columns.

    Returns:
        None.
    """
    import matplotlib.pyplot as plt

    time_s = analysis.get_column("time_s")
    diameter_um = analysis.get_column("diameter_um_filt")
    left_edge_um = analysis.get_column("left_edge_um")
    right_edge_um = analysis.get_column("right_edge_um")

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(time_s, diameter_um, label="diameter_um_filt")
    axes[0].set_ylabel("Diameter (um)")
    axes[0].legend()
    axes[0].set_title("Diameter vs time (ROI-local)")

    axes[1].plot(time_s, left_edge_um, label="left_edge_um")
    axes[1].plot(time_s, right_edge_um, label="right_edge_um")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Position (um)")
    axes[1].legend()
    axes[1].set_title("Edges vs time (ROI-local)")

    plt.tight_layout()
    plt.show()


def main() -> None:
    """Run the manual single-file diameter workflow.

    Returns:
        None.
    """
    path = "/Users/cudmore/Sites/cloudscope/example-data"
    acq_image_list = AcqImageList(path)
    path = acq_image_list.file_list[1]

    print("Diameter detection schema:")
    for field in DiameterAnalysis.get_detection_schema():
        print(
            f"  {field.name}: default={field.default!r}, "
            f"type={field.value_type.value}, methods={field.methods}, "
            f"choices={field.choices}"
        )

    acq_image = AcqImage(path)
    analysis = run_diameter_analysis(acq_image)

    print("Analysis key:", analysis.key)
    print("Summary:", analysis.result.summary)
    print("Columns:", analysis.get_table_columns())
    print("Overlay traces:", [trace.trace_id for trace in analysis.get_overlay_traces()])

    logger.info("saving")
    acq_image.save()

    plot_diameter_results(analysis)

    logger.info("reloading ...")
    reloaded = AcqImage(path)
    loaded = reloaded.analysis_set.get_required(analysis.key)
    print("  Reloaded:")
    print("  Analysis key:", loaded.key)
    print("  Summary:", loaded.result.summary)
    print("  Columns:", loaded.get_table_columns())


if __name__ == "__main__":
    main()
