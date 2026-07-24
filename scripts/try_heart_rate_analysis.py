"""Exercise AcqStore heart-rate analysis from the command line.

Heart-rate analysis is seeded by an existing ``radon_velocity`` analysis for the
same channel/ROI. This script runs a Radon velocity analysis, then runs
heart-rate analysis on the resulting velocity series, prints the summary,
saves the AcqImage sidecar, reloads it, and prints the reloaded summary.

Edit ``DATA_PATH`` before running.

Run:

    uv run python scripts/try_heart_rate_analysis.py
"""

from __future__ import annotations

import numpy as np

from acqstore.acq_image import AcqImage
from acqstore.acq_image import AcqImageList
from acqstore.acq_image.analysis import HeartRateAnalysis
from acqstore.acq_image.analysis.heart_rate_analysis.plotting.mpl_plots import (
    plot_segment_series,
    plot_summary,
)
from acqstore.acq_image.analysis.heart_rate_analysis.plotting.plotly_plots import (
    plot_segment_series_plotly,
    plot_summary_plotly,
)
from acqstore.acq_image.analysis.model import AnalysisKey, AnalysisRunContext
from acqstore.acq_image.analysis import RadonVelocityAnalysis
from acqstore.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)
setup_logging()

DATA_PATH = "/Users/cudmore/Sites/cloudscope/example-data"
CHANNEL = 0
WINDOW_WIDTH = 64
PLOT_RESULTS = True
USE_PLOTLY = True


def run_velocity(acq_image: AcqImage, *, channel: int, roi_id: int) -> RadonVelocityAnalysis:
    """Create and run a Radon velocity analysis to seed heart-rate analysis.

    Args:
        acq_image: Acquisition image.
        channel: Channel index.
        roi_id: ROI identifier.

    Returns:
        Completed Radon velocity analysis.
    """
    detection_params = RadonVelocityAnalysis.get_default_detection_params()
    detection_params["window_width"] = WINDOW_WIDTH

    key = AnalysisKey(RadonVelocityAnalysis.analysis_name, channel, roi_id)
    acq_image.analysis_set.remove(key)
    analysis = acq_image.analysis_set.create(
        RadonVelocityAnalysis.analysis_name,
        channel=channel,
        roi_id=roi_id,
        detection_params=detection_params,
    )
    context = AnalysisRunContext(
        progress_callback=lambda fraction, message: logger.info(f"velocity progress={fraction}: {message}")
    )
    acq_image.analysis_set.run_analysis(analysis.key, context=context)
    if not isinstance(analysis, RadonVelocityAnalysis):
        raise TypeError(f"Expected RadonVelocityAnalysis, got {type(analysis).__name__}")
    return analysis


def run_heart_rate(acq_image: AcqImage, *, channel: int, roi_id: int) -> HeartRateAnalysis:
    """Create and run heart-rate analysis seeded by the velocity analysis.

    Args:
        acq_image: Acquisition image.
        channel: Channel index.
        roi_id: ROI identifier.

    Returns:
        Completed heart-rate analysis.
    """
    key = AnalysisKey(HeartRateAnalysis.analysis_name, channel, roi_id)
    acq_image.analysis_set.remove(key)
    analysis = acq_image.analysis_set.create(
        HeartRateAnalysis.analysis_name,
        channel=channel,
        roi_id=roi_id,
    )
    context = AnalysisRunContext(
        progress_callback=lambda fraction, message: logger.info(f"heart-rate progress={fraction}: {message}")
    )
    acq_image.analysis_set.run_analysis(analysis.key, context=context)
    if not isinstance(analysis, HeartRateAnalysis):
        raise TypeError(f"Expected HeartRateAnalysis, got {type(analysis).__name__}")
    return analysis


def report_velocity_diagnostics(velocity: RadonVelocityAnalysis) -> None:
    """Print sampling diagnostics for the velocity series feeding heart-rate.

    Args:
        velocity: Completed Radon velocity analysis.

    Returns:
        None.
    """
    plot_data = velocity.get_plot_data()
    if plot_data is None:
        print("Velocity analysis produced no plot data.")
        return
    t = np.asarray(plot_data.x, dtype=float)
    v = np.asarray(plot_data.y, dtype=float)
    finite = np.isfinite(t) & np.isfinite(v)
    n_valid = int(np.sum(finite))
    print("Velocity series diagnostics:")
    print(f"  n_total = {t.size}")
    print(f"  n_valid = {n_valid} (heart-rate core requires >= 256)")
    if n_valid >= 2:
        dt = np.diff(t[finite])
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if dt.size:
            fs = 1.0 / float(np.median(dt))
            print(f"  effective sample rate ~= {fs:.2f} Hz (Nyquist {fs / 2:.2f} Hz)")
            print("  heart-rate band 240-600 bpm = 4.0-10.0 Hz")


def print_summary(label: str, summary: dict) -> None:
    """Print a heart-rate summary dictionary.

    Args:
        label: Label printed before the summary.
        summary: Heart-rate analysis summary dictionary.

    Returns:
        None.
    """
    print(label)
    print(f"  status = {summary.get('status')!r} ({summary.get('status_note')!r})")
    print(f"  n_valid = {summary.get('n_valid')} / {summary.get('n_total')}")
    for method in ("lomb", "welch"):
        block = summary.get(method) or {}
        print(
            f"  {method}: bpm={block.get('bpm')} f_hz={block.get('f_hz')} "
            f"snr={block.get('snr')} status={block.get('status')!r}"
        )
    print(f"  agreement = {summary.get('agreement')}")


def plot_heart_rate_diagnostics(velocity: RadonVelocityAnalysis, heart_rate: HeartRateAnalysis) -> None:
    """Plot heart-rate diagnostic figures with Plotly or matplotlib.

    Args:
        velocity: Completed parent velocity analysis.
        heart_rate: Completed heart-rate analysis with detection params.

    Returns:
        None.
    """
    if not PLOT_RESULTS:
        return
    plot_data = velocity.get_plot_data()
    if plot_data is None:
        print("Heart-rate plots skipped: velocity plot data is unavailable.")
        return

    title = f"{heart_rate.key.analysis_name} ch={heart_rate.key.channel} roi={heart_rate.key.roi_id}"
    if USE_PLOTLY:
        summary_fig = plot_summary_plotly(
            plot_data.x,
            plot_data.y,
            params=heart_rate.detection_params,
            title=title,
        )
        summary_fig.show()
        if bool(heart_rate.detection_params.get("do_segments", False)):
            segment_fig = plot_segment_series_plotly(
                plot_data.x,
                plot_data.y,
                params=heart_rate.detection_params,
                title=f"{title} | segment HR",
            )
            segment_fig.show()
        return

    from matplotlib import pyplot as plt

    plot_summary(
        plot_data.x,
        plot_data.y,
        params=heart_rate.detection_params,
        title=title,
    )
    if bool(heart_rate.detection_params.get("do_segments", False)):
        plot_segment_series(
            plot_data.x,
            plot_data.y,
            params=heart_rate.detection_params,
            title=f"{title} | segment HR",
        )
    plt.show()


def main() -> None:
    """Run the manual velocity -> heart-rate workflow on one file."""
    acq_image_list = AcqImageList(DATA_PATH)
    path = acq_image_list.file_list[1]
    print(f"Using file: {path}")

    print("Heart-rate detection schema:")
    for field in HeartRateAnalysis.get_detection_schema():
        print(f"  {field.name}: default={field.default!r} type={field.value_type.value} unit={field.unit}")

    acq_image = AcqImage(path)
    new_roi = acq_image.rois.create_rect_roi(name="hr_test", note="heart-rate test")
    roi_id = new_roi.roi_id

    velocity = run_velocity(acq_image, channel=CHANNEL, roi_id=roi_id)
    report_velocity_diagnostics(velocity)

    heart_rate = run_heart_rate(acq_image, channel=CHANNEL, roi_id=roi_id)
    print("Heart-rate analysis key:", heart_rate.key)
    print_summary("Summary before save:", heart_rate.result.summary)
    plot_heart_rate_diagnostics(velocity, heart_rate)

    acq_image.save()

    reloaded = AcqImage(path)
    loaded = reloaded.analysis_set.get_required(heart_rate.key)
    print_summary("Summary after reload:", loaded.result.summary)


if __name__ == "__main__":
    main()
