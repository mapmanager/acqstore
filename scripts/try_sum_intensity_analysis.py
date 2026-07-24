"""Run and plot one sum-intensity analysis from example data.

Edit the constants near the top of the file before running.

Run:

    uv run python scripts/try_sum_intensity_analysis.py
"""

from __future__ import annotations

from pathlib import Path

import plotly.graph_objects as go

from acqstore.acq_image import AcqImage
from acqstore.acq_image.analysis.sum_intensity_analysis.sum_intensity_analysis import (
    SumIntensityAnalysis,
)
from acqstore.acq_image.analysis.sum_intensity_analysis.sum_intensity_core import (
    SumIntensityEventPointKey,
    SumIntensityTraceKey,
)
from acqstore.acq_image.analysis.sum_intensity_analysis.sum_intensity_presets import (
    SumIntensityPresetName,
)
from acqstore.acq_image.analysis.model import AnalysisRunContext, BaseAnalysis
from acqstore.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)
setup_logging()

# Edit this path locally. The ROI defaults to the full image.
SOURCE_PATH = Path("/Users/cudmore/Dropbox/data/rabbit-ca-variance/raw-data/jan-12-2022/Control/220110n_0003.tif.frames/220110n_0003.tif")

# Edit this path locally. The ROI defaults to the full image.
# SOURCE_PATH = Path("/Users/cudmore/Dropbox/data/rabbit-ca-variance/raw-data/jan-12-2022/Control/220110n_0003.tif.frames/220110n_0003.tif")
# SOURCE_PATH = Path("/Users/cudmore/Dropbox/data/rabbit-ca-variance/raw-data/jan-12-2022/Control/220110n_0005.tif.frames/220110n_0005.tif")
SOURCE_PATH = Path("/Users/cudmore/Dropbox/data/rabbit-ca-variance/raw-data/jan-12-2022/Control/220110n_0009.tif.frames/220110n_0009.tif")
SOURCE_PATH = Path("/Users/cudmore/Dropbox/data/rabbit-ca-variance/raw-data/jan-12-2022/Thapsigargin/220110n_0055.tif.frames/220110n_0055.tif")
SOURCE_PATH = Path("/Users/cudmore/Dropbox/data/rabbit-ca-variance/raw-data/jan-12-2022/Thapsigargin/220110n_0056.tif.frames/220110n_0056.tif")

# Built-in presets now live in AcqStore. Edit this value locally while trying data.
PRESET_NAME = SumIntensityPresetName.SLOW


def run_sum_intensity_analysis(acq_image: AcqImage, preset_name: SumIntensityPresetName) -> SumIntensityAnalysis:
    """Create and run sum-intensity analysis on one file.

    Args:
        acq_image: Acquisition image.
        preset_name: Built-in sum-intensity preset name.

    Returns:
        Completed sum-intensity analysis.

    Raises:
        KeyError: If preset name is unknown.
    """
    channel = 0
    roi = acq_image.rois.create_rect_roi(name="sum_intensity_test", note="sum intensity test")
    context = AnalysisRunContext(
        progress_callback=lambda fraction, message: print(f"  progress={fraction}: {message}")
    )
    analysis = acq_image.analysis_set.create_and_run(
        SumIntensityAnalysis,
        channel=channel,
        roi_id=roi.roi_id,
        detection_params=SumIntensityAnalysis.get_detection_preset_params(preset_name),
        replace_existing=True,
        context=context,
    )
    if not isinstance(analysis, SumIntensityAnalysis):
        raise TypeError(f"Expected SumIntensityAnalysis, got {type(analysis).__name__}")
    return analysis


def plot_sum_intensity_results(analysis: BaseAnalysis) -> None:
    """Plot trace, derivative, onsets, and peaks with Plotly.

    Args:
        analysis: Completed sum-intensity analysis.
        output_html: HTML file to write.

    Returns:
        None.
    """
    if analysis.result.table is None:
        raise ValueError("analysis has no result table")

    table = analysis.result.table

    fig = go.Figure()

    norm_trace = analysis.get_trace(SumIntensityTraceKey.NORM_SUM_INTENSITY)
    fig.add_trace(
        go.Scatter(
            x=norm_trace.x,
            y=norm_trace.y,
            mode="lines",
            name=norm_trace.name,
            visible="legendonly",
        )
    )

    df_f_trace = analysis.get_trace(SumIntensityTraceKey.DF_F_SIGNAL)
    fig.add_trace(
        go.Scatter(
            x=df_f_trace.x,
            y=df_f_trace.y,
            mode="lines",
            name=df_f_trace.name,
        )
    )

    derivative_trace = analysis.get_trace(SumIntensityTraceKey.D_DF_F_SIGNAL)
    fig.add_trace(
        go.Scatter(
            x=derivative_trace.x,
            y=derivative_trace.y,
            mode="lines",
            name=derivative_trace.name,
            yaxis="y2",
        )
    )

    onset_points = analysis.get_event_points(SumIntensityEventPointKey.ONSETS)
    peak_points = analysis.get_event_points(SumIntensityEventPointKey.PEAKS)
    fig.add_trace(
        go.Scatter(
            x=onset_points.x,
            y=onset_points.y,
            mode="markers",
            name=onset_points.name,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=peak_points.x,
            y=peak_points.y,
            mode="markers",
            name=peak_points.name,
        )
    )

    for width_trace in analysis.get_width_trace():
        visible = True if width_trace.metadata.get("fraction") == 0.5 else "legendonly"
        fig.add_trace(
            go.Scatter(
                x=width_trace.x,
                y=width_trace.y,
                mode="lines",
                name=width_trace.name,
                visible=visible,
                connectgaps=False,
            )
        )

    f0_baseline = analysis.result.summary.get("f0_baseline")
    baseline_method = analysis.result.summary.get("baseline_method")
    baseline_percentile = analysis.result.summary.get("baseline_percentile")
    annotation = (
        f"F0={float(f0_baseline):.6g} "
        f"({baseline_method}, p={float(baseline_percentile):.1f})"
        if f0_baseline is not None and baseline_percentile is not None
        else "F0 unavailable"
    )
    fig.update_layout(
        title="CloudScope sum intensity analysis",
        xaxis_title="Time (s)",
        yaxis_title="df/f0",
        yaxis2={"title": "1/s", "overlaying": "y", "side": "right"},
        annotations=[
            {
                "text": annotation,
                "xref": "paper",
                "yref": "paper",
                "x": 0.01,
                "y": 0.99,
                "showarrow": False,
                "align": "left",
                "bgcolor": "rgba(255,255,255,0.7)",
            }
        ],
    )

    # fig.write_html(output_html)
    # print(f"wrote {output_html}")
    fig.show()


def main() -> None:
    """Run the manual single-file sum-intensity workflow.

    Returns:
        None.
    """
    print("Sum-intensity detection schema:")
    for field in SumIntensityAnalysis.get_detection_schema():
        print(
            f"  {field.name}: default={field.default!r}, "
            f"type={field.value_type.value}, unit={field.unit}, choices={field.choices}"
        )

    acq_image = AcqImage(SOURCE_PATH)
    analysis = run_sum_intensity_analysis(acq_image, preset_name=PRESET_NAME)
    print("Analysis key:", analysis.key)
    
    print("Summary:")
    from pprint import pprint
    pprint(analysis.result.summary)

    print("Columns:", analysis.get_table_columns())
    print("Detected peaks:", len(analysis.get_peak_events()))
    plot_sum_intensity_results(analysis)


if __name__ == "__main__":
    main()
