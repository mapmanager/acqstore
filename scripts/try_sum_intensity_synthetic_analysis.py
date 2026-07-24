"""Exercise sum-intensity analysis on synthetic data.

This script intentionally runs the NumPy-only core algorithm, not the
``AcqImage`` wrapper. It is a development tool for validating the backend API
that CloudScope views will later consume: traces, event points, width traces,
summary values, and F0 baseline information.

The Plotly figure uses a two-pane layout:

* left pane: the synthetic 2D ``(time, space)`` image analyzed by the core
  algorithm.
* right pane: stacked analysis plots. The upper plot shows df/f0, derivative,
  event markers, and width overlays. The lower plot shows the filtered and
  detrended normalized intensity traces plus the scalar F0 baseline used to
  calculate df/f0.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from acqstore.acq_image.analysis.sum_intensity_analysis.sum_intensity_core import (
    PeakWidthLevel,
    SumIntensityEventPointKey,
    SumIntensitySummaryKey,
    SumIntensityTraceKey,
    run_sum_intensity,
)
from acqstore.acq_image.analysis.sum_intensity_analysis.synthetic.synthetic_config import (
    SyntheticSumIntensityConfig,
)
from acqstore.acq_image.analysis.sum_intensity_analysis.synthetic.synthetic_generator import (
    make_synthetic_sum_intensity_image,
)


def main() -> None:
    """Run synthetic sum-intensity analysis and display a Plotly figure.

    Returns:
        None.
    """
    synthetic = make_synthetic_sum_intensity_image(
        SyntheticSumIntensityConfig(
            event_times_sec=(1.0, 2.4, 3.8, 5.6, 7.5),
            event_jitter_sec=0.015,
            event_amplitude=350.0,
            event_amplitude_jitter_fraction=0.10,
            noise_sigma=3.0,
            spatial_noise_sigma=2.0,
            pop_probability=0.001,
            pop_amplitude=160.0,
            seed=7,
        )
    )
    params = {
        "window_radius_points": 0,
        "filter_method": "median",
        "median_filter_kernel_points": 3,
        "detrend_method": "single_exponential",
        "baseline_method": "percentile",
        "baseline_percentile": 20.0,
        "manual_f0_baseline": 1.0,
        "baseline_min_value": 1e-12,
        "detection_method": "derivative_threshold",
        "polarity": "positive",
        "detection_source": SumIntensityTraceKey.DF_F_SIGNAL.value,
        "absolute_threshold": 0.1,
        "derivative_threshold_per_sec": 3.0,
        "refractory_period_ms": 500.0,
        "peak_search_window_ms": 300.0,
        "width_search_window_ms": 900.0,
        "level_fractions": "0.1,0.2,0.5,0.8,0.9",
    }
    result = run_sum_intensity(
        synthetic.image,
        detection_params=params,
        physical_units=(synthetic.seconds_per_line, synthetic.um_per_pixel),
    )

    df_f = result.get_trace(SumIntensityTraceKey.DF_F_SIGNAL)
    derivative = result.get_trace(SumIntensityTraceKey.D_DF_F_SIGNAL)
    filtered = result.get_trace(SumIntensityTraceKey.FILTERED_NORM_SUM_INTENSITY)
    detrended = result.get_trace(SumIntensityTraceKey.DETRENDED_NORM_SUM_INTENSITY)
    onsets = result.get_event_points(SumIntensityEventPointKey.ONSETS)
    peaks = result.get_event_points(SumIntensityEventPointKey.PEAKS)
    width_50 = result.get_width_trace(PeakWidthLevel.WIDTH_50)
    all_widths = result.get_width_trace()
    f0 = float(result.get_summary_value(SumIntensitySummaryKey.F0_BASELINE))

    print("Synthetic events:")
    print(synthetic.ground_truth_events)
    print("\nAnalysis summary:")
    for key in (
        SumIntensitySummaryKey.NUM_PEAKS,
        SumIntensitySummaryKey.F0_BASELINE,
        SumIntensitySummaryKey.BASELINE_METHOD,
        SumIntensitySummaryKey.BASELINE_PERCENTILE,
        SumIntensitySummaryKey.MANUAL_F0_BASELINE,
        SumIntensitySummaryKey.DETECTION_SOURCE,
        SumIntensitySummaryKey.WARNINGS,
        SumIntensitySummaryKey.ERRORS,
    ):
        print(f"  {key.value}: {result.get_summary_value(key)}")

    space_um = np.arange(synthetic.image.shape[1], dtype=float) * synthetic.um_per_pixel

    fig = make_subplots(
        rows=2,
        cols=2,
        column_widths=[0.42, 0.58],
        row_heights=[0.68, 0.32],
        specs=[[{"type": "heatmap", "rowspan": 2}, {"type": "xy"}], [None, {"type": "xy"}]],
        subplot_titles=(
            "Synthetic image (time × space)",
            "df/f0, derivative, peaks, and width overlays",
            "F0 baseline on normalized intensity trace",
        ),
        horizontal_spacing=0.08,
        vertical_spacing=0.10,
    )

    fig.add_trace(
        go.Heatmap(
            x=space_um,
            y=synthetic.time_sec,
            z=synthetic.image,
            colorscale="Viridis",
            colorbar={"title": "Intensity"},
            name="Synthetic image",
            showscale=True,
        ),
        row=1,
        col=1,
    )

    fig.add_scatter(
        x=synthetic.time_sec,
        y=synthetic.ideal_df_f_trace,
        mode="lines",
        name="Ground truth ideal df/f0",
        row=1,
        col=2,
    )
    fig.add_scatter(x=df_f.x, y=df_f.y, mode="lines", name=df_f.name, row=1, col=2)
    fig.add_scatter(
        x=derivative.x,
        y=derivative.y,
        mode="lines",
        name=derivative.name,
        yaxis="y3",
        row=1,
        col=2,
    )
    fig.add_scatter(x=onsets.x, y=onsets.y, mode="markers", name=onsets.name, row=1, col=2)
    fig.add_scatter(x=peaks.x, y=peaks.y, mode="markers", name=peaks.name, row=1, col=2)
    fig.add_scatter(
        x=width_50.x,
        y=width_50.y,
        mode="lines",
        name=f"{width_50.name} segments",
        connectgaps=False,
        row=1,
        col=2,
    )
    for width_trace in all_widths:
        if width_trace.metadata.get("fraction") == 0.5:
            continue
        fig.add_scatter(
            x=width_trace.x,
            y=width_trace.y,
            mode="lines",
            name=f"{width_trace.name} segments",
            connectgaps=False,
            visible="legendonly",
            row=1,
            col=2,
        )

    fig.add_scatter(
        x=filtered.x,
        y=filtered.y,
        mode="lines",
        name=filtered.name,
        row=2,
        col=2,
    )
    fig.add_scatter(
        x=detrended.x,
        y=detrended.y,
        mode="lines",
        name=detrended.name,
        row=2,
        col=2,
    )
    fig.add_scatter(
        x=[float(detrended.x[0]), float(detrended.x[-1])],
        y=[f0, f0],
        mode="lines",
        name=f"F0 baseline = {f0:.3g}",
        row=2,
        col=2,
    )

    fig.update_xaxes(title_text="Space (µm)", row=1, col=1)
    fig.update_yaxes(title_text="Time (s)", autorange="reversed", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=1, col=2)
    fig.update_yaxes(title_text="df/f0", row=1, col=2)
    fig.update_xaxes(title_text="Time (s)", row=2, col=2)
    fig.update_yaxes(title_text="Mean line intensity", row=2, col=2)
    fig.update_layout(
        title="Synthetic sum-intensity analysis",
        yaxis3={
            "title": "d(df/f0)/dt (1/s)",
            "overlaying": "y2",
            "side": "right",
            "showgrid": False,
        },
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.23},
        margin={"l": 70, "r": 80, "t": 85, "b": 120},
    )
    fig.show()


if __name__ == "__main__":
    main()
