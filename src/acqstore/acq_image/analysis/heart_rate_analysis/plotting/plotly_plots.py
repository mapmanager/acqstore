"""Plotly diagnostic plots for heart-rate analysis.

These helpers return ``plotly.graph_objects.Figure`` objects for scripts and
notebooks. They stay in the ``acqstore`` backend layer and do not import
CloudScope GUI, NiceGUI, or reusable widget modules.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from acqstore.acq_image.analysis.heart_rate_analysis.plotting.plot_data import (
    compute_lomb_spectrum,
    compute_preprocessing,
    compute_segment_series,
    compute_welch_spectrum,
)


def plot_velocity_overview_plotly(
    time_s: Sequence[float],
    velocity: Sequence[float],
    *,
    params: dict[str, Any] | None = None,
    title: str = "",
) -> go.Figure:
    """Return a Plotly overview of raw and preprocessed velocity traces.

    Args:
        time_s: Time samples in seconds.
        velocity: Velocity samples aligned to ``time_s``.
        params: Optional heart-rate detection parameters.
        title: Optional plot title.

    Returns:
        Plotly figure object.
    """
    fig = go.Figure()
    data = compute_preprocessing(time_s, velocity, params=params)
    _add_overview_traces(fig, data)
    fig.update_layout(
        title=title or "Velocity preprocessing for HR",
        xaxis_title="time (s)",
        yaxis_title="velocity / a.u.",
        legend={"orientation": "h"},
        margin={"l": 60, "r": 20, "t": 50, "b": 50},
    )
    fig.add_hline(y=0.0, line_width=1)
    return fig


def plot_welch_psd_plotly(
    time_s: Sequence[float],
    velocity: Sequence[float],
    *,
    params: dict[str, Any] | None = None,
    title: str = "",
) -> go.Figure:
    """Return a Plotly Welch PSD plot with peak and QC annotation.

    Args:
        time_s: Time samples in seconds.
        velocity: Velocity samples aligned to ``time_s``.
        params: Optional heart-rate detection parameters.
        title: Optional plot title.

    Returns:
        Plotly figure object.

    Raises:
        ValueError: If there are not enough samples for Welch diagnostics.
    """
    fig = go.Figure()
    data = compute_welch_spectrum(time_s, velocity, params=params)
    _add_spectrum_trace(fig, data, "Welch PSD")
    fig.update_layout(
        title=title or "Heart-rate PSD (Welch)",
        xaxis_title="frequency (Hz)",
        yaxis_title="PSD",
        margin={"l": 70, "r": 20, "t": 60, "b": 50},
    )
    fig.update_xaxes(range=[0.0, float(min(np.max(data.frequency_hz), data.params.band_hz[1] * 1.6))])
    _add_peak_marker(fig, data)
    _add_qc_annotation(fig, data.estimate)
    return fig


def plot_lomb_periodogram_plotly(
    time_s: Sequence[float],
    velocity: Sequence[float],
    *,
    params: dict[str, Any] | None = None,
    title: str = "",
) -> go.Figure:
    """Return a Plotly Lomb-Scargle periodogram with peak and QC annotation.

    Args:
        time_s: Time samples in seconds.
        velocity: Velocity samples aligned to ``time_s``.
        params: Optional heart-rate detection parameters.
        title: Optional plot title.

    Returns:
        Plotly figure object.

    Raises:
        ValueError: If there are not enough samples for Lomb-Scargle diagnostics.
    """
    fig = go.Figure()
    data = compute_lomb_spectrum(time_s, velocity, params=params)
    _add_spectrum_trace(fig, data, "Lomb-Scargle")
    fig.update_layout(
        title=title or "Heart-rate periodogram (Lomb-Scargle)",
        xaxis_title="frequency (Hz)",
        yaxis_title="normalized power",
        margin={"l": 70, "r": 20, "t": 60, "b": 50},
    )
    fig.update_xaxes(range=[float(data.params.band_hz[0] * 0.8), float(data.params.band_hz[1] * 1.2)])
    _add_peak_marker(fig, data)
    _add_qc_annotation(fig, data.estimate)
    return fig


def plot_segment_series_plotly(
    time_s: Sequence[float],
    velocity: Sequence[float],
    *,
    params: dict[str, Any] | None = None,
    title: str = "",
) -> go.Figure:
    """Return a Plotly windowed segment heart-rate series.

    Args:
        time_s: Time samples in seconds.
        velocity: Velocity samples aligned to ``time_s``.
        params: Optional heart-rate detection parameters.
        title: Optional plot title.

    Returns:
        Plotly figure object.
    """
    seg = compute_segment_series(time_s, velocity, params=params)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=seg["t_center"],
            y=seg["bpm"],
            mode="lines+markers",
            name="segment HR (bpm)",
        )
    )
    fig.update_layout(
        title=title or "Segment HR series",
        xaxis_title="time (s)",
        yaxis_title="HR (bpm)",
        margin={"l": 60, "r": 20, "t": 50, "b": 50},
    )
    return fig


def plot_summary_plotly(
    time_s: Sequence[float],
    velocity: Sequence[float],
    *,
    params: dict[str, Any] | None = None,
    title: str = "",
) -> go.Figure:
    """Return a three-panel Plotly heart-rate diagnostic summary.

    Args:
        time_s: Time samples in seconds.
        velocity: Velocity samples aligned to ``time_s``.
        params: Optional heart-rate detection parameters.
        title: Optional figure title.

    Returns:
        Plotly figure object containing overview, Welch PSD, and Lomb-Scargle
        periodogram panels.

    Raises:
        ValueError: If spectral diagnostic plots cannot be computed.
    """
    overview = compute_preprocessing(time_s, velocity, params=params)
    welch = compute_welch_spectrum(time_s, velocity, params=params)
    lomb = compute_lomb_spectrum(time_s, velocity, params=params)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=False,
        subplot_titles=(
            "Velocity preprocessing",
            "Welch PSD",
            "Lomb-Scargle periodogram",
        ),
        vertical_spacing=0.08,
    )
    _add_overview_traces(fig, overview, row=1, col=1)
    _add_spectrum_trace(fig, welch, "Welch PSD", row=2, col=1)
    _add_spectrum_trace(fig, lomb, "Lomb-Scargle", row=3, col=1)

    fig.add_hline(y=0.0, line_width=1, row=1, col=1)
    _add_peak_marker(fig, welch, row=2, col=1)
    _add_peak_marker(fig, lomb, row=3, col=1)
    _add_qc_annotation(fig, welch.estimate, x=0.98, y=0.62)
    _add_qc_annotation(fig, lomb.estimate, x=0.98, y=0.27)

    fig.update_xaxes(title_text="time (s)", row=1, col=1)
    fig.update_yaxes(title_text="velocity / a.u.", row=1, col=1)
    # Share a common frequency x-axis across the Welch and Lomb panels so their
    # peaks line up vertically for direct comparison. Matching the axes also
    # links interactive zoom/pan between the two spectra.
    shared_freq_range = [
        float(welch.params.band_hz[0] * 0.8),
        float(welch.params.band_hz[1] * 1.2),
    ]
    fig.update_xaxes(title_text="frequency (Hz)", range=shared_freq_range, row=2, col=1)
    fig.update_yaxes(title_text="PSD", row=2, col=1)
    fig.update_xaxes(
        title_text="frequency (Hz)",
        range=shared_freq_range,
        matches="x2",
        row=3,
        col=1,
    )
    fig.update_yaxes(title_text="normalized power", row=3, col=1)
    fig.update_layout(
        title=title or "Heart-rate diagnostics",
        height=900,
        legend={"orientation": "h"},
        margin={"l": 70, "r": 30, "t": 80, "b": 60},
    )
    return fig


def _add_overview_traces(fig: go.Figure, data: Any, *, row: int | None = None, col: int | None = None) -> None:
    """Add preprocessing overview traces to a Plotly figure."""
    traces = (
        go.Scatter(x=data.time_s, y=data.velocity, mode="lines", name="velocity (raw)", line={"width": 1.0}),
        go.Scatter(x=data.time_s, y=data.x_pre, mode="lines", name="preprocessed", line={"width": 1.2}),
        go.Scatter(x=data.time_s, y=data.x_interp, mode="lines", name="interp small gaps", line={"width": 1.2}),
        go.Scatter(
            x=data.time_s,
            y=data.x_bandpassed,
            mode="lines",
            name=f"bandpassed {data.band_hz[0]:.1f}-{data.band_hz[1]:.1f} Hz",
            line={"width": 1.6},
        ),
    )
    for trace in traces:
        fig.add_trace(trace, row=row, col=col)


def _add_spectrum_trace(
    fig: go.Figure,
    data: Any,
    name: str,
    *,
    row: int | None = None,
    col: int | None = None,
) -> None:
    """Add one spectrum line trace to a Plotly figure."""
    fig.add_trace(
        go.Scatter(
            x=data.frequency_hz,
            y=data.power,
            mode="lines",
            name=name,
            line={"width": 1.6},
        ),
        row=row,
        col=col,
    )


def _add_peak_marker(fig: go.Figure, data: Any, *, row: int | None = None, col: int | None = None) -> None:
    """Add a vertical peak marker to a Plotly figure."""
    fig.add_vline(
        x=float(data.f_peak_hz),
        line_width=2,
        annotation_text=f"peak {data.f_peak_hz:.2f} Hz ({60 * data.f_peak_hz:.0f} bpm), snr={data.snr:.1f}",
        annotation_position="top right",
        row=row,
        col=col,
    )


def _add_qc_annotation(
    fig: go.Figure,
    estimate: Any,
    *,
    x: float = 0.98,
    y: float = 0.98,
) -> None:
    """Add compact QC annotation to a Plotly figure when an estimate exists."""
    if estimate is None:
        return
    band_concentration = estimate.band_concentration
    bc_value = float("nan") if band_concentration is None else float(band_concentration)
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=x,
        y=y,
        text=(
            f"bpm={estimate.bpm:.1f}<br>"
            f"snr={estimate.snr:.2f}<br>"
            f"edge={'YES' if estimate.edge_flag else 'no'}<br>"
            f"bc={bc_value:.3f}"
        ),
        showarrow=False,
        xanchor="right",
        yanchor="top",
        align="right",
        bgcolor="rgba(255,255,255,0.75)",
        bordercolor="rgba(0,0,0,0.2)",
        borderwidth=1,
        font={"size": 11},
    )
