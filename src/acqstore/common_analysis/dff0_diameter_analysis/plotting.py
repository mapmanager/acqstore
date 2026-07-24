"""Plotly figures for triggered reporter/diameter analysis."""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .analysis import Dff0DiameterAnalysis


def build_overview_figure(analysis: Dff0DiameterAnalysis) -> go.Figure:
    """Build linked reporter and diameter traces with event markers."""
    _layout_height = 400  # 700

    dataset = analysis.dataset
    reporter_time = dataset.reporter["time_sec"]
    diameter_time = dataset.diameter["time_s"]
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    figure.add_trace(go.Scatter(x=reporter_time, y=dataset.reporter["df_f_signal"], name="df/f0"), row=1, col=1)
    figure.add_trace(go.Scatter(x=diameter_time, y=dataset.diameter["diameter_um_raw"], name="diameter raw", opacity=0.45), row=2, col=1)
    figure.add_trace(go.Scatter(x=diameter_time, y=analysis.diameter_filtered, name="diameter filtered"), row=2, col=1)
    onset_x = [event.onset_time_sec for event in dataset.events]
    onset_y = [dataset.reporter.iloc[event.onset_index]["df_f_signal"] for event in dataset.events]
    figure.add_trace(go.Scatter(x=onset_x, y=onset_y, mode="markers", name="reporter seeds"), row=1, col=1)
    ext_x = [event.extremum_time_sec for event in analysis.triggered_events if event.extremum_time_sec is not None]
    ext_y = [event.extremum_value for event in analysis.triggered_events if event.extremum_value is not None]
    figure.add_trace(go.Scatter(x=ext_x, y=ext_y, mode="markers", name="diameter extrema"), row=2, col=1)
    figure.update_yaxes(title_text="df/f0", row=1, col=1)
    figure.update_yaxes(title_text="diameter (um)", row=2, col=1)
    figure.update_xaxes(title_text="seconds", row=2, col=1)
    
    _title = '' # title=dataset.source_name
    figure.update_layout(
        height=_layout_height,
        title=_title,
        hovermode="x unified",
        margin=dict(l=45, r=20, t=30, b=45),
    )
    return figure


def build_event_figure(analysis: Dff0DiameterAnalysis, event_index: int) -> go.Figure:
    """Build a diagnostic plot for one selected triggered event."""
    event = analysis.triggered_events[event_index]
    dataset = analysis.dataset
    start, stop = event.window_start_index, event.window_stop_index
    time = dataset.diameter["time_s"].to_numpy(dtype=float)
    relative_time = time[start:stop] - event.seed_time_sec
    raw = dataset.diameter["diameter_um_raw"].to_numpy(dtype=float)[start:stop]
    filtered = analysis.diameter_filtered[start:stop]
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=relative_time, y=raw, name="diameter raw", opacity=0.45))
    figure.add_trace(go.Scatter(x=relative_time, y=filtered, name="diameter filtered"))
    figure.add_vline(x=0.0, line_dash="dash", annotation_text="reporter seed")
    if event.baseline_start_index is not None and event.baseline_stop_index is not None:
        figure.add_vrect(
            x0=time[event.baseline_start_index] - event.seed_time_sec,
            x1=time[event.baseline_stop_index - 1] - event.seed_time_sec,
            opacity=0.12,
            line_width=0,
            annotation_text="baseline",
        )
    if event.extremum_time_sec is not None and event.extremum_value is not None:
        figure.add_trace(go.Scatter(
            x=[event.extremum_time_sec - event.seed_time_sec],
            y=[event.extremum_value], mode="markers", name="extremum",
        ))
    if event.recovery_time_sec is not None and event.recovery_index is not None:
        figure.add_trace(go.Scatter(
            x=[event.recovery_time_sec - event.seed_time_sec],
            y=[analysis.diameter_filtered[event.recovery_index]],
            mode="markers", name="recovery",
        ))
    figure.update_layout(
        title=f"Triggered event {event_index + 1}: {event.status.value}",
        xaxis_title="time from reporter seed (s)", yaxis_title="diameter (um)",
        hovermode="x unified",
    )
    return figure


def build_metric_vs_time_figure(analysis: Dff0DiameterAnalysis, metric: str) -> go.Figure:
    """Plot one serialized event metric against seed time."""
    table = analysis.triggered_events_dataframe()
    if metric not in table.columns:
        raise ValueError(f"Unknown triggered-event metric: {metric}")
    figure = go.Figure(go.Scatter(x=table["seed_time_sec"], y=table[metric], mode="lines+markers"))
    figure.update_layout(xaxis_title="seed time (s)", yaxis_title=metric, title=f"{metric} vs recording time")
    return figure
