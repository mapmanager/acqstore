"""Plotly figures for continuous reporter/diameter coupling analysis."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .continuous_analysis import Dff0DiameterContinuousAnalysis


def build_continuous_coupling_figure(
    analysis: Dff0DiameterContinuousAnalysis,
) -> go.Figure:
    """Build continuous traces and lagged-correlation curve in two rows.

    The first row overlays filtered reporter and diameter traces using separate
    left and right y-axes. The second row shows normalized Pearson correlation
    versus lag.

    Args:
        analysis: Completed continuous coupling analysis.

    Returns:
        Plotly figure with shared horizontal layout.
    """
    dataset = analysis.dataset
    result = analysis.result
    time = dataset.reporter["time_sec"].to_numpy(dtype=float)

    figure = make_subplots(
        rows=2,
        cols=1,
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
        row_heights=[0.62, 0.38],
        vertical_spacing=0.10,
    )
    figure.add_trace(
        go.Scatter(x=time, y=analysis.reporter_filtered, name="df/f0 filtered"),
        row=1,
        col=1,
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(x=time, y=analysis.diameter_filtered, name="diameter filtered"),
        row=1,
        col=1,
        secondary_y=True,
    )

    correlation_y = [np.nan if value is None else value for value in result.correlation]
    figure.add_trace(
        go.Scatter(
            x=result.lag_seconds,
            y=correlation_y,
            mode="lines",
            name="Pearson r",
        ),
        row=2,
        col=1,
    )
    figure.add_vline(x=0.0, line_dash="dash", row=2, col=1)
    _add_lag_marker(
        figure,
        lag_sec=result.strongest_negative_lag_sec,
        correlation=result.strongest_negative_correlation,
        name="strongest negative",
    )
    _add_lag_marker(
        figure,
        lag_sec=result.strongest_positive_lag_sec,
        correlation=result.strongest_positive_correlation,
        name="strongest positive",
    )
    _add_lag_marker(
        figure,
        lag_sec=result.strongest_absolute_lag_sec,
        correlation=result.strongest_absolute_correlation,
        name="strongest absolute",
    )

    figure.update_yaxes(title_text="df/f0", row=1, col=1, secondary_y=False)
    figure.update_yaxes(
        title_text="diameter (um)", row=1, col=1, secondary_y=True
    )
    figure.update_yaxes(title_text="Pearson r", range=[-1.05, 1.05], row=2, col=1)
    figure.update_xaxes(title_text="time (s)", row=1, col=1)
    figure.update_xaxes(
        title_text="lag (s): positive means df/f0 leads diameter",
        row=2,
        col=1,
    )
    figure.update_layout(
        height=720,
        hovermode="x unified",
        margin=dict(l=55, r=55, t=35, b=50),
    )
    return figure


def build_shifted_overlay_figure(
    analysis: Dff0DiameterContinuousAnalysis,
    *,
    lag_points: int | None = None,
) -> go.Figure:
    """Overlay standardized signals after shifting diameter by one lag.

    Args:
        analysis: Completed continuous coupling analysis.
        lag_points: Lag to visualize. Defaults to the strongest absolute lag.

    Returns:
        Plotly figure containing overlapping standardized signals.
    """
    selected_lag = (
        analysis.result.strongest_absolute_lag_points
        if lag_points is None
        else lag_points
    )
    if selected_lag is None:
        raise ValueError("No valid lag is available for shifted overlay")

    time = analysis.dataset.reporter["time_sec"].to_numpy(dtype=float)
    reporter = _standardize(analysis.reporter_filtered)
    diameter = _standardize(analysis.diameter_filtered)
    reporter_segment, diameter_segment, time_segment = _shift_for_display(
        time,
        reporter,
        diameter,
        selected_lag,
    )

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(x=time_segment, y=reporter_segment, name="df/f0 standardized")
    )
    figure.add_trace(
        go.Scatter(
            x=time_segment,
            y=diameter_segment,
            name=f"diameter standardized, lag {selected_lag} points",
        )
    )
    figure.update_layout(
        xaxis_title="time (s)",
        yaxis_title="standardized value",
        hovermode="x unified",
        margin=dict(l=55, r=25, t=35, b=50),
    )
    return figure


def _add_lag_marker(
    figure: go.Figure,
    *,
    lag_sec: float | None,
    correlation: float | None,
    name: str,
) -> None:
    """Add one optional summary marker to the lag plot."""
    if lag_sec is None or correlation is None:
        return
    figure.add_trace(
        go.Scatter(
            x=[lag_sec],
            y=[correlation],
            mode="markers",
            name=name,
        ),
        row=2,
        col=1,
    )


def _standardize(values: np.ndarray) -> np.ndarray:
    """Return finite-value z scores while preserving invalid samples."""
    result = np.asarray(values, dtype=float).copy()
    finite = np.isfinite(result)
    if np.count_nonzero(finite) < 2:
        return result
    std = float(np.std(result[finite]))
    if np.isclose(std, 0.0):
        result[finite] = 0.0
        return result
    result[finite] = (result[finite] - float(np.mean(result[finite]))) / std
    return result


def _shift_for_display(
    time: np.ndarray,
    reporter: np.ndarray,
    diameter: np.ndarray,
    lag_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return overlapping standardized traces using the lag sign convention."""
    if lag_points > 0:
        return reporter[:-lag_points], diameter[lag_points:], time[:-lag_points]
    if lag_points < 0:
        offset = -lag_points
        return reporter[offset:], diameter[:-offset], time[offset:]
    return reporter, diameter, time
