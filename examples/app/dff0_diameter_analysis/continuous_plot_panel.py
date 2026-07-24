"""Right-side plot panel for continuous coupling analysis."""

from __future__ import annotations

from typing import Any

from nicegui import ui

from acqstore.common_analysis.dff0_diameter_analysis.continuous_analysis import (
    Dff0DiameterContinuousAnalysis,
)
from acqstore.common_analysis.dff0_diameter_analysis.continuous_plotting import (
    build_continuous_coupling_figure,
    build_shifted_overlay_figure,
)


class ContinuousPlotPanel:
    """Own annotations and Plotly figures for continuous analysis."""

    def __init__(self) -> None:
        """Create the persistent parent container."""
        self._container: Any = ui.column().classes("w-full gap-3 p-3")
        with self._container:
            ui.label("Select an analysis hit and click Plot continuous coupling.").classes(
                "text-subtitle1"
            )

    def show_analysis(self, analysis: Dff0DiameterContinuousAnalysis) -> None:
        """Clear and rebuild continuous-analysis annotations and figures.

        Args:
            analysis: Completed continuous coupling analysis.
        """
        result = analysis.result
        self._container.clear()
        with self._container:
            ui.label(analysis.dataset.source_name).classes("text-h5")
            ui.label(
                "Positive lag means df/f0 leads and diameter follows."
            ).classes("text-caption")
            ui.label(
                "Zero lag: "
                f"{_format_value(result.zero_lag_correlation)} | "
                "Strongest negative: "
                f"r={_format_value(result.strongest_negative_correlation)}, "
                f"lag={_format_seconds(result.strongest_negative_lag_sec)} | "
                "Strongest absolute: "
                f"r={_format_value(result.strongest_absolute_correlation)}, "
                f"lag={_format_seconds(result.strongest_absolute_lag_sec)}"
            ).classes("text-caption")
            ui.plotly(build_continuous_coupling_figure(analysis)).classes("w-full")

            ui.label("Shifted standardized overlay").classes("text-h6")
            ui.plotly(build_shifted_overlay_figure(analysis)).classes("w-full")

    def show_error(self, message: str) -> None:
        """Replace contents with a visible error message.

        Args:
            message: Human-readable error text.
        """
        self._container.clear()
        with self._container:
            ui.label("Unable to plot continuous analysis").classes(
                "text-h6 text-negative"
            )
            ui.label(message)


def _format_value(value: float | None) -> str:
    """Format an optional numeric summary value."""
    return "n/a" if value is None else f"{value:.4f}"


def _format_seconds(value: float | None) -> str:
    """Format an optional lag in seconds."""
    return "n/a" if value is None else f"{value:.4f} s"
