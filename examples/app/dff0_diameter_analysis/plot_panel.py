"""Reusable right-side plot panel for the standalone analysis app."""

from __future__ import annotations

from typing import Any

from nicegui import ui

from acqstore.common_analysis.dff0_diameter_analysis.analysis import (
    Dff0DiameterAnalysis,
)
from acqstore.common_analysis.dff0_diameter_analysis.plotting import (
    build_event_figure,
    build_metric_vs_time_figure,
    build_overview_figure,
)


class PlotPanel:
    """Own labels and Plotly figures displayed to the right of the splitter."""

    def __init__(self) -> None:
        self._container: Any = None
        self._build()

    def _build(self) -> None:
        """Create the persistent parent container for plot rebuilding."""
        self._container = ui.column().classes("w-full gap-3 p-3")
        with self._container:
            ui.label("Select an analysis hit and click Plot.").classes("text-subtitle1")

    def show_analysis(
        self,
        analysis: Dff0DiameterAnalysis,
        *,
        event_index: int,
        metric_name: str,
    ) -> None:
        """Clear and rebuild all analysis annotations and figures.

        Args:
            analysis: Completed paired reporter/diameter analysis.
            event_index: Zero-based event to show in the diagnostic figure.
            metric_name: Serialized event field plotted against recording time.
        """
        self._container.clear()
        with self._container:
            ui.label(analysis.dataset.source_name).classes("text-h5")
            # ui.label(str(analysis.get_alignment_summary())).classes("text-caption")

            # ui.label("Overview").classes("text-h6")
            ui.plotly(build_overview_figure(analysis)).classes("w-full")

            ui.label(f"Triggered event {event_index + 1}").classes("text-h6")
            ui.plotly(build_event_figure(analysis, event_index)).classes("w-full")

            ui.label("Metric versus recording time").classes("text-h6")
            ui.plotly(
                build_metric_vs_time_figure(analysis, metric_name)
            ).classes("w-full")

    def show_error(self, message: str) -> None:
        """Replace plot contents with a visible error message.

        Args:
            message: Human-readable error text.
        """
        self._container.clear()
        with self._container:
            ui.label("Unable to plot analysis").classes("text-h6 text-negative")
            ui.label(message)
