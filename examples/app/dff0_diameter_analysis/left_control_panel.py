"""Reusable NiceGUI controls for selecting and configuring an analysis."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nicegui import ui

from acqstore.common_analysis.dff0_diameter_analysis.models import (
    EventDirection,
    SignalFilterMethod,
    TriggeredEventParams,
)

from .analysis_hits import AnalysisHit


METRIC_OPTIONS: tuple[str, ...] = (
    "baseline_value",
    "baseline_slope_per_sec",
    "amplitude",
    "time_to_extremum_from_seed_sec",
    "extremum_to_recovery_sec",
    "baseline_adjusted_auc_seed_to_stop",
)


class LeftControlPanel:
    """Build and own the left-side selection and parameter controls.

    Args:
        hits: Available paired-analysis hits.
        on_row_clicked: Callback for AG Grid row-click events.
        on_plot_requested: Callback invoked by the Plot button.
    """

    def __init__(
        self,
        *,
        hits: list[AnalysisHit],
        on_row_clicked: Callable[[Any], None],
        on_plot_requested: Callable[[], None],
    ) -> None:
        self._hits = hits
        self._on_row_clicked = on_row_clicked
        self._on_plot_requested = on_plot_requested

        self.selected_file_label: Any = None
        self.selected_channel_label: Any = None
        self.selected_roi_label: Any = None
        self.hit_grid: Any = None
        self.event_index_input: Any = None
        self.direction_select: Any = None
        self.filter_select: Any = None
        self.median_kernel_input: Any = None
        self.savgol_window_input: Any = None
        self.savgol_order_input: Any = None
        self.pre_points_input: Any = None
        self.post_points_input: Any = None
        self.post_search_points_input: Any = None
        self.baseline_start_input: Any = None
        self.baseline_stop_input: Any = None
        self.recovery_fraction_input: Any = None
        self.metric_select: Any = None

        self._build()

    def _build(self) -> None:
        """Create all NiceGUI elements in the current parent context."""
        with ui.column().classes("w-full h-full gap-3 p-3"):
            ui.button("Plot", on_click=self._on_plot_requested).classes("w-full")

            # ui.label("Analysis selections").classes("text-h6")
            with ui.column().classes("w-full shrink-0").style("height: 16rem;"):
                self.hit_grid = ui.aggrid(
                    {
                        "columnDefs": [
                            {
                                "headerName": "File",
                                "field": "file_name",
                                "flex": 1,
                                "minWidth": 180,
                            },
                            {
                                "headerName": "Ch",
                                "field": "channel",
                                "width": 70,
                            },
                            {
                                "headerName": "ROI",
                                "field": "roi_id",
                                "width": 70,
                            },
                            {
                                "headerName": "Peaks",
                                "field": "peak_count",
                                "width": 85,
                            },
                        ],
                        "rowData": [hit.to_grid_row() for hit in self._hits],
                        "rowSelection": "single",
                        "defaultColDef": {
                            "cellStyle": {"fontSize": "9pt"},
                            "headerStyle": {"fontSize": "9pt"},
                        },
                        "rowHeight": 22,
                        "headerHeight": 24,
                        "suppressCellFocus": True,
                        "suppressRowHoverHighlight": True,
                        ":getRowId": "(params) => String(params.data.hit_id)",
                    },
                    auto_size_columns=False,
                ).classes("w-full h-full min-w-0 min-h-0").style("height: 100%;")
            self.hit_grid.on("rowClicked", self._on_row_clicked, ["data"])

            with ui.card().classes("w-full").props("flat bordered").style("padding: 6px 8px;"):
                with ui.column().classes("gap-0"):
                    ui.label("Selected hit").classes("text-subtitle2")
                    self.selected_file_label = ui.label("File: none")
                    self.selected_channel_label = ui.label("Channel: —")
                    self.selected_roi_label = ui.label("ROI: —")

            ui.separator()
            ui.label("Triggered-event parameters").classes("text-h6")

            with ui.grid(columns=2).classes("w-full gap-1"):
                self.event_index_input = ui.number(
                    "Event index",
                    value=0,
                    min=0,
                    precision=0,
                ).classes("w-full")
                self.direction_select = ui.select(
                    [item.value for item in EventDirection],
                    value=EventDirection.NEGATIVE.value,
                    label="Direction",
                ).classes("w-full")
                self.filter_select = ui.select(
                    [item.value for item in SignalFilterMethod],
                    value=SignalFilterMethod.MEDIAN.value,
                    label="Filter",
                ).classes("w-full")
                self.median_kernel_input = ui.number(
                    "Median kernel points",
                    value=3,
                    min=1,
                    precision=0,
                ).classes("w-full")
                self.savgol_window_input = ui.number(
                    "Sav-Gol window points",
                    value=11,
                    min=3,
                    precision=0,
                ).classes("w-full")
                self.savgol_order_input = ui.number(
                    "Sav-Gol order",
                    value=3,
                    min=0,
                    precision=0,
                ).classes("w-full")
                self.pre_points_input = ui.number(
                    "Pre points",
                    value=50,
                    min=0,
                    precision=0,
                ).classes("w-full")
                self.post_points_input = ui.number(
                    "Post points",
                    value=500,
                    min=1,
                    precision=0,
                ).classes("w-full")
                self.post_search_points_input = ui.number(
                    "Extremum search points",
                    value=250,
                    min=1,
                    precision=0,
                ).classes("w-full")
                self.baseline_start_input = ui.number(
                    "Baseline start offset",
                    value=-50,
                    precision=0,
                ).classes("w-full")
                self.baseline_stop_input = ui.number(
                    "Baseline stop offset",
                    value=0,
                    precision=0,
                ).classes("w-full")
                self.recovery_fraction_input = ui.number(
                    "Recovery fraction",
                    value=0.9,
                    min=0.01,
                    max=1.0,
                    step=0.05,
                ).classes("w-full")

            self.metric_select = ui.select(
                list(METRIC_OPTIONS),
                value="time_to_extremum_from_seed_sec",
                label="Metric versus recording time",
            ).classes("w-full")

    def set_selected_hit(self, hit: AnalysisHit) -> None:
        """Populate controls and labels from a selected analysis hit.

        Args:
            hit: Selected file/channel/ROI hit.
        """
        self.selected_file_label.set_text(f"File: {hit.file_name}")
        self.selected_channel_label.set_text(f"Channel: {hit.channel}")
        self.selected_roi_label.set_text(f"ROI: {hit.roi_id}")
        self.event_index_input.set_value(0)
        self.event_index_input.props(f"max={max(0, hit.peak_count - 1)}")

    def get_triggered_event_params(self) -> TriggeredEventParams:
        """Create validated analysis parameters from current control values.

        Returns:
            Generic triggered-event parameters.
        """
        return TriggeredEventParams(
            direction=EventDirection(str(self.direction_select.value)),
            pre_points=int(self.pre_points_input.value),
            post_points=int(self.post_points_input.value),
            post_search_window_points=int(self.post_search_points_input.value),
            baseline_start_offset_points=int(self.baseline_start_input.value),
            baseline_stop_offset_points=int(self.baseline_stop_input.value),
            filter_method=SignalFilterMethod(str(self.filter_select.value)),
            median_kernel_points=int(self.median_kernel_input.value),
            savgol_window_points=int(self.savgol_window_input.value),
            savgol_polyorder=int(self.savgol_order_input.value),
            recovery_fraction=float(self.recovery_fraction_input.value),
        )

    def get_event_index(self) -> int:
        """Return the currently requested zero-based event index."""
        return int(self.event_index_input.value)

    def set_event_index(self, event_index: int) -> None:
        """Set the zero-based event index control.

        Args:
            event_index: Event index clamped by the caller.
        """
        self.event_index_input.set_value(event_index)

    def get_metric_name(self) -> str:
        """Return the selected event metric field name."""
        return str(self.metric_select.value)
