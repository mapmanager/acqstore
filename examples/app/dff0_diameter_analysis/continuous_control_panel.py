"""NiceGUI controls for continuous lagged-correlation analysis."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nicegui import ui

from acqstore.common_analysis.dff0_diameter_analysis.models import (
    LaggedCorrelationParams,
    SignalFilterMethod,
)

from .analysis_hits import AnalysisHit


class ContinuousControlPanel:
    """Build and own continuous-analysis selection and parameter controls."""

    def __init__(
        self,
        *,
        hits: list[AnalysisHit],
        on_row_clicked: Callable[[Any], None],
        on_plot_requested: Callable[[], None],
    ) -> None:
        """Initialize the left control panel.

        Args:
            hits: Available paired-analysis hits.
            on_row_clicked: Callback for AG Grid row-click events.
            on_plot_requested: Callback invoked by the Plot button.
        """
        self._hits = hits
        self._on_row_clicked = on_row_clicked
        self._on_plot_requested = on_plot_requested

        self.hit_grid: Any = None
        self.selected_file_label: Any = None
        self.selected_channel_label: Any = None
        self.selected_roi_label: Any = None
        self.max_lag_points_input: Any = None
        self.minimum_overlap_points_input: Any = None
        self.reporter_filter_select: Any = None
        self.reporter_median_kernel_input: Any = None
        self.reporter_savgol_window_input: Any = None
        self.reporter_savgol_order_input: Any = None
        self.diameter_filter_select: Any = None
        self.diameter_median_kernel_input: Any = None
        self.diameter_savgol_window_input: Any = None
        self.diameter_savgol_order_input: Any = None
        self.remove_linear_trend_checkbox: Any = None
        self._build()

    def _build(self) -> None:
        """Create all controls in the current NiceGUI parent context."""
        with ui.column().classes("w-full h-full gap-3 p-3"):
            ui.button("Plot continuous coupling", on_click=self._on_plot_requested).classes(
                "w-full"
            )

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
                            {"headerName": "Ch", "field": "channel", "width": 70},
                            {"headerName": "ROI", "field": "roi_id", "width": 70},
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

            with ui.card().classes("w-full").props("flat bordered").style(
                "padding: 6px 8px;"
            ):
                with ui.column().classes("gap-0"):
                    ui.label("Selected hit").classes("text-subtitle2")
                    self.selected_file_label = ui.label("File: none")
                    self.selected_channel_label = ui.label("Channel: —")
                    self.selected_roi_label = ui.label("ROI: —")

            ui.separator()
            ui.label("Lagged-correlation parameters").classes("text-h6")

            with ui.grid(columns=2).classes("w-full gap-1"):
                self.max_lag_points_input = ui.number(
                    "Maximum lag points", value=250, min=0, precision=0
                ).classes("w-full")
                self.minimum_overlap_points_input = ui.number(
                    "Minimum overlap points", value=100, min=2, precision=0
                ).classes("w-full")

                self.reporter_filter_select = ui.select(
                    [item.value for item in SignalFilterMethod],
                    value=SignalFilterMethod.MEDIAN.value,
                    label="Reporter filter",
                ).classes("w-full")
                self.reporter_median_kernel_input = ui.number(
                    "Reporter median kernel", value=3, min=1, precision=0
                ).classes("w-full")
                self.reporter_savgol_window_input = ui.number(
                    "Reporter Sav-Gol window", value=15, min=3, precision=0
                ).classes("w-full")
                self.reporter_savgol_order_input = ui.number(
                    "Reporter Sav-Gol order", value=4, min=0, precision=0
                ).classes("w-full")

                self.diameter_filter_select = ui.select(
                    [item.value for item in SignalFilterMethod],
                    value=SignalFilterMethod.SAVGOL.value,
                    label="Diameter filter",
                ).classes("w-full")
                self.diameter_median_kernel_input = ui.number(
                    "Diameter median kernel", value=3, min=1, precision=0
                ).classes("w-full")
                self.diameter_savgol_window_input = ui.number(
                    "Diameter Sav-Gol window", value=15, min=3, precision=0
                ).classes("w-full")
                self.diameter_savgol_order_input = ui.number(
                    "Diameter Sav-Gol order", value=4, min=0, precision=0
                ).classes("w-full")

            self.remove_linear_trend_checkbox = ui.checkbox(
                "Remove linear trend from both signals", value=False
            )

    def set_selected_hit(self, hit: AnalysisHit) -> None:
        """Update labels from one selected analysis hit.

        Args:
            hit: Selected file/channel/ROI hit.
        """
        self.selected_file_label.set_text(f"File: {hit.file_name}")
        self.selected_channel_label.set_text(f"Channel: {hit.channel}")
        self.selected_roi_label.set_text(f"ROI: {hit.roi_id}")

    def get_params(self) -> LaggedCorrelationParams:
        """Create immutable continuous-analysis parameters from controls."""
        return LaggedCorrelationParams(
            max_lag_points=int(self.max_lag_points_input.value),
            minimum_overlap_points=int(self.minimum_overlap_points_input.value),
            reporter_filter_method=SignalFilterMethod(
                str(self.reporter_filter_select.value)
            ),
            reporter_median_kernel_points=int(
                self.reporter_median_kernel_input.value
            ),
            reporter_savgol_window_points=int(
                self.reporter_savgol_window_input.value
            ),
            reporter_savgol_polyorder=int(self.reporter_savgol_order_input.value),
            diameter_filter_method=SignalFilterMethod(
                str(self.diameter_filter_select.value)
            ),
            diameter_median_kernel_points=int(
                self.diameter_median_kernel_input.value
            ),
            diameter_savgol_window_points=int(
                self.diameter_savgol_window_input.value
            ),
            diameter_savgol_polyorder=int(self.diameter_savgol_order_input.value),
            remove_linear_trend=bool(self.remove_linear_trend_checkbox.value),
        )
