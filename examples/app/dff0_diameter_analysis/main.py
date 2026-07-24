"""Standalone NiceGUI app for triggered and continuous coupling analyses."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from nicegui import ui

from acqstore.acq_image.acq_image_list import AcqImageList
from acqstore.common_analysis.dff0_diameter_analysis.analysis import (
    Dff0DiameterAnalysis,
)
from acqstore.common_analysis.dff0_diameter_analysis.continuous_analysis import (
    Dff0DiameterContinuousAnalysis,
)

from .analysis_hits import find_analysis_hits
from .app_state import AppState
from .continuous_control_panel import ContinuousControlPanel
from .continuous_plot_panel import ContinuousPlotPanel
from .left_control_panel import LeftControlPanel
from .plot_panel import PlotPanel


SOURCE_PATH = Path(
    "/Users/cudmore/Sites/cloudscope-data/data-samples/diameter-sample-data"
)

NATIVE = False
NATIVE_WINDOW_SIZE = (1000, 800)


@dataclass(slots=True)
class TriggeredPageContext:
    """Bind triggered-analysis state to page controls and plots."""

    state: AppState
    controls: LeftControlPanel | None = None
    plots: PlotPanel | None = None


@dataclass(slots=True)
class ContinuousPageContext:
    """Bind continuous-analysis state to page controls and plots."""

    state: AppState
    controls: ContinuousControlPanel | None = None
    plots: ContinuousPlotPanel | None = None


def handle_triggered_hit_row_clicked(
    event: Any,
    *,
    context: TriggeredPageContext,
) -> None:
    """Update triggered-page state after an AG Grid row click.

    Args:
        event: NiceGUI event containing selected row data.
        context: Triggered page-local state and components.
    """
    if context.controls is None:
        return
    hit = _select_hit_from_event(event, context.state)
    if hit is not None:
        context.controls.set_selected_hit(hit)


def handle_continuous_hit_row_clicked(
    event: Any,
    *,
    context: ContinuousPageContext,
) -> None:
    """Update continuous-page state after an AG Grid row click.

    Args:
        event: NiceGUI event containing selected row data.
        context: Continuous page-local state and components.
    """
    if context.controls is None:
        return
    hit = _select_hit_from_event(event, context.state)
    if hit is not None:
        context.controls.set_selected_hit(hit)


def handle_triggered_plot_requested(*, context: TriggeredPageContext) -> None:
    """Run and plot the selected triggered-event analysis.

    Args:
        context: Triggered page-local state and components.
    """
    controls = context.controls
    plots = context.plots
    if controls is None or plots is None:
        return

    hit = context.state.selected_hit
    if hit is None:
        ui.notify("Select an analysis row before plotting.", type="warning")
        return

    try:
        analysis = Dff0DiameterAnalysis.from_acq_image(
            acq_image=hit.acq_image,
            channel=hit.channel,
            roi_id=hit.roi_id,
            triggered_event_params=controls.get_triggered_event_params(),
        )
        if not analysis.triggered_events:
            raise ValueError("The selected analysis contains no reporter seed events.")

        max_event_index = len(analysis.triggered_events) - 1
        event_index = min(max(controls.get_event_index(), 0), max_event_index)
        controls.set_event_index(event_index)

        context.state.analysis = analysis
        plots.show_analysis(
            analysis,
            event_index=event_index,
            metric_name=controls.get_metric_name(),
        )
    except (TypeError, ValueError, KeyError) as exc:
        context.state.analysis = None
        plots.show_error(str(exc))
        ui.notify(str(exc), type="negative")


def handle_continuous_plot_requested(*, context: ContinuousPageContext) -> None:
    """Run and plot the selected continuous coupling analysis.

    Args:
        context: Continuous page-local state and components.
    """
    controls = context.controls
    plots = context.plots
    if controls is None or plots is None:
        return

    hit = context.state.selected_hit
    if hit is None:
        ui.notify("Select an analysis row before plotting.", type="warning")
        return

    try:
        analysis = Dff0DiameterContinuousAnalysis.from_acq_image(
            acq_image=hit.acq_image,
            channel=hit.channel,
            roi_id=hit.roi_id,
            params=controls.get_params(),
        )
        context.state.continuous_analysis = analysis
        plots.show_analysis(analysis)
    except (TypeError, ValueError, KeyError) as exc:
        context.state.continuous_analysis = None
        plots.show_error(str(exc))
        ui.notify(str(exc), type="negative")


def _select_hit_from_event(event: Any, state: AppState):
    """Resolve and select one ``AnalysisHit`` from AG Grid event data."""
    row_data = event.args.get("data") if isinstance(event.args, dict) else None
    if not isinstance(row_data, dict):
        ui.notify("The selected grid row did not contain row data.", type="warning")
        return None

    hit_id = row_data.get("hit_id")
    if not isinstance(hit_id, str):
        ui.notify("The selected grid row did not contain a hit identifier.", type="warning")
        return None

    try:
        return state.select_hit(hit_id)
    except KeyError:
        ui.notify("The selected analysis hit is no longer available.", type="negative")
        return None


def load_app_state(source_path: Path) -> AppState:
    """Load acquisitions and discover all paired analysis hits.

    Args:
        source_path: Folder containing acquisition files and analysis sidecars.

    Returns:
        Initialized page state.
    """
    acq_image_list = AcqImageList(
        str(source_path),
        load_images=False,
        load_analysis_csv=True,
    )
    return AppState(
        acq_image_list=acq_image_list,
        analysis_hits=find_analysis_hits(acq_image_list),
    )


def _build_page_header(*, title: str, state: AppState) -> None:
    """Build a shared page header and simple analysis-page navigation."""
    with ui.row().classes("w-full items-center justify-between px-3 pt-3"):
        ui.label(title).classes("text-h5")
        with ui.row().classes("gap-2"):
            ui.link("Triggered events", "/")
            ui.link("Continuous coupling", "/continuous")
    ui.label(f"Source: {SOURCE_PATH}").classes("text-caption px-3")
    ui.label(f"Paired analysis hits: {len(state.analysis_hits)}").classes(
        "text-caption px-3"
    )


def _load_state_or_show_error() -> AppState | None:
    """Load page state or render a visible folder-loading error."""
    try:
        return load_app_state(SOURCE_PATH)
    except (OSError, RuntimeError, ValueError) as exc:
        ui.label("Unable to load acquisition folder").classes("text-h5 text-negative")
        ui.label(str(exc))
        ui.label(str(SOURCE_PATH)).classes("text-caption")
        return None


@ui.page("/")
def home_page() -> None:
    """Build the standalone triggered-event analysis page."""
    from nicewidgets.gui_defaults import setUpGuiDefaults

    setUpGuiDefaults(text_size="text-xs")
    state = _load_state_or_show_error()
    if state is None:
        return

    context = TriggeredPageContext(state=state)
    _build_page_header(title="DFF0 / Diameter Triggered Event Analysis", state=state)

    with ui.splitter(value=32).classes("w-full h-[calc(100vh-90px)]") as splitter:
        with splitter.before:
            context.controls = LeftControlPanel(
                hits=state.analysis_hits,
                on_row_clicked=partial(
                    handle_triggered_hit_row_clicked,
                    context=context,
                ),
                on_plot_requested=partial(
                    handle_triggered_plot_requested,
                    context=context,
                ),
            )
        with splitter.after:
            context.plots = PlotPanel()


@ui.page("/continuous")
def continuous_page() -> None:
    """Build the standalone continuous lagged-correlation page."""
    from nicewidgets.gui_defaults import setUpGuiDefaults

    setUpGuiDefaults(text_size="text-xs")
    state = _load_state_or_show_error()
    if state is None:
        return

    context = ContinuousPageContext(state=state)
    _build_page_header(title="DFF0 / Diameter Continuous Coupling", state=state)

    with ui.splitter(value=32).classes("w-full h-[calc(100vh-90px)]") as splitter:
        with splitter.before:
            context.controls = ContinuousControlPanel(
                hits=state.analysis_hits,
                on_row_clicked=partial(
                    handle_continuous_hit_row_clicked,
                    context=context,
                ),
                on_plot_requested=partial(
                    handle_continuous_plot_requested,
                    context=context,
                ),
            )
        with splitter.after:
            context.plots = ContinuousPlotPanel()


def run_app(*, native: bool = NATIVE) -> None:
    """Run the NiceGUI app in browser or native desktop mode.

    Args:
        native: When true, launch a native window with the configured size.
    """
    run_kwargs: dict[str, object] = {}
    if native:
        run_kwargs.update(native=True, window_size=NATIVE_WINDOW_SIZE)
    ui.run(**run_kwargs)


if __name__ in {"__main__", "__mp_main__"}:
    run_app()
