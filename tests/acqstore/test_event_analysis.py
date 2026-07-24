"""Tests for Radon-dependent event analysis."""

from __future__ import annotations

from acqstore.acq_image.analysis.event_analysis.event_analysis import (
    EventAnalysis,
    EventStats,
    calculate_window_stats,
)
from acqstore.acq_image.analysis.model import AnalysisPlotData, AnalysisResult, BaseAnalysis


class DummyRadonAnalysis(BaseAnalysis):
    """Minimal parent analysis exposing canonical plot data."""

    analysis_name = "radon_velocity"

    def __init__(self, plot_data: AnalysisPlotData) -> None:
        """Create the dummy parent analysis.

        Args:
            plot_data: Plot data returned by ``get_plot_data``.
        """
        super().__init__(channel=0, roi_id=1)
        self._plot_data = plot_data

    def run(self, data_provider, *, context=None, dependencies=None) -> AnalysisResult:
        """Return the existing empty result."""
        return self.result

    def get_plot_data(self) -> AnalysisPlotData:
        """Return configured plot data."""
        return self._plot_data


def test_calculate_window_stats_returns_null_values_for_empty_window() -> None:
    """Empty windows should produce strict-JSON-compatible null stats."""
    plot_data = AnalysisPlotData(
        x=(0.0, 1.0, 2.0),
        y=(10.0, 20.0, 30.0),
        x_label="Time (s)",
        y_label="Velocity",
    )

    stats = calculate_window_stats(plot_data, 5.0, 6.0, True, True)

    assert stats == EventStats()
    assert stats.to_json_dict() == {
        "n": 0,
        "mean": None,
        "min": None,
        "max": None,
        "std": None,
        "se": None,
        "cv": None,
    }


def test_add_event_uses_parent_plot_data_and_configured_windows() -> None:
    """Adding an event should fill event/pre/post stats from plot data."""
    plot_data = AnalysisPlotData(
        x=(0.0, 1.0, 2.0, 3.0, 4.0),
        y=(10.0, 20.0, 30.0, 40.0, 50.0),
        x_label="Time (s)",
        y_label="Velocity",
    )
    analysis = EventAnalysis(channel=0, roi_id=1, detection_params={"pre_post_win_sec": 1.0})

    event = analysis.add_event(1.0, 3.0, plot_data=plot_data)

    assert event.event_stats.n == 3
    assert event.event_stats.mean == 30.0
    assert event.pre_win_stats.n == 1
    assert event.pre_win_stats.mean == 10.0
    assert event.post_win_stats.n == 1
    assert event.post_win_stats.mean == 50.0


def test_reanalysis_updates_stats_but_preserves_event_coordinates() -> None:
    """Running event analysis after parent changes should preserve x0/x1."""
    old_plot_data = AnalysisPlotData(
        x=(0.0, 1.0, 2.0, 3.0),
        y=(1.0, 2.0, 3.0, 4.0),
        x_label="Time (s)",
        y_label="Velocity",
    )
    new_plot_data = AnalysisPlotData(
        x=(0.0, 1.0, 2.0, 3.0),
        y=(10.0, 20.0, 30.0, 40.0),
        x_label="Time (s)",
        y_label="Velocity",
    )
    analysis = EventAnalysis(channel=0, roi_id=1, detection_params={"pre_post_win_sec": 1.0})
    analysis.add_event(1.0, 2.0, plot_data=old_plot_data)

    result = analysis.run(
        data_provider=object(),
        dependencies={"radon_velocity": DummyRadonAnalysis(new_plot_data)},
    )
    [event] = analysis.get_events()

    assert event.x0 == 1.0
    assert event.x1 == 2.0
    assert event.event_stats.mean == 25.0
    assert result.summary["events"][0]["event_stats"]["mean"] == 25.0


def test_load_json_requires_current_summary_shape() -> None:
    """Old event JSON without stats should fail instead of being migrated."""
    analysis = EventAnalysis(channel=0, roi_id=1)

    try:
        analysis.load_json_dict(
            {
                "analysis_name": "event",
                "channel": 0,
                "roi_id": 1,
                "detection_params": {"pre_post_win_sec": 1.0},
                "summary": {
                    "version": 1,
                    "events": [{"id": 1, "x0": 0.0, "x1": 1.0, "event_type": "user"}],
                },
            }
        )
    except ValueError as exc:
        assert "Unsupported event summary version" in str(exc)
    else:  # pragma: no cover - defensive assertion branch
        raise AssertionError("old event JSON loaded unexpectedly")
