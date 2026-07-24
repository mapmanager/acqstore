"""Tests for the generic triggered-event engine."""

from __future__ import annotations

import numpy as np

from acqstore.common_analysis.dff0_diameter_analysis.models import (
    EventDirection,
    SignalFilterMethod,
    TriggeredEvent,
    TriggeredEventParams,
)
from acqstore.common_analysis.dff0_diameter_analysis.triggered_events import (
    analyze_triggered_events,
)


def test_negative_event_measures_pre_and_post_seed() -> None:
    """A negative response should yield baseline, extremum, and recovery."""
    time = np.arange(30, dtype=float) * 0.1
    signal = np.full(30, 10.0)
    signal[8:11] = [9.8, 9.9, 10.0]
    signal[10:18] = [10.0, 9.5, 9.0, 8.0, 7.0, 8.0, 9.2, 10.0]
    params = TriggeredEventParams(
        direction=EventDirection.NEGATIVE,
        pre_points=5,
        post_points=12,
        post_search_window_points=6,
        baseline_start_offset_points=-5,
        baseline_stop_offset_points=0,
        filter_method=SignalFilterMethod.NONE,
        recovery_fraction=0.9,
    )

    event = analyze_triggered_events(time, signal, [10], params)[0]

    assert event.baseline_value == 10.0
    assert event.pre_seed_value == 9.9
    assert event.extremum_index == 14
    assert event.amplitude == 3.0
    assert event.time_to_extremum_from_seed_points == 4
    assert event.recovery_detected
    assert event.recovery_index == 17


def test_next_seed_truncates_event_and_extremum_search() -> None:
    """A current event must not consume samples from the next seed."""
    time = np.arange(40, dtype=float) * 0.01
    signal = np.full(40, 5.0)
    signal[10:20] = np.linspace(5.0, 3.0, 10)
    signal[20:30] = np.linspace(2.0, 5.0, 10)
    params = TriggeredEventParams(
        pre_points=5,
        post_points=25,
        post_search_window_points=25,
        filter_method=SignalFilterMethod.NONE,
    )

    events = analyze_triggered_events(time, signal, [10, 20], params)

    assert events[0].window_stop_index == 20
    assert events[0].truncated_by_next_seed
    assert events[0].extremum_index < 20


def test_serialization_round_trip() -> None:
    """TriggeredEvent serialization should preserve the full schema."""
    time = np.arange(20, dtype=float)
    signal = np.r_[np.ones(5), np.linspace(1.0, 0.0, 5), np.linspace(0.0, 1.0, 10)]
    event = analyze_triggered_events(
        time,
        signal,
        [5],
        TriggeredEventParams(
            pre_points=5,
            post_points=14,
            post_search_window_points=8,
            filter_method=SignalFilterMethod.NONE,
        ),
    )[0]

    restored = TriggeredEvent.from_dict(event.to_dict())

    assert restored == event


def test_savgol_filter_is_supported() -> None:
    """Savitzky-Golay filtering should run in the initial implementation."""
    time = np.arange(31, dtype=float) * 0.1
    signal = 10.0 - np.exp(-((np.arange(31) - 15) / 4.0) ** 2)
    events = analyze_triggered_events(
        time,
        signal,
        [10],
        TriggeredEventParams(
            pre_points=5,
            post_points=20,
            post_search_window_points=12,
            filter_method=SignalFilterMethod.SAVGOL,
            savgol_window_points=7,
            savgol_polyorder=2,
        ),
    )

    assert len(events) == 1
    assert events[0].extremum_index is not None
