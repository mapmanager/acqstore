"""Event analysis for AcqImage channel/ROI selections.

Event analysis stores user- or algorithm-defined x-axis intervals and derives
per-event statistics from a required parent Radon velocity analysis for the same
``(channel, roi_id)`` selection. The parent analysis is accessed through the
backend ``get_plot_data()`` API so event statistics do not depend on
Radon-specific table column names.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import pandas as pd

from acqstore.acq_image.analysis.data_provider import AnalysisDataProvider
from acqstore.acq_image.analysis.model import (
    AnalysisPlotData,
    AnalysisResult,
    AnalysisRunContext,
    BaseAnalysis,
    DetectionParamSchema,
    DetectionValueType,
)
from acqstore.acq_image.analysis.registry import register_analysis_class


RADON_VELOCITY_ANALYSIS_NAME = "radon_velocity"
EVENT_SUMMARY_VERSION = 2
DEFAULT_PRE_POST_WIN_SEC = 1.0
STAT_KEYS = ("n", "mean", "min", "max", "std", "se", "cv")


class EventType(StrEnum):
    """Supported acquisition-image event categories."""

    USER = "user"
    RISE = "rise"
    FALL = "fall"
    TRANSIENT = "transient"


@dataclass(frozen=True, slots=True)
class EventStats:
    """Summary statistics for one event-analysis x window.

    Args:
        n: Number of finite samples included in the window.
        mean: Arithmetic mean, or None when ``n == 0``.
        min: Minimum sample value, or None when ``n == 0``.
        max: Maximum sample value, or None when ``n == 0``.
        std: Sample standard deviation, or 0.0 for one sample, or None when
            ``n == 0``.
        se: Standard error of the mean, or None when ``n == 0``.
        cv: Coefficient of variation ``std / abs(mean)``, or None when
            unavailable.
    """

    n: int = 0
    mean: float | None = None
    min: float | None = None
    max: float | None = None
    std: float | None = None
    se: float | None = None
    cv: float | None = None

    def to_json_dict(self) -> dict[str, object]:
        """Return a strict-JSON-compatible dictionary.

        Returns:
            Mapping with stable statistic keys. Missing numeric values are
            encoded as None so sidecar JSON uses standard ``null`` values.
        """
        return {
            "n": int(self.n),
            "mean": self.mean,
            "min": self.min,
            "max": self.max,
            "std": self.std,
            "se": self.se,
            "cv": self.cv,
        }

    @classmethod
    def from_json_dict(cls, record: dict[str, Any]) -> EventStats:
        """Create stats from a JSON mapping.

        Args:
            record: JSON stats dictionary containing all statistic keys.

        Returns:
            Parsed stats.

        Raises:
            KeyError: If required statistic keys are missing.
        """
        missing = [key for key in STAT_KEYS if key not in record]
        if missing:
            raise KeyError(f"event stats missing keys: {missing}")
        return cls(
            n=int(record["n"]),
            mean=_optional_float(record["mean"]),
            min=_optional_float(record["min"]),
            max=_optional_float(record["max"]),
            std=_optional_float(record["std"]),
            se=_optional_float(record["se"]),
            cv=_optional_float(record["cv"]),
        )


@dataclass(frozen=True, slots=True)
class AcqImageEvent:
    """One x-axis interval event for an acquisition image analysis selection.

    Args:
        id: Stable integer event identifier within one event store.
        x0: First x-coordinate in analysis plot coordinates.
        x1: Second x-coordinate in analysis plot coordinates.
        event_type: Semantic event category.
        event_stats: Statistics from the event window ``[x_min, x_max]``.
        pre_win_stats: Statistics from the pre-event window
            ``[x0 - pre_post_win_sec, x0)``.
        post_win_stats: Statistics from the post-event window
            ``(x1, x1 + pre_post_win_sec]``.
    """

    id: int
    x0: float
    x1: float
    event_type: EventType = EventType.USER
    event_stats: EventStats = field(default_factory=EventStats)
    pre_win_stats: EventStats = field(default_factory=EventStats)
    post_win_stats: EventStats = field(default_factory=EventStats)

    def __post_init__(self) -> None:
        """Validate event fields.

        Raises:
            ValueError: If ``id`` is not positive or x coordinates are equal.
        """
        if int(self.id) < 1:
            raise ValueError(f"event id must be >= 1, got {self.id!r}")
        if float(self.x0) == float(self.x1):
            raise ValueError("event x0 and x1 must differ")

    @property
    def x_min(self) -> float:
        """Return the smaller x coordinate."""
        return min(float(self.x0), float(self.x1))

    @property
    def x_max(self) -> float:
        """Return the larger x coordinate."""
        return max(float(self.x0), float(self.x1))

    @property
    def duration(self) -> float:
        """Return the interval duration in x-axis units."""
        return abs(float(self.x1) - float(self.x0))

    def with_stats(self, plot_data: AnalysisPlotData, pre_post_win_sec: float) -> AcqImageEvent:
        """Return this event with stats recalculated from parent plot data.

        Args:
            plot_data: Parent analysis x/y data.
            pre_post_win_sec: Pre/post window width in seconds.

        Returns:
            New immutable event with recalculated stats.
        """
        return AcqImageEvent(
            id=self.id,
            x0=self.x0,
            x1=self.x1,
            event_type=self.event_type,
            event_stats=calculate_window_stats(plot_data, self.x_min, self.x_max, True, True),
            pre_win_stats=calculate_window_stats(
                plot_data,
                float(self.x0) - float(pre_post_win_sec),
                float(self.x0),
                True,
                False,
            ),
            post_win_stats=calculate_window_stats(
                plot_data,
                float(self.x1),
                float(self.x1) + float(pre_post_win_sec),
                False,
                True,
            ),
        )

    def to_json_dict(self) -> dict[str, object]:
        """Return a JSON-serializable event record.

        Returns:
            Dictionary containing event identity, interval coordinates,
            duration, and derived statistics.
        """
        return {
            "id": int(self.id),
            "x0": float(self.x0),
            "x1": float(self.x1),
            "event_type": self.event_type.value,
            "duration": self.duration,
            "event_stats": self.event_stats.to_json_dict(),
            "pre_win_stats": self.pre_win_stats.to_json_dict(),
            "post_win_stats": self.post_win_stats.to_json_dict(),
        }

    @classmethod
    def from_json_dict(cls, record: dict[str, Any]) -> AcqImageEvent:
        """Create an event from a JSON-like mapping.

        Args:
            record: Event record from analysis summary JSON.

        Returns:
            Parsed event.

        Raises:
            KeyError: If required fields are missing.
            ValueError: If values are invalid.
        """
        return cls(
            id=int(record["id"]),
            x0=float(record["x0"]),
            x1=float(record["x1"]),
            event_type=EventType(str(record["event_type"])),
            event_stats=EventStats.from_json_dict(dict(record["event_stats"])),
            pre_win_stats=EventStats.from_json_dict(dict(record["pre_win_stats"])),
            post_win_stats=EventStats.from_json_dict(dict(record["post_win_stats"])),
        )


class AcqImageEventStore:
    """Backend store for ``AcqImageEvent`` records.

    This class intentionally has no GUI concepts such as selection, visibility,
    or style. The next id is derived from current contents rather than persisted
    as separate state.
    """

    def __init__(self, events: list[AcqImageEvent] | None = None) -> None:
        """Initialize the store.

        Args:
            events: Optional initial events.
        """
        self._events: dict[int, AcqImageEvent] = {}
        if events:
            self.add_events(events)

    def __len__(self) -> int:
        """Return the number of stored events."""
        return len(self._events)

    def _next_id(self) -> int:
        """Return the next integer id.

        Returns:
            One greater than the current max id, or 1 for an empty store.
        """
        if not self._events:
            return 1
        return max(self._events) + 1

    def add_event(
        self,
        x0: float,
        x1: float,
        *,
        plot_data: AnalysisPlotData,
        pre_post_win_sec: float,
        event_type: EventType = EventType.USER,
        event_id: int | None = None,
    ) -> AcqImageEvent:
        """Add one x-interval event.

        Args:
            x0: First x-coordinate.
            x1: Second x-coordinate.
            plot_data: Parent analysis x/y data used for stats.
            pre_post_win_sec: Pre/post window width in seconds.
            event_type: Event category.
            event_id: Optional caller-supplied id. When omitted, the store uses
                the next integer id.

        Returns:
            Created event.

        Raises:
            ValueError: If the id already exists or coordinates are invalid.
        """
        new_id = self._next_id() if event_id is None else int(event_id)
        if new_id in self._events:
            raise ValueError(f"event id already exists: {new_id}")
        event = AcqImageEvent(id=new_id, x0=float(x0), x1=float(x1), event_type=event_type)
        event = event.with_stats(plot_data, pre_post_win_sec)
        self._events[event.id] = event
        return event

    def add_events(self, events: list[AcqImageEvent]) -> list[AcqImageEvent]:
        """Add multiple already-calculated events.

        Args:
            events: Events to add. Ids must be unique.

        Returns:
            Added events in input order.
        """
        added: list[AcqImageEvent] = []
        for event in events:
            if event.id in self._events:
                raise ValueError(f"event id already exists: {event.id}")
            self._events[int(event.id)] = event
            added.append(event)
        return added

    def delete_event(self, event_id: int) -> None:
        """Delete one event by id.

        Args:
            event_id: Event id to delete.

        Raises:
            KeyError: If the id is not present.
        """
        try:
            del self._events[int(event_id)]
        except KeyError:
            raise KeyError(f"event id not found: {event_id}") from None

    def update_event(
        self,
        event_id: int,
        *,
        plot_data: AnalysisPlotData,
        pre_post_win_sec: float,
        x0: float | None = None,
        x1: float | None = None,
        event_type: EventType | None = None,
    ) -> AcqImageEvent:
        """Update one event and recalculate its stats.

        Args:
            event_id: Event id to update.
            plot_data: Parent analysis x/y data used for stats.
            pre_post_win_sec: Pre/post window width in seconds.
            x0: Replacement first coordinate, or None to keep current.
            x1: Replacement second coordinate, or None to keep current.
            event_type: Replacement category, or None to keep current.

        Returns:
            Updated event.

        Raises:
            KeyError: If the id is not present.
        """
        eid = int(event_id)
        event = self.get_required(eid)
        updated = AcqImageEvent(
            id=event.id,
            x0=event.x0 if x0 is None else float(x0),
            x1=event.x1 if x1 is None else float(x1),
            event_type=event.event_type if event_type is None else event_type,
        ).with_stats(plot_data, pre_post_win_sec)
        self._events[eid] = updated
        return updated

    def reanalyze_events(self, plot_data: AnalysisPlotData, pre_post_win_sec: float) -> list[AcqImageEvent]:
        """Recalculate stats for all events without changing x coordinates.

        Args:
            plot_data: Parent analysis x/y data used for stats.
            pre_post_win_sec: Pre/post window width in seconds.

        Returns:
            Updated events sorted by id.
        """
        updated = [event.with_stats(plot_data, pre_post_win_sec) for event in self.get_events()]
        self._events = {event.id: event for event in updated}
        return updated

    def get_events(self) -> list[AcqImageEvent]:
        """Return events sorted by id.

        Returns:
            List of events.
        """
        return [self._events[key] for key in sorted(self._events)]

    def get_required(self, event_id: int) -> AcqImageEvent:
        """Return one event or raise.

        Args:
            event_id: Event id.

        Returns:
            Matching event.

        Raises:
            KeyError: If the id is not present.
        """
        eid = int(event_id)
        try:
            return self._events[eid]
        except KeyError:
            raise KeyError(f"event id not found: {event_id}") from None

    def to_summary_dict(self) -> dict[str, object]:
        """Return structured summary JSON for analysis persistence.

        Returns:
            Summary dictionary with version and event records.
        """
        return {
            "version": EVENT_SUMMARY_VERSION,
            "parent_analysis_name": RADON_VELOCITY_ANALYSIS_NAME,
            "events": [event.to_json_dict() for event in self.get_events()],
        }

    @classmethod
    def from_summary_dict(cls, summary: dict[str, Any]) -> AcqImageEventStore:
        """Create a store from analysis summary JSON.

        Args:
            summary: Structured summary from ``AnalysisResult.summary``.

        Returns:
            Hydrated event store.

        Raises:
            ValueError: If the summary version is not current.
        """
        version = int(summary["version"])
        if version != EVENT_SUMMARY_VERSION:
            raise ValueError(f"Unsupported event summary version: {version}")
        records = summary["events"]
        events = [AcqImageEvent.from_json_dict(dict(record)) for record in records]
        return cls(events)


@register_analysis_class
class EventAnalysis(BaseAnalysis):
    """Event interval analysis dependent on Radon velocity plot data."""

    analysis_name = "event"
    analysis_version = EVENT_SUMMARY_VERSION
    summary_columns = (
        "analysis_date",
        "analysis_time",
        "analysis_version",
        "version",
        "parent_analysis_name",
        "num_events",
        "user_events",
        "rise_events",
        "fall_events",
        "transient_events",
        "mean_duration",
    )
    exclusive_group = None
    depends_on = (RADON_VELOCITY_ANALYSIS_NAME,)
    detection_schema = (
        DetectionParamSchema(
            name="pre_post_win_sec",
            display_name="Pre/post window",
            value_type=DetectionValueType.FLOAT,
            default=DEFAULT_PRE_POST_WIN_SEC,
            description="Window width used before and after each event for derived statistics.",
            unit="s",
        ),
    )

    def __init__(
        self,
        *,
        channel: int,
        roi_id: int,
        detection_params: dict[str, object] | None = None,
    ) -> None:
        """Create an event analysis.

        Args:
            channel: Channel index.
            roi_id: ROI identifier.
            detection_params: Optional detection parameters.
        """
        super().__init__(channel=channel, roi_id=roi_id, detection_params=detection_params)
        self.events = AcqImageEventStore()
        self._sync_summary()

    @property
    def pre_post_win_sec(self) -> float:
        """Return the configured pre/post stats window in seconds."""
        value = float(self.detection_params["pre_post_win_sec"])
        if value < 0:
            raise ValueError("pre_post_win_sec must be >= 0")
        return value

    def set_detection_params(self, detection_params: dict[str, object]) -> None:
        """Replace detection parameters and mark this analysis dirty.

        Args:
            detection_params: New detection parameter mapping.
        """
        params = self.get_default_detection_params()
        self.validate_detection_params(detection_params)
        params.update(detection_params)
        self.detection_params = params
        _ = self.pre_post_win_sec
        self.set_dirty()

    def run(
        self,
        data_provider: AnalysisDataProvider,
        *,
        context: AnalysisRunContext | None = None,
        dependencies: dict[str, BaseAnalysis] | None = None,
    ) -> AnalysisResult:
        """Recalculate event stats from the required parent analysis.

        Args:
            data_provider: Unused analysis data provider.
            context: Optional run context.
            dependencies: Dependency mapping containing ``radon_velocity``.

        Returns:
            Current analysis result with recalculated event statistics.

        Raises:
            ValueError: If dependency data is missing or has no plot data.
        """
        _ = data_provider
        if context is not None:
            context.report_progress(0.0, "Preparing event analysis")
            context.raise_if_cancelled()
        plot_data = self._required_parent_plot_data(dependencies)
        self.events.reanalyze_events(plot_data, self.pre_post_win_sec)
        self._sync_summary()
        self.result.summary = self.finalize_summary(self.result.summary)
        self.set_dirty()
        if context is not None:
            context.report_progress(1.0, "Event analysis complete")
        return self.result

    def add_event(
        self,
        x0: float,
        x1: float,
        *,
        plot_data: AnalysisPlotData,
        event_type: EventType = EventType.USER,
        event_id: int | None = None,
    ) -> AcqImageEvent:
        """Add one event and mark analysis dirty.

        Args:
            x0: First x-coordinate.
            x1: Second x-coordinate.
            plot_data: Parent plot data used to calculate stats.
            event_type: Event category.
            event_id: Optional caller-supplied id.

        Returns:
            Created event.
        """
        event = self.events.add_event(
            x0,
            x1,
            plot_data=plot_data,
            pre_post_win_sec=self.pre_post_win_sec,
            event_type=event_type,
            event_id=event_id,
        )
        self._sync_summary()
        self.set_dirty()
        return event

    def add_events(self, events: list[AcqImageEvent]) -> list[AcqImageEvent]:
        """Add multiple already-calculated events and mark analysis dirty.

        Args:
            events: Events to add.

        Returns:
            Added events.
        """
        added = self.events.add_events(events)
        self._sync_summary()
        self.set_dirty()
        return added

    def delete_event(self, event_id: int) -> None:
        """Delete one event and mark analysis dirty.

        Args:
            event_id: Event id to delete.
        """
        self.events.delete_event(event_id)
        self._sync_summary()
        self.set_dirty()

    def update_event(
        self,
        event_id: int,
        *,
        plot_data: AnalysisPlotData,
        x0: float | None = None,
        x1: float | None = None,
        event_type: EventType | None = None,
    ) -> AcqImageEvent:
        """Update one event and mark analysis dirty.

        Args:
            event_id: Event id.
            plot_data: Parent plot data used to calculate stats.
            x0: Optional replacement x0.
            x1: Optional replacement x1.
            event_type: Optional replacement category.

        Returns:
            Updated event.
        """
        event = self.events.update_event(
            event_id,
            plot_data=plot_data,
            pre_post_win_sec=self.pre_post_win_sec,
            x0=x0,
            x1=x1,
            event_type=event_type,
        )
        self._sync_summary()
        self.set_dirty()
        return event

    def get_events(self) -> list[AcqImageEvent]:
        """Return current events.

        Returns:
            List of events sorted by id.
        """
        return self.events.get_events()

    def get_summary_values(self) -> dict[str, object]:
        """Return flat event summary values for analysis pools.

        Returns:
            Mapping with exactly the keys declared in
            :attr:`summary_columns`. The nested event records are summarized as
            scalar counts and mean event duration.
        """
        summary = self.result.summary
        records = summary.get("events")
        events = records if isinstance(records, list) else []
        type_counts = {event_type.value: 0 for event_type in EventType}
        durations: list[float] = []
        for record in events:
            if not isinstance(record, dict):
                continue
            event_type = str(record.get("event_type", ""))
            if event_type in type_counts:
                type_counts[event_type] += 1
            duration = record.get("duration")
            if duration is not None:
                try:
                    durations.append(float(duration))
                except (TypeError, ValueError):
                    continue

        values: dict[str, object] = {
            "analysis_date": summary.get("analysis_date", pd.NA),
            "analysis_time": summary.get("analysis_time", pd.NA),
            "analysis_version": summary.get("analysis_version", pd.NA),
            "version": summary.get("version", pd.NA),
            "parent_analysis_name": summary.get("parent_analysis_name", pd.NA),
            "num_events": len(events),
            "user_events": type_counts[EventType.USER.value],
            "rise_events": type_counts[EventType.RISE.value],
            "fall_events": type_counts[EventType.FALL.value],
            "transient_events": type_counts[EventType.TRANSIENT.value],
            "mean_duration": (sum(durations) / len(durations)) if durations else pd.NA,
        }
        return {key: values.get(key, pd.NA) for key in self.get_summary_columns()}

    def load_json_dict(self, record: dict[str, Any]) -> None:
        """Load event analysis state from a JSON analysis record.

        Args:
            record: Analysis sidecar record.
        """
        detection_params = dict(record["detection_params"])
        self.set_detection_params(detection_params)
        self.result.summary = dict(record["summary"])
        self.events = AcqImageEventStore.from_summary_dict(self.result.summary)
        self._sync_summary()
        self.set_clean()

    def _sync_summary(self) -> None:
        """Synchronize ``AnalysisResult.summary`` with the event store."""
        self.result.summary = self.events.to_summary_dict()

    @staticmethod
    def _required_parent_plot_data(dependencies: dict[str, BaseAnalysis] | None) -> AnalysisPlotData:
        """Return required parent plot data or raise.

        Args:
            dependencies: Analysis dependencies keyed by analysis name.

        Returns:
            Parent analysis plot data.

        Raises:
            ValueError: If the required dependency or its plot data is missing.
        """
        if dependencies is None or RADON_VELOCITY_ANALYSIS_NAME not in dependencies:
            raise ValueError("Event analysis requires radon_velocity dependency")
        parent = dependencies[RADON_VELOCITY_ANALYSIS_NAME]
        plot_data = parent.get_plot_data()
        if plot_data is None:
            raise ValueError("Event analysis requires radon_velocity plot data")
        return plot_data


def calculate_window_stats(
    plot_data: AnalysisPlotData,
    x0: float,
    x1: float,
    include_left: bool,
    include_right: bool,
) -> EventStats:
    """Calculate stats for finite y samples inside an x window.

    Args:
        plot_data: Parent analysis x/y data.
        x0: Left window boundary.
        x1: Right window boundary.
        include_left: Whether the left boundary is inclusive.
        include_right: Whether the right boundary is inclusive.

    Returns:
        Event statistics. Empty windows return ``n=0`` and None values.
    """
        
    left = min(float(x0), float(x1))
    right = max(float(x0), float(x1))
    values: list[float] = []
    for x_value, y_value in zip(plot_data.x, plot_data.y, strict=True):
        x = float(x_value)
        y = float(y_value)
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        left_ok = x >= left if include_left else x > left
        right_ok = x <= right if include_right else x < right
        if left_ok and right_ok:
            values.append(y)
    return _stats_from_values(values)


def _stats_from_values(values: list[float]) -> EventStats:
    """Return event stats for a list of finite values.

    Args:
        values: Finite samples.

    Returns:
        Derived event statistics.
    """
    n = len(values)
    if n == 0:
        return EventStats()
    mean = sum(values) / n
    minimum = min(values)
    maximum = max(values)
    if n == 1:
        std = 0.0
    else:
        variance = sum((value - mean) ** 2 for value in values) / (n - 1)
        std = math.sqrt(variance)
    se = std / math.sqrt(n)
    cv = None if mean == 0 else std / abs(mean)
    return EventStats(
        n=n,
        mean=float(mean),
        min=float(minimum),
        max=float(maximum),
        std=float(std),
        se=float(se),
        cv=None if cv is None else float(cv),
    )


def _optional_float(value: Any) -> float | None:
    """Return ``value`` as float, preserving None.

    Args:
        value: JSON value.

    Returns:
        Float value or None.
    """
    if value is None:
        return None
    return float(value)
