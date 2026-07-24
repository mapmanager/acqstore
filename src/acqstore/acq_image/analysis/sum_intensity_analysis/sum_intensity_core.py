"""Pure sum-intensity and peak-detection algorithms.

The functions in this module operate on NumPy arrays and plain Python values.
They intentionally do not import from :mod:`acqstore` so they can be tested and
reused independently of acquisition-image persistence or GUI code.

Axis convention for line-scan kymographs:

* axis 0 is time, with spacing ``physical_units[0]`` in seconds per line.
* axis 1 is space, with spacing ``physical_units[1]`` in microns per pixel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit

Polarity = Literal["positive", "negative"]
DetectionMethod = Literal["derivative_threshold", "absolute_threshold"]
FilterMethod = Literal["none", "median"]
DetrendMethod = Literal["none", "single_exponential"]
BaselineMethod = Literal["percentile", "manual"]


class SumIntensityTraceKey(StrEnum):
    """Named continuous traces produced by sum-intensity analysis.

    The enum is the public API for trace lookup. It intentionally hides raw
    DataFrame column strings from callers such as GUI views, scripts, and
    downstream analysis code.

    Members:
        SUM_INTENSITY: Spatial row sum after optional rolling row averaging.
        NORM_SUM_INTENSITY: Mean line intensity, ``sum_intensity / n_space``.
        FILTERED_NORM_SUM_INTENSITY: Normalized trace after optional filtering.
        DETRENDED_NORM_SUM_INTENSITY: Filtered normalized trace after optional
            bleaching detrend.
        DF_F_SIGNAL: Delta-F over F0 signal calculated from the detrended or
            filtered normalized trace.
        D_DF_F_SIGNAL: Time derivative of ``DF_F_SIGNAL`` in ``1 / second``.
    """

    SUM_INTENSITY = "sum_intensity"
    NORM_SUM_INTENSITY = "norm_sum_intensity"
    FILTERED_NORM_SUM_INTENSITY = "filtered_norm_sum_intensity"
    DETRENDED_NORM_SUM_INTENSITY = "detrended_norm_sum_intensity"
    DF_F_SIGNAL = "df_f_signal"
    D_DF_F_SIGNAL = "d_df_f_signal"


class SumIntensityEventPointKey(StrEnum):
    """Named sparse event-point collections.

    Event points are one point per detected peak event. They are primarily used
    for plotting markers on top of continuous traces without exposing the JSON
    representation of ``PeakEvent``.

    Members:
        ONSETS: Accepted onset locations, plotted at onset time and value.
        PEAKS: Refined peak locations, plotted at peak time and value.
    """

    ONSETS = "onsets"
    PEAKS = "peaks"


class PeakWidthLevel(StrEnum):
    """Named fractional peak-width measurements.

    Width levels are restricted to the standard CloudScope fractions so callers
    do not pass unvalidated arbitrary floats. The stored measurement represents
    the left and right crossings of the requested fraction of peak amplitude.

    Members:
        WIDTH_10: Width at 10 percent of peak amplitude.
        WIDTH_20: Width at 20 percent of peak amplitude.
        WIDTH_50: Width at 50 percent of peak amplitude.
        WIDTH_80: Width at 80 percent of peak amplitude.
        WIDTH_90: Width at 90 percent of peak amplitude.
    """

    WIDTH_10 = "width_10"
    WIDTH_20 = "width_20"
    WIDTH_50 = "width_50"
    WIDTH_80 = "width_80"
    WIDTH_90 = "width_90"


class SumIntensitySummaryKey(StrEnum):
    """Named scalar values stored in the sum-intensity summary.

    Summary values are compact run-level measurements or metadata. Full traces
    live in the result table, and per-peak measurements live in
    ``peak_events``.

    Members:
        NUM_PEAKS: Number of accepted peak events.
        F0_BASELINE: Scalar F0 value used to calculate ``df_f_signal``.
        BASELINE_METHOD: F0 calculation method.
        BASELINE_PERCENTILE: Percentile used when ``baseline_method`` is
            ``"percentile"``.
        MANUAL_F0_BASELINE: User-supplied F0 value used when ``baseline_method``
            is ``"manual"``.
        DETECTION_SOURCE: Trace key selected for onset detection.
        SECONDS_PER_LINE: Time spacing in seconds.
        WARNINGS: Analysis-level non-fatal warnings.
        ERRORS: Analysis-level non-fatal algorithm errors.
    """

    NUM_PEAKS = "num_peaks"
    F0_BASELINE = "f0_baseline"
    BASELINE_METHOD = "baseline_method"
    BASELINE_PERCENTILE = "baseline_percentile"
    MANUAL_F0_BASELINE = "manual_f0_baseline"
    DETECTION_SOURCE = "detection_source"
    SECONDS_PER_LINE = "seconds_per_line"
    WARNINGS = "warnings"
    ERRORS = "errors"


WIDTH_LEVEL_FRACTIONS: dict[PeakWidthLevel, float] = {
    PeakWidthLevel.WIDTH_10: 0.1,
    PeakWidthLevel.WIDTH_20: 0.2,
    PeakWidthLevel.WIDTH_50: 0.5,
    PeakWidthLevel.WIDTH_80: 0.8,
    PeakWidthLevel.WIDTH_90: 0.9,
}


TRACE_DEFINITIONS: dict[SumIntensityTraceKey, dict[str, str]] = {
    SumIntensityTraceKey.SUM_INTENSITY: {
        "display_name": "Sum intensity",
        "description": "Spatial row sum after optional rolling row averaging.",
        "y_label": "Intensity sum",
        "units": "image intensity",
    },
    SumIntensityTraceKey.NORM_SUM_INTENSITY: {
        "display_name": "Normalized sum intensity",
        "description": "Mean line intensity, sum_intensity divided by spatial pixel count.",
        "y_label": "Mean line intensity",
        "units": "image intensity",
    },
    SumIntensityTraceKey.FILTERED_NORM_SUM_INTENSITY: {
        "display_name": "Filtered normalized sum intensity",
        "description": "Normalized trace after optional median filtering.",
        "y_label": "Mean line intensity",
        "units": "image intensity",
    },
    SumIntensityTraceKey.DETRENDED_NORM_SUM_INTENSITY: {
        "display_name": "Detrended normalized sum intensity",
        "description": "Filtered normalized trace after optional bleaching detrend.",
        "y_label": "Detrended mean line intensity",
        "units": "image intensity",
    },
    SumIntensityTraceKey.DF_F_SIGNAL: {
        "display_name": "df/f0 signal",
        "description": "Delta-F over F0 signal calculated from the detrended or filtered trace.",
        "y_label": "df/f0",
        "units": "fraction",
    },
    SumIntensityTraceKey.D_DF_F_SIGNAL: {
        "display_name": "Derivative of df/f0",
        "description": "Time derivative of df/f0.",
        "y_label": "d(df/f0)/dt",
        "units": "1/s",
    },
}


@dataclass(frozen=True, slots=True)
class ResultTrace:
    """Continuous trace ready for plotting or downstream analysis.

    Args:
        key: Trace key identifying the source column.
        name: Human-readable trace name.
        x: X-axis values, usually time in seconds.
        y: Trace values.
        x_label: X-axis label.
        y_label: Y-axis label.
        metadata: Optional extra plotting hints. Backend code does not assume a
            specific plotting library.
    """

    key: SumIntensityTraceKey | str
    name: str
    x: np.ndarray
    y: np.ndarray
    x_label: str
    y_label: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ResultPoints:
    """Sparse event points ready for plotting as markers.

    Args:
        key: Event point key identifying the point collection.
        name: Human-readable point collection name.
        x: Point x-axis values.
        y: Point y-axis values.
        x_label: X-axis label.
        y_label: Y-axis label.
        metadata: Optional extra plotting hints.
    """

    key: SumIntensityEventPointKey
    name: str
    x: np.ndarray
    y: np.ndarray
    x_label: str
    y_label: str
    metadata: dict[str, object] = field(default_factory=dict)


SUM_INTENSITY_TABLE_COLUMNS: tuple[str, ...] = (
    "time_index",
    "time_sec",
    "sum_intensity",
    "norm_sum_intensity",
    "filtered_norm_sum_intensity",
    "detrended_norm_sum_intensity",
    "f0_baseline",
    "df_f_signal",
    "d_df_f_signal",
    "is_onset",
    "is_peak",
    "onset_index",
    "peak_id",
)


@dataclass(frozen=True, slots=True)
class EventFeature:
    """Event-local scalar measurement with failure provenance.

    Args:
        value: Numeric measurement value, or None when unavailable.
        status: Measurement status. ``"ok"`` means the value is valid; other
            values identify expected scientific failure modes.
        reason: Human-readable reason when ``status`` is not ``"ok"``.
    """

    value: float | None
    status: str = "ok"
    reason: str = ""

    def to_json_dict(self) -> dict[str, object]:
        """Return a JSON-serializable feature record.

        Returns:
            Dictionary containing value, status, and reason.
        """
        return {
            "value": _optional_float(self.value),
            "status": str(self.status),
            "reason": str(self.reason),
        }

    @classmethod
    def from_json_dict(cls, record: dict[str, Any]) -> EventFeature:
        """Create a feature record from JSON-like data.

        Args:
            record: JSON dictionary created by :meth:`to_json_dict`.

        Returns:
            Parsed event feature.
        """
        return cls(
            value=_optional_float(record.get("value")),
            status=str(record.get("status", "ok")),
            reason=str(record.get("reason", "")),
        )


@dataclass(frozen=True, slots=True)
class LevelCrossing:
    """Peak level-crossing measurement at one amplitude fraction.

    Args:
        fraction: Requested fraction of peak amplitude, for example ``0.5``.
        value: Signal value corresponding to the requested fraction, or None
            when the crossing could not be evaluated.
        left_index: Interpolated rising-side index, or None when unavailable.
        right_index: Interpolated falling-side index, or None when unavailable.
        width: ``right_index - left_index`` in points, or None when either side
            is unavailable.
        width_sec: Width in seconds, or None when width is unavailable.
        status: Measurement status. ``"ok"`` means width is valid; other
            values identify expected scientific failure modes.
    """

    fraction: float
    value: float | None
    left_index: float | None
    right_index: float | None
    width: float | None
    width_sec: float | None = None
    status: str = "ok"

    def to_json_dict(self) -> dict[str, object]:
        """Return a JSON-serializable level-crossing record.

        Returns:
            Dictionary with stable keys for sidecar JSON storage.
        """
        return {
            "fraction": float(self.fraction),
            "value": _optional_float(self.value),
            "left_index": _optional_float(self.left_index),
            "right_index": _optional_float(self.right_index),
            "width": _optional_float(self.width),
            "width_sec": _optional_float(self.width_sec),
            "status": str(self.status),
        }

    @classmethod
    def from_json_dict(cls, record: dict[str, Any]) -> LevelCrossing:
        """Create a crossing record from JSON-like data.

        Args:
            record: JSON dictionary created by :meth:`to_json_dict`.

        Returns:
            Parsed crossing record.
        """
        return cls(
            fraction=float(record["fraction"]),
            value=_optional_float(record.get("value")),
            left_index=_optional_float(record.get("left_index")),
            right_index=_optional_float(record.get("right_index")),
            width=_optional_float(record.get("width")),
            width_sec=_optional_float(record.get("width_sec")),
            status=str(record.get("status", "ok")),
        )


@dataclass(frozen=True, slots=True)
class PeakEvent:
    """One detected sum-intensity peak/event.

    Args:
        peak_id: Stable one-based event identifier within one analysis run.
        status: Event-level status string.
        warnings: Event-local warnings for measurements that failed normally.
        onset_index: Time index where the detector accepted event onset.
        onset_time_sec: Onset time in seconds.
        onset_value: Detection-signal value at onset.
        peak_index: Refined peak index, or None when peak refinement failed.
        peak_time_sec: Peak time in seconds, or None when unavailable.
        peak_value: Detection-signal value at the peak, or None when
            unavailable.
        peak_amplitude: Peak amplitude relative to onset, or None when
            unavailable.
        detection_method: Detection method used to accept the onset.
        baseline_mean: Mean detection-source value before onset.
        baseline_std: Standard deviation of detection-source values before onset.
        rise_10_90_sec: Time from left 10% to left 90% crossing.
        decay_90_10_sec: Time from right 90% to right 10% crossing.
        decay_time_sec: Default decay-time measurement, currently decay 90→10.
        max_rise_slope: Maximum signed derivative from onset to peak.
        max_decay_slope: Maximum signed derivative from peak to right 10% crossing.
        auc: Area under event between left and right 10% crossings.
        prominence: Peak value relative to the pre-onset baseline mean.
        level_crossings: Requested fractional width measurements.
        onset_to_onset_interval_sec: Interval from the previous accepted onset,
            or None for the first event.
        peak_to_peak_interval_sec: Interval from the previous refined peak, or
            None when unavailable.
    """

    peak_id: int
    status: str
    warnings: tuple[str, ...]
    onset_index: int
    onset_time_sec: float
    onset_value: float
    peak_index: int | None
    peak_time_sec: float | None
    peak_value: float | None
    peak_amplitude: float | None
    detection_method: str
    baseline_mean: EventFeature
    baseline_std: EventFeature
    rise_10_90_sec: EventFeature
    decay_90_10_sec: EventFeature
    decay_time_sec: EventFeature
    max_rise_slope: EventFeature
    max_decay_slope: EventFeature
    auc: EventFeature
    prominence: EventFeature
    level_crossings: tuple[LevelCrossing, ...] = field(default_factory=tuple)
    onset_to_onset_interval_sec: float | None = None
    peak_to_peak_interval_sec: float | None = None

    def to_json_dict(self) -> dict[str, object]:
        """Return a JSON-serializable event record.

        Returns:
            Dictionary with stable keys for sidecar JSON storage.
        """
        return {
            "peak_id": int(self.peak_id),
            "status": str(self.status),
            "warnings": list(self.warnings),
            "onset": {
                "index": int(self.onset_index),
                "time_sec": float(self.onset_time_sec),
                "value": float(self.onset_value),
            },
            "peak": {
                "index": None if self.peak_index is None else int(self.peak_index),
                "time_sec": _optional_float(self.peak_time_sec),
                "value": _optional_float(self.peak_value),
                "amplitude": _optional_float(self.peak_amplitude),
            },
            "detection_method": str(self.detection_method),
            "features": {
                "baseline_mean": self.baseline_mean.to_json_dict(),
                "baseline_std": self.baseline_std.to_json_dict(),
                "rise_10_90_sec": self.rise_10_90_sec.to_json_dict(),
                "decay_90_10_sec": self.decay_90_10_sec.to_json_dict(),
                "decay_time_sec": self.decay_time_sec.to_json_dict(),
                "max_rise_slope": self.max_rise_slope.to_json_dict(),
                "max_decay_slope": self.max_decay_slope.to_json_dict(),
                "auc": self.auc.to_json_dict(),
                "prominence": self.prominence.to_json_dict(),
            },
            "level_crossings": [crossing.to_json_dict() for crossing in self.level_crossings],
            "intervals": {
                "onset_to_onset_interval_sec": _optional_float(
                    self.onset_to_onset_interval_sec
                ),
                "peak_to_peak_interval_sec": _optional_float(
                    self.peak_to_peak_interval_sec
                ),
            },
        }

    @classmethod
    def from_json_dict(cls, record: dict[str, Any]) -> PeakEvent:
        """Create an event from JSON-like data.

        Args:
            record: JSON dictionary created by :meth:`to_json_dict`.

        Returns:
            Parsed event.
        """
        onset = dict(record["onset"])
        peak = dict(record["peak"])
        intervals = dict(record.get("intervals", {}))
        features = dict(record.get("features", {}))

        def _feature(name: str) -> EventFeature:
            feature_record = features.get(name)
            if feature_record is None:
                return EventFeature(value=None, status="not_available", reason="missing_from_json")
            return EventFeature.from_json_dict(dict(feature_record))

        return cls(
            peak_id=int(record["peak_id"]),
            status=str(record["status"]),
            warnings=tuple(str(item) for item in record.get("warnings", ())),
            onset_index=int(onset["index"]),
            onset_time_sec=float(onset["time_sec"]),
            onset_value=float(onset["value"]),
            peak_index=None if peak.get("index") is None else int(peak["index"]),
            peak_time_sec=_optional_float(peak.get("time_sec")),
            peak_value=_optional_float(peak.get("value")),
            peak_amplitude=_optional_float(peak.get("amplitude")),
            detection_method=str(record["detection_method"]),
            baseline_mean=_feature("baseline_mean"),
            baseline_std=_feature("baseline_std"),
            rise_10_90_sec=_feature("rise_10_90_sec"),
            decay_90_10_sec=_feature("decay_90_10_sec"),
            decay_time_sec=_feature("decay_time_sec"),
            max_rise_slope=_feature("max_rise_slope"),
            max_decay_slope=_feature("max_decay_slope"),
            auc=_feature("auc"),
            prominence=_feature("prominence"),
            level_crossings=tuple(
                LevelCrossing.from_json_dict(dict(item))
                for item in record.get("level_crossings", ())
            ),
            onset_to_onset_interval_sec=_optional_float(
                intervals.get("onset_to_onset_interval_sec")
            ),
            peak_to_peak_interval_sec=_optional_float(
                intervals.get("peak_to_peak_interval_sec")
            ),
        )


@dataclass(frozen=True, slots=True)
class SumIntensityCoreResult:
    """Pure sum-intensity analysis output.

    This object is the public result returned by the NumPy-only core algorithm.
    It intentionally exposes plotting-friendly accessors so scripts, tests, and
    future CloudScope views can use the scientific API without parsing raw
    DataFrame columns or summary dictionaries.

    Args:
        summary: JSON-serializable summary including run metadata, scalar
            results, analysis-level warnings/errors, and peak-event records.
        table: Per-timepoint result table.
        events: Parsed event objects mirrored in ``summary``.
    """

    summary: dict[str, Any]
    table: pd.DataFrame
    events: tuple[PeakEvent, ...]

    def get_peak_events(self) -> tuple[PeakEvent, ...]:
        """Return parsed peak-event records.

        Returns:
            Tuple of peak events. Empty when no peaks were detected.
        """
        return self.events

    def get_trace(self, key: SumIntensityTraceKey) -> ResultTrace:
        """Return one named continuous trace.

        Args:
            key: Trace key to retrieve.

        Returns:
            Result trace with ``time_sec`` as x values.

        Raises:
            KeyError: If the result table is missing a required column.
        """
        if "time_sec" not in self.table.columns:
            raise KeyError("Sum intensity trace requires 'time_sec' column")
        if key.value not in self.table.columns:
            raise KeyError(f"Sum intensity trace column is missing: {key.value!r}")
        definition = TRACE_DEFINITIONS[key]
        return ResultTrace(
            key=key,
            name=definition["display_name"],
            x=self.table["time_sec"].to_numpy(dtype=float),
            y=self.table[key.value].to_numpy(dtype=float),
            x_label="Time (s)",
            y_label=definition["y_label"],
            metadata={
                "description": definition["description"],
                "units": definition["units"],
            },
        )

    def get_event_points(self, key: SumIntensityEventPointKey) -> ResultPoints:
        """Return sparse event marker points.

        Args:
            key: Event point collection to retrieve.

        Returns:
            Result points for plotting event markers.

        Raises:
            KeyError: If the point key is unknown.
        """
        events = self.get_peak_events()
        if key == SumIntensityEventPointKey.ONSETS:
            return ResultPoints(
                key=key,
                name="Onsets",
                x=_optional_event_array(events, "onset_time_sec"),
                y=_optional_event_array(events, "onset_value"),
                x_label="Time (s)",
                y_label="Detection source",
                metadata={"description": "Accepted onset threshold crossings."},
            )
        if key == SumIntensityEventPointKey.PEAKS:
            return ResultPoints(
                key=key,
                name="Peaks",
                x=_optional_event_array(events, "peak_time_sec"),
                y=_optional_event_array(events, "peak_value"),
                x_label="Time (s)",
                y_label="Detection source",
                metadata={"description": "Refined peak locations."},
            )
        raise KeyError(f"Unknown sum-intensity event point key: {key!r}")

    def get_width_trace(
        self,
        peak_width_level: PeakWidthLevel | None = None,
    ) -> ResultTrace | tuple[ResultTrace, ...]:
        """Return NaN-separated width segment traces.

        Args:
            peak_width_level: Specific width level to return. When None, traces
                for all standard width levels are returned.

        Returns:
            One ``ResultTrace`` when ``peak_width_level`` is supplied, otherwise
            a tuple of traces for all levels.
        """
        if peak_width_level is None:
            return tuple(self.get_width_trace(level) for level in PeakWidthLevel)
        fraction = WIDTH_LEVEL_FRACTIONS[peak_width_level]
        x_values: list[float] = []
        y_values: list[float] = []
        seconds_per_line = float(
            self.summary.get(SumIntensitySummaryKey.SECONDS_PER_LINE.value, 1.0)
        )
        for event in self.get_peak_events():
            crossing = _find_event_crossing(event, fraction)
            if crossing is None or crossing.status != "ok":
                continue
            if crossing.left_index is None or crossing.right_index is None or crossing.value is None:
                continue
            x_values.extend(
                [
                    float(crossing.left_index) * seconds_per_line,
                    float(crossing.right_index) * seconds_per_line,
                    float("nan"),
                ]
            )
            y_values.extend([float(crossing.value), float(crossing.value), float("nan")])
        return ResultTrace(
            key=peak_width_level.value,
            name=f"Peak {peak_width_level.value.replace('_', ' ')}",
            x=np.asarray(x_values, dtype=float),
            y=np.asarray(y_values, dtype=float),
            x_label="Time (s)",
            y_label="Detection source",
            metadata={
                "fraction": fraction,
                "trace_type": "width_segments",
                "connectgaps": False,
            },
        )

    def get_summary_value(self, key: SumIntensitySummaryKey) -> object:
        """Return one named summary value.

        Args:
            key: Summary key to retrieve.

        Returns:
            Stored summary value, or None when the key is absent.
        """
        return self.summary.get(key.value)



def run_sum_intensity(
    image: np.ndarray,
    *,
    detection_params: dict[str, Any],
    physical_units: tuple[float, float],
) -> SumIntensityCoreResult:
    """Run sum-intensity analysis on one ROI-cropped kymograph.

    Args:
        image: Two-dimensional image with shape ``(time, space)``.
        detection_params: Flat detection parameter mapping.
        physical_units: ``(seconds_per_line, um_per_pixel)``. Only seconds per
            line is used by this first-pass analysis; the spatial unit is kept
            as explicit input for consistency with other kymograph analyses.

    Returns:
        Summary, table, and parsed peak events.

    Raises:
        ValueError: If image, physical units, or detection parameters are
            invalid.
    """
    arr = np.asarray(image, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"image must be 2D (time, space), got shape={arr.shape!r}")
    if arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError(f"image must be non-empty in both axes, got shape={arr.shape!r}")

    seconds_per_line, um_per_pixel = (float(physical_units[0]), float(physical_units[1]))
    if seconds_per_line <= 0 or um_per_pixel <= 0:
        raise ValueError("physical_units must be strictly positive")

    params = _validated_detection_params(detection_params)
    warnings: list[str] = []
    errors: list[str] = []

    n_time, n_space = arr.shape
    time_index = np.arange(n_time, dtype=int)
    time_sec = time_index.astype(float) * seconds_per_line

    row_sum = np.sum(arr, axis=1, dtype=float)
    sum_intensity = _rolling_mean_clipped(row_sum, radius=int(params["window_radius_points"]))
    norm_sum_intensity = sum_intensity / float(n_space)

    filtered_norm = _filter_trace(
        norm_sum_intensity,
        filter_method=str(params["filter_method"]),
        median_kernel_points=int(params["median_filter_kernel_points"]),
        warnings=warnings,
    )
    detrended_norm = _detrend_trace(
        filtered_norm,
        time_sec=time_sec,
        detrend_method=str(params["detrend_method"]),
        warnings=warnings,
        errors=errors,
    )
    f0_baseline, df_f_signal = _calculate_df_f_signal(
        detrended_norm,
        baseline_method=str(params["baseline_method"]),
        baseline_percentile=float(params["baseline_percentile"]),
        manual_f0_baseline=float(params["manual_f0_baseline"]),
        baseline_min_value=float(params["baseline_min_value"]),
        warnings=warnings,
        errors=errors,
    )
    trace_map = {
        SumIntensityTraceKey.SUM_INTENSITY.value: sum_intensity,
        SumIntensityTraceKey.NORM_SUM_INTENSITY.value: norm_sum_intensity,
        SumIntensityTraceKey.FILTERED_NORM_SUM_INTENSITY.value: filtered_norm,
        SumIntensityTraceKey.DETRENDED_NORM_SUM_INTENSITY.value: detrended_norm,
        SumIntensityTraceKey.DF_F_SIGNAL.value: df_f_signal,
    }
    detection_source = str(params["detection_source"])
    detection_signal = np.asarray(trace_map[detection_source], dtype=float)
    d_df_f_signal = np.gradient(df_f_signal, time_sec)
    d_detection_signal = np.gradient(detection_signal, time_sec)

    onset_indices = _detect_onsets(
        detection_signal=detection_signal,
        derivative_signal=d_detection_signal,
        method=str(params["detection_method"]),
        polarity=str(params["polarity"]),
        absolute_threshold=float(params["absolute_threshold"]),
        derivative_threshold_per_sec=float(params["derivative_threshold_per_sec"]),
    )
    refractory_points = _duration_ms_to_points(
        float(params["refractory_period_ms"]), seconds_per_line
    )
    accepted_onsets = _apply_refractory(onset_indices, refractory_points=refractory_points)

    peak_search_window_points = _duration_ms_to_points(
        float(params["peak_search_window_ms"]), seconds_per_line
    )
    width_search_window_points = _duration_ms_to_points(
        float(params["width_search_window_ms"]), seconds_per_line
    )
    level_fractions = _parse_level_fractions(str(params["level_fractions"]))
    baseline_window_points = _duration_ms_to_points(
        float(params["baseline_window_ms"]), seconds_per_line
    )
    events = _build_events(
        detection_signal=detection_signal,
        derivative_signal=d_detection_signal,
        time_sec=time_sec,
        onset_indices=accepted_onsets,
        detection_method=str(params["detection_method"]),
        polarity=str(params["polarity"]),
        peak_search_window_points=peak_search_window_points,
        width_search_window_points=width_search_window_points,
        baseline_window_points=baseline_window_points,
        level_fractions=level_fractions,
    )

    table = _build_table(
        time_index=time_index,
        time_sec=time_sec,
        sum_intensity=sum_intensity,
        norm_sum_intensity=norm_sum_intensity,
        filtered_norm_sum_intensity=filtered_norm,
        detrended_norm_sum_intensity=detrended_norm,
        f0_baseline=f0_baseline,
        df_f_signal=df_f_signal,
        d_df_f_signal=d_df_f_signal,
        events=events,
    )
    summary = _build_summary(
        table=table,
        events=events,
        warnings=warnings,
        errors=errors,
        params=params,
        seconds_per_line=seconds_per_line,
        n_space=n_space,
        f0_baseline=f0_baseline,
    )
    return SumIntensityCoreResult(summary=summary, table=table, events=events)


def _validated_detection_params(params: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize core detection parameters.

    Args:
        params: Flat detection parameter mapping.

    Returns:
        Normalized copy of ``params``.

    Raises:
        ValueError: If any parameter value is invalid.
    """
    normalized = dict(params)
    if int(normalized["window_radius_points"]) < 0:
        raise ValueError("window_radius_points must be >= 0")
    if normalized["filter_method"] not in {"none", "median"}:
        raise ValueError("filter_method must be 'none' or 'median'")
    kernel = int(normalized["median_filter_kernel_points"])
    if kernel < 1:
        raise ValueError("median_filter_kernel_points must be >= 1")
    if normalized["detrend_method"] not in {"none", "single_exponential"}:
        raise ValueError("detrend_method must be 'none' or 'single_exponential'")
    if normalized["baseline_method"] not in {"percentile", "manual"}:
        raise ValueError("baseline_method must be 'percentile' or 'manual'")
    baseline_percentile = float(normalized["baseline_percentile"])
    if baseline_percentile < 0.0 or baseline_percentile > 100.0:
        raise ValueError("baseline_percentile must be between 0 and 100")
    if float(normalized["baseline_min_value"]) <= 0.0:
        raise ValueError("baseline_min_value must be > 0")
    manual_f0 = float(normalized.get("manual_f0_baseline", 1.0))
    if normalized["baseline_method"] == "manual" and manual_f0 == 0.0:
        raise ValueError("manual_f0_baseline must be non-zero when baseline_method is manual")
    normalized["manual_f0_baseline"] = manual_f0
    if normalized["detection_method"] not in {"derivative_threshold", "absolute_threshold"}:
        raise ValueError("detection_method must be 'derivative_threshold' or 'absolute_threshold'")
    if normalized["polarity"] not in {"positive", "negative"}:
        raise ValueError("polarity must be 'positive' or 'negative'")
    detection_source = str(normalized.get("detection_source", SumIntensityTraceKey.DF_F_SIGNAL.value))
    allowed_detection_sources = {
        SumIntensityTraceKey.SUM_INTENSITY.value,
        SumIntensityTraceKey.NORM_SUM_INTENSITY.value,
        SumIntensityTraceKey.FILTERED_NORM_SUM_INTENSITY.value,
        SumIntensityTraceKey.DETRENDED_NORM_SUM_INTENSITY.value,
        SumIntensityTraceKey.DF_F_SIGNAL.value,
    }
    if detection_source not in allowed_detection_sources:
        raise ValueError(
            "detection_source must be one of "
            f"{tuple(sorted(allowed_detection_sources))!r}"
        )
    normalized["detection_source"] = detection_source
    if float(normalized["baseline_window_ms"]) < 0:
        raise ValueError("baseline_window_ms must be >= 0")
    if float(normalized["refractory_period_ms"]) < 0:
        raise ValueError("refractory_period_ms must be >= 0")
    if float(normalized["peak_search_window_ms"]) <= 0:
        raise ValueError("peak_search_window_ms must be > 0")
    if float(normalized["width_search_window_ms"]) <= 0:
        raise ValueError("width_search_window_ms must be > 0")
    _parse_level_fractions(str(normalized["level_fractions"]))
    return normalized


def _rolling_mean_clipped(values: np.ndarray, *, radius: int) -> np.ndarray:
    """Return clipped-edge rolling mean for a one-dimensional trace.

    Args:
        values: One-dimensional input trace.
        radius: Number of samples on each side of the center point.

    Returns:
        Rolling mean with the same length as ``values``.
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("values must be 1D")
    if radius < 0:
        raise ValueError("radius must be >= 0")
    if radius == 0 or arr.size == 0:
        return arr.copy()

    indices = np.arange(arr.size)
    starts = np.maximum(0, indices - radius)
    stops = np.minimum(arr.size, indices + radius + 1)
    cumsum = np.concatenate(([0.0], np.cumsum(arr, dtype=float)))
    totals = cumsum[stops] - cumsum[starts]
    counts = stops - starts
    return totals / counts.astype(float)


def _filter_trace(
    values: np.ndarray,
    *,
    filter_method: str,
    median_kernel_points: int,
    warnings: list[str],
) -> np.ndarray:
    """Filter a one-dimensional normalized intensity trace.

    Args:
        values: Trace to filter.
        filter_method: ``"none"`` or ``"median"``.
        median_kernel_points: Median-filter kernel size in points.
        warnings: Mutable list receiving non-fatal warnings.

    Returns:
        Filtered trace.
    """
    if filter_method == "none":
        return np.asarray(values, dtype=float).copy()

    kernel = int(median_kernel_points)
    if kernel <= 1:
        return np.asarray(values, dtype=float).copy()
    if kernel % 2 == 0:
        kernel += 1
        warnings.append(
            f"median_filter_kernel_points was even; using {kernel} points instead"
        )
    return np.asarray(median_filter(values, size=kernel, mode="nearest"), dtype=float)


def _single_exponential(time_sec: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Evaluate ``a * exp(-b * t) + c``.

    Args:
        time_sec: Time axis in seconds.
        a: Exponential amplitude.
        b: Exponential decay rate.
        c: Constant offset.

    Returns:
        Exponential trace.
    """
    return a * np.exp(-b * time_sec) + c


def _detrend_trace(
    values: np.ndarray,
    *,
    time_sec: np.ndarray,
    detrend_method: str,
    warnings: list[str],
    errors: list[str],
) -> np.ndarray:
    """Optionally remove a slow trend from a trace.

    Args:
        values: Input trace after filtering.
        time_sec: Time axis in seconds.
        detrend_method: ``"none"`` or ``"single_exponential"``.
        warnings: Mutable list receiving non-fatal warnings.
        errors: Mutable list receiving algorithm-step failures.

    Returns:
        Detrended trace, or the original values when detrending is disabled or
        expected fitting failure occurs.
    """
    trace = np.asarray(values, dtype=float)
    if detrend_method == "none":
        return trace.copy()
    if trace.size < 4 or np.allclose(trace, trace[0]):
        warnings.append("single_exponential_detrend skipped; trace is too short or constant")
        return trace.copy()

    try:
        amplitude_guess = float(trace[0] - trace[-1])
        decay_guess = 1.0 / max(float(time_sec[-1] - time_sec[0]), np.finfo(float).eps)
        offset_guess = float(trace[-1])
        params, _cov = curve_fit(
            _single_exponential,
            time_sec,
            trace,
            p0=(amplitude_guess, decay_guess, offset_guess),
            maxfev=10_000,
        )
        fitted = _single_exponential(time_sec, *params)
        return trace - fitted + float(fitted[0])
    except (RuntimeError, ValueError, FloatingPointError, OverflowError) as exc:
        message = (
            "single_exponential_detrend failed; using filtered_norm_sum_intensity "
            f"instead ({type(exc).__name__}: {exc})"
        )
        errors.append(message)
        return trace.copy()



def _calculate_df_f_signal(
    values: np.ndarray,
    *,
    baseline_method: str,
    baseline_percentile: float,
    manual_f0_baseline: float,
    baseline_min_value: float,
    warnings: list[str],
    errors: list[str],
) -> tuple[float, np.ndarray]:
    """Calculate a delta-F over F0 trace from a one-dimensional signal.

    Args:
        values: Input trace after filtering and optional detrending.
        baseline_method: Baseline estimator name. Supports ``"percentile"``
            and ``"manual"``.
        baseline_percentile: Percentile used to estimate scalar ``F0`` when
            ``baseline_method`` is ``"percentile"``.
        manual_f0_baseline: User-supplied scalar F0 when ``baseline_method`` is
            ``"manual"``.
        baseline_min_value: Small positive floor used when the estimated
            baseline is zero or too close to zero.
        warnings: Mutable list receiving non-fatal warnings.
        errors: Mutable list receiving algorithm-step failures.

    Returns:
        Tuple of ``(f0_baseline, df_f_signal)`` where ``df_f_signal`` has the
        same shape as ``values``.
    """
    trace = np.asarray(values, dtype=float)
    if trace.size == 0:
        raise ValueError("values must be non-empty")
    if baseline_method not in {"percentile", "manual"}:
        raise ValueError("baseline_method must be 'percentile' or 'manual'")

    finite = trace[np.isfinite(trace)]
    if finite.size == 0:
        errors.append(
            "df_f_baseline failed; trace has no finite values, using zeros for df_f_signal"
        )
        return float(baseline_min_value), np.zeros_like(trace, dtype=float)

    if baseline_method == "manual":
        f0 = float(manual_f0_baseline)
    else:
        f0 = float(np.percentile(finite, baseline_percentile))
    if not np.isfinite(f0):
        errors.append(
            "df_f_baseline failed; percentile baseline was not finite, using baseline_min_value"
        )
        f0 = float(baseline_min_value)
    elif abs(f0) < float(baseline_min_value):
        warnings.append(
            "df_f_baseline was close to zero; using baseline_min_value to avoid division by zero"
        )
        f0 = float(baseline_min_value) if f0 >= 0 else -float(baseline_min_value)

    return f0, (trace - f0) / f0


def _detect_onsets(
    *,
    detection_signal: np.ndarray,
    derivative_signal: np.ndarray,
    method: str,
    polarity: str,
    absolute_threshold: float,
    derivative_threshold_per_sec: float,
) -> list[int]:
    """Detect candidate onset indices before refractory filtering.

    Args:
        detection_signal: Trace used for absolute-threshold detection.
        derivative_signal: Trace derivative used for derivative-threshold
            detection.
        method: Detection method name.
        polarity: ``"positive"`` or ``"negative"``.
        absolute_threshold: Absolute signal threshold.
        derivative_threshold_per_sec: Derivative threshold in signal units per second.

    Returns:
        Candidate onset indices.
    """
    trace = derivative_signal if method == "derivative_threshold" else detection_signal
    threshold = derivative_threshold_per_sec if method == "derivative_threshold" else absolute_threshold
    if polarity == "negative":
        above = trace <= -abs(float(threshold))
    else:
        above = trace >= float(threshold)
    if above.size == 0:
        return []
    starts = np.flatnonzero(above & np.concatenate(([True], ~above[:-1])))
    return [int(index) for index in starts]


def _apply_refractory(indices: list[int], *, refractory_points: int) -> list[int]:
    """Reject candidate onsets inside the onset-to-onset refractory period.

    Args:
        indices: Candidate onset indices in ascending order.
        refractory_points: Minimum accepted onset-to-onset spacing in points.

    Returns:
        Accepted onset indices.
    """
    accepted: list[int] = []
    last_accepted: int | None = None
    for index in indices:
        if last_accepted is None or index > last_accepted + refractory_points:
            accepted.append(index)
            last_accepted = index
    return accepted


def _duration_ms_to_points(duration_ms: float, seconds_per_line: float) -> int:
    """Convert a duration in milliseconds to nearest sample count.

    Args:
        duration_ms: Duration in milliseconds.
        seconds_per_line: Sampling interval in seconds.

    Returns:
        Non-negative sample count.
    """
    return max(0, int(round((duration_ms / 1000.0) / seconds_per_line)))


def _parse_level_fractions(value: str) -> tuple[float, ...]:
    """Parse comma-separated level fractions.

    Args:
        value: Comma-separated fractions such as ``"0.1,0.5,0.9"``.

    Returns:
        Fractions as floats.

    Raises:
        ValueError: If no valid fractions are supplied or a value is outside
            the open interval ``(0, 1)``.
    """
    fractions: list[float] = []
    for part in value.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        fraction = float(stripped)
        if fraction <= 0.0 or fraction >= 1.0:
            raise ValueError("level_fractions values must be between 0 and 1")
        fractions.append(fraction)
    if not fractions:
        raise ValueError("level_fractions must include at least one fraction")
    return tuple(fractions)


def _build_events(
    *,
    detection_signal: np.ndarray,
    derivative_signal: np.ndarray,
    time_sec: np.ndarray,
    onset_indices: list[int],
    detection_method: str,
    polarity: str,
    peak_search_window_points: int,
    width_search_window_points: int,
    baseline_window_points: int,
    level_fractions: tuple[float, ...],
) -> tuple[PeakEvent, ...]:
    """Build event records from accepted onsets.

    Args:
        detection_signal: Trace used for feature extraction.
        derivative_signal: Derivative of ``detection_signal`` in signal units per second.
        time_sec: Time axis in seconds.
        onset_indices: Accepted onset indices.
        detection_method: Detection method used for onset detection.
        polarity: Peak polarity.
        peak_search_window_points: Forward peak search window in points.
        width_search_window_points: Forward width-crossing search window in points.
        baseline_window_points: Backward baseline window before onset in points.
        level_fractions: Requested amplitude fractions for width measures.

    Returns:
        Peak events with interval statistics.
    """
    events: list[PeakEvent] = []
    previous_onset_time: float | None = None
    previous_peak_time: float | None = None
    for zero_based, onset_index in enumerate(onset_indices):
        next_onset = onset_indices[zero_based + 1] if zero_based + 1 < len(onset_indices) else None
        event = _build_one_event(
            detection_signal=detection_signal,
            derivative_signal=derivative_signal,
            time_sec=time_sec,
            peak_id=zero_based + 1,
            onset_index=onset_index,
            next_onset_index=next_onset,
            detection_method=detection_method,
            polarity=polarity,
            peak_search_window_points=peak_search_window_points,
            width_search_window_points=width_search_window_points,
            baseline_window_points=baseline_window_points,
            level_fractions=level_fractions,
            previous_onset_time=previous_onset_time,
            previous_peak_time=previous_peak_time,
        )
        events.append(event)
        previous_onset_time = event.onset_time_sec
        if event.peak_time_sec is not None:
            previous_peak_time = event.peak_time_sec
    return tuple(events)


def _build_one_event(
    *,
    detection_signal: np.ndarray,
    derivative_signal: np.ndarray,
    time_sec: np.ndarray,
    peak_id: int,
    onset_index: int,
    next_onset_index: int | None,
    detection_method: str,
    polarity: str,
    peak_search_window_points: int,
    width_search_window_points: int,
    baseline_window_points: int,
    level_fractions: tuple[float, ...],
    previous_onset_time: float | None,
    previous_peak_time: float | None,
) -> PeakEvent:
    """Build one event record.

    Args:
        detection_signal: Trace used for feature extraction.
        derivative_signal: Derivative of ``detection_signal`` in signal units per second.
        time_sec: Time axis in seconds.
        peak_id: One-based event identifier.
        onset_index: Accepted onset index.
        next_onset_index: Next accepted onset index, if available.
        detection_method: Detection method name.
        polarity: Peak polarity.
        peak_search_window_points: Forward peak search window in points.
        width_search_window_points: Forward width-crossing search window in points.
        baseline_window_points: Backward baseline window before onset in points.
        level_fractions: Requested amplitude fractions.
        previous_onset_time: Previous event onset time for interval stats.
        previous_peak_time: Previous event peak time for interval stats.

    Returns:
        Event record.
    """
    n_time = detection_signal.size
    onset_value = float(detection_signal[onset_index])
    onset_time = float(time_sec[onset_index])
    search_stop = min(n_time, onset_index + max(1, peak_search_window_points) + 1)
    search_values = detection_signal[onset_index:search_stop]
    warnings: list[str] = []

    if search_values.size == 0:
        warnings.append("No samples available for peak refinement")
        peak_index = None
        peak_value = None
        peak_time = None
        amplitude = None
        crossings = tuple(
            LevelCrossing(
                fraction=fraction,
                value=None,
                left_index=None,
                right_index=None,
                width=None,
                width_sec=None,
                status="peak_not_found",
            )
            for fraction in level_fractions
        )
        status = "peak_not_found"
    else:
        local_index = int(np.argmin(search_values) if polarity == "negative" else np.argmax(search_values))
        peak_index = onset_index + local_index
        peak_value = float(detection_signal[peak_index])
        peak_time = float(time_sec[peak_index])
        amplitude = float(onset_value - peak_value if polarity == "negative" else peak_value - onset_value)
        if amplitude <= 0:
            warnings.append("Peak amplitude was not positive after refinement")
            status = "non_positive_amplitude"
        else:
            status = "ok"
        crossings = tuple(
            _measure_level_crossing(
                detection_signal=detection_signal,
                onset_index=onset_index,
                peak_index=peak_index,
                next_onset_index=next_onset_index,
                width_search_window_points=width_search_window_points,
                fraction=fraction,
                onset_value=onset_value,
                peak_value=peak_value,
                polarity=polarity,
                seconds_per_line=_seconds_per_line_from_time_axis(time_sec),
            )
            for fraction in level_fractions
        )

    baseline_mean, baseline_std = _measure_baseline_features(
        detection_signal=detection_signal,
        onset_index=onset_index,
        baseline_window_points=baseline_window_points,
    )
    prominence = _measure_prominence(
        baseline_mean=baseline_mean,
        peak_value=peak_value,
        polarity=polarity,
    )
    rise_10_90 = _measure_rise_10_90(crossings=crossings, time_sec=time_sec)
    decay_90_10 = _measure_decay_90_10(crossings=crossings, time_sec=time_sec)
    decay_time = EventFeature(
        value=decay_90_10.value,
        status=decay_90_10.status,
        reason=decay_90_10.reason,
    )
    max_rise_slope = _measure_max_rise_slope(
        derivative_signal=derivative_signal,
        onset_index=onset_index,
        peak_index=peak_index,
        polarity=polarity,
    )
    max_decay_slope = _measure_max_decay_slope(
        derivative_signal=derivative_signal,
        peak_index=peak_index,
        right_10_index=_right_crossing_index(crossings, 0.1),
        polarity=polarity,
    )
    auc = _measure_auc(
        detection_signal=detection_signal,
        time_sec=time_sec,
        crossings=crossings,
        onset_value=onset_value,
        polarity=polarity,
    )

    onset_interval = None if previous_onset_time is None else onset_time - previous_onset_time
    peak_interval = (
        None
        if previous_peak_time is None or peak_time is None
        else peak_time - previous_peak_time
    )
    return PeakEvent(
        peak_id=peak_id,
        status=status,
        warnings=tuple(warnings),
        onset_index=onset_index,
        onset_time_sec=onset_time,
        onset_value=onset_value,
        peak_index=peak_index,
        peak_time_sec=peak_time,
        peak_value=peak_value,
        peak_amplitude=amplitude,
        detection_method=detection_method,
        baseline_mean=baseline_mean,
        baseline_std=baseline_std,
        rise_10_90_sec=rise_10_90,
        decay_90_10_sec=decay_90_10,
        decay_time_sec=decay_time,
        max_rise_slope=max_rise_slope,
        max_decay_slope=max_decay_slope,
        auc=auc,
        prominence=prominence,
        level_crossings=crossings,
        onset_to_onset_interval_sec=onset_interval,
        peak_to_peak_interval_sec=peak_interval,
    )


def _seconds_per_line_from_time_axis(time_sec: np.ndarray) -> float:
    """Return the uniform time spacing represented by a result time axis.

    Args:
        time_sec: Time axis in seconds.

    Returns:
        Time spacing in seconds per line. Single-sample axes return ``0.0``
        because no width measurement can be valid without two crossings.
    """
    if time_sec.size < 2:
        return 0.0
    return float(time_sec[1] - time_sec[0])


def _measure_baseline_features(
    *,
    detection_signal: np.ndarray,
    onset_index: int,
    baseline_window_points: int,
) -> tuple[EventFeature, EventFeature]:
    """Measure pre-onset baseline mean and standard deviation.

    Args:
        detection_signal: Trace used for event feature extraction.
        onset_index: Accepted event onset index.
        baseline_window_points: Number of points before onset to inspect.

    Returns:
        Tuple of ``(baseline_mean, baseline_std)`` feature records.
    """
    start = max(0, int(onset_index) - max(0, int(baseline_window_points)))
    stop = int(onset_index)
    baseline = np.asarray(detection_signal[start:stop], dtype=float)
    finite = baseline[np.isfinite(baseline)]
    if finite.size == 0:
        failed = EventFeature(
            value=None,
            status="failed",
            reason="insufficient_baseline_samples",
        )
        return failed, failed
    mean = EventFeature(value=float(np.mean(finite)))
    if finite.size < 2:
        std = EventFeature(
            value=None,
            status="failed",
            reason="insufficient_baseline_samples",
        )
    else:
        std = EventFeature(value=float(np.std(finite, ddof=1)))
    return mean, std


def _measure_prominence(
    *,
    baseline_mean: EventFeature,
    peak_value: float | None,
    polarity: str,
) -> EventFeature:
    """Measure peak prominence relative to pre-onset baseline mean.

    Args:
        baseline_mean: Baseline mean feature record.
        peak_value: Refined peak value, or None when unavailable.
        polarity: Peak polarity.
        seconds_per_line: Time spacing in seconds per line.

    Returns:
        Prominence feature record.
    """
    if baseline_mean.status != "ok" or baseline_mean.value is None:
        return EventFeature(value=None, status="failed", reason="baseline_mean_unavailable")
    if peak_value is None:
        return EventFeature(value=None, status="failed", reason="peak_value_unavailable")
    value = float(peak_value) - float(baseline_mean.value)
    if polarity == "negative":
        value = float(baseline_mean.value) - float(peak_value)
    return EventFeature(value=value)


def _measure_rise_10_90(
    *,
    crossings: tuple[LevelCrossing, ...],
    time_sec: np.ndarray,
) -> EventFeature:
    """Measure rise time from 10 percent to 90 percent level crossing.

    Args:
        crossings: Event level-crossing measurements.
        time_sec: Time axis in seconds.

    Returns:
        Rise-time feature record.
    """
    left_10 = _left_crossing_index(crossings, 0.1)
    left_90 = _left_crossing_index(crossings, 0.9)
    if left_10 is None or left_90 is None:
        return EventFeature(value=None, status="failed", reason="left_10_or_left_90_unavailable")
    return EventFeature(value=_time_at_fractional_index(time_sec, left_90) - _time_at_fractional_index(time_sec, left_10))


def _measure_decay_90_10(
    *,
    crossings: tuple[LevelCrossing, ...],
    time_sec: np.ndarray,
) -> EventFeature:
    """Measure decay time from 90 percent to 10 percent level crossing.

    Args:
        crossings: Event level-crossing measurements.
        time_sec: Time axis in seconds.

    Returns:
        Decay-time feature record.
    """
    right_90 = _right_crossing_index(crossings, 0.9)
    right_10 = _right_crossing_index(crossings, 0.1)
    if right_90 is None or right_10 is None:
        return EventFeature(value=None, status="failed", reason="right_90_or_right_10_unavailable")
    return EventFeature(value=_time_at_fractional_index(time_sec, right_10) - _time_at_fractional_index(time_sec, right_90))


def _measure_max_rise_slope(
    *,
    derivative_signal: np.ndarray,
    onset_index: int,
    peak_index: int | None,
    polarity: str,
) -> EventFeature:
    """Measure maximum signed rise slope from onset to peak.

    Args:
        derivative_signal: Derivative of detection signal.
        onset_index: Event onset index.
        peak_index: Refined peak index, or None.
        polarity: Peak polarity.
        seconds_per_line: Time spacing in seconds per line.

    Returns:
        Rise-slope feature record.
    """
    if peak_index is None or peak_index <= onset_index:
        return EventFeature(value=None, status="failed", reason="peak_index_unavailable")
    segment = np.asarray(derivative_signal[onset_index : peak_index + 1], dtype=float)
    finite = segment[np.isfinite(segment)]
    if finite.size == 0:
        return EventFeature(value=None, status="failed", reason="no_finite_derivative_samples")
    value = np.min(finite) if polarity == "negative" else np.max(finite)
    return EventFeature(value=float(value))


def _measure_max_decay_slope(
    *,
    derivative_signal: np.ndarray,
    peak_index: int | None,
    right_10_index: float | None,
    polarity: str,
) -> EventFeature:
    """Measure maximum signed decay slope from peak to right 10 percent crossing.

    Args:
        derivative_signal: Derivative of detection signal.
        peak_index: Refined peak index, or None.
        right_10_index: Interpolated right 10 percent crossing index, or None.
        polarity: Peak polarity.
        seconds_per_line: Time spacing in seconds per line.

    Returns:
        Decay-slope feature record.
    """
    if peak_index is None:
        return EventFeature(value=None, status="failed", reason="peak_index_unavailable")
    if right_10_index is None:
        return EventFeature(value=None, status="failed", reason="right_10_crossing_unavailable")
    stop = min(derivative_signal.size, int(np.ceil(float(right_10_index))) + 1)
    if stop <= peak_index:
        return EventFeature(value=None, status="failed", reason="empty_peak_to_right_10_window")
    segment = np.asarray(derivative_signal[peak_index:stop], dtype=float)
    finite = segment[np.isfinite(segment)]
    if finite.size == 0:
        return EventFeature(value=None, status="failed", reason="no_finite_derivative_samples")
    value = np.max(finite) if polarity == "negative" else np.min(finite)
    return EventFeature(value=float(value))


def _measure_auc(
    *,
    detection_signal: np.ndarray,
    time_sec: np.ndarray,
    crossings: tuple[LevelCrossing, ...],
    onset_value: float,
    polarity: str,
) -> EventFeature:
    """Measure event area between left and right 10 percent crossings.

    Args:
        detection_signal: Trace used for event feature extraction.
        time_sec: Time axis in seconds.
        crossings: Event level-crossing measurements.
        onset_value: Event onset value used as integration baseline.
        polarity: Peak polarity.
        seconds_per_line: Time spacing in seconds per line.

    Returns:
        AUC feature record.
    """
    left_10 = _left_crossing_index(crossings, 0.1)
    right_10 = _right_crossing_index(crossings, 0.1)
    if left_10 is None or right_10 is None:
        return EventFeature(value=None, status="failed", reason="left_10_or_right_10_unavailable")
    if right_10 <= left_10:
        return EventFeature(value=None, status="failed", reason="invalid_10_percent_window")
    x, y = _interpolated_segment(
        values=detection_signal,
        time_sec=time_sec,
        left_index=left_10,
        right_index=right_10,
    )
    if x.size < 2:
        return EventFeature(value=None, status="failed", reason="insufficient_auc_samples")
    if polarity == "negative":
        area_values = np.maximum(float(onset_value) - y, 0.0)
    else:
        area_values = np.maximum(y - float(onset_value), 0.0)
    return EventFeature(value=float(np.trapezoid(area_values, x)))


def _left_crossing_index(crossings: tuple[LevelCrossing, ...], fraction: float) -> float | None:
    """Return left crossing index for one fraction when status is ok.

    Args:
        crossings: Event level-crossing measurements.
        fraction: Fraction to match.

    Returns:
        Interpolated left index, or None.
    """
    crossing = _find_crossing_by_fraction(crossings, fraction)
    if crossing is None or crossing.status != "ok":
        return None
    return crossing.left_index


def _right_crossing_index(crossings: tuple[LevelCrossing, ...], fraction: float) -> float | None:
    """Return right crossing index for one fraction when status is ok.

    Args:
        crossings: Event level-crossing measurements.
        fraction: Fraction to match.

    Returns:
        Interpolated right index, or None.
    """
    crossing = _find_crossing_by_fraction(crossings, fraction)
    if crossing is None or crossing.status != "ok":
        return None
    return crossing.right_index


def _find_crossing_by_fraction(
    crossings: tuple[LevelCrossing, ...],
    fraction: float,
) -> LevelCrossing | None:
    """Return crossing matching a requested fraction.

    Args:
        crossings: Event level-crossing measurements.
        fraction: Fraction to match.

    Returns:
        Matching crossing, or None.
    """
    for crossing in crossings:
        if abs(float(crossing.fraction) - float(fraction)) < 1e-12:
            return crossing
    return None


def _time_at_fractional_index(time_sec: np.ndarray, index: float) -> float:
    """Return interpolated time for a fractional sample index.

    Args:
        time_sec: Time axis in seconds.
        index: Fractional sample index.

    Returns:
        Interpolated time in seconds.
    """
    if time_sec.size == 1:
        return float(time_sec[0])
    clipped = min(max(float(index), 0.0), float(time_sec.size - 1))
    return float(np.interp(clipped, np.arange(time_sec.size, dtype=float), time_sec))


def _interpolated_segment(
    *,
    values: np.ndarray,
    time_sec: np.ndarray,
    left_index: float,
    right_index: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return trace segment with interpolated endpoints.

    Args:
        values: One-dimensional signal.
        time_sec: Time axis in seconds.
        left_index: Fractional left sample index.
        right_index: Fractional right sample index.

    Returns:
        Tuple of ``(time, values)`` arrays including interpolated endpoints.
    """
    left = float(left_index)
    right = float(right_index)
    start = int(np.floor(left)) + 1
    stop = int(np.floor(right)) + 1
    interior_indices = np.arange(start, stop, dtype=float)
    all_indices = np.concatenate(([left], interior_indices, [right]))
    all_indices = np.unique(all_indices)
    x = np.asarray([_time_at_fractional_index(time_sec, idx) for idx in all_indices], dtype=float)
    y = np.interp(all_indices, np.arange(values.size, dtype=float), np.asarray(values, dtype=float))
    return x, y


def _measure_level_crossing(
    *,
    detection_signal: np.ndarray,
    onset_index: int,
    peak_index: int,
    next_onset_index: int | None,
    width_search_window_points: int,
    fraction: float,
    onset_value: float,
    peak_value: float,
    polarity: str,
    seconds_per_line: float,
) -> LevelCrossing:
    """Measure one peak width fraction.

    Args:
        detection_signal: Trace used for feature extraction.
        onset_index: Event onset index.
        peak_index: Event peak index.
        next_onset_index: Next event onset index, if available.
        width_search_window_points: Maximum forward search window for the falling-side crossing.
        fraction: Requested amplitude fraction.
        onset_value: Signal value at onset.
        peak_value: Signal value at peak.
        polarity: Peak polarity.
        seconds_per_line: Time spacing in seconds per line.

    Returns:
        Level-crossing measurement with local failure status when needed.
    """
    level_value = onset_value + float(fraction) * (peak_value - onset_value)
    left = _find_crossing(
        detection_signal,
        start=onset_index,
        stop=peak_index,
        level_value=level_value,
        polarity=polarity,
        direction="rising",
    )
    right_stop = min(detection_signal.size - 1, peak_index + max(1, int(width_search_window_points)))
    if next_onset_index is not None:
        right_stop = min(right_stop, next_onset_index)
    right = _find_crossing(
        detection_signal,
        start=peak_index,
        stop=right_stop,
        level_value=level_value,
        polarity=polarity,
        direction="falling",
    )
    if left is not None and right is not None:
        return LevelCrossing(
            fraction=float(fraction),
            value=float(level_value),
            left_index=float(left),
            right_index=float(right),
            width=float(right - left),
            width_sec=float(right - left) * float(seconds_per_line),
            status="ok",
        )
    if left is None and right is None:
        status = "both_not_found"
    elif left is None:
        status = "left_not_found"
    else:
        status = "right_not_found_within_width_search_window"
    return LevelCrossing(
        fraction=float(fraction),
        value=float(level_value),
        left_index=_optional_float(left),
        right_index=_optional_float(right),
        width=None,
        width_sec=None,
        status=status,
    )


def _find_crossing(
    values: np.ndarray,
    *,
    start: int,
    stop: int,
    level_value: float,
    polarity: str,
    direction: Literal["rising", "falling"],
) -> float | None:
    """Find an interpolated level crossing.

    Args:
        values: One-dimensional trace.
        start: First index to search.
        stop: Last index to search.
        level_value: Requested crossing value.
        polarity: Peak polarity.
        direction: Rising-side or falling-side crossing.

    Returns:
        Interpolated crossing index, or None when unavailable.
    """
    if stop <= start:
        return None
    indices = range(start, stop)
    for i in indices:
        y0 = float(values[i])
        y1 = float(values[i + 1])
        if _segment_crosses(y0, y1, level_value=level_value, polarity=polarity, direction=direction):
            if y1 == y0:
                return float(i)
            return float(i) + (float(level_value) - y0) / (y1 - y0)
    return None


def _segment_crosses(
    y0: float,
    y1: float,
    *,
    level_value: float,
    polarity: str,
    direction: Literal["rising", "falling"],
) -> bool:
    """Return whether one segment crosses a requested level.

    Args:
        y0: Segment start value.
        y1: Segment end value.
        level_value: Requested crossing value.
        polarity: Peak polarity.
        direction: Rising-side or falling-side crossing.

    Returns:
        True when the segment crosses the level in the expected direction.
    """
    if polarity == "negative":
        if direction == "rising":
            return y0 >= level_value and y1 <= level_value
        return y0 <= level_value and y1 >= level_value
    if direction == "rising":
        return y0 <= level_value and y1 >= level_value
    return y0 >= level_value and y1 <= level_value


def _build_table(
    *,
    time_index: np.ndarray,
    time_sec: np.ndarray,
    sum_intensity: np.ndarray,
    norm_sum_intensity: np.ndarray,
    filtered_norm_sum_intensity: np.ndarray,
    detrended_norm_sum_intensity: np.ndarray,
    f0_baseline: float,
    df_f_signal: np.ndarray,
    d_df_f_signal: np.ndarray,
    events: tuple[PeakEvent, ...],
) -> pd.DataFrame:
    """Build the per-timepoint result table.

    Args:
        time_index: Integer time indices.
        time_sec: Time in seconds.
        sum_intensity: Raw or windowed row-sum intensity.
        norm_sum_intensity: Sum intensity divided by spatial pixel count.
        filtered_norm_sum_intensity: Filtered normalized trace.
        detrended_norm_sum_intensity: Detrended normalized trace used for df/f0.
        f0_baseline: Scalar F0 baseline used for df/f0.
        df_f_signal: Delta-F over F0 trace.
        d_df_f_signal: Derivative of ``df_f_signal`` in ``1 / second``.
        events: Detected peak events.

    Returns:
        Result table with stable columns.
    """
    table = pd.DataFrame(
        {
            "time_index": time_index.astype(int),
            "time_sec": time_sec.astype(float),
            "sum_intensity": sum_intensity.astype(float),
            "norm_sum_intensity": norm_sum_intensity.astype(float),
            "filtered_norm_sum_intensity": filtered_norm_sum_intensity.astype(float),
            "detrended_norm_sum_intensity": detrended_norm_sum_intensity.astype(float),
            "f0_baseline": np.full(time_index.size, float(f0_baseline)),
            "df_f_signal": df_f_signal.astype(float),
            "d_df_f_signal": d_df_f_signal.astype(float),
            "is_onset": np.zeros(time_index.size, dtype=bool),
            "is_peak": np.zeros(time_index.size, dtype=bool),
            "onset_index": np.full(time_index.size, np.nan),
            "peak_id": np.full(time_index.size, np.nan),
        },
        columns=SUM_INTENSITY_TABLE_COLUMNS,
    )
    for event in events:
        table.loc[event.onset_index, "is_onset"] = True
        table.loc[event.onset_index, "onset_index"] = int(event.onset_index)
        table.loc[event.onset_index, "peak_id"] = int(event.peak_id)
        if event.peak_index is not None:
            table.loc[event.peak_index, "is_peak"] = True
            table.loc[event.peak_index, "onset_index"] = int(event.onset_index)
            table.loc[event.peak_index, "peak_id"] = int(event.peak_id)
    return table

def _build_summary(
    *,
    table: pd.DataFrame,
    events: tuple[PeakEvent, ...],
    warnings: list[str],
    errors: list[str],
    params: dict[str, Any],
    seconds_per_line: float,
    n_space: int,
    f0_baseline: float,
) -> dict[str, Any]:
    """Build the JSON summary for one analysis run.

    Args:
        table: Per-timepoint result table.
        events: Detected peak events.
        warnings: Analysis-level warnings.
        errors: Analysis-level non-fatal errors.
        params: Validated detection parameters.
        seconds_per_line: Time spacing in seconds.
        n_space: Number of spatial pixels in the ROI image.
        f0_baseline: Scalar F0 baseline used to calculate df/f0.

    Returns:
        JSON-serializable summary dictionary.
    """
    amplitudes = [event.peak_amplitude for event in events if event.peak_amplitude is not None]
    status = "ok"
    if errors:
        status = "ok_with_errors"
    elif warnings or any(event.status != "ok" for event in events):
        status = "ok_with_warnings"
    return {
        "status": status,
        "num_timepoints": int(len(table)),
        "num_peaks": int(len(events)),
        "num_space_pixels": int(n_space),
        "seconds_per_line": float(seconds_per_line),
        "f0_baseline": float(f0_baseline),
        "baseline_method": str(params["baseline_method"]),
        "baseline_percentile": float(params["baseline_percentile"]),
        "manual_f0_baseline": float(params["manual_f0_baseline"]),
        "peak_amplitude_mean": _nanmean_or_none(amplitudes),
        "peak_amplitude_median": _nanmedian_or_none(amplitudes),
        "warnings": list(warnings),
        "errors": list(errors),
        "detrend_method": str(params["detrend_method"]),
        "detection_method": str(params["detection_method"]),
        "detection_source": str(params["detection_source"]),
        "peak_search_window_ms": float(params["peak_search_window_ms"]),
        "width_search_window_ms": float(params["width_search_window_ms"]),
        "baseline_window_ms": float(params["baseline_window_ms"]),
        "peak_events": [event.to_json_dict() for event in events],
    }


def _nanmean_or_none(values: list[float]) -> float | None:
    """Return mean or None for empty input.

    Args:
        values: Numeric values.

    Returns:
        Mean, or None when no values exist.
    """
    if not values:
        return None
    return float(np.nanmean(np.asarray(values, dtype=float)))


def _nanmedian_or_none(values: list[float]) -> float | None:
    """Return median or None for empty input.

    Args:
        values: Numeric values.

    Returns:
        Median, or None when no values exist.
    """
    if not values:
        return None
    return float(np.nanmedian(np.asarray(values, dtype=float)))


def _optional_float(value: object) -> float | None:
    """Return a float or None.

    Args:
        value: Numeric value or None-like value.

    Returns:
        Float value, or None when input is None or NaN.
    """
    if value is None:
        return None
    numeric = float(value)
    if np.isnan(numeric):
        return None
    return numeric


def _optional_event_array(events: tuple[PeakEvent, ...], attr: str) -> np.ndarray:
    """Return event attribute values as a float array with missing values removed.

    Args:
        events: Peak events.
        attr: Event attribute name.

    Returns:
        NumPy float array containing finite event values.
    """
    values: list[float] = []
    for event in events:
        value = getattr(event, attr)
        if value is None:
            continue
        values.append(float(value))
    return np.asarray(values, dtype=float)


def _find_event_crossing(event: PeakEvent, fraction: float) -> LevelCrossing | None:
    """Return the crossing matching a requested fraction.

    Args:
        event: Peak event to inspect.
        fraction: Fraction to match.

    Returns:
        Matching level crossing, or None.
    """
    for crossing in event.level_crossings:
        if abs(float(crossing.fraction) - float(fraction)) < 1e-12:
            return crossing
    return None
