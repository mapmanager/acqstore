"""Public BaseAnalysis wrapper for sum-intensity peak detection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from acqstore.acq_image.analysis.data_provider import AnalysisDataProvider
from acqstore.acq_image.analysis.model import (
    AnalysisPlotData,
    AnalysisResult,
    AnalysisRunContext,
    BaseAnalysis,
    DetectionParamCategory,
    DetectionParamSchema,
    DetectionValueType,
)
from acqstore.acq_image.analysis.registry import register_analysis_class
from acqstore.acq_image.analysis.sum_intensity_analysis.sum_intensity_core import (
    EventFeature,
    LevelCrossing,
    PeakEvent,
    PeakWidthLevel,
    ResultPoints,
    ResultTrace,
    SumIntensityEventPointKey,
    SumIntensitySummaryKey,
    SumIntensityTraceKey,
    WIDTH_LEVEL_FRACTIONS,
    run_sum_intensity,
)
from acqstore.acq_image.analysis.sum_intensity_analysis.sum_intensity_features import (
    SumIntensityFeatureSchema,
    get_sum_intensity_feature_schema,
    get_sum_intensity_feature_schema_dataframe,
)
from acqstore.acq_image.analysis.sum_intensity_analysis.sum_intensity_presets import (
    SumIntensityDetectionPreset,
    SumIntensityPresetName,
    get_sum_intensity_detection_preset,
    get_sum_intensity_detection_preset_params,
    list_sum_intensity_detection_presets,
)


@register_analysis_class
class SumIntensityAnalysis(BaseAnalysis):
    """Measure normalized line intensity and detect peaks/events.

    The analysis runs on a full-resolution rectangular ROI crop. Rows are
    interpreted as time and columns as distance. The raw row-sum trace is stored
    for debugging, but peak detection uses normalized intensity
    ``sum_intensity / image.shape[1]`` so ROIs with different spatial widths are
    more comparable.

    Args:
        channel: Zero-based channel index for analysis.
        roi_id: ROI identifier for analysis.
        detection_params: Optional detection parameters. Missing values are
            filled from ``detection_schema`` defaults.
    """

    analysis_name = "sum_intensity"
    analysis_version = 1
    summary_columns = (
        "analysis_date",
        "analysis_time",
        "analysis_version",
        "status",
        "num_timepoints",
        "num_peaks",
        "num_space_pixels",
        "seconds_per_line",
        "f0_baseline",
        "baseline_method",
        "baseline_percentile",
        "manual_f0_baseline",
        "detrend_method",
        "detection_method",
        "detection_source",
        "peak_search_window_ms",
        "width_search_window_ms",
        "baseline_window_ms",
        "peak_amplitude_mean",
        "peak_amplitude_median",
        "errors",
    )
    detection_schema = (
        DetectionParamSchema(
            name="window_radius_points",
            display_name="Window Radius",
            value_type=DetectionValueType.INT,
            default=6,
            unit="points",
            description="Radius around each time row used for rolling row-sum averaging.",
            category=DetectionParamCategory.PREPROCESSING,
        ),
        DetectionParamSchema(
            name="filter_method",
            display_name="Filter Method",
            value_type=DetectionValueType.ENUM,
            default="median",
            choices=("none", "median"),
            description="Optional pre-detection trace filter applied to normalized intensity.",
            category=DetectionParamCategory.PREPROCESSING,
        ),
        DetectionParamSchema(
            name="median_filter_kernel_points",
            display_name="Median Filter Kernel",
            value_type=DetectionValueType.INT,
            default=3,
            unit="points",
            description="Median filter kernel size in time points. Even values are rounded up.",
            methods=("median",),
            category=DetectionParamCategory.PREPROCESSING,
        ),
        DetectionParamSchema(
            name="detrend_method",
            display_name="Detrend Method",
            value_type=DetectionValueType.ENUM,
            default="single_exponential",
            choices=("none", "single_exponential"),
            description="Optional bleach-trend removal before detection.",
            category=DetectionParamCategory.PREPROCESSING,
        ),
        DetectionParamSchema(
            name="baseline_method",
            display_name="F0 Baseline Method",
            value_type=DetectionValueType.ENUM,
            default="percentile",
            choices=("percentile", "manual"),
            description="Method used to estimate scalar F0 for delta-F over F0.",
            category=DetectionParamCategory.PREPROCESSING,
        ),
        DetectionParamSchema(
            name="baseline_percentile",
            display_name="F0 Baseline Percentile",
            value_type=DetectionValueType.FLOAT,
            default=20.0,
            unit="percentile",
            description="Percentile of the filtered and detrended trace used as F0.",
            methods=("percentile",),
            category=DetectionParamCategory.PREPROCESSING,
        ),
        DetectionParamSchema(
            name="manual_f0_baseline",
            display_name="Manual F0 Baseline",
            value_type=DetectionValueType.FLOAT,
            default=1.0,
            description=(
                "User-supplied scalar F0 used when baseline_method is manual. "
                "Units are the same as the filtered/detrended normalized intensity trace."
            ),
            methods=("manual",),
            category=DetectionParamCategory.PREPROCESSING,
        ),
        DetectionParamSchema(
            name="baseline_window_ms",
            display_name="Baseline Window",
            value_type=DetectionValueType.FLOAT,
            default=100.0,
            unit="ms",
            description=(
                "Backward window before each detected onset used to calculate "
                "event-local baseline_mean, baseline_std, and prominence."
            ),
            category=DetectionParamCategory.PREPROCESSING,
        ),
        DetectionParamSchema(
            name="baseline_min_value",
            display_name="F0 Minimum Value",
            value_type=DetectionValueType.FLOAT,
            default=1e-12,
            description="Small positive floor used to avoid division by zero in df/f0.",
            visible=False,
            category=DetectionParamCategory.PREPROCESSING,
        ),
        DetectionParamSchema(
            name="detection_method",
            display_name="Detection Method",
            value_type=DetectionValueType.ENUM,
            default="derivative_threshold",
            choices=("derivative_threshold", "absolute_threshold"),
            description="Peak onset detector.",
            category=DetectionParamCategory.PEAK_DETECTION,
        ),
        DetectionParamSchema(
            name="polarity",
            display_name="Polarity",
            value_type=DetectionValueType.ENUM,
            default="positive",
            choices=("positive", "negative"),
            description="Expected peak polarity in the detection signal.",
            category=DetectionParamCategory.PEAK_DETECTION,
        ),
        DetectionParamSchema(
            name="detection_source",
            display_name="Detection Source",
            value_type=DetectionValueType.ENUM,
            default=SumIntensityTraceKey.DF_F_SIGNAL.value,
            choices=(
                SumIntensityTraceKey.SUM_INTENSITY.value,
                SumIntensityTraceKey.NORM_SUM_INTENSITY.value,
                SumIntensityTraceKey.FILTERED_NORM_SUM_INTENSITY.value,
                SumIntensityTraceKey.DETRENDED_NORM_SUM_INTENSITY.value,
                SumIntensityTraceKey.DF_F_SIGNAL.value,
            ),
            description=(
                "Continuous trace used for onset detection. Derivative-threshold "
                "detection uses the time derivative of this selected trace."
            ),
            category=DetectionParamCategory.PEAK_DETECTION,
        ),
        DetectionParamSchema(
            name="absolute_threshold",
            display_name="Absolute Threshold",
            value_type=DetectionValueType.FLOAT,
            default=0.0,
            description="Detection-signal threshold used by absolute_threshold detection.",
            methods=("absolute_threshold",),
            category=DetectionParamCategory.PEAK_DETECTION,
        ),
        DetectionParamSchema(
            name="derivative_threshold_per_sec",
            display_name="Derivative Threshold",
            value_type=DetectionValueType.FLOAT,
            default=3.0,
            unit="1/s",
            description="Derivative threshold in selected detection-source units per second.",
            methods=("derivative_threshold",),
            category=DetectionParamCategory.PEAK_DETECTION,
        ),
        DetectionParamSchema(
            name="refractory_period_ms",
            display_name="Refractory Period",
            value_type=DetectionValueType.FLOAT,
            default=10.0,
            unit="ms",
            description="Minimum accepted onset-to-onset interval.",
            category=DetectionParamCategory.PEAK_DETECTION,
        ),
        DetectionParamSchema(
            name="peak_search_window_ms",
            display_name="Peak Search Window",
            value_type=DetectionValueType.FLOAT,
            default=50.0,
            unit="ms",
            description="Forward search window used to refine peak index after onset.",
            category=DetectionParamCategory.PEAK_DETECTION,
        ),
        DetectionParamSchema(
            name="width_search_window_ms",
            display_name="Width Search Window",
            value_type=DetectionValueType.FLOAT,
            default=750.0,
            unit="ms",
            description=(
                "Maximum forward search from the refined peak to find falling-side "
                "fractional width crossings. Missing crossings within this window "
                "are stored as level-crossing failures."
            ),
            category=DetectionParamCategory.PEAK_DETECTION,
        ),
        DetectionParamSchema(
            name="level_fractions",
            display_name="Level Fractions",
            value_type=DetectionValueType.STR,
            default="0.1,0.2,0.5,0.8,0.9",
            description="Comma-separated peak-amplitude fractions for width measurements.",
            visible=False,
            category=DetectionParamCategory.PEAK_DETECTION,
        ),
    )

    @classmethod
    def get_detection_presets(cls) -> tuple[SumIntensityDetectionPreset, ...]:
        """Return built-in detection presets for this analysis type.

        Returns:
            Tuple of immutable preset descriptors in stable UI order.
        """
        return list_sum_intensity_detection_presets()

    @classmethod
    def get_detection_preset(
        cls,
        name: SumIntensityPresetName | str,
    ) -> SumIntensityDetectionPreset:
        """Return one built-in detection preset.

        Args:
            name: Preset enum value or its string value.

        Returns:
            Matching preset descriptor.

        Raises:
            KeyError: If ``name`` is not a built-in preset.
        """
        return get_sum_intensity_detection_preset(name)

    @classmethod
    def get_detection_preset_params(
        cls,
        name: SumIntensityPresetName | str,
    ) -> dict[str, object]:
        """Return a copied detection-parameter mapping for one preset.

        Args:
            name: Preset enum value or its string value.

        Returns:
            Complete detection-parameter dictionary suitable for constructing or
            updating a ``SumIntensityAnalysis`` instance.

        Raises:
            KeyError: If ``name`` is not a built-in preset.
        """
        params = get_sum_intensity_detection_preset_params(name)
        cls.validate_detection_params(params)
        return params

    @classmethod
    def get_feature_schema(cls) -> tuple[SumIntensityFeatureSchema, ...]:
        """Return event-level result feature schema entries.

        Returns:
            Tuple of feature schema records in stable report order.
        """
        return get_sum_intensity_feature_schema()

    @classmethod
    def get_feature_schema_dataframe(cls) -> pd.DataFrame:
        """Return event-level result feature schema as a DataFrame.

        Returns:
            DataFrame with one row per documented event-level feature.
        """
        return get_sum_intensity_feature_schema_dataframe()

    def __init__(
        self,
        *,
        channel: int,
        roi_id: int,
        detection_params: dict[str, object] | None = None,
    ) -> None:
        """Create a sum-intensity analysis instance.

        Args:
            channel: Channel index for analysis.
            roi_id: ROI identifier for analysis.
            detection_params: Optional detection parameters.
        """
        super().__init__(channel=channel, roi_id=roi_id, detection_params=detection_params)

    def run(
        self,
        data_provider: AnalysisDataProvider,
        *,
        context: AnalysisRunContext | None = None,
        dependencies: dict[str, BaseAnalysis] | None = None,
    ) -> AnalysisResult:
        """Run sum-intensity peak detection on one ROI crop.

        Args:
            data_provider: Provider for ROI image data and physical spacing.
                ``get_roi_image`` must return a 2D ``(time, space)`` array.
            context: Optional progress/cancellation context.
            dependencies: Unused for sum-intensity analysis.

        Returns:
            Populated analysis result. The result table includes normalized
            intensity, detection signal, onset mask, and peak mask columns.
        """
        _ = dependencies
        context = context or AnalysisRunContext()
        context.raise_if_cancelled()
        context.report_progress(0.0, "Loading ROI image")
        image = data_provider.get_roi_image(channel=self.key.channel, roi_id=self.key.roi_id)
        physical_units = data_provider.get_image_physical_units()
        context.report_progress(0.25, "Running sum intensity analysis")
        result = run_sum_intensity(
            image,
            detection_params=self.detection_params,
            physical_units=physical_units,
        )
        context.raise_if_cancelled()
        self.result.summary = self.finalize_summary(result.summary)
        self.result.table = result.table
        self.set_dirty()
        context.report_progress(1.0, "Sum intensity analysis complete")
        return self.result

    @classmethod
    def get_pool_summary_columns(cls) -> tuple[str, ...]:
        """Return scalar summary columns for sum-intensity pool tables.

        This additive pool-facing API keeps the existing rich summary and JSON
        APIs unchanged while exposing table-safe scalar columns. List-like
        summary values are flattened for collection-level DataFrame caches.

        Returns:
            Tuple of scalar summary column names in stable order.
        """
        columns: list[str] = []
        for column in cls.get_summary_columns():
            if column == "errors":
                columns.extend(["error_count", "errors_text"])
            else:
                columns.append(column)
        return tuple(columns)

    def get_pool_summary_values(self) -> dict[str, object]:
        """Return scalar summary values for sum-intensity pool tables.

        Returns:
            Mapping whose keys match :meth:`get_pool_summary_columns`. Values
            are scalar objects suitable for pandas table cells.
        """
        values: dict[str, object] = {}
        for column in self.get_summary_columns():
            if column == "errors":
                errors = self.result.summary.get("errors", ())
                if errors is pd.NA or errors is None:
                    error_values: tuple[str, ...] = ()
                elif isinstance(errors, (list, tuple)):
                    error_values = tuple(str(item) for item in errors)
                else:
                    error_values = (str(errors),)
                values["error_count"] = len(error_values)
                values["errors_text"] = "; ".join(error_values)
            else:
                values[column] = self.result.summary.get(column, pd.NA)
        return values

    @classmethod
    def get_pool_peak_columns(cls) -> tuple[str, ...]:
        """Return scalar peak-row columns for sum-intensity pool tables.

        Returns:
            Tuple of flattened peak-event column names. Feature columns are
            generated from :meth:`get_feature_schema` so future event-level
            features propagate to the pool API without pool changes.
        """
        base_columns = (
            "peak_id",
            "peak_status",
            "peak_warning_count",
            "peak_warnings_text",
            "onset_index",
            "onset_time_sec",
            "onset_value",
            "peak_index",
            "peak_time_sec",
            "peak_value",
            "peak_amplitude",
            "peak_detection_method",
            "onset_to_onset_interval_sec",
            "peak_to_peak_interval_sec",
        )
        feature_columns: list[str] = []
        for schema in cls.get_feature_schema():
            feature_columns.extend(
                [
                    schema.name,
                    f"{schema.name}_status",
                    f"{schema.name}_reason",
                ]
            )
        width_columns: list[str] = []
        for level in PeakWidthLevel:
            prefix = level.value
            width_columns.extend(
                [
                    f"{prefix}_value",
                    f"{prefix}_left_index",
                    f"{prefix}_right_index",
                    f"{prefix}_points",
                    f"{prefix}_sec",
                    f"{prefix}_status",
                ]
            )
        return base_columns + tuple(feature_columns) + tuple(width_columns)

    def get_pool_peak_rows(self) -> tuple[dict[str, object], ...]:
        """Return flattened scalar rows for detected sum-intensity peaks.

        Returns:
            Tuple of dictionaries, one per detected peak. Empty when the
            analysis has no peak events. Each dictionary has exactly the keys
            returned by :meth:`get_pool_peak_columns`.
        """
        return tuple(self._pool_peak_row(event) for event in self.get_peak_events())

    def get_peak_events(self) -> tuple[PeakEvent, ...]:
        """Return parsed peak-event records from the result summary.

        Returns:
            Tuple of peak events. Empty when analysis has not been run or when
            no peaks were detected.
        """
        records = self.result.summary.get("peak_events", ())
        return tuple(PeakEvent.from_json_dict(dict(record)) for record in records)

    @classmethod
    def _pool_peak_row(cls, event: PeakEvent) -> dict[str, object]:
        row: dict[str, object] = {
            "peak_id": int(event.peak_id),
            "peak_status": str(event.status),
            "peak_warning_count": len(event.warnings),
            "peak_warnings_text": "; ".join(str(item) for item in event.warnings),
            "onset_index": int(event.onset_index),
            "onset_time_sec": float(event.onset_time_sec),
            "onset_value": float(event.onset_value),
            "peak_index": pd.NA if event.peak_index is None else int(event.peak_index),
            "peak_time_sec": _pool_optional_float(event.peak_time_sec),
            "peak_value": _pool_optional_float(event.peak_value),
            "peak_amplitude": _pool_optional_float(event.peak_amplitude),
            "peak_detection_method": str(event.detection_method),
            "onset_to_onset_interval_sec": _pool_optional_float(
                event.onset_to_onset_interval_sec
            ),
            "peak_to_peak_interval_sec": _pool_optional_float(
                event.peak_to_peak_interval_sec
            ),
        }
        for schema in cls.get_feature_schema():
            feature = getattr(event, schema.name)
            row.update(_pool_feature_values(schema.name, feature))
        crossings_by_prefix = {
            _pool_width_prefix(crossing): crossing for crossing in event.level_crossings
        }
        for level in PeakWidthLevel:
            row.update(_pool_width_values(level.value, crossings_by_prefix.get(level.value)))
        return {column: row.get(column, pd.NA) for column in cls.get_pool_peak_columns()}

    def get_trace(self, key: SumIntensityTraceKey) -> ResultTrace:
        """Return one named continuous trace.

        Args:
            key: Trace key to retrieve.

        Returns:
            Result trace with ``time_sec`` as x values.

        Raises:
            ValueError: If analysis has no result table.
            KeyError: If the requested trace column is missing.
        """
        if self.result.table is None:
            raise ValueError("analysis has no result table")
        table = self.result.table
        if "time_sec" not in table.columns:
            raise KeyError("Sum intensity trace requires 'time_sec' column")
        if key.value not in table.columns:
            raise KeyError(f"Sum intensity trace column is missing: {key.value!r}")
        definition = _TRACE_DEFINITIONS[key]
        return ResultTrace(
            key=key,
            name=definition["display_name"],
            x=table["time_sec"].to_numpy(dtype=float),
            y=table[key.value].to_numpy(dtype=float),
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

        Raises:
            KeyError: If a requested width level is unknown.
        """
        if peak_width_level is None:
            return tuple(self.get_width_trace(level) for level in PeakWidthLevel)
        fraction = WIDTH_LEVEL_FRACTIONS[peak_width_level]
        x_values: list[float] = []
        y_values: list[float] = []
        for event in self.get_peak_events():
            crossing = _find_event_crossing(event, fraction)
            if crossing is None or crossing.status != "ok":
                continue
            if crossing.left_index is None or crossing.right_index is None or crossing.value is None:
                continue
            seconds_per_line = float(
                self.result.summary.get(SumIntensitySummaryKey.SECONDS_PER_LINE.value, 1.0)
            )
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
        return self.result.summary.get(key.value)

    def get_percentile_f0_baseline(self, percentile: float | None = None) -> float:
        """Return the percentile-estimated F0 from the stored detrended trace.

        This recomputes scalar F0 with the same signal and flooring rules used by
        ``baseline_method="percentile"``, even when the last run used manual F0.
        It does not mutate detection params or re-run analysis.

        Args:
            percentile: Percentile in ``[0, 100]``. When ``None``, uses the
                analysis ``baseline_percentile`` from the result summary when
                present, otherwise from ``detection_params``.

        Returns:
            Scalar F0 in detrended normalized intensity units.

        Raises:
            ValueError: If analysis has no result table, percentile is out of
                range, or the detrended trace has no finite values.
            KeyError: If the detrended trace column is missing.
        """
        if self.result.table is None:
            raise ValueError("analysis has no result table")
        if percentile is None:
            summary_percentile = self.result.summary.get(
                SumIntensitySummaryKey.BASELINE_PERCENTILE.value
            )
            if summary_percentile is None:
                percentile = float(self.detection_params["baseline_percentile"])
            else:
                percentile = float(summary_percentile)
        percentile = float(percentile)
        if percentile < 0.0 or percentile > 100.0:
            raise ValueError("percentile must be between 0 and 100")
        trace = self.get_trace(SumIntensityTraceKey.DETRENDED_NORM_SUM_INTENSITY)
        values = np.asarray(trace.y, dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError("detrended trace has no finite values")
        f0 = float(np.percentile(finite, percentile))
        baseline_min_value = float(self.detection_params["baseline_min_value"])
        if not np.isfinite(f0):
            return float(baseline_min_value)
        if abs(f0) < baseline_min_value:
            return float(baseline_min_value) if f0 >= 0 else -float(baseline_min_value)
        return f0

    def get_plot_data(self) -> AnalysisPlotData | None:
        """Return canonical df/f0 plot data.

        Returns:
            Plot data using ``time_sec`` for the x axis and ``df_f_signal`` for
            the y axis. Returns None when the analysis has no table output yet.

        Raises:
            KeyError: If the table is present but missing required columns.
        """
        if self.result.table is None:
            return None
        trace = self.get_trace(SumIntensityTraceKey.DF_F_SIGNAL)
        return AnalysisPlotData(
            x=tuple(float(value) for value in trace.x.tolist()),
            y=tuple(float(value) for value in trace.y.tolist()),
            x_label=trace.x_label,
            y_label=trace.y_label,
            series_name=trace.name,
        )


def _pool_optional_float(value: object) -> object:
    """Return a scalar float or ``pandas.NA`` for pool table cells.

    Args:
        value: Optional numeric value.

    Returns:
        Float value when present, otherwise ``pandas.NA``.
    """
    if value is None:
        return pd.NA
    return float(value)


def _pool_feature_values(prefix: str, feature: EventFeature) -> dict[str, object]:
    """Return scalar pool cells for one event feature.

    Args:
        prefix: Feature name used as the value column prefix.
        feature: Event feature to flatten.

    Returns:
        Mapping with value, status, and reason cells.
    """
    return {
        prefix: _pool_optional_float(feature.value),
        f"{prefix}_status": str(feature.status),
        f"{prefix}_reason": str(feature.reason),
    }


def _pool_width_prefix(crossing: LevelCrossing) -> str:
    """Return the pool column prefix for one level crossing.

    Args:
        crossing: Level-crossing measurement.

    Returns:
        Prefix such as ``"width_50"``.
    """
    percent = int(round(float(crossing.fraction) * 100.0))
    return f"width_{percent}"


def _pool_width_values(prefix: str, crossing: LevelCrossing | None) -> dict[str, object]:
    """Return scalar pool cells for one peak width level.

    Args:
        prefix: Width prefix such as ``"width_50"``.
        crossing: Level-crossing measurement, or None when missing.

    Returns:
        Mapping with scalar width cells.
    """
    if crossing is None:
        return {
            f"{prefix}_value": pd.NA,
            f"{prefix}_left_index": pd.NA,
            f"{prefix}_right_index": pd.NA,
            f"{prefix}_points": pd.NA,
            f"{prefix}_sec": pd.NA,
            f"{prefix}_status": "missing",
        }
    return {
        f"{prefix}_value": _pool_optional_float(crossing.value),
        f"{prefix}_left_index": _pool_optional_float(crossing.left_index),
        f"{prefix}_right_index": _pool_optional_float(crossing.right_index),
        f"{prefix}_points": _pool_optional_float(crossing.width),
        f"{prefix}_sec": _pool_optional_float(crossing.width_sec),
        f"{prefix}_status": str(crossing.status),
    }


_TRACE_DEFINITIONS: dict[SumIntensityTraceKey, dict[str, str]] = {
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


def _find_event_crossing(event: PeakEvent, fraction: float):
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
