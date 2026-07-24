"""Event-level feature schema for sum-intensity analysis.

This module documents scalar features measured for each detected
sum-intensity peak event. Continuous traces such as ``df_f_signal`` are
intentionally documented separately by trace definitions in
``sum_intensity_core``. The feature schema is for event-level measurements such
as rise time, decay time, prominence, and AUC that appear on ``PeakEvent``
records and future event-level result tables.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum

import pandas as pd


class SumIntensityFeatureCategory(StrEnum):
    """Categories for event-level sum-intensity result features.

    Members:
        BASELINE: Baseline measurements made before event onset.
        AMPLITUDE: Amplitude-like measurements relative to baseline or onset.
        KINETICS: Time-domain event shape measurements.
        SLOPE: Derivative-based event shape measurements.
        AREA: Integrated event area measurements.
    """

    BASELINE = "Baseline"
    AMPLITUDE = "Amplitude"
    KINETICS = "Kinetics"
    SLOPE = "Slope"
    AREA = "Area"


@dataclass(frozen=True, slots=True)
class SumIntensityFeatureSchema:
    """Schema entry for one event-level sum-intensity result feature.

    Args:
        name: Stable feature name used in JSON and event-feature records.
        display_name: Human-readable feature label for reports and GUIs.
        value_type: Human-readable value type, for example ``"float"``.
        unit: Feature unit, or None when dimensionless.
        description: Scientific meaning of the feature.
        algorithm: Human-readable calculation description.
        category: Scientific feature category.
    """

    name: str
    display_name: str
    value_type: str
    unit: str | None
    description: str
    algorithm: str
    category: SumIntensityFeatureCategory

    def to_dict(self) -> dict[str, object]:
        """Return a plain dictionary representation.

        Returns:
            JSON-serializable feature schema dictionary.
        """
        record = asdict(self)
        record["category"] = self.category.value
        return record


SUM_INTENSITY_FEATURE_SCHEMA: tuple[SumIntensityFeatureSchema, ...] = (
    SumIntensityFeatureSchema(
        name="baseline_mean",
        display_name="Baseline Mean",
        value_type="float",
        unit="detection source",
        description="Mean detection-source value immediately before event onset.",
        algorithm="Mean of detection_source samples in [onset - baseline_window_ms, onset).",
        category=SumIntensityFeatureCategory.BASELINE,
    ),
    SumIntensityFeatureSchema(
        name="baseline_std",
        display_name="Baseline Std",
        value_type="float",
        unit="detection source",
        description="Standard deviation of detection-source values immediately before event onset.",
        algorithm="Sample standard deviation of detection_source samples in [onset - baseline_window_ms, onset).",
        category=SumIntensityFeatureCategory.BASELINE,
    ),
    SumIntensityFeatureSchema(
        name="prominence",
        display_name="Prominence",
        value_type="float",
        unit="detection source",
        description="Peak amplitude relative to the pre-onset baseline mean.",
        algorithm="For positive peaks, peak_value - baseline_mean. For negative peaks, baseline_mean - peak_value.",
        category=SumIntensityFeatureCategory.AMPLITUDE,
    ),
    SumIntensityFeatureSchema(
        name="rise_10_90_sec",
        display_name="Rise 10→90",
        value_type="float",
        unit="s",
        description="Rising-phase time from 10% to 90% of peak amplitude.",
        algorithm="left_90_time_sec - left_10_time_sec using fractional level crossings.",
        category=SumIntensityFeatureCategory.KINETICS,
    ),
    SumIntensityFeatureSchema(
        name="decay_90_10_sec",
        display_name="Decay 90→10",
        value_type="float",
        unit="s",
        description="Falling-phase time from 90% to 10% of peak amplitude.",
        algorithm="right_10_time_sec - right_90_time_sec using fractional level crossings.",
        category=SumIntensityFeatureCategory.KINETICS,
    ),
    SumIntensityFeatureSchema(
        name="decay_time_sec",
        display_name="Decay Time",
        value_type="float",
        unit="s",
        description="Default decay-time measurement for the event.",
        algorithm="Alias of decay_90_10_sec for the first implementation.",
        category=SumIntensityFeatureCategory.KINETICS,
    ),
    SumIntensityFeatureSchema(
        name="max_rise_slope",
        display_name="Max Rise Slope",
        value_type="float",
        unit="detection source/s",
        description="Maximum signed derivative during the onset-to-peak phase.",
        algorithm="Positive peaks use max derivative from onset to peak; negative peaks use min derivative from onset to peak.",
        category=SumIntensityFeatureCategory.SLOPE,
    ),
    SumIntensityFeatureSchema(
        name="max_decay_slope",
        display_name="Max Decay Slope",
        value_type="float",
        unit="detection source/s",
        description="Maximum signed derivative during the peak-to-right-10% decay phase.",
        algorithm="Positive peaks use min derivative from peak to right_10; negative peaks use max derivative from peak to right_10. Fails if right_10 is unavailable.",
        category=SumIntensityFeatureCategory.SLOPE,
    ),
    SumIntensityFeatureSchema(
        name="auc",
        display_name="AUC",
        value_type="float",
        unit="detection source*s",
        description="Area under the event above onset value.",
        algorithm="Integral from left_10 to right_10 of max(signal - onset_value, 0) for positive peaks, or max(onset_value - signal, 0) for negative peaks.",
        category=SumIntensityFeatureCategory.AREA,
    ),
)


def get_sum_intensity_feature_schema() -> tuple[SumIntensityFeatureSchema, ...]:
    """Return event-level feature schema entries.

    Returns:
        Tuple of immutable feature schema records in stable report order.
    """
    return SUM_INTENSITY_FEATURE_SCHEMA


def get_sum_intensity_feature_schema_dataframe() -> pd.DataFrame:
    """Return event-level feature schema as a DataFrame.

    Returns:
        DataFrame with one row per feature and columns suitable for reports,
        GUI tooltips, or documentation tables.
    """
    return pd.DataFrame([schema.to_dict() for schema in SUM_INTENSITY_FEATURE_SCHEMA])
