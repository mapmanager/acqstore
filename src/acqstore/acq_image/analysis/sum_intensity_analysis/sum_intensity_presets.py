"""Detection-parameter presets for sum-intensity analysis.

This module defines the public backend API for named sum-intensity detection
presets. Presets are intended for scripting and GUI defaults only: they return
complete detection-parameter dictionaries that callers may copy into an analysis
or edit before running. The preset registry is immutable from the caller's point
of view, and every accessor returns a fresh copy of the parameter mapping so UI
edits never mutate global preset state.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from acqstore.acq_image.analysis.sum_intensity_analysis.sum_intensity_core import (
    SumIntensityTraceKey,
)


class SumIntensityPresetName(StrEnum):
    """Stable names for built-in sum-intensity detection presets.

    The names are intentionally small and backend-native. CloudScope GUI code can
    expose the accompanying display names while storing these enum values in
    state or user preferences.

    Members:
        FAST: Shorter windows for faster events or exploratory detection.
        MEDIUM: General-purpose first-pass preset.
        SLOW: Longer windows and refractory period for slower biological events.
    """

    FAST = "fast"
    MEDIUM = "medium"
    SLOW = "slow"


@dataclass(frozen=True, slots=True)
class SumIntensityDetectionPreset:
    """One named sum-intensity detection preset.

    Args:
        name: Stable enum value for programmatic lookup.
        display_name: Human-readable name for UI presentation.
        description: Short explanation of the intended use.
        params: Complete detection-parameter mapping. Callers receive copied
            mappings from public accessors and may edit those copies before
            running analysis.
    """

    name: SumIntensityPresetName
    display_name: str
    description: str
    params: dict[str, Any]

    def copy_params(self) -> dict[str, Any]:
        """Return an independent copy of the preset parameters.

        Returns:
            Deep copy of the preset's detection-parameter mapping.
        """
        return deepcopy(self.params)


def list_sum_intensity_detection_presets() -> tuple[SumIntensityDetectionPreset, ...]:
    """Return all built-in sum-intensity detection presets.

    Returns:
        Tuple of presets in stable UI order: fast, medium, slow.
    """
    return _PRESETS


def get_sum_intensity_detection_preset(
    name: SumIntensityPresetName | str,
) -> SumIntensityDetectionPreset:
    """Return one named sum-intensity detection preset.

    Args:
        name: Preset enum value or its string value.

    Returns:
        Matching immutable preset descriptor.

    Raises:
        KeyError: If ``name`` is not a built-in preset.
    """
    try:
        preset_name = SumIntensityPresetName(name)
    except ValueError as exc:
        raise KeyError(f"Unknown sum-intensity detection preset: {name!r}") from exc
    return _PRESETS_BY_NAME[preset_name]


def get_sum_intensity_detection_preset_params(
    name: SumIntensityPresetName | str,
) -> dict[str, Any]:
    """Return a copied detection-parameter mapping for one preset.

    Args:
        name: Preset enum value or its string value.

    Returns:
        Complete detection-parameter dictionary suitable for
        ``SumIntensityAnalysis`` construction or ``run_sum_intensity``.

    Raises:
        KeyError: If ``name`` is not a built-in preset.
    """
    return get_sum_intensity_detection_preset(name).copy_params()


def _base_params() -> dict[str, Any]:
    """Return shared default detection parameters for all presets.

    Returns:
        Complete detection-parameter mapping before kinetic-specific overrides.
    """
    return {
        "window_radius_points": 0,
        "filter_method": "median",
        "median_filter_kernel_points": 3,
        "detrend_method": "single_exponential",
        "baseline_method": "percentile",
        "baseline_percentile": 20.0,
        "manual_f0_baseline": 1.0,
        "baseline_min_value": 1e-12,
        "baseline_window_ms": 100.0,
        "detection_method": "derivative_threshold",
        "polarity": "positive",
        "detection_source": SumIntensityTraceKey.DF_F_SIGNAL.value,
        "absolute_threshold": 0.0,
        "derivative_threshold_per_sec": 1.0,
        "refractory_period_ms": 10.0,
        "peak_search_window_ms": 50.0,
        "width_search_window_ms": 150.0,
        "level_fractions": "0.1,0.2,0.5,0.8,0.9",
    }


def _params_with(**overrides: Any) -> dict[str, Any]:
    """Return shared params updated with preset-specific overrides.

    Args:
        **overrides: Detection-parameter values to override.

    Returns:
        Complete detection-parameter mapping.
    """
    params = _base_params()
    params.update(overrides)
    return params


_PRESETS: tuple[SumIntensityDetectionPreset, ...] = (
    SumIntensityDetectionPreset(
        name=SumIntensityPresetName.FAST,
        display_name="Fast events",
        description=(
            ""
            # "Faster refractory and search windows for slower biological kinetics. "
        ),
        params=_params_with(
            # window_radius_points=0,
            refractory_period_ms=10.0,
            peak_search_window_ms=50.0,
            width_search_window_ms=150.0,
        ),
    ),
    SumIntensityDetectionPreset(
        name=SumIntensityPresetName.MEDIUM,
        display_name="Medium events",
        description=(
            ""
            # "Medium refractory and search windows for slower biological kinetics. "
        ),
        params=_params_with(
            # window_radius_points=2,
            refractory_period_ms=250.0,
            peak_search_window_ms=150.0,
            width_search_window_ms=500.0,
        ),
    ),
    SumIntensityDetectionPreset(
        name=SumIntensityPresetName.SLOW,
        display_name="Slow events",
        description=(
            ""
            # "Longer refractory and search windows for slower biological kinetics. "
        ),
        params=_params_with(
            # window_radius_points=8,
            refractory_period_ms=500.0,
            peak_search_window_ms=250.0,
            width_search_window_ms=750.0,
        ),
    ),
)

_PRESETS_BY_NAME: dict[SumIntensityPresetName, SumIntensityDetectionPreset] = {
    preset.name: preset for preset in _PRESETS
}
