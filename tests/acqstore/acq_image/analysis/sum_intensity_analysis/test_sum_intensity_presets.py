"""Tests for sum-intensity detection preset API."""

from __future__ import annotations

import pytest

from acqstore.acq_image.analysis.sum_intensity_analysis.sum_intensity_analysis import (
    SumIntensityAnalysis,
)
from acqstore.acq_image.analysis.sum_intensity_analysis.sum_intensity_core import (
    SumIntensitySummaryKey,
    SumIntensityTraceKey,
    run_sum_intensity,
)
from acqstore.acq_image.analysis.sum_intensity_analysis.sum_intensity_presets import (
    SumIntensityPresetName,
    get_sum_intensity_detection_preset,
    get_sum_intensity_detection_preset_params,
    list_sum_intensity_detection_presets,
)
from acqstore.acq_image.analysis.sum_intensity_analysis.synthetic.synthetic_config import (
    SyntheticSumIntensityConfig,
)
from acqstore.acq_image.analysis.sum_intensity_analysis.synthetic.synthetic_generator import (
    make_synthetic_sum_intensity_image,
)


def test_list_presets_returns_stable_builtin_order() -> None:
    """Preset list should expose fast, medium, and slow in stable order."""
    presets = list_sum_intensity_detection_presets()

    assert tuple(preset.name for preset in presets) == (
        SumIntensityPresetName.FAST,
        SumIntensityPresetName.MEDIUM,
        SumIntensityPresetName.SLOW,
    )
    assert [preset.display_name for preset in presets] == [
        "Fast events",
        "Medium events",
        "Slow events",
    ]


def test_get_preset_accepts_enum_and_rejects_unknown_name() -> None:
    """Preset lookup should be enum-backed and fail fast on bad names."""
    preset = get_sum_intensity_detection_preset(SumIntensityPresetName.SLOW)

    assert preset.name is SumIntensityPresetName.SLOW
    with pytest.raises(KeyError, match="Unknown sum-intensity detection preset"):
        get_sum_intensity_detection_preset("manual_f0")


def test_preset_params_are_complete_valid_and_independent_copies() -> None:
    """Preset params should validate against current analysis schema."""
    expected_keys = set(SumIntensityAnalysis.get_default_detection_params())

    params = get_sum_intensity_detection_preset_params(SumIntensityPresetName.MEDIUM)
    params_again = get_sum_intensity_detection_preset_params(SumIntensityPresetName.MEDIUM)

    assert set(params) == expected_keys
    SumIntensityAnalysis.validate_detection_params(params)
    assert params["baseline_method"] == "percentile"
    assert "manual_f0" not in [preset.name.value for preset in list_sum_intensity_detection_presets()]

    params["derivative_threshold_per_sec"] = 999.0
    assert params_again["derivative_threshold_per_sec"] != 999.0


def test_analysis_class_forwards_detection_preset_api() -> None:
    """SumIntensityAnalysis should expose preset helpers for GUI callers."""
    params = SumIntensityAnalysis.get_detection_preset_params("fast")

    assert params["detection_source"] == SumIntensityTraceKey.DF_F_SIGNAL.value
    analysis = SumIntensityAnalysis(channel=0, roi_id=1, detection_params=params)
    assert analysis.detection_params["refractory_period_ms"] == 10.0
    assert SumIntensityAnalysis.get_detection_preset("slow").name is SumIntensityPresetName.SLOW
    assert len(SumIntensityAnalysis.get_detection_presets()) == 3


def test_medium_preset_runs_synthetic_core_analysis() -> None:
    """Medium preset should run the core algorithm on deterministic synthetic data."""
    data = make_synthetic_sum_intensity_image(
        SyntheticSumIntensityConfig(
            event_times_sec=(1.0, 2.4, 3.8),
            event_amplitude=350.0,
            noise_sigma=1.0,
            spatial_noise_sigma=1.0,
            pop_probability=0.0,
            seed=12,
        )
    )
    result = run_sum_intensity(
        data.image,
        detection_params=SumIntensityAnalysis.get_detection_preset_params(
            SumIntensityPresetName.MEDIUM
        ),
        physical_units=(data.seconds_per_line, data.um_per_pixel),
    )

    assert result.get_summary_value(SumIntensitySummaryKey.NUM_PEAKS) == 3
