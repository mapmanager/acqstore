"""Tests for sum-intensity detection parameter schema metadata."""

from __future__ import annotations

from acqstore.acq_image.analysis.model import DetectionParamCategory
from acqstore.acq_image.analysis.sum_intensity_analysis.sum_intensity_analysis import (
    SumIntensityAnalysis,
)


def test_sum_intensity_schema_categories_are_declared() -> None:
    """Every sum-intensity detection parameter should have a category."""
    schema = SumIntensityAnalysis.get_detection_schema()

    assert schema
    assert all(field.category is not None for field in schema)
    assert {field.category for field in schema} == {
        DetectionParamCategory.PREPROCESSING,
        DetectionParamCategory.PEAK_DETECTION,
    }


def test_sum_intensity_schema_category_order_is_linear() -> None:
    """Categories should appear in schema order without interleaving."""
    categories = [field.category for field in SumIntensityAnalysis.get_detection_schema()]
    first_peak = categories.index(DetectionParamCategory.PEAK_DETECTION)

    assert all(category is DetectionParamCategory.PREPROCESSING for category in categories[:first_peak])
    assert all(category is DetectionParamCategory.PEAK_DETECTION for category in categories[first_peak:])


def test_sum_intensity_hidden_advanced_fields_remain_in_schema() -> None:
    """Advanced hidden fields should remain backend-valid detection params."""
    schema_by_name = {field.name: field for field in SumIntensityAnalysis.get_detection_schema()}

    assert schema_by_name["baseline_window_ms"].visible is True
    assert schema_by_name["baseline_window_ms"].category is DetectionParamCategory.PREPROCESSING
    assert schema_by_name["baseline_min_value"].visible is False
    assert schema_by_name["baseline_min_value"].category is DetectionParamCategory.PREPROCESSING
    assert schema_by_name["level_fractions"].visible is False
    assert schema_by_name["level_fractions"].category is DetectionParamCategory.PEAK_DETECTION

    defaults = SumIntensityAnalysis.get_default_detection_params()
    assert "baseline_window_ms" in defaults
    assert "baseline_min_value" in defaults
    assert "level_fractions" in defaults
    SumIntensityAnalysis.validate_detection_params(defaults)


def test_detection_schema_dataframe_includes_category_values() -> None:
    """Schema DataFrame should expose category values for scripting/frontends."""
    df = SumIntensityAnalysis.get_detection_schema_dataframe()

    assert "category" in df.columns
    assert df.loc["detrend_method", "category"] == DetectionParamCategory.PREPROCESSING.value
    assert df.loc["detection_method", "category"] == DetectionParamCategory.PEAK_DETECTION.value
