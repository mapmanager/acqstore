"""Tests for DetectionParamSchema.methods backward compatibility."""

from acqstore.acq_image.analysis.model import DetectionParamSchema, DetectionValueType
from acqstore.acq_image.analysis.velocity_analysis.radon_velocity_analysis import RadonVelocityAnalysis


def test_radon_schema_methods_default_none() -> None:
    """Existing analyses should keep methods=None on schema entries."""
    schema = RadonVelocityAnalysis.get_detection_schema()
    assert schema[0].methods is None


def test_detection_param_schema_accepts_methods() -> None:
    """methods field should be optional on schema entries."""
    entry = DetectionParamSchema(
        name="demo",
        display_name="Demo",
        value_type=DetectionValueType.INT,
        default=1,
        methods=("threshold_width",),
    )
    assert entry.methods == ("threshold_width",)
