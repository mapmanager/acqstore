"""Tests for analysis registry behavior."""

from collections.abc import Iterator

import pytest

from acqstore.acq_image.analysis import registry


@pytest.fixture
def empty_registry() -> Iterator[None]:
    """Clear and restore the registry around one test."""
    saved_registry = dict(registry._ANALYSIS_REGISTRY)
    saved_builtins_registered = registry._BUILTINS_REGISTERED

    registry.clear_analysis_registry()
    try:
        yield
    finally:
        registry._ANALYSIS_REGISTRY.clear()
        registry._ANALYSIS_REGISTRY.update(saved_registry)
        registry._BUILTINS_REGISTERED = saved_builtins_registered


def test_get_analysis_class_registers_builtin_analyses(empty_registry: None) -> None:
    """Built-in analysis classes should resolve without caller-side imports."""
    assert registry.get_analysis_class("radon_velocity").__name__ == "RadonVelocityAnalysis"
    assert registry.get_analysis_class("diameter").__name__ == "DiameterAnalysis"
    assert registry.get_analysis_class("heart_rate").__name__ == "HeartRateAnalysis"
    assert registry.get_analysis_class("event").__name__ == "EventAnalysis"


def test_get_analysis_registry_registers_builtin_analyses(empty_registry: None) -> None:
    """The registry snapshot should include all built-in analyses."""
    analysis_registry = registry.get_analysis_registry()

    assert {
        "radon_velocity",
        "diameter",
        "heart_rate",
        "event",
    }.issubset(analysis_registry)
