"""Analysis plugin registry."""

from __future__ import annotations

from typing import TypeVar

from acqstore.acq_image.analysis.model import BaseAnalysis
from acqstore.utils.logging import get_logger
logger = get_logger(__name__)

AnalysisType = TypeVar("AnalysisType", bound=BaseAnalysis)

_ANALYSIS_REGISTRY: dict[str, type[BaseAnalysis]] = {}
_BUILTINS_REGISTERED = False


def register_analysis_class(cls: type[AnalysisType]) -> type[AnalysisType]:
    """Register an analysis class.

    Args:
        cls: Analysis class to register. It must define ``analysis_name``.

    Returns:
        The same class, allowing decorator-style registration.

    Raises:
        ValueError: If the class has no name or the name is already registered.
    """
    analysis_name = getattr(cls, "analysis_name", None)
    if not analysis_name:
        raise ValueError(f"{cls.__name__} is missing analysis_name")

    if analysis_name in _ANALYSIS_REGISTRY:
        raise ValueError(f"Duplicate analysis_name: {analysis_name!r}")

    _ANALYSIS_REGISTRY[str(analysis_name)] = cls
    return cls


def register_builtin_analyses() -> None:
    """Register all built-in production analysis classes.

    The registry is intentionally self-populating so callers can load an
    ``AcqImage`` without importing analysis modules for side effects first.
    Function-local imports avoid import cycles with modules that use the
    ``register_analysis_class`` decorator.

    Returns:
        None.
    """
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return

    from acqstore.acq_image.analysis.diameter_analysis.diameter_analysis import DiameterAnalysis
    from acqstore.acq_image.analysis.event_analysis.event_analysis import EventAnalysis
    from acqstore.acq_image.analysis.heart_rate_analysis.heart_rate_analysis import (
        HeartRateAnalysis,
    )
    from acqstore.acq_image.analysis.sum_intensity_analysis.sum_intensity_analysis import (
        SumIntensityAnalysis,
    )
    from acqstore.acq_image.analysis.velocity_analysis.radon_velocity_analysis import (
        RadonVelocityAnalysis,
    )

    for cls in (
        RadonVelocityAnalysis,
        DiameterAnalysis,
        HeartRateAnalysis,
        EventAnalysis,
        SumIntensityAnalysis,
    ):
        _ANALYSIS_REGISTRY.setdefault(str(cls.analysis_name), cls)

    _BUILTINS_REGISTERED = True


def get_analysis_class(analysis_name: str) -> type[BaseAnalysis]:
    """Return registered analysis class by name.

    Args:
        analysis_name: Analysis type name.

    Returns:
        Registered analysis class.

    Raises:
        KeyError: If no class is registered for ``analysis_name``.
    """
    register_builtin_analyses()
    try:
        return _ANALYSIS_REGISTRY[analysis_name]
    except KeyError:
        logger.error(f'did not understand "{analysis_name}" available analysis names are {list(_ANALYSIS_REGISTRY.keys())}')
        raise KeyError(f"No analysis class registered for {analysis_name!r}") from None


def get_analysis_registry() -> dict[str, type[BaseAnalysis]]:
    """Return a copy of the analysis registry.

    Returns:
        Mapping from analysis name to analysis class.
    """
    register_builtin_analyses()
    return dict(_ANALYSIS_REGISTRY)


def clear_analysis_registry() -> None:
    """Clear the registry.

    This is intended for unit tests.

    Returns:
        None.
    """
    global _BUILTINS_REGISTERED
    _ANALYSIS_REGISTRY.clear()
    _BUILTINS_REGISTERED = False
