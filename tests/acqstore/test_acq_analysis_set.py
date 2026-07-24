"""Tests for AcqAnalysisSet dependency enforcement."""

from __future__ import annotations

import pytest

from acqstore.acq_image.acq_analysis_set import AcqAnalysisSet
from acqstore.acq_image.analysis.event_analysis.event_analysis import EventAnalysis
from acqstore.acq_image.analysis.model import (
    AnalysisPlotData,
    AnalysisResult,
    BaseAnalysis,
    DetectionParamSchema,
    DetectionValueType,
)
from acqstore.acq_image.analysis.registry import (
    _ANALYSIS_REGISTRY,
    register_analysis_class,
)


class DummyRadonAnalysis(BaseAnalysis):
    """Minimal Radon velocity dependency for event-analysis creation."""

    analysis_name = "radon_velocity"

    def run(self, data_provider, *, context=None, dependencies=None) -> AnalysisResult:
        """Return the existing empty result."""
        return self.result

    def get_plot_data(self) -> AnalysisPlotData:
        """Return simple velocity plot data."""
        return AnalysisPlotData(
            x=(0.0, 1.0),
            y=(1.0, 2.0),
            x_label="Time (s)",
            y_label="Velocity",
        )


def test_event_analysis_creation_requires_matching_radon_velocity() -> None:
    """Event analysis should not be created without its Radon dependency."""
    analysis_set = AcqAnalysisSet("example.tif")

    try:
        analysis_set.create(EventAnalysis.analysis_name, channel=0, roi_id=1)
    except ValueError as exc:
        assert "requires 'radon_velocity'" in str(exc)
    else:  # pragma: no cover - defensive assertion branch
        raise AssertionError("event analysis created without Radon dependency")


def test_event_analysis_creation_succeeds_with_matching_radon_velocity() -> None:
    """Event analysis should be creatable when matching Radon exists."""
    analysis_set = AcqAnalysisSet("example.tif")
    analysis_set.add(DummyRadonAnalysis(channel=0, roi_id=1))

    analysis = analysis_set.create(EventAnalysis.analysis_name, channel=0, roi_id=1)

    assert isinstance(analysis, EventAnalysis)


class RunnableAnalysis(BaseAnalysis):
    """Minimal registered analysis with a detection param for create_and_run tests."""

    analysis_name = "create_and_run_dummy"
    detection_schema = (
        DetectionParamSchema(
            name="window",
            display_name="Window",
            value_type=DetectionValueType.INT,
            default=32,
        ),
    )

    def __init__(self, *, channel: int, roi_id: int, detection_params=None) -> None:
        super().__init__(channel=channel, roi_id=roi_id, detection_params=detection_params)
        self.use_multiprocessing = True

    def set_execution_options(self, *, use_multiprocessing: bool = True) -> None:
        """Record an execution option for assertions."""
        self.use_multiprocessing = bool(use_multiprocessing)

    def run(self, data_provider, *, context=None, dependencies=None) -> AnalysisResult:
        """Record that the analysis ran and echo its window parameter."""
        self.result.summary["ran"] = True
        self.result.summary["window"] = self.detection_params["window"]
        self.result.summary["use_multiprocessing"] = self.use_multiprocessing
        return self.result


@pytest.fixture
def runnable_analysis_cls():
    """Register ``RunnableAnalysis`` for the duration of one test."""
    register_analysis_class(RunnableAnalysis)
    try:
        yield RunnableAnalysis
    finally:
        _ANALYSIS_REGISTRY.pop(RunnableAnalysis.analysis_name, None)


def _set_with_provider() -> AcqAnalysisSet:
    """Return an analysis set with a non-None placeholder data provider."""
    return AcqAnalysisSet("example.tif", data_provider=object())


def test_create_and_run_creates_and_runs(runnable_analysis_cls) -> None:
    """create_and_run should create the analysis, run it, and store it."""
    analysis_set = _set_with_provider()

    analysis = analysis_set.create_and_run(runnable_analysis_cls, channel=0, roi_id=1)

    assert isinstance(analysis, runnable_analysis_cls)
    assert analysis.result.summary["ran"] is True
    assert analysis_set.get(analysis.key) is analysis


def test_create_and_run_accepts_name_string(runnable_analysis_cls) -> None:
    """create_and_run should accept a registered analysis name string."""
    analysis_set = _set_with_provider()

    analysis = analysis_set.create_and_run("create_and_run_dummy", channel=0, roi_id=1)

    assert analysis.key.analysis_name == "create_and_run_dummy"


def test_create_and_run_merges_partial_detection_params(runnable_analysis_cls) -> None:
    """Partial detection params should merge over schema defaults."""
    analysis_set = _set_with_provider()

    analysis = analysis_set.create_and_run(
        runnable_analysis_cls,
        channel=0,
        roi_id=1,
        detection_params={"window": 64},
    )

    assert analysis.detection_params["window"] == 64
    assert analysis.result.summary["window"] == 64


def test_create_and_run_duplicate_raises(runnable_analysis_cls) -> None:
    """A duplicate identity should raise when replace_existing is False."""
    analysis_set = _set_with_provider()
    analysis_set.create_and_run(runnable_analysis_cls, channel=0, roi_id=1)

    with pytest.raises(ValueError):
        analysis_set.create_and_run(runnable_analysis_cls, channel=0, roi_id=1)


def test_create_and_run_replace_existing(runnable_analysis_cls) -> None:
    """replace_existing should replace and rerun a matching analysis."""
    analysis_set = _set_with_provider()
    first = analysis_set.create_and_run(
        runnable_analysis_cls, channel=0, roi_id=1, detection_params={"window": 16}
    )

    second = analysis_set.create_and_run(
        runnable_analysis_cls,
        channel=0,
        roi_id=1,
        detection_params={"window": 64},
        replace_existing=True,
    )

    assert second is not first
    assert analysis_set.get(second.key) is second
    assert second.detection_params["window"] == 64
    assert len(analysis_set.as_list()) == 1


def test_create_and_run_without_data_provider_raises_and_leaves_set_unchanged(
    runnable_analysis_cls,
) -> None:
    """Missing data provider should raise before any mutation."""
    analysis_set = AcqAnalysisSet("example.tif")

    with pytest.raises(RuntimeError):
        analysis_set.create_and_run(runnable_analysis_cls, channel=0, roi_id=1)

    assert analysis_set.as_list() == []


def test_create_and_run_invalid_detection_params_does_not_mutate(
    runnable_analysis_cls,
) -> None:
    """Invalid detection params should raise before any mutation."""
    analysis_set = _set_with_provider()

    with pytest.raises(KeyError):
        analysis_set.create_and_run(
            runnable_analysis_cls,
            channel=0,
            roi_id=1,
            detection_params={"unknown": 1},
        )

    assert analysis_set.as_list() == []


def test_create_and_run_unregistered_name_raises() -> None:
    """An unregistered analysis name should raise KeyError."""
    analysis_set = _set_with_provider()

    with pytest.raises(KeyError):
        analysis_set.create_and_run("not_registered", channel=0, roi_id=1)


def test_create_and_run_rejects_invalid_type() -> None:
    """A non-str, non-class analysis argument should raise TypeError."""
    analysis_set = _set_with_provider()

    with pytest.raises(TypeError):
        analysis_set.create_and_run(123, channel=0, roi_id=1)  # type: ignore[arg-type]


def test_create_and_run_applies_execution_options(runnable_analysis_cls) -> None:
    """execution_options should be forwarded to set_execution_options."""
    analysis_set = _set_with_provider()

    analysis = analysis_set.create_and_run(
        runnable_analysis_cls,
        channel=0,
        roi_id=1,
        execution_options={"use_multiprocessing": False},
    )

    assert analysis.use_multiprocessing is False
    assert analysis.result.summary["use_multiprocessing"] is False


def test_create_and_run_unknown_execution_option_does_not_mutate(
    runnable_analysis_cls,
) -> None:
    """An unknown execution option should raise before any mutation."""
    analysis_set = _set_with_provider()

    with pytest.raises(TypeError):
        analysis_set.create_and_run(
            runnable_analysis_cls,
            channel=0,
            roi_id=1,
            execution_options={"nope": True},
        )

    assert analysis_set.as_list() == []


def test_create_and_run_execution_options_unsupported_type_raises(
    runnable_analysis_cls,
) -> None:
    """execution_options on an analysis without the setter should raise TypeError."""
    analysis_set = _set_with_provider()

    with pytest.raises(TypeError):
        analysis_set.create_and_run(
            EventAnalysis,
            channel=0,
            roi_id=1,
            execution_options={"use_multiprocessing": False},
        )

    assert analysis_set.as_list() == []


def test_create_and_run_missing_dependency_raises_and_leaves_set_unchanged() -> None:
    """A missing dependency should raise before mutation (event needs radon)."""
    analysis_set = _set_with_provider()

    with pytest.raises(ValueError):
        analysis_set.create_and_run(EventAnalysis, channel=0, roi_id=1)

    assert analysis_set.as_list() == []


def test_get_analysis_resolves_by_class(runnable_analysis_cls) -> None:
    """get_analysis should find an analysis by its class, channel, and ROI."""
    analysis_set = _set_with_provider()
    created = analysis_set.create_and_run(runnable_analysis_cls, channel=0, roi_id=1)

    found = analysis_set.get_analysis(runnable_analysis_cls, channel=0, roi_id=1)

    assert found is created


def test_get_analysis_resolves_by_name_string(runnable_analysis_cls) -> None:
    """get_analysis should accept a registered analysis name string."""
    analysis_set = _set_with_provider()
    created = analysis_set.create_and_run(runnable_analysis_cls, channel=0, roi_id=1)

    found = analysis_set.get_analysis("create_and_run_dummy", channel=0, roi_id=1)

    assert found is created


def test_get_analysis_missing_raises_key_error(runnable_analysis_cls) -> None:
    """get_analysis should raise KeyError when no matching analysis exists."""
    analysis_set = _set_with_provider()

    with pytest.raises(KeyError):
        analysis_set.get_analysis(runnable_analysis_cls, channel=0, roi_id=1)


def test_get_analysis_rejects_invalid_type() -> None:
    """A non-str, non-class analysis argument should raise TypeError."""
    analysis_set = _set_with_provider()

    with pytest.raises(TypeError):
        analysis_set.get_analysis(123, channel=0, roi_id=1)  # type: ignore[arg-type]


def test_find_analysis_resolves_by_class(runnable_analysis_cls) -> None:
    """find_analysis should find an analysis by its class, channel, and ROI."""
    analysis_set = _set_with_provider()
    created = analysis_set.create_and_run(runnable_analysis_cls, channel=0, roi_id=1)

    found = analysis_set.find_analysis(runnable_analysis_cls, channel=0, roi_id=1)

    assert found is created


def test_find_analysis_resolves_by_name_string(runnable_analysis_cls) -> None:
    """find_analysis should accept a registered analysis name string."""
    analysis_set = _set_with_provider()
    created = analysis_set.create_and_run(runnable_analysis_cls, channel=0, roi_id=1)

    found = analysis_set.find_analysis("create_and_run_dummy", channel=0, roi_id=1)

    assert found is created


def test_find_analysis_missing_returns_none(runnable_analysis_cls) -> None:
    """find_analysis should return None when no matching analysis exists."""
    analysis_set = _set_with_provider()

    assert analysis_set.find_analysis(runnable_analysis_cls, channel=0, roi_id=1) is None


def test_find_analysis_rejects_invalid_type() -> None:
    """A non-str, non-class analysis argument should raise TypeError."""
    analysis_set = _set_with_provider()

    with pytest.raises(TypeError):
        analysis_set.find_analysis(123, channel=0, roi_id=1)  # type: ignore[arg-type]
