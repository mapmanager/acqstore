"""Tests for AnalysisRunContext."""

import pytest

from acqstore.acq_image.analysis.model import AnalysisCancelled, AnalysisRunContext


def test_progress_callback_is_called() -> None:
    """AnalysisRunContext should call progress callbacks."""
    seen: list[tuple[float | None, str]] = []
    context = AnalysisRunContext(progress_callback=lambda f, m: seen.append((f, m)))

    context.report_progress(0.5, "Halfway")

    assert seen == [(0.5, "Halfway")]


def test_cancel_callback_raises() -> None:
    """AnalysisRunContext should raise when cancelled."""
    context = AnalysisRunContext(cancel_callback=lambda: True)

    with pytest.raises(AnalysisCancelled):
        context.raise_if_cancelled()


def test_is_cancelled_false_without_callback() -> None:
    """AnalysisRunContext should not be cancelled by default."""
    assert not AnalysisRunContext().is_cancelled()
