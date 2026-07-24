"""Tests for pure Radon velocity core."""

from __future__ import annotations

import numpy as np
import pytest

from acqstore.acq_image.analysis.velocity_analysis.radon_core import (
    RADON_VELOCITY_COLUMNS,
    RadonVelocityCancelled,
    run_radon_velocity,
)


def _synthetic_kymograph() -> np.ndarray:
    """Return a deterministic synthetic kymograph.

    Returns:
        Two-dimensional test image.
    """
    y, x = np.indices((96, 32))
    return (np.sin((x + y) / 8.0) * 100.0 + 200.0).astype(np.float32)


def test_run_radon_velocity_single_process_returns_expected_columns() -> None:
    """Radon core should return the expected result table columns."""
    result = run_radon_velocity(
        _synthetic_kymograph(),
        window_width=16,
        physical_units=(0.001, 0.2),
        use_multiprocessing=False,
    )

    assert tuple(result.table.columns) == RADON_VELOCITY_COLUMNS
    assert len(result.table) > 0
    # assert result.summary["window_width"] == 16
    assert result.summary["num_windows"] == len(result.table)


def test_run_radon_velocity_reports_progress() -> None:
    """Radon core should invoke progress callbacks."""
    seen: list[tuple[float | None, str]] = []

    run_radon_velocity(
        _synthetic_kymograph(),
        window_width=16,
        physical_units=(0.001, 0.2),
        progress_callback=lambda fraction, message: seen.append((fraction, message)),
        use_multiprocessing=False,
    )

    assert seen
    assert seen[-1][0] == pytest.approx(1.0)


def test_run_radon_velocity_cancels() -> None:
    """Radon core should raise when cancellation is requested."""
    with pytest.raises(RadonVelocityCancelled):
        run_radon_velocity(
            _synthetic_kymograph(),
            window_width=16,
            physical_units=(0.001, 0.2),
            cancel_callback=lambda: True,
            use_multiprocessing=False,
        )
