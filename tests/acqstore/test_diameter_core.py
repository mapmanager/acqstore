"""Tests for pure diameter analysis core."""

from __future__ import annotations

import numpy as np
import pytest

from acqstore.acq_image.analysis.diameter_analysis.diameter_core import (
    DiameterCancelled,
    run_diameter,
)


def _synthetic_kymograph(*, n_time: int = 40, n_space: int = 80) -> np.ndarray:
    """Return a simple bright band kymograph."""
    image = np.zeros((n_time, n_space), dtype=np.float32)
    image[:, 30:50] = 200.0
    return image


def _default_params(**overrides: object) -> dict[str, object]:
    params: dict[str, object] = {
        "window_rows_odd": 5,
        "stride": 2,
        "binning_method": "mean",
        "polarity": "bright_on_dark",
        "diameter_method": "threshold_width",
        "post_filter_kernel_size": 3,
        "threshold_mode": "half_max",
        "threshold_value": 0.0,
        "gradient_sigma": 1.5,
        "gradient_kernel": "central_diff",
        "gradient_min_edge_strength": 0.02,
        "enable_motion_gating": True,
        "max_edge_shift_um": 2.0,
        "max_diameter_change_um": 2.0,
        "max_center_shift_um": 2.0,
    }
    params.update(overrides)
    return params


def test_run_diameter_threshold_width_populates_table() -> None:
    """Threshold width should return ROI-local rows and summary."""
    result = run_diameter(
        _synthetic_kymograph(),
        detection_params=_default_params(),
        physical_units=(0.001, 0.2),
        use_threads=False,
    )
    assert result.summary["num_rows"] > 0
    assert "time_s" in result.table.columns
    assert "diameter_um_filt" in result.table.columns
    assert "qc_edge_violation" in result.table.columns
    assert "qc_flags" not in result.table.columns


def test_run_diameter_gradient_edges() -> None:
    """Gradient edges method should run without error."""
    result = run_diameter(
        _synthetic_kymograph(),
        detection_params=_default_params(diameter_method="gradient_edges"),
        physical_units=(0.001, 0.2),
        use_threads=False,
    )
    assert result.summary["num_rows"] > 0


def test_run_diameter_respects_post_filter_kernel_size() -> None:
    """Larger post-filter kernel should still return filtered diameter column."""
    small = run_diameter(
        _synthetic_kymograph(),
        detection_params=_default_params(post_filter_kernel_size=3),
        physical_units=(0.001, 0.2),
        use_threads=False,
    )
    large = run_diameter(
        _synthetic_kymograph(),
        detection_params=_default_params(post_filter_kernel_size=5),
        physical_units=(0.001, 0.2),
        use_threads=False,
    )
    assert "diameter_um_filt" in small.table.columns
    assert "diameter_um_filt" in large.table.columns


def test_run_diameter_cancel_raises() -> None:
    """Cancellation callback should stop the run."""
    with pytest.raises(DiameterCancelled):
        run_diameter(
            _synthetic_kymograph(n_time=200),
            detection_params=_default_params(stride=1),
            physical_units=(0.001, 0.2),
            use_threads=False,
            cancel_callback=lambda: True,
        )


def test_run_diameter_invalid_kernel_raises() -> None:
    """Invalid post_filter_kernel_size should fail fast."""
    with pytest.raises(ValueError, match="post_filter_kernel_size"):
        run_diameter(
            _synthetic_kymograph(),
            detection_params=_default_params(post_filter_kernel_size=4),
            physical_units=(0.001, 0.2),
            use_threads=False,
        )
