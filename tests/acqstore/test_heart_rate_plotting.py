"""Tests for heart-rate diagnostic plotting helpers."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pytest

from acqstore.acq_image.analysis.heart_rate_analysis.plotting.plot_data import (
    compute_lomb_spectrum,
    compute_preprocessing,
    compute_segment_series,
    compute_welch_spectrum,
)
from acqstore.acq_image.analysis.heart_rate_analysis.plotting.plotly_plots import (
    plot_segment_series_plotly,
    plot_summary_plotly,
)

EXPECTED_FREQ_HZ = 6.0
EXPECTED_BPM = 360.0
FS_HZ = 100.0


def _require_matplotlib() -> None:
    """Skip the calling test when optional Matplotlib is not installed."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)


def _synthetic_velocity(n_samples: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    """Return synthetic velocity data with a 360 bpm component.

    Args:
        n_samples: Number of samples.

    Returns:
        Tuple ``(time_s, velocity)``.
    """
    time_s = np.arange(n_samples, dtype=float) / FS_HZ
    velocity = 2.0 + np.sin(2.0 * np.pi * EXPECTED_FREQ_HZ * time_s)
    return time_s, velocity


def test_compute_preprocessing_defaults() -> None:
    """Preprocessing should preserve input length and derive the default band."""
    time_s, velocity = _synthetic_velocity()

    data = compute_preprocessing(time_s, velocity)

    assert data.time_s.shape == time_s.shape
    assert data.velocity.shape == velocity.shape
    assert data.x_pre.shape == velocity.shape
    assert data.x_interp.shape == velocity.shape
    assert data.x_bandpassed.shape == velocity.shape
    assert data.fs_hz == pytest.approx(FS_HZ)
    assert data.band_hz == pytest.approx((4.0, 10.0))


def test_compute_spectra_recover_expected_frequency() -> None:
    """Welch and Lomb diagnostic spectra should recover the synthetic frequency."""
    time_s, velocity = _synthetic_velocity()

    welch = compute_welch_spectrum(time_s, velocity)
    lomb = compute_lomb_spectrum(time_s, velocity)

    assert welch.f_peak_hz == pytest.approx(EXPECTED_FREQ_HZ, abs=0.5)
    assert lomb.f_peak_hz == pytest.approx(EXPECTED_FREQ_HZ, abs=0.1)
    assert welch.estimate is not None
    assert lomb.estimate is not None
    assert welch.estimate.bpm == pytest.approx(EXPECTED_BPM, abs=30.0)
    assert lomb.estimate.bpm == pytest.approx(EXPECTED_BPM, abs=10.0)


def test_plotly_summary_returns_go_figure() -> None:
    """Plotly summary helper should return a multi-trace figure object."""
    time_s, velocity = _synthetic_velocity()

    fig = plot_summary_plotly(time_s, velocity, title="synthetic")

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 6
    assert fig.layout.title.text == "synthetic"


def test_plotly_summary_shares_frequency_xaxis() -> None:
    """Welch and Lomb panels in the summary figure should share frequency x-axis."""
    time_s, velocity = _synthetic_velocity()

    fig = plot_summary_plotly(time_s, velocity)

    welch_xaxis = fig.layout.xaxis2
    lomb_xaxis = fig.layout.xaxis3
    assert welch_xaxis.range == lomb_xaxis.range
    assert lomb_xaxis.matches == "x2"


def test_plotly_segment_series_returns_go_figure() -> None:
    """Plotly segment helper should return a single trace figure."""
    time_s, velocity = _synthetic_velocity()

    fig = plot_segment_series_plotly(time_s, velocity)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].name == "segment HR (bpm)"


def test_mpl_summary_returns_figure_and_axes() -> None:
    """Matplotlib summary helper should return a figure and three axes."""
    _require_matplotlib()
    from matplotlib import pyplot as plt

    from acqstore.acq_image.analysis.heart_rate_analysis.plotting.mpl_plots import (
        plot_summary,
    )

    time_s, velocity = _synthetic_velocity()

    fig, axes = plot_summary(time_s, velocity, title="synthetic")

    assert len(axes) == 3
    assert len(axes[0].lines) >= 4
    assert len(axes[1].lines) >= 2
    assert len(axes[2].lines) >= 2
    plt.close(fig)


def test_mpl_segment_series_returns_axes() -> None:
    """Matplotlib segment helper should return axes with one plotted line."""
    _require_matplotlib()
    from matplotlib import pyplot as plt

    from acqstore.acq_image.analysis.heart_rate_analysis.plotting.mpl_plots import (
        plot_segment_series,
    )

    time_s, velocity = _synthetic_velocity()

    ax = plot_segment_series(time_s, velocity)

    assert len(ax.lines) == 1
    plt.close(ax.figure)


def test_compute_segment_series_has_expected_keys() -> None:
    """Segment series compute helper should return the expected array keys."""
    time_s, velocity = _synthetic_velocity()

    seg = compute_segment_series(time_s, velocity)

    assert set(seg) == {
        "t_center",
        "bpm",
        "snr",
        "valid_frac",
        "edge_flag",
        "band_concentration",
    }
