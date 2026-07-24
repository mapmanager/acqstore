"""Matplotlib diagnostic plots for heart-rate analysis.

These helpers are intended for Python scripts and notebooks. They recompute
plot-specific preprocessing and spectra from a velocity time-series, then render
matplotlib figures. They do not import or depend on CloudScope GUI code.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from acqstore.acq_image.analysis.heart_rate_analysis.plotting.plot_data import (
    compute_lomb_spectrum,
    compute_preprocessing,
    compute_segment_series,
    compute_welch_spectrum,
)


def plot_velocity_overview(
    time_s: Sequence[float],
    velocity: Sequence[float],
    *,
    params: dict[str, Any] | None = None,
    title: str = "",
    ax: Any = None,
) -> Any:
    """Plot raw and preprocessed velocity traces.

    Args:
        time_s: Time samples in seconds.
        velocity: Velocity samples aligned to ``time_s``.
        params: Optional heart-rate detection parameters.
        title: Optional plot title.
        ax: Optional matplotlib axes target.

    Returns:
        Matplotlib axes containing the overview plot.
    """
    from matplotlib import pyplot as plt

    data = compute_preprocessing(time_s, velocity, params=params)
    created_ax = ax is None
    if ax is None:
        ax = plt.subplot(1, 1, 1)

    ax.plot(data.time_s, data.velocity, linewidth=0.8, label="velocity (raw)")
    ax.plot(data.time_s, data.x_pre, linewidth=1.0, label="preprocessed")
    ax.plot(data.time_s, data.x_interp, linewidth=1.0, label="interp small gaps")
    ax.plot(
        data.time_s,
        data.x_bandpassed,
        linewidth=1.2,
        label=f"bandpassed {data.band_hz[0]:.1f}-{data.band_hz[1]:.1f} Hz",
    )
    ax.axhline(0, linewidth=0.5)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("velocity / a.u.")
    ax.set_title(title or "Velocity preprocessing for HR")
    ax.legend(loc="best")

    if created_ax:
        plt.tight_layout()
    return ax


def plot_welch_psd(
    time_s: Sequence[float],
    velocity: Sequence[float],
    *,
    params: dict[str, Any] | None = None,
    title: str = "",
    ax: Any = None,
) -> Any:
    """Plot Welch PSD with peak marker and QC annotation.

    Args:
        time_s: Time samples in seconds.
        velocity: Velocity samples aligned to ``time_s``.
        params: Optional heart-rate detection parameters.
        title: Optional plot title.
        ax: Optional matplotlib axes target.

    Returns:
        Matplotlib axes containing the Welch PSD plot.

    Raises:
        ValueError: If there are not enough samples for Welch diagnostics.
    """
    from matplotlib import pyplot as plt

    data = compute_welch_spectrum(time_s, velocity, params=params)
    created_ax = ax is None
    if ax is None:
        ax = plt.subplot(1, 1, 1)

    ax.plot(data.frequency_hz, data.power, linewidth=1.0, label="Welch PSD")
    ax.axvline(
        data.f_peak_hz,
        linewidth=1.0,
        label=f"peak {data.f_peak_hz:.2f} Hz ({60 * data.f_peak_hz:.0f} bpm), snr={data.snr:.1f}",
    )
    ax.set_xlim(0.0, min(float(np.max(data.frequency_hz)), data.params.band_hz[1] * 1.6))
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title(title or "Heart-rate PSD (Welch)")
    _add_qc_text(ax, data.estimate)
    ax.legend(loc="best")

    if created_ax:
        plt.tight_layout()
    return ax


def plot_lomb_periodogram(
    time_s: Sequence[float],
    velocity: Sequence[float],
    *,
    params: dict[str, Any] | None = None,
    title: str = "",
    ax: Any = None,
) -> Any:
    """Plot Lomb-Scargle periodogram with peak marker and QC annotation.

    Args:
        time_s: Time samples in seconds.
        velocity: Velocity samples aligned to ``time_s``.
        params: Optional heart-rate detection parameters.
        title: Optional plot title.
        ax: Optional matplotlib axes target.

    Returns:
        Matplotlib axes containing the Lomb-Scargle periodogram.

    Raises:
        ValueError: If there are not enough samples for Lomb-Scargle diagnostics.
    """
    from matplotlib import pyplot as plt

    data = compute_lomb_spectrum(time_s, velocity, params=params)
    created_ax = ax is None
    if ax is None:
        ax = plt.subplot(1, 1, 1)

    ax.plot(data.frequency_hz, data.power, linewidth=1.0, label="Lomb-Scargle")
    ax.axvline(
        data.f_peak_hz,
        linewidth=1.0,
        label=f"peak {data.f_peak_hz:.2f} Hz ({60 * data.f_peak_hz:.0f} bpm), snr={data.snr:.1f}",
    )
    ax.set_xlim(data.params.band_hz[0] * 0.8, data.params.band_hz[1] * 1.2)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("normalized power")
    ax.set_title(title or "Heart-rate periodogram (Lomb-Scargle)")
    _add_qc_text(ax, data.estimate)
    ax.legend(loc="best")

    if created_ax:
        plt.tight_layout()
    return ax


def plot_segment_series(
    time_s: Sequence[float],
    velocity: Sequence[float],
    *,
    params: dict[str, Any] | None = None,
    title: str = "",
    ax: Any = None,
) -> Any:
    """Plot windowed segment heart-rate series.

    Args:
        time_s: Time samples in seconds.
        velocity: Velocity samples aligned to ``time_s``.
        params: Optional heart-rate detection parameters.
        title: Optional plot title.
        ax: Optional matplotlib axes target.

    Returns:
        Matplotlib axes containing the segment HR series.
    """
    from matplotlib import pyplot as plt

    seg = compute_segment_series(time_s, velocity, params=params)
    created_ax = ax is None
    if ax is None:
        ax = plt.subplot(1, 1, 1)

    ax.plot(seg["t_center"], seg["bpm"], marker="o", linewidth=1.0, label="segment HR (bpm)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("HR (bpm)")
    ax.set_title(title or "Segment HR series")
    ax.legend(loc="best")

    if created_ax:
        plt.tight_layout()
    return ax


def plot_summary(
    time_s: Sequence[float],
    velocity: Sequence[float],
    *,
    params: dict[str, Any] | None = None,
    title: str = "",
) -> tuple[Any, list[Any]]:
    """Plot overview, Welch PSD, and Lomb-Scargle diagnostics.

    Args:
        time_s: Time samples in seconds.
        velocity: Velocity samples aligned to ``time_s``.
        params: Optional heart-rate detection parameters.
        title: Optional base plot title.

    Returns:
        Tuple ``(figure, axes)`` where axes contains three matplotlib axes.

    Raises:
        ValueError: If spectral diagnostic plots cannot be computed.
    """
    from matplotlib import pyplot as plt

    fig, axes_array = plt.subplots(3, 1, figsize=(10, 7))
    axes = list(axes_array)
    plot_velocity_overview(time_s, velocity, params=params, title=f"{title} | overview" if title else "", ax=axes[0])
    plot_welch_psd(time_s, velocity, params=params, title=f"{title} | Welch PSD" if title else "", ax=axes[1])
    plot_lomb_periodogram(
        time_s,
        velocity,
        params=params,
        title=f"{title} | Lomb-Scargle" if title else "",
        ax=axes[2],
    )
    fig.tight_layout()
    return fig, axes


def _add_qc_text(ax: Any, estimate: Any) -> None:
    """Add a compact QC annotation to an axes when an estimate is available."""
    if estimate is None:
        return
    band_concentration = estimate.band_concentration
    bc_value = float("nan") if band_concentration is None else float(band_concentration)
    qc_text = (
        f"bpm={estimate.bpm:.1f}\n"
        f"snr={estimate.snr:.2f}\n"
        f"edge={'YES' if estimate.edge_flag else 'no'}\n"
        f"bc={bc_value:.3f}"
    )
    ax.text(
        0.98,
        0.97,
        qc_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7},
    )
