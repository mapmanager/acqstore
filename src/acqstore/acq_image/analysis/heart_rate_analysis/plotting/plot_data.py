"""Numpy-only diagnostic data helpers for heart-rate plots.

The functions in this module recompute plot-oriented preprocessing and spectra
from a velocity time-series. They intentionally do not read persisted
``HeartRateAnalysis.summary`` debug arrays because those arrays are not stored
in the AcqImage sidecar JSON. These helpers are intended for scripts, notebooks,
and higher-level plotting modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from acqstore.acq_image.analysis.heart_rate_analysis.heart_rate_analysis import (
    HeartRateAnalysis,
    LOMB_METHOD,
    WELCH_METHOD,
)
from acqstore.acq_image.analysis.heart_rate_analysis.heart_rate_core import (
    HeartRateEstimate,
    bandpass_filter,
    detrend_finite,
    dominant_freq_lombscargle,
    dominant_freq_welch,
    estimate_fs,
    estimate_heart_rate_global,
    estimate_heart_rate_segment_series,
    interpolate_small_gaps,
    winsorize_mad,
)
from acqstore.acq_image.analysis.heart_rate_analysis.heart_rate_params import (
    HeartRateCoreParams,
    normalize_heart_rate_detection_params,
)


@dataclass(frozen=True, slots=True)
class HeartRatePreprocessingData:
    """Preprocessed velocity traces used for diagnostic plots.

    Args:
        time_s: Original time samples in seconds.
        velocity: Original velocity samples.
        x_pre: Velocity after optional absolute value, MAD winsorization, and
            finite-value detrending.
        x_interp: ``x_pre`` after small-gap interpolation.
        x_bandpassed: Band-passed trace aligned to ``time_s`` with NaNs where
            filtering was not possible.
        fs_hz: Estimated sample rate in Hz.
        band_hz: Heart-rate band in Hz.
        params: Normalized heart-rate core parameters used for computation.
    """

    time_s: np.ndarray
    velocity: np.ndarray
    x_pre: np.ndarray
    x_interp: np.ndarray
    x_bandpassed: np.ndarray
    fs_hz: float
    band_hz: tuple[float, float]
    params: HeartRateCoreParams


@dataclass(frozen=True, slots=True)
class HeartRateSpectrumData:
    """One heart-rate spectral diagnostic.

    Args:
        method: Estimator method label.
        frequency_hz: Frequency axis in Hz.
        power: Spectral power aligned to ``frequency_hz``.
        f_peak_hz: Detected peak frequency in Hz.
        snr: Peak signal-to-noise metric.
        estimate: Optional global estimate from the core estimator.
        params: Normalized heart-rate core parameters used for computation.
    """

    method: str
    frequency_hz: np.ndarray
    power: np.ndarray
    f_peak_hz: float
    snr: float
    estimate: HeartRateEstimate | None
    params: HeartRateCoreParams


def default_heart_rate_detection_params() -> dict[str, Any]:
    """Return default heart-rate detection parameters.

    Returns:
        Default detection parameter mapping from ``HeartRateAnalysis``.
    """
    return HeartRateAnalysis.get_default_detection_params()


def normalize_plot_params(params: dict[str, Any] | None = None) -> HeartRateCoreParams:
    """Normalize optional plotting parameters.

    Args:
        params: Optional heart-rate detection parameter mapping. When omitted,
            ``HeartRateAnalysis`` defaults are used.

    Returns:
        Normalized heart-rate core parameters.
    """
    if params is None:
        params = default_heart_rate_detection_params()
    return normalize_heart_rate_detection_params(params)


def compute_preprocessing(
    time_s: Sequence[float],
    velocity: Sequence[float],
    *,
    params: dict[str, Any] | None = None,
) -> HeartRatePreprocessingData:
    """Compute preprocessing traces used by heart-rate diagnostic plots.

    Args:
        time_s: Time samples in seconds.
        velocity: Velocity samples aligned to ``time_s``.
        params: Optional heart-rate detection parameters.

    Returns:
        Preprocessing data with raw, preprocessed, interpolated, and band-passed
        traces.

    Raises:
        ValueError: If the input arrays have mismatched shapes or sampling rate
            cannot be estimated.
    """
    t = np.asarray(time_s, dtype=float)
    v = np.asarray(velocity, dtype=float)
    if t.shape != v.shape:
        raise ValueError("time_s and velocity must have matching shapes.")

    normalized = normalize_plot_params(params)
    fs_hz = estimate_fs(t[np.isfinite(t)])
    x_pre = np.abs(v) if normalized.use_abs else v
    x_pre = winsorize_mad(x_pre, k=normalized.outlier_k_mad)
    x_pre = detrend_finite(x_pre)
    x_interp = interpolate_small_gaps(t, x_pre, max_gap_sec=normalized.interp_max_gap_sec)

    finite = np.isfinite(x_interp)
    x_bandpassed = np.full_like(x_interp, np.nan, dtype=float)
    if int(np.sum(finite)) > 10:
        x_bandpassed[finite] = bandpass_filter(
            x_interp[finite],
            fs_hz,
            band_hz=normalized.band_hz,
            order=normalized.bandpass_order,
        )

    return HeartRatePreprocessingData(
        time_s=t,
        velocity=v,
        x_pre=x_pre,
        x_interp=x_interp,
        x_bandpassed=x_bandpassed,
        fs_hz=float(fs_hz),
        band_hz=normalized.band_hz,
        params=normalized,
    )


def compute_welch_spectrum(
    time_s: Sequence[float],
    velocity: Sequence[float],
    *,
    params: dict[str, Any] | None = None,
) -> HeartRateSpectrumData:
    """Compute Welch PSD diagnostic data.

    Args:
        time_s: Time samples in seconds.
        velocity: Velocity samples aligned to ``time_s``.
        params: Optional heart-rate detection parameters.

    Returns:
        Welch spectrum data with peak and estimate metadata.

    Raises:
        ValueError: If fewer than 256 finite samples are available after
            preprocessing/interpolation.
    """
    pre = compute_preprocessing(time_s, velocity, params=params)
    finite = np.isfinite(pre.x_interp)
    x = pre.x_interp[finite]
    if x.size < 256:
        raise ValueError("Not enough finite samples for Welch PSD plot.")

    xf = bandpass_filter(
        x,
        pre.fs_hz,
        band_hz=pre.band_hz,
        order=pre.params.bandpass_order,
    )
    nperseg = int(np.clip(round(pre.fs_hz * pre.params.nperseg_sec), 128, 8192))
    f_peak, snr, frequency_hz, power = dominant_freq_welch(
        xf,
        pre.fs_hz,
        band_hz=pre.band_hz,
        nperseg=nperseg,
    )
    estimate, _debug = estimate_heart_rate_global(
        pre.time_s,
        pre.velocity,
        method=WELCH_METHOD,
        **pre.params.to_core_kwargs(),
    )
    return HeartRateSpectrumData(
        method=WELCH_METHOD,
        frequency_hz=frequency_hz,
        power=power,
        f_peak_hz=float(f_peak),
        snr=float(snr),
        estimate=estimate,
        params=pre.params,
    )


def compute_lomb_spectrum(
    time_s: Sequence[float],
    velocity: Sequence[float],
    *,
    params: dict[str, Any] | None = None,
) -> HeartRateSpectrumData:
    """Compute Lomb-Scargle periodogram diagnostic data.

    Args:
        time_s: Time samples in seconds.
        velocity: Velocity samples aligned to ``time_s``.
        params: Optional heart-rate detection parameters.

    Returns:
        Lomb-Scargle spectrum data with peak and estimate metadata.

    Raises:
        ValueError: If fewer than 256 finite samples are available after
            preprocessing.
    """
    pre = compute_preprocessing(time_s, velocity, params=params)
    finite = np.isfinite(pre.time_s) & np.isfinite(pre.x_pre)
    if int(np.sum(finite)) < 256:
        raise ValueError("Not enough finite samples for Lomb-Scargle plot.")

    f_peak, snr, frequency_hz, power = dominant_freq_lombscargle(
        pre.time_s[finite],
        pre.x_pre[finite],
        band_hz=pre.band_hz,
        n_freq=pre.params.lomb_n_freq,
    )
    estimate, _debug = estimate_heart_rate_global(
        pre.time_s,
        pre.velocity,
        method=LOMB_METHOD,
        **pre.params.to_core_kwargs(),
    )
    return HeartRateSpectrumData(
        method=LOMB_METHOD,
        frequency_hz=frequency_hz,
        power=power,
        f_peak_hz=float(f_peak),
        snr=float(snr),
        estimate=estimate,
        params=pre.params,
    )


def compute_segment_series(
    time_s: Sequence[float],
    velocity: Sequence[float],
    *,
    params: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Compute windowed segment heart-rate series for diagnostic plots.

    Args:
        time_s: Time samples in seconds.
        velocity: Velocity samples aligned to ``time_s``.
        params: Optional heart-rate detection parameters.

    Returns:
        Dictionary of numpy arrays with ``t_center``, ``bpm``, ``snr``,
        ``valid_frac``, ``edge_flag``, and ``band_concentration``.
    """
    normalized = normalize_plot_params(params)
    local_params = default_heart_rate_detection_params()
    if params is not None:
        local_params.update(params)
    return estimate_heart_rate_segment_series(
        time_s,
        velocity,
        method=WELCH_METHOD,
        seg_win_sec=float(local_params["seg_win_sec"]),
        seg_step_sec=float(local_params["seg_step_sec"]),
        seg_min_valid_frac=float(local_params["seg_min_valid_frac"]),
        **normalized.to_core_kwargs(),
    )
