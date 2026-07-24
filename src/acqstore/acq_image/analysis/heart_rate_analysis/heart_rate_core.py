"""Core heart-rate estimation algorithms for velocity time-series.

This module is the numeric core of CloudScope heart-rate analysis. It lives in
the ``acqstore`` backend layer and owns the scientific computation only; it has
no knowledge of ``AcqImage``, ROIs, persistence, or GUI concerns. The
``HeartRateAnalysis`` wrapper adapts these functions to the analysis framework.

The estimators operate on ``(time_s, velocity)`` arrays. Time is in seconds and
velocity is in physical units. Heart rate is reported in beats-per-minute (bpm)
and in Hz. Two spectral methods are provided:

- ``"lombscargle"``: Lomb-Scargle periodogram, robust to missing samples.
- ``"welch"``: Welch PSD after small-gap interpolation and band-pass filtering.

The algorithms in this module are ported verbatim from the original Radon
velocity heart-rate implementation and are intended to remain numerically
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence

import numpy as np

try:
    from scipy import signal
except Exception as e:  # pragma: no cover
    signal = None  # type: ignore


@dataclass(frozen=True)
class HeartRateEstimate:
    """Heart rate estimate.

    bpm, f_hz:
        Estimated heart rate.
    t_start, t_end:
        Time bounds of the analyzed interval.
    n_samples:
        Number of original samples in the interval.
    n_valid:
        Number of finite samples used.
    snr:
        Confidence-ish metric (bigger is better; method-dependent).
    method:
        'welch' (after interpolation+bandpass) or 'lombscargle' (handles gaps).
    """
    bpm: float
    f_hz: float
    t_start: float
    t_end: float
    n_samples: int
    n_valid: int
    snr: float
    method: str
    edge_flag: bool = False
    edge_hz_distance: Optional[float] = None
    band_concentration: Optional[float] = None


class HRStatus(str, Enum):
    """Structured status labels for HR estimation and summary classification."""

    OK = "ok"
    INSUFFICIENT_VALID = "insufficient_valid"
    NO_PEAK_LOMB = "no_peak_lomb"
    NO_PEAK_WELCH = "no_peak_welch"
    METHOD_DISAGREE = "method_disagree"
    OTHER_ERROR = "other_error"


def _effective_edge_margin_hz(
    band_hz: tuple[float, float],
    edge_margin_hz: Optional[float],
) -> float:
    lo, hi = band_hz
    if edge_margin_hz is None:
        return float(max(0.2, 0.05 * (hi - lo)))
    return float(max(0.0, edge_margin_hz))


def _compute_qc_metrics_from_spectrum(
    f_axis: np.ndarray,
    power_axis: np.ndarray,
    *,
    f_peak: float,
    band_hz: tuple[float, float],
    edge_margin_hz: Optional[float],
    peak_half_width_hz: float,
) -> tuple[bool, Optional[float], Optional[float]]:
    """Compute edge-flag and band-concentration metrics for one peak.

    Args:
        f_axis: Frequency axis in Hz.
        power_axis: Spectrum/power values aligned to ``f_axis``.
        f_peak: Peak frequency in Hz.
        band_hz: Frequency band ``(lo, hi)`` in Hz.
        edge_margin_hz: Optional explicit edge margin in Hz. If ``None``, uses
            ``max(0.2, 0.05 * (hi - lo))``.
        peak_half_width_hz: Half width for local neighborhood integration in Hz.

    Returns:
        tuple[bool, Optional[float], Optional[float]]: ``(edge_flag,
        edge_hz_distance, band_concentration)``.
    """
    lo, hi = band_hz
    margin = _effective_edge_margin_hz(band_hz, edge_margin_hz)
    edge_dist = float(min(abs(float(f_peak) - lo), abs(hi - float(f_peak))))
    edge_flag = bool(float(f_peak) <= (lo + margin) or float(f_peak) >= (hi - margin))

    m_band = (f_axis >= lo) & (f_axis <= hi)
    if not np.any(m_band):
        return edge_flag, edge_dist, None

    f_band = f_axis[m_band]
    p_band = power_axis[m_band]
    total_power = float(np.sum(p_band))
    if not np.isfinite(total_power) or total_power <= 0:
        return edge_flag, edge_dist, None

    half_width = float(max(0.0, peak_half_width_hz))
    m_peak = np.abs(f_band - float(f_peak)) <= half_width
    local_power = float(np.sum(p_band[m_peak])) if np.any(m_peak) else 0.0
    band_concentration = float(local_power / total_power)
    return edge_flag, edge_dist, band_concentration


def estimate_fs(time_s: np.ndarray) -> float:
    """Estimate sample rate (Hz) from median dt."""
    dt = np.diff(time_s)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        raise ValueError("Cannot estimate sampling rate: time array has no positive finite diffs.")
    return 1.0 / float(np.median(dt))


def winsorize_mad(x: np.ndarray, *, k: float = 4.0) -> np.ndarray:
    """Clip outliers using MAD around the median."""
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad == 0:
        return x.copy()
    sigma = 1.4826 * mad
    lo = med - k * sigma
    hi = med + k * sigma
    return np.clip(x, lo, hi)


def detrend_finite(x: np.ndarray) -> np.ndarray:
    """Remove mean from finite values."""
    x = np.asarray(x, dtype=float)
    out = x.copy()
    m = np.isfinite(out)
    if np.any(m):
        out[m] = out[m] - float(np.mean(out[m]))
    return out


def bandpass_filter(
    x: np.ndarray,
    fs: float,
    *,
    band_hz: tuple[float, float],
    order: int = 3,
) -> np.ndarray:
    """Butterworth band-pass filter (zero-phase). Requires SciPy."""
    if signal is None:  # pragma: no cover
        raise ImportError("scipy is required for bandpass_filter")
    lo, hi = band_hz
    if lo <= 0 or hi <= 0 or hi <= lo:
        raise ValueError(f"Invalid band_hz={band_hz}")
    nyq = 0.5 * fs
    b, a = signal.butter(order, [lo / nyq, hi / nyq], btype="bandpass")
    return signal.filtfilt(b, a, x)


def dominant_freq_welch(
    x: np.ndarray,
    fs: float,
    *,
    band_hz: tuple[float, float],
    nperseg: int,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Dominant frequency via Welch PSD inside band_hz. Returns (f_peak, snr, f, Pxx)."""
    if signal is None:  # pragma: no cover
        raise ImportError("scipy is required for dominant_freq_welch")

    f, Pxx = signal.welch(x, fs=fs, nperseg=min(nperseg, x.size), detrend=False, scaling="density")
    lo, hi = band_hz
    m = (f >= lo) & (f <= hi)
    if not np.any(m):
        return float("nan"), 0.0, f, Pxx
    f_band = f[m]
    P_band = Pxx[m]
    i_peak = int(np.argmax(P_band))
    f_peak = float(f_band[i_peak])
    med = float(np.median(P_band)) if P_band.size else 0.0
    snr = float(P_band[i_peak] / med) if med > 0 else float("inf")
    return f_peak, snr, f, Pxx


def dominant_freq_lombscargle(
    t: np.ndarray,
    x: np.ndarray,
    *,
    band_hz: tuple[float, float],
    n_freq: int = 512,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Dominant frequency via Lomb–Scargle periodogram (handles missing samples).

    Returns (f_peak, snr, f_grid, power).
    """
    if signal is None:  # pragma: no cover
        raise ImportError("scipy is required for dominant_freq_lombscargle")

    lo, hi = band_hz
    if lo <= 0 or hi <= lo:
        raise ValueError(f"Invalid band_hz={band_hz}")

    # Frequency grid in rad/s for scipy.signal.lombscargle
    f_grid = np.linspace(lo, hi, int(n_freq), dtype=float)
    w = 2.0 * np.pi * f_grid

    # lombscargle expects mean-removed input; do that anyway
    x0 = x - float(np.mean(x))
    power = signal.lombscargle(t, x0, w, precenter=False, normalize=True)

    i_peak = int(np.argmax(power))
    f_peak = float(f_grid[i_peak])

    med = float(np.median(power)) if power.size else 0.0
    snr = float(power[i_peak] / med) if med > 0 else float("inf")
    return f_peak, snr, f_grid, power


def interpolate_small_gaps(
    t: np.ndarray,
    x: np.ndarray,
    *,
    max_gap_sec: float,
) -> np.ndarray:
    """Linearly interpolate over NaN gaps shorter than max_gap_sec; leave longer gaps as NaN."""
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)

    m = np.isfinite(x) & np.isfinite(t)
    if np.sum(m) < 2:
        return x.copy()

    # Identify NaN runs and their duration; fill only if short
    out = x.copy()
    isn = ~np.isfinite(out)
    if not np.any(isn):
        return out

    idx = np.arange(out.size)
    valid_idx = idx[m]
    valid_t = t[m]
    valid_x = out[m]

    # Global linear interpolation for all NaNs
    interp = np.interp(t, valid_t, valid_x)

    # Decide which NaNs are in short gaps
    nan_idx = np.flatnonzero(isn)
    # Run-length encode nan_idx
    runs = []
    s = int(nan_idx[0])
    prev = int(nan_idx[0])
    for i in nan_idx[1:]:
        i = int(i)
        if i == prev + 1:
            prev = i
        else:
            runs.append((s, prev))
            s = prev = i
    runs.append((s, prev))

    for a, b in runs:
        gap_dur = float(t[b] - t[a]) if b > a else 0.0
        if gap_dur <= max_gap_sec:
            out[a:b+1] = interp[a:b+1]
        # else: leave as NaN
    return out


def estimate_heart_rate_global(
    time_s: Sequence[float],
    velocity: Sequence[float],
    *,
    bpm_band: tuple[float, float] = (240.0, 600.0),
    use_abs: bool = True,
    outlier_k_mad: float = 4.0,
    method: str = "lombscargle",
    lomb_n_freq: int = 512,
    interp_max_gap_sec: float = 0.05,
    bandpass_order: int = 3,
    nperseg_sec: float = 2.0,
    edge_margin_hz: Optional[float] = None,
    peak_half_width_hz: float = 0.5,
) -> tuple[Optional[HeartRateEstimate], dict]:
    """Estimate global heart rate from one trace using Lomb or Welch.

    Args:
        time_s: Time samples in seconds.
        velocity: Velocity samples aligned to ``time_s``.
        bpm_band: Allowed heart-rate band in bpm.
        use_abs: Whether to analyze absolute velocity.
        outlier_k_mad: MAD winsorization factor.
        method: ``"lombscargle"`` or ``"welch"``.
        lomb_n_freq: Lomb-Scargle frequency grid resolution.
        interp_max_gap_sec: Max interpolation gap for Welch mode.
        bandpass_order: Welch bandpass filter order.
        nperseg_sec: Welch PSD segment duration in seconds.
        edge_margin_hz: Optional edge margin override in Hz.
        peak_half_width_hz: Half-width around detected peak for concentration metric.

    Returns:
        tuple[Optional[HeartRateEstimate], dict]: Estimate and debug payload.
        The debug dict always contains structured ``status`` (``HRStatus`` value
        string). It contains ``reason`` when estimate is ``None`` and contains
        spectral arrays required for plotting when available.
    """
    t = np.asarray(time_s, dtype=float)
    v = np.asarray(velocity, dtype=float)
    n = int(t.size)

    x0 = np.abs(v) if use_abs else v
    x0 = winsorize_mad(x0, k=outlier_k_mad)
    x0 = detrend_finite(x0)

    m = np.isfinite(t) & np.isfinite(x0)
    n_valid = int(np.sum(m))
    if n_valid < 256:
        return None, {
            "status": HRStatus.INSUFFICIENT_VALID.value,
            "reason": "not_enough_valid_samples",
            "note": "not enough valid samples for reliable HR estimate",
            "n_valid": n_valid,
        }

    lo_bpm, hi_bpm = bpm_band
    band_hz = (lo_bpm / 60.0, hi_bpm / 60.0)

    if method.lower() == "lombscargle":
        f_peak, snr, f_grid, power = dominant_freq_lombscargle(t[m], x0[m], band_hz=band_hz, n_freq=lomb_n_freq)
        if not np.isfinite(f_peak):
            return None, {
                "status": HRStatus.NO_PEAK_LOMB.value,
                "reason": "no_lomb_peak_in_band",
                "note": "no Lomb-Scargle peak found in analysis band",
                "n_valid": n_valid,
                "band_hz": band_hz,
                "f_grid": f_grid,
                "power": power,
                "x_used": x0,
                "mask": m,
            }
        edge_flag, edge_dist, band_concentration = _compute_qc_metrics_from_spectrum(
            f_grid,
            power,
            f_peak=f_peak,
            band_hz=band_hz,
            edge_margin_hz=edge_margin_hz,
            peak_half_width_hz=peak_half_width_hz,
        )
        est = HeartRateEstimate(
            bpm=float(60.0 * f_peak),
            f_hz=float(f_peak),
            t_start=float(np.nanmin(t[m])),
            t_end=float(np.nanmax(t[m])),
            n_samples=n,
            n_valid=n_valid,
            snr=float(snr),
            method="lombscargle",
            edge_flag=edge_flag,
            edge_hz_distance=edge_dist,
            band_concentration=band_concentration,
        )
        dbg = {
            "status": HRStatus.OK.value,
            "note": "",
            "band_hz": band_hz,
            "f_grid": f_grid,
            "power": power,
            "x_used": x0,
            "mask": m,
            "edge_flag": edge_flag,
            "edge_hz_distance": edge_dist,
            "band_concentration": band_concentration,
            "edge_margin_hz": _effective_edge_margin_hz(band_hz, edge_margin_hz),
            "peak_half_width_hz": float(peak_half_width_hz),
        }
        return est, dbg

    if method.lower() == "welch":
        try:
            x_interp = interpolate_small_gaps(t, x0, max_gap_sec=interp_max_gap_sec)
            m_welch = np.isfinite(t) & np.isfinite(x_interp)
            n_valid_welch = int(np.sum(m_welch))
            if n_valid_welch < 256:
                return None, {
                    "status": HRStatus.INSUFFICIENT_VALID.value,
                    "reason": "not_enough_valid_samples_after_interp",
                    "note": "not enough valid samples after interpolation",
                    "n_valid": n_valid_welch,
                    "x_used": x0,
                    "x_interp": x_interp,
                    "mask": m_welch,
                    "band_hz": band_hz,
                }

            fs = estimate_fs(t[m_welch])
            xf = bandpass_filter(x_interp[m_welch], fs, band_hz=band_hz, order=bandpass_order)
            nperseg = int(np.clip(round(fs * nperseg_sec), 128, 8192))
            f_peak, snr, f, Pxx = dominant_freq_welch(xf, fs, band_hz=band_hz, nperseg=nperseg)
            edge_flag, edge_dist, band_concentration = _compute_qc_metrics_from_spectrum(
                f,
                Pxx,
                f_peak=f_peak,
                band_hz=band_hz,
                edge_margin_hz=edge_margin_hz,
                peak_half_width_hz=peak_half_width_hz,
            )
            if not np.isfinite(f_peak):
                return None, {
                    "status": HRStatus.NO_PEAK_WELCH.value,
                    "reason": "no_welch_peak_in_band",
                    "note": "no Welch peak found in analysis band",
                    "n_valid": n_valid_welch,
                    "x_used": x0,
                    "x_interp": x_interp,
                    "x_bandpassed": xf,
                    "mask": m_welch,
                    "band_hz": band_hz,
                    "fs_hz": float(fs),
                    "nperseg": int(nperseg),
                    "f": f,
                    "Pxx": Pxx,
                }

            est = HeartRateEstimate(
                bpm=float(60.0 * f_peak),
                f_hz=float(f_peak),
                t_start=float(np.nanmin(t[m_welch])),
                t_end=float(np.nanmax(t[m_welch])),
                n_samples=n,
                n_valid=n_valid_welch,
                snr=float(snr),
                method="welch",
                edge_flag=edge_flag,
                edge_hz_distance=edge_dist,
                band_concentration=band_concentration,
            )
            dbg = {
                "status": HRStatus.OK.value,
                "note": "",
                "band_hz": band_hz,
                "fs_hz": float(fs),
                "nperseg": int(nperseg),
                "f": f,
                "Pxx": Pxx,
                "x_used": x0,
                "x_interp": x_interp,
                "x_bandpassed": xf,
                "mask": m_welch,
                "edge_flag": edge_flag,
                "edge_hz_distance": edge_dist,
                "band_concentration": band_concentration,
                "edge_margin_hz": _effective_edge_margin_hz(band_hz, edge_margin_hz),
                "peak_half_width_hz": float(peak_half_width_hz),
            }
            return est, dbg
        except Exception as e:
            return None, {
                "status": HRStatus.OTHER_ERROR.value,
                "reason": f"welch_failed: {e}",
                "note": "Welch processing failed",
                "x_used": x0,
                "band_hz": band_hz,
            }

    raise ValueError(f"Unknown method={method!r}")


def estimate_heart_rate_segment_series(
    time_s: Sequence[float],
    velocity: Sequence[float],
    *,
    method: str = "welch",
    bpm_band: tuple[float, float] = (240.0, 600.0),
    use_abs: bool = True,
    outlier_k_mad: float = 4.0,
    lomb_n_freq: int = 512,
    interp_max_gap_sec: float = 0.05,
    bandpass_order: int = 3,
    nperseg_sec: float = 2.0,
    edge_margin_hz: Optional[float] = None,
    peak_half_width_hz: float = 0.5,
    seg_win_sec: float = 6.0,
    seg_step_sec: float = 1.0,
    seg_min_valid_frac: float = 0.5,
) -> dict[str, np.ndarray]:
    """Estimate windowed HR time-series for QC and non-stationarity checks.

    Args:
        time_s: Time samples in seconds.
        velocity: Velocity samples aligned to ``time_s``.
        method: Global estimator method for each window.
        bpm_band: Analysis heart-rate bounds in bpm.
        use_abs: Whether to analyze absolute velocity.
        outlier_k_mad: MAD clip factor.
        lomb_n_freq: Lomb-Scargle frequency grid size.
        interp_max_gap_sec: Maximum interpolation gap for Welch path.
        bandpass_order: Welch bandpass order.
        nperseg_sec: Welch segment size in seconds.
        edge_margin_hz: Optional edge margin override.
        peak_half_width_hz: Half-width around peak for band concentration metric.
        seg_win_sec: Window length in seconds.
        seg_step_sec: Window step in seconds.
        seg_min_valid_frac: Minimum finite-sample fraction required per window.

    Returns:
        dict[str, np.ndarray]: Arrays with keys ``t_center``, ``bpm``, ``snr``,
        ``valid_frac``, ``edge_flag``, ``band_concentration``.
    """
    t = np.asarray(time_s, dtype=float)
    v = np.asarray(velocity, dtype=float)
    finite_all = np.isfinite(t) & np.isfinite(v)

    if t.size == 0 or not np.any(finite_all):
        empty = np.array([], dtype=float)
        return {
            "t_center": empty,
            "bpm": empty,
            "snr": empty,
            "valid_frac": empty,
            "edge_flag": empty,
            "band_concentration": empty,
        }

    t_start = float(np.nanmin(t[finite_all]))
    t_end = float(np.nanmax(t[finite_all]))
    win = float(seg_win_sec)
    step = float(seg_step_sec)
    min_frac = float(np.clip(seg_min_valid_frac, 0.0, 1.0))
    if win <= 0 or step <= 0:
        raise ValueError("seg_win_sec and seg_step_sec must be > 0.")

    centers: list[float] = []
    bpm_list: list[float] = []
    snr_list: list[float] = []
    valid_frac_list: list[float] = []
    edge_list: list[float] = []
    bc_list: list[float] = []

    left = t_start
    while (left + win) <= (t_end + 1e-9):
        right = left + win
        m_win = (t >= left) & (t < right)
        if not np.any(m_win):
            left += step
            continue
        tv = t[m_win]
        vv = v[m_win]
        finite = np.isfinite(tv) & np.isfinite(vv)
        frac = float(np.sum(finite) / tv.size) if tv.size else 0.0

        center = float((left + right) * 0.5)
        centers.append(center)
        valid_frac_list.append(frac)

        if frac < min_frac:
            bpm_list.append(float("nan"))
            snr_list.append(float("nan"))
            edge_list.append(float("nan"))
            bc_list.append(float("nan"))
            left += step
            continue

        est, _dbg = estimate_heart_rate_global(
            tv,
            vv,
            bpm_band=bpm_band,
            use_abs=use_abs,
            outlier_k_mad=outlier_k_mad,
            method=method,
            lomb_n_freq=lomb_n_freq,
            interp_max_gap_sec=interp_max_gap_sec,
            bandpass_order=bandpass_order,
            nperseg_sec=nperseg_sec,
            edge_margin_hz=edge_margin_hz,
            peak_half_width_hz=peak_half_width_hz,
        )
        if est is None:
            bpm_list.append(float("nan"))
            snr_list.append(float("nan"))
            edge_list.append(float("nan"))
            bc_list.append(float("nan"))
        else:
            bpm_list.append(float(est.bpm))
            snr_list.append(float(est.snr))
            edge_list.append(1.0 if bool(est.edge_flag) else 0.0)
            bc = est.band_concentration
            bc_list.append(float("nan") if bc is None else float(bc))

        left += step

    return {
        "t_center": np.asarray(centers, dtype=float),
        "bpm": np.asarray(bpm_list, dtype=float),
        "snr": np.asarray(snr_list, dtype=float),
        "valid_frac": np.asarray(valid_frac_list, dtype=float),
        "edge_flag": np.asarray(edge_list, dtype=float),
        "band_concentration": np.asarray(bc_list, dtype=float),
    }
