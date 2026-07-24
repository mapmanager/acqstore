"""Parameter normalization for heart-rate analysis.

This module owns the small translation layer between serialized CloudScope
heart-rate ``detection_params`` and the keyword arguments expected by the
numeric heart-rate core. It is shared by the persisted analysis wrapper and the
diagnostic plotting helpers so scripts, notebooks, and analysis runs use the
same parameter semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

EDGE_MARGIN_AUTO_SENTINEL = -1.0


@dataclass(frozen=True, slots=True)
class HeartRateCoreParams:
    """Normalized heart-rate parameters for core computation.

    Args:
        bpm_band: Lower/upper heart-rate bounds in beats per minute.
        use_abs: Whether to analyze absolute velocity.
        outlier_k_mad: MAD winsorization factor.
        lomb_n_freq: Number of frequencies in the Lomb-Scargle grid.
        interp_max_gap_sec: Maximum NaN gap interpolated for Welch processing.
        bandpass_order: Butterworth band-pass order for Welch processing.
        nperseg_sec: Welch PSD segment duration in seconds.
        edge_margin_hz: Optional edge margin in Hz. ``None`` means auto.
        peak_half_width_hz: Half-width around the detected peak used for band
            concentration.
    """

    bpm_band: tuple[float, float]
    use_abs: bool
    outlier_k_mad: float
    lomb_n_freq: int
    interp_max_gap_sec: float
    bandpass_order: int
    nperseg_sec: float
    edge_margin_hz: float | None
    peak_half_width_hz: float

    @property
    def band_hz(self) -> tuple[float, float]:
        """Return the configured heart-rate band in Hz.

        Returns:
            Tuple ``(low_hz, high_hz)``.
        """
        return (float(self.bpm_band[0]) / 60.0, float(self.bpm_band[1]) / 60.0)

    def to_core_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments accepted by core estimator functions.

        Returns:
            Dictionary suitable for ``estimate_heart_rate_global`` and related
            core helpers.
        """
        return {
            "bpm_band": self.bpm_band,
            "use_abs": self.use_abs,
            "outlier_k_mad": self.outlier_k_mad,
            "lomb_n_freq": self.lomb_n_freq,
            "interp_max_gap_sec": self.interp_max_gap_sec,
            "bandpass_order": self.bandpass_order,
            "nperseg_sec": self.nperseg_sec,
            "edge_margin_hz": self.edge_margin_hz,
            "peak_half_width_hz": self.peak_half_width_hz,
        }


def normalize_heart_rate_detection_params(params: dict[str, Any]) -> HeartRateCoreParams:
    """Normalize serialized heart-rate detection parameters.

    Args:
        params: Detection parameter mapping from ``HeartRateAnalysis``.

    Returns:
        Normalized immutable parameter object for core computation.
    """
    edge_margin_raw = float(params["edge_margin_hz"])
    edge_margin_hz = None if edge_margin_raw < 0.0 else edge_margin_raw
    return HeartRateCoreParams(
        bpm_band=(float(params["bpm_min"]), float(params["bpm_max"])),
        use_abs=bool(params["use_abs"]),
        outlier_k_mad=float(params["outlier_k_mad"]),
        lomb_n_freq=int(params["lomb_n_freq"]),
        interp_max_gap_sec=float(params["interp_max_gap_sec"]),
        bandpass_order=int(params["bandpass_order"]),
        nperseg_sec=float(params["nperseg_sec"]),
        edge_margin_hz=edge_margin_hz,
        peak_half_width_hz=float(params["peak_half_width_hz"]),
    )
