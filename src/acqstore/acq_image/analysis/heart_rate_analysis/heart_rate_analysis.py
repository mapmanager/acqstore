"""Heart-rate analysis for AcqImage channel/ROI selections.

Heart-rate analysis estimates a global heart rate from the velocity time-series
produced by a required parent ``radon_velocity`` analysis for the same
``(channel, roi_id)`` selection. The parent velocity is read through the backend
``get_plot_data()`` API so heart-rate code never depends on Radon-specific table
column names.

This analysis produces a small JSON-serializable summary dictionary only. It
does not produce a CSV table, so persistence is handled entirely through the
AcqImage sidecar JSON (``detection_params`` and ``summary``). Two spectral
methods are evaluated, ``"lombscargle"`` and ``"welch"``, and their agreement is
reported as a quality-control metric.

The numeric computation lives in :mod:`heart_rate_core`; this module only adapts
that core to the analysis framework.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from acqstore.acq_image.analysis.data_provider import AnalysisDataProvider
from acqstore.acq_image.analysis.model import (
    AnalysisPlotData,
    AnalysisResult,
    AnalysisRunContext,
    BaseAnalysis,
    DetectionParamSchema,
    DetectionValueType,
)
from acqstore.acq_image.analysis.registry import register_analysis_class
from acqstore.acq_image.analysis.heart_rate_analysis.heart_rate_core import (
    HRStatus,
    HeartRateEstimate,
    estimate_heart_rate_global,
    estimate_heart_rate_segment_series,
)
from acqstore.acq_image.analysis.heart_rate_analysis.heart_rate_params import (
    EDGE_MARGIN_AUTO_SENTINEL,
    normalize_heart_rate_detection_params,
)

RADON_VELOCITY_ANALYSIS_NAME = "radon_velocity"
HEART_RATE_SUMMARY_VERSION = 1
LOMB_METHOD = "lombscargle"
WELCH_METHOD = "welch"


@register_analysis_class
class HeartRateAnalysis(BaseAnalysis):
    """Heart-rate analysis dependent on Radon velocity plot data.

    The analysis runs on the velocity-versus-time series of the parent
    ``radon_velocity`` analysis for the same ``(channel, roi_id)``. Time is in
    seconds and heart rate is reported in beats-per-minute (bpm) and Hz. Both the
    Lomb-Scargle and Welch estimators are evaluated on every run, and a compact
    summary dictionary is stored in :attr:`AnalysisResult.summary`. No CSV table
    is produced; results are persisted only through the AcqImage sidecar JSON.

    The detection parameter ``edge_margin_hz`` uses ``-1.0`` as a sentinel that
    means "auto" (the core then derives a default edge margin from the band
    width). Heart-rate band bounds are supplied as ``bpm_min`` and ``bpm_max``.

    Args:
        channel: Zero-based channel index for analysis.
        roi_id: ROI identifier for analysis.
        detection_params: Optional detection parameters. Missing values are
            filled from ``detection_schema`` defaults.
    """

    analysis_name = "heart_rate"
    analysis_version = HEART_RATE_SUMMARY_VERSION
    summary_columns = (
        "analysis_date",
        "analysis_time",
        "analysis_version",
        "version",
        "n_total",
        "n_valid",
        "valid_frac",
        "t_min",
        "t_max",
        "status",
        "status_note",
        "lomb_bpm",
        "lomb_f_hz",
        "lomb_snr",
        "lomb_edge_flag",
        "lomb_edge_hz_distance",
        "lomb_band_concentration",
        "lomb_status",
        "lomb_status_note",
        "welch_bpm",
        "welch_f_hz",
        "welch_snr",
        "welch_edge_flag",
        "welch_edge_hz_distance",
        "welch_band_concentration",
        "welch_status",
        "welch_status_note",
        "agreement_delta_bpm",
        "agreement_abs_delta_bpm",
        "agreement_delta_hz",
        "agreement_agree_ok",
        "segments_n_windows",
        "segments_n_valid_windows",
        "segments_median_bpm",
        "segments_iqr_bpm",
    )
    exclusive_group = None
    depends_on = (RADON_VELOCITY_ANALYSIS_NAME,)
    detection_schema = (
        DetectionParamSchema(
            name="bpm_min",
            display_name="Heart rate min",
            value_type=DetectionValueType.FLOAT,
            default=240.0,
            description="Lower heart-rate bound of the analysis band.",
            unit="bpm",
        ),
        DetectionParamSchema(
            name="bpm_max",
            display_name="Heart rate max",
            value_type=DetectionValueType.FLOAT,
            default=600.0,
            description="Upper heart-rate bound of the analysis band.",
            unit="bpm",
        ),
        DetectionParamSchema(
            name="use_abs",
            display_name="Use absolute velocity",
            value_type=DetectionValueType.BOOL,
            default=True,
            description="Analyze absolute velocity instead of signed velocity.",
        ),
        DetectionParamSchema(
            name="outlier_k_mad",
            display_name="Outlier clip (MAD)",
            value_type=DetectionValueType.FLOAT,
            default=4.0,
            description="MAD winsorization factor applied during preprocessing.",
        ),
        DetectionParamSchema(
            name="lomb_n_freq",
            display_name="Lomb frequencies",
            value_type=DetectionValueType.INT,
            default=512,
            description="Number of frequencies in the Lomb-Scargle grid.",
        ),
        DetectionParamSchema(
            name="interp_max_gap_sec",
            display_name="Max interp gap",
            value_type=DetectionValueType.FLOAT,
            default=0.05,
            description="Maximum NaN gap interpolated for the Welch path.",
            unit="s",
        ),
        DetectionParamSchema(
            name="bandpass_order",
            display_name="Bandpass order",
            value_type=DetectionValueType.INT,
            default=3,
            description="Butterworth band-pass order for the Welch path.",
        ),
        DetectionParamSchema(
            name="nperseg_sec",
            display_name="Welch segment",
            value_type=DetectionValueType.FLOAT,
            default=2.0,
            description="Welch PSD segment duration.",
            unit="s",
        ),
        DetectionParamSchema(
            name="edge_margin_hz",
            display_name="Edge margin",
            value_type=DetectionValueType.FLOAT,
            default=EDGE_MARGIN_AUTO_SENTINEL,
            description="Edge margin in Hz for edge flagging. Use -1.0 for auto.",
            unit="Hz",
        ),
        DetectionParamSchema(
            name="peak_half_width_hz",
            display_name="Peak half width",
            value_type=DetectionValueType.FLOAT,
            default=0.5,
            description="Half-width around the peak used for band concentration.",
            unit="Hz",
        ),
        DetectionParamSchema(
            name="agree_tol_bpm",
            display_name="Agreement tolerance",
            value_type=DetectionValueType.FLOAT,
            default=30.0,
            description="Maximum Lomb-vs-Welch bpm delta considered agreement.",
            unit="bpm",
        ),
        DetectionParamSchema(
            name="do_segments",
            display_name="Compute segments",
            value_type=DetectionValueType.BOOL,
            default=False,
            description="Compute a compact windowed segment summary.",
        ),
        DetectionParamSchema(
            name="seg_win_sec",
            display_name="Segment window",
            value_type=DetectionValueType.FLOAT,
            default=6.0,
            description="Segment window length when segments are computed.",
            unit="s",
        ),
        DetectionParamSchema(
            name="seg_step_sec",
            display_name="Segment step",
            value_type=DetectionValueType.FLOAT,
            default=1.0,
            description="Segment window step when segments are computed.",
            unit="s",
        ),
        DetectionParamSchema(
            name="seg_min_valid_frac",
            display_name="Segment min valid",
            value_type=DetectionValueType.FLOAT,
            default=0.5,
            description="Minimum finite-sample fraction required per segment window.",
        ),
    )

    def set_detection_params(self, detection_params: dict[str, Any]) -> None:
        """Replace detection parameters and mark this analysis dirty.

        Args:
            detection_params: New detection parameter mapping.

        Returns:
            None.
        """
        params = self.get_default_detection_params()
        self.validate_detection_params(detection_params)
        params.update(detection_params)
        self.detection_params = params
        self.set_dirty()

    def run(
        self,
        data_provider: AnalysisDataProvider,
        *,
        context: AnalysisRunContext | None = None,
        dependencies: dict[str, BaseAnalysis] | None = None,
    ) -> AnalysisResult:
        """Estimate heart rate from the required parent velocity analysis.

        Args:
            data_provider: Unused analysis data provider.
            context: Optional progress/cancellation context.
            dependencies: Dependency mapping containing ``radon_velocity``.

        Returns:
            Current analysis result with the heart-rate summary populated. The
            result has no table output.

        Raises:
            ValueError: If the velocity dependency or its plot data is missing.
        """
        _ = data_provider
        if context is not None:
            context.report_progress(0.0, "Preparing heart-rate analysis")
            context.raise_if_cancelled()

        plot_data = self._required_parent_plot_data(dependencies)
        time_s = np.asarray(plot_data.x, dtype=float)
        velocity = np.asarray(plot_data.y, dtype=float)

        self.result.summary = self.finalize_summary(self._build_summary(time_s, velocity))
        self.result.table = None
        self.set_dirty()

        if context is not None:
            context.report_progress(1.0, "Heart-rate analysis complete")
        return self.result

    def load_json_dict(self, record: dict[str, Any]) -> None:
        """Load heart-rate analysis state from a JSON analysis record.

        Args:
            record: Analysis sidecar record containing detection params and
                summary.

        Returns:
            None.
        """
        self.set_detection_params(dict(record.get("detection_params", {})))
        self.result.summary = dict(record.get("summary", {}))
        self.set_clean()

    def get_summary_values(self) -> dict[str, object]:
        """Return flat heart-rate summary values for analysis pools.

        Returns:
            Mapping with exactly the keys declared in
            :attr:`summary_columns`. Nested method, agreement, and segment
            summary blocks are projected into scalar columns.
        """
        summary = self.result.summary
        lomb = summary.get("lomb") if isinstance(summary.get("lomb"), dict) else {}
        welch = summary.get("welch") if isinstance(summary.get("welch"), dict) else {}
        agreement = summary.get("agreement") if isinstance(summary.get("agreement"), dict) else {}
        segments = (
            summary.get("segments_summary")
            if isinstance(summary.get("segments_summary"), dict)
            else {}
        )
        values: dict[str, object] = {
            "analysis_date": summary.get("analysis_date", pd.NA),
            "analysis_time": summary.get("analysis_time", pd.NA),
            "analysis_version": summary.get("analysis_version", pd.NA),
            "version": summary.get("version", pd.NA),
            "n_total": summary.get("n_total", pd.NA),
            "n_valid": summary.get("n_valid", pd.NA),
            "valid_frac": summary.get("valid_frac", pd.NA),
            "t_min": summary.get("t_min", pd.NA),
            "t_max": summary.get("t_max", pd.NA),
            "status": summary.get("status", pd.NA),
            "status_note": summary.get("status_note", pd.NA),
            "lomb_bpm": lomb.get("bpm", pd.NA),
            "lomb_f_hz": lomb.get("f_hz", pd.NA),
            "lomb_snr": lomb.get("snr", pd.NA),
            "lomb_edge_flag": lomb.get("edge_flag", pd.NA),
            "lomb_edge_hz_distance": lomb.get("edge_hz_distance", pd.NA),
            "lomb_band_concentration": lomb.get("band_concentration", pd.NA),
            "lomb_status": lomb.get("status", pd.NA),
            "lomb_status_note": lomb.get("status_note", pd.NA),
            "welch_bpm": welch.get("bpm", pd.NA),
            "welch_f_hz": welch.get("f_hz", pd.NA),
            "welch_snr": welch.get("snr", pd.NA),
            "welch_edge_flag": welch.get("edge_flag", pd.NA),
            "welch_edge_hz_distance": welch.get("edge_hz_distance", pd.NA),
            "welch_band_concentration": welch.get("band_concentration", pd.NA),
            "welch_status": welch.get("status", pd.NA),
            "welch_status_note": welch.get("status_note", pd.NA),
            "agreement_delta_bpm": agreement.get("delta_bpm", pd.NA),
            "agreement_abs_delta_bpm": agreement.get("abs_delta_bpm", pd.NA),
            "agreement_delta_hz": agreement.get("delta_hz", pd.NA),
            "agreement_agree_ok": agreement.get("agree_ok", pd.NA),
            "segments_n_windows": segments.get("n_windows", pd.NA),
            "segments_n_valid_windows": segments.get("n_valid_windows", pd.NA),
            "segments_median_bpm": segments.get("median_bpm", pd.NA),
            "segments_iqr_bpm": segments.get("iqr_bpm", pd.NA),
        }
        return {key: values.get(key, pd.NA) for key in self.get_summary_columns()}

    def _build_summary(self, time_s: np.ndarray, velocity: np.ndarray) -> dict[str, Any]:
        """Build the JSON-serializable heart-rate summary dictionary.

        Args:
            time_s: Velocity time samples in seconds.
            velocity: Velocity samples aligned to ``time_s``.

        Returns:
            Summary dictionary as defined by the heart-rate summary schema.
        """
        params = self.detection_params
        core_params = normalize_heart_rate_detection_params(params)
        agree_tol_bpm = float(params["agree_tol_bpm"])

        finite = np.isfinite(time_s) & np.isfinite(velocity)
        n_total = int(time_s.size)
        n_valid = int(np.sum(finite))
        valid_frac = float(n_valid / n_total) if n_total else 0.0
        t_min: float | None = None
        t_max: float | None = None
        if n_valid > 0:
            t_min = float(np.nanmin(time_s[finite]))
            t_max = float(np.nanmax(time_s[finite]))

        common_kwargs = core_params.to_core_kwargs()

        lomb_est, lomb_dbg = estimate_heart_rate_global(time_s, velocity, method=LOMB_METHOD, **common_kwargs)
        welch_est, welch_dbg = estimate_heart_rate_global(time_s, velocity, method=WELCH_METHOD, **common_kwargs)

        lomb_block = _method_block(LOMB_METHOD, lomb_est, lomb_dbg)
        welch_block = _method_block(WELCH_METHOD, welch_est, welch_dbg)

        agreement = _agreement_block(lomb_block, welch_block, agree_tol_bpm=agree_tol_bpm)
        status, status_note = _classify_status(lomb_block, welch_block, agree_tol_bpm=agree_tol_bpm)

        summary: dict[str, Any] = {
            "version": HEART_RATE_SUMMARY_VERSION,
            "n_total": n_total,
            "n_valid": n_valid,
            "valid_frac": valid_frac,
            "t_min": t_min,
            "t_max": t_max,
            "lomb": lomb_block,
            "welch": welch_block,
            "agreement": agreement,
            "status": status.value,
            "status_note": status_note,
        }

        if bool(params["do_segments"]):
            summary["segments_summary"] = self._segments_summary(time_s, velocity, common_kwargs)

        return summary

    def _segments_summary(
        self,
        time_s: np.ndarray,
        velocity: np.ndarray,
        common_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Build a compact windowed segment summary (no raw arrays).

        Args:
            time_s: Velocity time samples in seconds.
            velocity: Velocity samples aligned to ``time_s``.
            common_kwargs: Shared estimator keyword arguments.

        Returns:
            Compact segment summary dictionary with window counts and bpm stats.
        """
        params = self.detection_params
        seg = estimate_heart_rate_segment_series(
            time_s,
            velocity,
            method=WELCH_METHOD,
            seg_win_sec=float(params["seg_win_sec"]),
            seg_step_sec=float(params["seg_step_sec"]),
            seg_min_valid_frac=float(params["seg_min_valid_frac"]),
            **common_kwargs,
        )
        seg_bpm = np.asarray(seg["bpm"], dtype=float)
        valid = np.isfinite(seg_bpm)
        has_valid = bool(np.any(valid))
        q25 = float(np.nanpercentile(seg_bpm, 25)) if has_valid else None
        q75 = float(np.nanpercentile(seg_bpm, 75)) if has_valid else None
        return {
            "method": WELCH_METHOD,
            "n_windows": int(seg_bpm.size),
            "n_valid_windows": int(np.sum(valid)),
            "median_bpm": float(np.nanmedian(seg_bpm)) if has_valid else None,
            "iqr_bpm": float(q75 - q25) if (q25 is not None and q75 is not None) else None,
        }

    @staticmethod
    def _required_parent_plot_data(dependencies: dict[str, BaseAnalysis] | None) -> AnalysisPlotData:
        """Return required parent velocity plot data or raise.

        Args:
            dependencies: Analysis dependencies keyed by analysis name.

        Returns:
            Parent analysis plot data.

        Raises:
            ValueError: If the required dependency or its plot data is missing.
        """
        if dependencies is None or RADON_VELOCITY_ANALYSIS_NAME not in dependencies:
            raise ValueError("Heart-rate analysis requires radon_velocity dependency")
        parent = dependencies[RADON_VELOCITY_ANALYSIS_NAME]
        plot_data = parent.get_plot_data()
        if plot_data is None:
            raise ValueError("Heart-rate analysis requires radon_velocity plot data")
        return plot_data


def _method_block(
    method: str,
    estimate: HeartRateEstimate | None,
    debug: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build one method result block from core estimator outputs.

    Args:
        method: Method label.
        estimate: Optional estimate returned by the core analysis.
        debug: Optional debug dictionary returned by the core analysis.

    Returns:
        JSON-serializable per-method result block. Numeric fields are ``None``
        when the method produced no estimate.
    """
    dbg = debug or {}
    status = _coerce_hr_status(
        dbg.get("status"),
        default=HRStatus.OK if estimate is not None else HRStatus.OTHER_ERROR,
    )
    status_note = str(dbg.get("note", ""))

    if estimate is None:
        return {
            "method": method,
            "bpm": None,
            "f_hz": None,
            "snr": None,
            "edge_flag": None,
            "edge_hz_distance": None,
            "band_concentration": None,
            "n_samples": None,
            "n_valid": None,
            "t_start": None,
            "t_end": None,
            "status": status.value,
            "status_note": status_note,
            "reason": str(dbg.get("reason", "not_available")),
        }

    return {
        "method": method,
        "bpm": float(estimate.bpm),
        "f_hz": float(estimate.f_hz),
        "snr": float(estimate.snr),
        "edge_flag": bool(estimate.edge_flag),
        "edge_hz_distance": None if estimate.edge_hz_distance is None else float(estimate.edge_hz_distance),
        "band_concentration": None if estimate.band_concentration is None else float(estimate.band_concentration),
        "n_samples": int(estimate.n_samples),
        "n_valid": int(estimate.n_valid),
        "t_start": float(estimate.t_start),
        "t_end": float(estimate.t_end),
        "status": status.value,
        "status_note": status_note,
        "reason": None,
    }


def _agreement_block(
    lomb_block: dict[str, Any],
    welch_block: dict[str, Any],
    *,
    agree_tol_bpm: float,
) -> dict[str, Any] | None:
    """Build the Lomb-vs-Welch agreement block.

    Args:
        lomb_block: Lomb method result block.
        welch_block: Welch method result block.
        agree_tol_bpm: Maximum bpm delta considered agreement.

    Returns:
        Agreement dictionary, or ``None`` when either method lacks a bpm.
    """
    lomb_bpm = lomb_block.get("bpm")
    welch_bpm = welch_block.get("bpm")
    if lomb_bpm is None or welch_bpm is None:
        return None

    lomb_hz = lomb_block.get("f_hz") or 0.0
    welch_hz = welch_block.get("f_hz") or 0.0
    delta_bpm = float(welch_bpm) - float(lomb_bpm)
    abs_delta_bpm = float(abs(delta_bpm))
    return {
        "delta_bpm": delta_bpm,
        "abs_delta_bpm": abs_delta_bpm,
        "delta_hz": float(welch_hz) - float(lomb_hz),
        "agree_ok": bool(abs_delta_bpm <= float(agree_tol_bpm)),
        "agree_tol_bpm": float(agree_tol_bpm),
    }


def _classify_status(
    lomb_block: dict[str, Any],
    welch_block: dict[str, Any],
    *,
    agree_tol_bpm: float,
) -> tuple[HRStatus, str]:
    """Classify a heart-rate result into a cross-method rollup status.

    Args:
        lomb_block: Lomb method result block.
        welch_block: Welch method result block.
        agree_tol_bpm: Agreement threshold for the Lomb-vs-Welch bpm delta.

    Returns:
        tuple[HRStatus, str]: ``(status, status_note)``.
    """
    lomb_bpm = lomb_block.get("bpm")
    welch_bpm = welch_block.get("bpm")
    lomb_ok = lomb_bpm is not None
    welch_ok = welch_bpm is not None

    if lomb_ok and welch_ok:
        delta = abs(float(lomb_bpm) - float(welch_bpm))
        if delta > float(agree_tol_bpm):
            return HRStatus.METHOD_DISAGREE, f"abs delta bpm {delta:.1f} > tol {float(agree_tol_bpm):.1f}"
        return HRStatus.OK, ""

    if lomb_ok or welch_ok:
        return HRStatus.OK, ""

    lomb_status = _coerce_hr_status(lomb_block.get("status"), default=HRStatus.OTHER_ERROR)
    welch_status = _coerce_hr_status(welch_block.get("status"), default=HRStatus.OTHER_ERROR)

    if lomb_status is HRStatus.INSUFFICIENT_VALID or welch_status is HRStatus.INSUFFICIENT_VALID:
        note = str(lomb_block.get("status_note") or welch_block.get("status_note") or "")
        return HRStatus.INSUFFICIENT_VALID, note
    if lomb_status is HRStatus.NO_PEAK_LOMB:
        return HRStatus.NO_PEAK_LOMB, str(lomb_block.get("status_note") or "")
    if welch_status is HRStatus.NO_PEAK_WELCH:
        return HRStatus.NO_PEAK_WELCH, str(welch_block.get("status_note") or "")

    note_parts = []
    if lomb_block.get("status_note"):
        note_parts.append(f"lomb: {lomb_block['status_note']}")
    if welch_block.get("status_note"):
        note_parts.append(f"welch: {welch_block['status_note']}")
    note = "; ".join(note_parts) if note_parts else "no method estimate available"
    return HRStatus.OTHER_ERROR, note


def _coerce_hr_status(raw: Any, *, default: HRStatus) -> HRStatus:
    """Coerce a status payload into an ``HRStatus``.

    Args:
        raw: Raw status value from a debug dictionary or summary block.
        default: Fallback status when ``raw`` cannot be parsed.

    Returns:
        Parsed ``HRStatus`` value.
    """
    if isinstance(raw, HRStatus):
        return raw
    if isinstance(raw, str):
        try:
            return HRStatus(raw)
        except ValueError:
            return default
    return default
