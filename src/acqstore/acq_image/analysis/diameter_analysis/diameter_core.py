"""Pure diameter analysis on 2D kymograph ROI images.

Axis convention (line-scan kymographs; see ``docs-dev/acqstore_kymograph_axes.md``):

    - axis 0 (rows, dim0, Y) = time. Step = ``physical_units[0]`` (seconds per line).
    - axis 1 (columns, dim1, X) = space. Step = ``physical_units[1]`` (µm per pixel).

The input ``image`` is the ROI crop from ``AnalysisDataProvider.get_roi_image``.
All outputs use ROI-local coordinates (time and space origin at the ROI's first
row/column).

This module has no imports from ``acqstore``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

THREAD_CHUNK_SIZE = 512

DIAMETER_TABLE_COLUMNS: tuple[str, ...] = (
    "time_s",
    "center_row",
    "left_edge_px",
    "right_edge_px",
    "left_edge_um",
    "right_edge_um",
    "diameter_px",
    "diameter_um",
    "diameter_px_filt",
    "diameter_um_filt",
    "diameter_was_filtered",
    "peak",
    "baseline",
    "sum_intensity",
    "edge_strength_left",
    "edge_strength_right",
    "qc_score",
    "qc_edge_violation",
    "qc_diameter_violation",
    "qc_center_violation",
)


class DiameterCancelled(RuntimeError):
    """Raised when diameter analysis is cancelled."""


@dataclass(slots=True)
class _DiameterRow:
    center_row: int
    time_s: float
    left_edge_px: float
    right_edge_px: float
    diameter_px: float
    peak: float
    baseline: float
    edge_strength_left: float
    edge_strength_right: float
    diameter_px_filt: float
    diameter_was_filtered: bool
    qc_score: float
    qc_edge_violation: bool
    qc_diameter_violation: bool
    qc_center_violation: bool
    sum_intensity: float


@dataclass(frozen=True, slots=True)
class DiameterCoreResult:
    """Pure diameter analysis output.

    Args:
        summary: Small JSON-serializable summary dictionary.
        table: Per-row results in ROI-local coordinates.
    """

    summary: dict[str, Any]
    table: pd.DataFrame


def run_diameter(
    image: np.ndarray,
    *,
    detection_params: dict[str, Any],
    physical_units: tuple[float, float],
    progress_callback: Callable[[float | None, str], None] | None = None,
    cancel_callback: Callable[[], bool] | None = None,
    use_threads: bool = True,
    max_workers: int | None = None,
) -> DiameterCoreResult:
    """Run diameter analysis on one ROI-cropped 2D kymograph.

    Args:
        image: Two-dimensional array with shape ``(time, space)``.
        detection_params: Flat detection-parameter mapping.
        physical_units: ``(seconds_per_line, um_per_pixel)``.
        progress_callback: Optional callback ``(fraction, message)``.
        cancel_callback: Optional callback returning True when cancelled.
        use_threads: Whether to analyze center rows using a thread pool.
        max_workers: Optional worker count for the thread pool.

    Returns:
        Summary and per-row table.

    Raises:
        DiameterCancelled: If cancellation is requested.
        ValueError: If inputs or detection parameters are invalid.
    """
    arr = np.asarray(image, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"image must be 2D (time, space), got shape={arr.shape!r}")

    seconds_per_line, um_per_pixel = physical_units
    if seconds_per_line <= 0 or um_per_pixel <= 0:
        raise ValueError("physical_units must be strictly positive")

    params = _validated_detection_params(detection_params)
    n_time, _n_space = arr.shape
    t0, t1, x0, x1 = 0, n_time, 0, arr.shape[1]
    centers = list(range(t0, t1, int(params["stride"])))
    total = max(len(centers), 1)

    def _report(done: int, message: str) -> None:
        if progress_callback is not None:
            progress_callback(done / total, message)

    def _check_cancel() -> None:
        if cancel_callback is not None and cancel_callback():
            raise DiameterCancelled("Diameter analysis cancelled")

    if use_threads and len(centers) > 1:
        results = _analyze_threads(
            arr,
            centers,
            params,
            t0=t0,
            t1=t1,
            x0=x0,
            x1=x1,
            seconds_per_line=seconds_per_line,
            max_workers=max_workers,
            progress_callback=_report,
            cancel_callback=_check_cancel,
            total=total,
        )
    else:
        results = []
        for idx, center_row in enumerate(centers):
            _check_cancel()
            results.append(
                _analyze_center(
                    arr,
                    center_row,
                    params,
                    t0=t0,
                    t1=t1,
                    x0=x0,
                    x1=x1,
                    seconds_per_line=seconds_per_line,
                )
            )
            _report(idx + 1, f"Analyzed row {center_row}")

    results.sort(key=lambda row: row.center_row)

    if (
        params["diameter_method"] == "gradient_edges"
        and bool(params["enable_motion_gating"])
    ):
        _apply_motion_constraints(results=results, params=params, um_per_pixel=um_per_pixel)

    kernel_size = int(params["post_filter_kernel_size"])
    _apply_median_post_filter(results, kernel_size=kernel_size)

    table = _rows_to_dataframe(results, um_per_pixel=um_per_pixel)
    summary = _build_summary(table)
    return DiameterCoreResult(summary=summary, table=table)


def _validated_detection_params(params: dict[str, Any]) -> dict[str, Any]:
    window_rows_odd = int(params["window_rows_odd"])
    if window_rows_odd < 1 or window_rows_odd % 2 == 0:
        raise ValueError("window_rows_odd must be odd and >= 1")
    stride = int(params["stride"])
    if stride < 1:
        raise ValueError("stride must be >= 1")
    gradient_sigma = float(params["gradient_sigma"])
    if gradient_sigma < 0:
        raise ValueError("gradient_sigma must be >= 0")
    gradient_min = float(params["gradient_min_edge_strength"])
    if gradient_min < 0:
        raise ValueError("gradient_min_edge_strength must be >= 0")
    for name in ("max_edge_shift_um", "max_diameter_change_um", "max_center_shift_um"):
        if float(params[name]) < 0:
            raise ValueError(f"{name} must be >= 0")
    kernel_size = int(params["post_filter_kernel_size"])
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("post_filter_kernel_size must be odd and >= 3")
    if params["gradient_kernel"] != "central_diff":
        raise ValueError("gradient_kernel must be 'central_diff'")
    if params["threshold_mode"] == "absolute":
        threshold_value = float(params["threshold_value"])
        if not math.isfinite(threshold_value):
            raise ValueError("threshold_value must be finite when threshold_mode='absolute'")
    return params


def _analyze_threads(
    image: np.ndarray,
    centers: list[int],
    params: dict[str, Any],
    *,
    t0: int,
    t1: int,
    x0: int,
    x1: int,
    seconds_per_line: float,
    max_workers: int | None,
    progress_callback: Callable[[int, str], None],
    cancel_callback: Callable[[], None],
    total: int,
) -> list[_DiameterRow]:
    def _chunked(values: list[int], size: int) -> Iterable[list[int]]:
        for start in range(0, len(values), size):
            yield values[start : start + size]

    def _process_chunk(chunk: list[int]) -> list[_DiameterRow]:
        return [
            _analyze_center(
                image,
                center_row,
                params,
                t0=t0,
                t1=t1,
                x0=x0,
                x1=x1,
                seconds_per_line=seconds_per_line,
            )
            for center_row in chunk
        ]

    chunks = list(_chunked(centers, THREAD_CHUNK_SIZE))
    completed = 0
    rows: list[_DiameterRow] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for chunk_rows in executor.map(_process_chunk, chunks):
            cancel_callback()
            rows.extend(chunk_rows)
            completed += len(chunk_rows)
            progress_callback(completed, f"Analyzed {completed}/{total} rows")
    return rows


def _analyze_center(
    image: np.ndarray,
    center_row: int,
    params: dict[str, Any],
    *,
    t0: int,
    t1: int,
    x0: int,
    x1: int,
    seconds_per_line: float,
) -> _DiameterRow:
    window_half = int(params["window_rows_odd"]) // 2
    w0 = max(t0, center_row - window_half)
    w1 = min(t1, center_row + window_half + 1)
    window = image[w0:w1, x0:x1]

    if params["binning_method"] == "mean":
        profile = np.nanmean(window, axis=0)
    else:
        profile = np.nanmedian(window, axis=0)

    profile_proc = np.asarray(profile, dtype=float)
    if params["polarity"] == "dark_on_bright":
        profile_proc = np.nanmax(profile_proc) - profile_proc

    sum_intensity = float(np.sum(profile_proc))
    baseline = float(np.nanpercentile(profile_proc, 10))
    peak = float(np.nanpercentile(profile_proc, 90))

    if params["diameter_method"] == "threshold_width":
        left_edge, right_edge, diameter, edge_flags = _threshold_width(profile_proc, params, x0)
        edge_strength_left = math.nan
        edge_strength_right = math.nan
    elif params["diameter_method"] == "gradient_edges":
        (
            left_edge,
            right_edge,
            diameter,
            edge_strength_left,
            edge_strength_right,
            edge_flags,
        ) = _gradient_edges(profile_proc, params, x0)
    else:
        raise ValueError(f"Unsupported diameter_method={params['diameter_method']!r}")

    qc_score, qc_edge_violation = _qc_metrics(
        profile=profile_proc,
        baseline=baseline,
        peak=peak,
        edge_flags=edge_flags,
        edge_strength_left=edge_strength_left,
        edge_strength_right=edge_strength_right,
        edge_strength_threshold=float(params["gradient_min_edge_strength"]),
    )

    return _DiameterRow(
        center_row=int(center_row),
        time_s=float(center_row * seconds_per_line),
        left_edge_px=float(left_edge),
        right_edge_px=float(right_edge),
        diameter_px=float(diameter),
        peak=peak,
        baseline=baseline,
        edge_strength_left=float(edge_strength_left),
        edge_strength_right=float(edge_strength_right),
        diameter_px_filt=float(diameter),
        diameter_was_filtered=False,
        qc_score=qc_score,
        qc_edge_violation=qc_edge_violation,
        qc_diameter_violation=False,
        qc_center_violation=False,
        sum_intensity=sum_intensity,
    )


def _apply_motion_constraints(
    *,
    results: list[_DiameterRow],
    params: dict[str, Any],
    um_per_pixel: float,
) -> None:
    if len(results) < 2:
        return

    edge_thr_px = float(params["max_edge_shift_um"]) / um_per_pixel
    diam_thr_px = float(params["max_diameter_change_um"]) / um_per_pixel
    center_thr_px = float(params["max_center_shift_um"]) / um_per_pixel

    for i in range(1, len(results)):
        prev = results[i - 1]
        cur = results[i]
        edge_violation = False
        diameter_violation = False
        center_violation = False

        prev_left = float(prev.left_edge_px)
        prev_right = float(prev.right_edge_px)
        cur_left = float(cur.left_edge_px)
        cur_right = float(cur.right_edge_px)

        if np.isfinite(cur_left) and np.isfinite(prev_left):
            if abs(cur_left - prev_left) > edge_thr_px:
                cur_left = math.nan
                edge_violation = True

        if np.isfinite(cur_right) and np.isfinite(prev_right):
            if abs(cur_right - prev_right) > edge_thr_px:
                cur_right = math.nan
                edge_violation = True

        prev_d = prev_right - prev_left if np.isfinite(prev_left) and np.isfinite(prev_right) else math.nan
        cur_d = cur_right - cur_left if np.isfinite(cur_left) and np.isfinite(cur_right) else math.nan
        if np.isfinite(cur_d) and np.isfinite(prev_d):
            if abs(cur_d - prev_d) > diam_thr_px:
                cur_left = math.nan
                cur_right = math.nan
                diameter_violation = True

        prev_c = 0.5 * (prev_left + prev_right) if np.isfinite(prev_left) and np.isfinite(prev_right) else math.nan
        cur_c = 0.5 * (cur_left + cur_right) if np.isfinite(cur_left) and np.isfinite(cur_right) else math.nan
        if np.isfinite(cur_c) and np.isfinite(prev_c):
            if abs(cur_c - prev_c) > center_thr_px:
                cur_left = math.nan
                cur_right = math.nan
                center_violation = True

        cur.left_edge_px = float(cur_left)
        cur.right_edge_px = float(cur_right)
        cur.diameter_px = (
            float(cur_right - cur_left)
            if np.isfinite(cur_left) and np.isfinite(cur_right)
            else math.nan
        )
        cur.diameter_px_filt = float(cur.diameter_px)
        cur.qc_edge_violation = edge_violation or cur.qc_edge_violation
        cur.qc_diameter_violation = diameter_violation
        cur.qc_center_violation = center_violation


def _apply_median_post_filter(results: list[_DiameterRow], *, kernel_size: int) -> None:
    if not results:
        return
    raw = np.asarray([row.diameter_px for row in results], dtype=float)
    filtered = _nan_safe_median_filter(raw, kernel_size=kernel_size)
    replaced = np.isfinite(raw) & np.isfinite(filtered) & (~np.isclose(raw, filtered))
    for i, row in enumerate(results):
        row.diameter_px_filt = float(filtered[i])
        row.diameter_was_filtered = bool(replaced[i])


def _threshold_width(
    profile: np.ndarray,
    params: dict[str, Any],
    x0: int,
) -> tuple[float, float, float, list[str]]:
    flags: list[str] = []
    if profile.size == 0 or not np.any(np.isfinite(profile)):
        return math.nan, math.nan, math.nan, ["empty_profile"]

    finite = profile[np.isfinite(profile)]
    baseline = float(np.nanpercentile(finite, 10))
    peak = float(np.nanpercentile(finite, 90))
    if params["threshold_mode"] == "half_max":
        threshold = baseline + 0.5 * (peak - baseline)
    elif params["threshold_mode"] == "absolute":
        threshold = float(params["threshold_value"])
    else:
        raise ValueError("threshold_mode must be 'half_max' or 'absolute'")

    above = profile >= threshold
    if not np.any(above):
        return math.nan, math.nan, math.nan, ["missing_left_edge", "missing_right_edge"]

    left_idx = int(np.argmax(above))
    right_idx = int(len(above) - 1 - np.argmax(above[::-1]))
    left_edge = float(left_idx + x0)
    right_edge = float(right_idx + x0)
    diameter = right_edge - left_edge

    if left_idx == 0:
        flags.append("missing_left_edge")
        left_edge = math.nan
    if right_idx == len(above) - 1:
        flags.append("missing_right_edge")
        right_edge = math.nan
    if not np.isfinite(left_edge) or not np.isfinite(right_edge):
        diameter = math.nan

    return left_edge, right_edge, diameter, flags


def _gradient_edges(
    profile: np.ndarray,
    params: dict[str, Any],
    x0: int,
) -> tuple[float, float, float, float, float, list[str]]:
    if profile.size == 0 or not np.any(np.isfinite(profile)):
        return math.nan, math.nan, math.nan, math.nan, math.nan, ["empty_profile"]

    finite_profile = _fill_nan_1d(profile)
    smooth = _smooth_profile(finite_profile, sigma=float(params["gradient_sigma"]))
    deriv = np.gradient(smooth)

    left_idx = int(np.argmax(deriv))
    right_idx = int(np.argmin(deriv))
    edge_strength_left = float(max(0.0, deriv[left_idx]))
    edge_strength_right = float(max(0.0, -deriv[right_idx]))

    flags: list[str] = []
    if left_idx >= right_idx:
        flags.append("gradient_invalid_order")
        return math.nan, math.nan, math.nan, edge_strength_left, edge_strength_right, flags

    if (
        edge_strength_left < float(params["gradient_min_edge_strength"])
        or edge_strength_right < float(params["gradient_min_edge_strength"])
    ):
        flags.append("gradient_low_edge_strength")

    left_edge = float(left_idx + x0)
    right_edge = float(right_idx + x0)
    diameter = float(right_edge - left_edge)
    return left_edge, right_edge, diameter, edge_strength_left, edge_strength_right, flags


def _qc_metrics(
    *,
    profile: np.ndarray,
    baseline: float,
    peak: float,
    edge_flags: list[str],
    edge_strength_left: float,
    edge_strength_right: float,
    edge_strength_threshold: float,
) -> tuple[float, bool]:
    flags = list(edge_flags)
    contrast = peak - baseline
    if not np.isfinite(contrast) or contrast <= 0:
        flags.append("low_contrast")

    pmin = float(np.nanmin(profile))
    pmax = float(np.nanmax(profile))
    dynamic_range = pmax - pmin
    if dynamic_range > 0:
        p01 = float(np.nanpercentile(profile, 1))
        p99 = float(np.nanpercentile(profile, 99))
        low_tail = (p01 - pmin) / dynamic_range
        high_tail = (pmax - p99) / dynamic_range
        saturated = bool(low_tail < 0.01 or high_tail < 0.01)
    else:
        saturated = True
    if saturated:
        flags.append("saturation")

    double_peak = False
    if profile.size >= 3 and np.isfinite(contrast) and contrast > 0:
        center = profile[1:-1]
        peaks = (center > profile[:-2]) & (center >= profile[2:])
        strong = center >= (baseline + 0.8 * contrast)
        double_peak = int(np.count_nonzero(peaks & strong)) >= 2
    if double_peak:
        flags.append("double_peak")

    if np.isfinite(edge_strength_left) and np.isfinite(edge_strength_right):
        min_strength = min(edge_strength_left, edge_strength_right)
        if min_strength < edge_strength_threshold:
            flags.append("gradient_low_edge_strength")
        if np.isfinite(contrast) and contrast > 0:
            strength_ratio = min_strength / (contrast + 1e-12)
            if strength_ratio < 0.2:
                flags.append("gradient_low_edge_strength")

    score = 1.0
    if "low_contrast" in flags:
        score -= 0.55
    if "missing_left_edge" in flags:
        score -= 0.25
    if "missing_right_edge" in flags:
        score -= 0.25
    if "gradient_invalid_order" in flags:
        score -= 0.35
    if "gradient_low_edge_strength" in flags:
        score -= 0.2
    if "saturation" in flags:
        score -= 0.1
    if "double_peak" in flags:
        score -= 0.15

    score = float(np.clip(score, 0.0, 1.0))
    edge_violation = any(
        flag in flags
        for flag in (
            "missing_left_edge",
            "missing_right_edge",
            "gradient_invalid_order",
            "gradient_low_edge_strength",
            "empty_profile",
        )
    )
    return score, edge_violation


def _nan_safe_median_filter(series: np.ndarray, kernel_size: int) -> np.ndarray:
    x = np.asarray(series, dtype=float)
    out = x.copy()
    half = kernel_size // 2
    for i in range(x.size):
        if not np.isfinite(x[i]):
            continue
        i0 = max(0, i - half)
        i1 = min(x.size, i + half + 1)
        valid = x[i0:i1][np.isfinite(x[i0:i1])]
        if valid.size > 0:
            out[i] = float(np.median(valid))
    return out


def _fill_nan_1d(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    mask = np.isfinite(out)
    if np.all(mask):
        return out
    if not np.any(mask):
        return np.zeros_like(out)
    x = np.arange(out.size, dtype=float)
    out[~mask] = np.interp(x[~mask], x[mask], out[mask])
    return out


def _smooth_profile(profile: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return profile.copy()
    try:
        from scipy.ndimage import gaussian_filter1d

        return gaussian_filter1d(profile, sigma=sigma, mode="nearest")
    except Exception:
        radius = max(1, int(round(3.0 * sigma)))
        x = np.arange(-radius, radius + 1, dtype=float)
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel /= np.sum(kernel)
        return np.convolve(profile, kernel, mode="same")


def _rows_to_dataframe(rows: list[_DiameterRow], *, um_per_pixel: float) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=list(DIAMETER_TABLE_COLUMNS))

    um = float(um_per_pixel)
    data = {
        "time_s": [row.time_s for row in rows],
        "center_row": [row.center_row for row in rows],
        "left_edge_px": [row.left_edge_px for row in rows],
        "right_edge_px": [row.right_edge_px for row in rows],
        "left_edge_um": [float(row.left_edge_px) * um for row in rows],
        "right_edge_um": [float(row.right_edge_px) * um for row in rows],
        "diameter_px": [row.diameter_px for row in rows],
        "diameter_um": [float(row.diameter_px) * um for row in rows],
        "diameter_px_filt": [row.diameter_px_filt for row in rows],
        "diameter_um_filt": [float(row.diameter_px_filt) * um for row in rows],
        "diameter_was_filtered": [row.diameter_was_filtered for row in rows],
        "peak": [row.peak for row in rows],
        "baseline": [row.baseline for row in rows],
        "sum_intensity": [row.sum_intensity for row in rows],
        "edge_strength_left": [row.edge_strength_left for row in rows],
        "edge_strength_right": [row.edge_strength_right for row in rows],
        "qc_score": [row.qc_score for row in rows],
        "qc_edge_violation": [row.qc_edge_violation for row in rows],
        "qc_diameter_violation": [row.qc_diameter_violation for row in rows],
        "qc_center_violation": [row.qc_center_violation for row in rows],
    }
    return pd.DataFrame(data)


def _build_summary(table: pd.DataFrame) -> dict[str, Any]:
    if table.empty:
        return {
            "num_rows": 0,
            "diameter_um_mean": math.nan,
            "diameter_um_median": math.nan,
            "diameter_um_cv": math.nan,
            "qc_score_mean": math.nan,
            "num_qc_edge_violations": 0,
            "num_qc_diameter_violations": 0,
            "num_qc_center_violations": 0,
        }

    y_col = "diameter_um_filt" if "diameter_um_filt" in table.columns else "diameter_um"
    diam = table[y_col].astype(float)
    valid = diam[np.isfinite(diam)]
    mean_val = float(np.nanmean(valid)) if valid.size else math.nan
    median_val = float(np.nanmedian(valid)) if valid.size else math.nan
    cv_val = math.nan
    if valid.size and np.isfinite(mean_val) and mean_val != 0.0:
        cv_val = float(np.nanstd(valid) / mean_val)

    return {
        "num_rows": int(len(table)),
        "diameter_um_mean": mean_val,
        "diameter_um_median": median_val,
        "diameter_um_cv": cv_val,
        "qc_score_mean": float(np.nanmean(table["qc_score"].astype(float))),
        "num_qc_edge_violations": int(table["qc_edge_violation"].sum()),
        "num_qc_diameter_violations": int(table["qc_diameter_violation"].sum()),
        "num_qc_center_violations": int(table["qc_center_violation"].sum()),
    }
