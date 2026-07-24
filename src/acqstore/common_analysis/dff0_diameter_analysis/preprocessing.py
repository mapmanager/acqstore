"""Signal preprocessing for triggered-event analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import medfilt, savgol_filter

from .models import DiameterFilterParams, SignalFilterMethod, TriggeredEventParams


def filter_signal(signal: np.ndarray, params: TriggeredEventParams) -> np.ndarray:
    """Return a filtered copy of one signal.

    Missing values (``NaN``) are allowed in the input.

    **Previous behavior:** any ``NaN`` in the input raised ``ValueError`` and no
    filter was applied.

    **Current behavior:**

    - ``SignalFilterMethod.NONE`` — returns a copy with ``NaN`` preserved at the
      same indices as the input (unchanged semantics aside from no longer raising).
    - ``SignalFilterMethod.MEDIAN`` and ``SignalFilterMethod.SAVGOL`` — missing
      samples are linearly interpolated across finite neighbors *before* filtering,
      so the returned array is fully finite when at least one input sample is
      finite. Callers that previously never received output because of the
      ``ValueError`` now get a continuous filtered trace; gaps are not retained
      in the filtered result (see raw input or ``filter_method=none`` for gaps).
      If every input sample is missing, filtered methods return zeros.

    Args:
        signal: One-dimensional numeric signal.
        params: Triggered-event parameters selecting the filter.

    Returns:
        Filtered floating-point array with the same length as the input.
    """
    values = np.asarray(signal, dtype=float)
    if values.ndim != 1:
        raise ValueError("signal must be one-dimensional")

    if params.filter_method is SignalFilterMethod.NONE:
        return values.copy()
    if params.filter_method is SignalFilterMethod.MEDIAN:
        kernel = params.median_kernel_points
        if kernel < 1 or kernel % 2 == 0:
            raise ValueError("median_kernel_points must be a positive odd integer")
        return medfilt(_fill_nan_1d(values), kernel_size=kernel)
    if params.filter_method is SignalFilterMethod.SAVGOL:
        window = params.savgol_window_points
        if window < 3 or window % 2 == 0:
            raise ValueError("savgol_window_points must be an odd integer >= 3")
        if params.savgol_polyorder < 0 or params.savgol_polyorder >= window:
            raise ValueError("savgol_polyorder must be >= 0 and less than window")
        if window > len(values):
            raise ValueError("savgol_window_points cannot exceed signal length")
        filled = _fill_nan_1d(values)
        return savgol_filter(filled, window_length=window, polyorder=params.savgol_polyorder)
    raise ValueError(f"Unsupported filter method: {params.filter_method}")


def _fill_nan_1d(values: np.ndarray) -> np.ndarray:
    """Linearly interpolate missing samples for one-dimensional filtering.

    Args:
        values: One-dimensional numeric array that may contain ``NaN``.

    Returns:
        Copy with non-finite samples filled by ``numpy.interp`` over finite
        neighbors. Returns the input unchanged when already finite. Returns
        zeros when no finite samples exist.
    """
    out = np.asarray(values, dtype=float).copy()
    mask = np.isfinite(out)
    if np.all(mask):
        return out
    if not np.any(mask):
        return np.zeros_like(out)
    x = np.arange(out.size, dtype=float)
    out[~mask] = np.interp(x[~mask], x[mask], out[mask])
    return out


def filter_diameter(series: pd.Series, params: DiameterFilterParams) -> pd.Series:
    """Compatibility wrapper used by the sidecar loader."""
    if params.method == "none":
        return pd.to_numeric(series, errors="coerce").astype(float)
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    trigger_params = TriggeredEventParams(
        filter_method=SignalFilterMethod(params.method),
        median_kernel_points=params.kernel_points,
    )
    return pd.Series(filter_signal(values, trigger_params), index=series.index)
