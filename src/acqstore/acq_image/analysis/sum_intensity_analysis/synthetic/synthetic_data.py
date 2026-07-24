"""Data containers returned by synthetic sum-intensity generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class SyntheticSumIntensityData:
    """Synthetic image and ground truth for sum-intensity analysis.

    Args:
        image: Synthetic image with shape ``(time, space)``. This is the input
            expected by the core sum-intensity algorithm.
        time_sec: Time axis in seconds, one value per image row.
        fluorescence_trace: One-dimensional mean fluorescence trace before
            expansion into an image.
        ideal_df_f_trace: Noise-free approximate df/f0 event trace before image
            noise and detection preprocessing.
        ground_truth_events: DataFrame with one row per synthetic event. Stable
            columns include ``event_id``, ``onset_time_sec``,
            ``peak_time_sec``, and ``amplitude``.
        seconds_per_line: Time spacing in seconds.
        um_per_pixel: Spatial spacing in microns.
        f0: Baseline fluorescence used by the synthetic model.
    """

    image: np.ndarray
    time_sec: np.ndarray
    fluorescence_trace: np.ndarray
    ideal_df_f_trace: np.ndarray
    ground_truth_events: pd.DataFrame
    seconds_per_line: float
    um_per_pixel: float
    f0: float
