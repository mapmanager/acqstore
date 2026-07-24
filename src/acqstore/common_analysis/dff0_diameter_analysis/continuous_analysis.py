"""High-level continuous coupling analysis for reporter and diameter traces."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .continuous_coupling import (
    analyze_lagged_correlation,
    preprocess_continuous_signals,
)
from .loading import load_dataset_from_acq_image, load_sidecars
from .models import (
    AnalysisSelection,
    DiameterFilterParams,
    Dff0DiameterDataset,
    LaggedCorrelationParams,
    LaggedCorrelationResult,
)


class Dff0DiameterContinuousAnalysis:
    """Analyze continuous reporter/diameter coupling for one channel and ROI."""

    def __init__(
        self,
        dataset: Dff0DiameterDataset,
        *,
        params: LaggedCorrelationParams | None = None,
    ) -> None:
        """Initialize and run the continuous lagged-correlation analysis.

        Args:
            dataset: Validated paired reporter/diameter dataset.
            params: Optional lagged-correlation parameters.
        """
        self._dataset = dataset
        self._params = params or LaggedCorrelationParams()
        self._reporter_filtered, self._diameter_filtered = (
            preprocess_continuous_signals(
                signal_a=dataset.reporter["df_f_signal"].to_numpy(dtype=float),
                signal_b=dataset.diameter["diameter_um_raw"].to_numpy(dtype=float),
                params=self._params,
            )
        )
        self._result = analyze_lagged_correlation(
            time=dataset.reporter["time_sec"].to_numpy(dtype=float),
            signal_a=dataset.reporter["df_f_signal"].to_numpy(dtype=float),
            signal_b=dataset.diameter["diameter_um_raw"].to_numpy(dtype=float),
            params=self._params,
        )

    @classmethod
    def from_sidecars(
        cls,
        *,
        diameter_csv: str | Path,
        reporter_csv: str | Path,
        analysis_json: str | Path,
        channel: int,
        roi_id: int,
        params: LaggedCorrelationParams | None = None,
    ) -> "Dff0DiameterContinuousAnalysis":
        """Create continuous analysis from three per-file sidecars."""
        dataset = load_sidecars(
            diameter_csv=diameter_csv,
            reporter_csv=reporter_csv,
            analysis_json=analysis_json,
            selection=AnalysisSelection(channel=channel, roi_id=roi_id),
            diameter_filter=DiameterFilterParams(method="none", kernel_points=1),
        )
        return cls(dataset, params=params)

    @classmethod
    def from_acq_image(
        cls,
        *,
        acq_image: Any,
        channel: int,
        roi_id: int,
        params: LaggedCorrelationParams | None = None,
    ) -> "Dff0DiameterContinuousAnalysis":
        """Create continuous analysis through public ``AcqImage`` APIs."""
        dataset = load_dataset_from_acq_image(
            acq_image=acq_image,
            channel=channel,
            roi_id=roi_id,
        )
        return cls(dataset, params=params)

    @property
    def dataset(self) -> Dff0DiameterDataset:
        """Return the validated paired source dataset."""
        return self._dataset

    @property
    def params(self) -> LaggedCorrelationParams:
        """Return immutable continuous-analysis parameters."""
        return self._params

    @property
    def result(self) -> LaggedCorrelationResult:
        """Return the lagged-correlation result."""
        return self._result

    @property
    def reporter_filtered(self) -> np.ndarray:
        """Return a copy of the locally filtered reporter trace."""
        return self._reporter_filtered.copy()

    @property
    def diameter_filtered(self) -> np.ndarray:
        """Return a copy of the locally filtered diameter trace."""
        return self._diameter_filtered.copy()
