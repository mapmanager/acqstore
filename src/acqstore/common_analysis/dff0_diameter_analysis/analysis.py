"""High-level paired reporter and diameter analysis interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .loading import load_dataset_from_acq_image, load_sidecars
from .models import (
    AnalysisSelection,
    DiameterFilterParams,
    Dff0DiameterDataset,
    TriggeredEvent,
    TriggeredEventParams,
)
from .preprocessing import filter_signal
from .triggered_events import analyze_triggered_events, events_to_dataframe


class MissingCrossAnalysisInputError(ValueError):
    """Raised when one selected AcqImage analysis input is unavailable."""


class Dff0DiameterAnalysis:
    """Analyze paired reporter and diameter results for one channel and ROI."""

    def __init__(
        self,
        dataset: Dff0DiameterDataset,
        *,
        triggered_event_params: TriggeredEventParams | None = None,
    ) -> None:
        """Initialize from a validated paired dataset."""
        self._dataset = dataset
        self._triggered_event_params = triggered_event_params or TriggeredEventParams()
        self._diameter_filtered = filter_signal(
            dataset.diameter["diameter_um_raw"].to_numpy(dtype=float),
            self._triggered_event_params,
        )
        self._triggered_events = tuple(
            analyze_triggered_events(
                time=dataset.diameter["time_s"].to_numpy(dtype=float),
                signal=dataset.diameter["diameter_um_raw"].to_numpy(dtype=float),
                seed_indices=[event.onset_index for event in dataset.events],
                params=self._triggered_event_params,
            )
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
        triggered_event_params: TriggeredEventParams | None = None,
    ) -> "Dff0DiameterAnalysis":
        """Create an analysis from three colocated per-file sidecars."""
        dataset = load_sidecars(
            diameter_csv=diameter_csv,
            reporter_csv=reporter_csv,
            analysis_json=analysis_json,
            selection=AnalysisSelection(channel=channel, roi_id=roi_id),
            diameter_filter=DiameterFilterParams(method="none", kernel_points=1),
        )
        return cls(dataset, triggered_event_params=triggered_event_params)

    @classmethod
    def from_acq_image(
        cls,
        *,
        acq_image: Any,
        channel: int,
        roi_id: int,
        triggered_event_params: TriggeredEventParams | None = None,
    ) -> "Dff0DiameterAnalysis":
        """Create an analysis from public AcqImage analysis APIs.

        Args:
            acq_image: Loaded ``AcqImage`` whose analysis tables are available.
            channel: Selected zero-based channel.
            roi_id: Selected ROI identifier.
            triggered_event_params: Optional generic event parameters.

        Raises:
            MissingCrossAnalysisInputError: If diameter or sum-intensity input
                is missing or has no loaded result table.
        """
        try:
            dataset = load_dataset_from_acq_image(
                acq_image=acq_image,
                channel=channel,
                roi_id=roi_id,
            )
        except ValueError as exc:
            raise MissingCrossAnalysisInputError(str(exc)) from exc
        return cls(dataset, triggered_event_params=triggered_event_params)

    @property
    def dataset(self) -> Dff0DiameterDataset:
        """Return the validated source dataset."""
        return self._dataset

    @property
    def triggered_event_params(self) -> TriggeredEventParams:
        """Return immutable triggered-event parameters."""
        return self._triggered_event_params

    @property
    def triggered_events(self) -> tuple[TriggeredEvent, ...]:
        """Return one measured diameter event per reporter seed."""
        return self._triggered_events

    @property
    def diameter_filtered(self) -> np.ndarray:
        """Return a copy of locally filtered diameter values."""
        return self._diameter_filtered.copy()

    def triggered_events_dataframe(self) -> pd.DataFrame:
        """Return one flat row per generic triggered event."""
        return events_to_dataframe(self._triggered_events)

    def get_alignment_summary(self) -> dict[str, object]:
        """Return compact source and alignment diagnostics."""
        reporter = self._dataset.reporter
        diameter = self._dataset.diameter
        return {
            "source_name": self._dataset.source_name,
            "channel": self._dataset.selection.channel,
            "roi_id": self._dataset.selection.roi_id,
            "num_points": len(reporter),
            "seconds_per_point": self._dataset.seconds_per_point,
            "duration_sec": float(reporter["time_sec"].iloc[-1]),
            "num_reporter_events": len(self._dataset.events),
            "num_missing_reporter": int(reporter["df_f_signal"].isna().sum()),
            "num_missing_diameter_raw": int(diameter["diameter_um_raw"].isna().sum()),
        }

    def get_reporter_events_dataframe(self) -> pd.DataFrame:
        """Return one flat row per selected upstream reporter event."""
        return pd.DataFrame(
            [{
                "peak_id": event.peak_id,
                "status": event.status,
                "onset_index": event.onset_index,
                "onset_time_sec": event.onset_time_sec,
                "onset_value": event.onset_value,
                "peak_index": event.peak_index,
                "peak_time_sec": event.peak_time_sec,
                "peak_value": event.peak_value,
                "peak_amplitude": event.peak_amplitude,
            } for event in self._dataset.events]
        )
