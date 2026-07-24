"""Tests for sum-intensity pool-facing APIs and AcqImageList cache."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from acqstore.acq_image.acq_analysis_set import AcqAnalysisSet
from acqstore.acq_image.acq_image_list import AcqImageList
from acqstore.acq_image.analysis.sum_intensity_analysis.sum_intensity_analysis import (
    SumIntensityAnalysis,
)
from acqstore.acq_image.analysis.sum_intensity_analysis.sum_intensity_core import (
    EventFeature,
    LevelCrossing,
    PeakEvent,
)
from acqstore.acq_image.metadata import ExperimentMetadata
from acqstore.analysis_pool.sum_intensity_analysis_pool import SumIntensityAnalysisPool


class _SingleChannelImages:
    """Minimal image-loader stand-in exposing one channel."""

    num_channels = 1

    def channels(self) -> list[int]:
        """Return fake channel indices."""
        return [0]


class _FakeRois:
    """Minimal ROI-set stand-in exposing ROI ids."""

    def __init__(self, roi_ids: list[int]) -> None:
        self._roi_ids = list(roi_ids)

    def get_roi_ids(self) -> list[int]:
        """Return fake ROI ids."""
        return list(self._roi_ids)


class _PoolFakeAcqImage:
    """AcqImage-like object for sum-intensity pool tests."""

    def __init__(self, path: str) -> None:
        self.path = str(Path(path).resolve())
        self.file_id = self.path
        self.name = Path(self.path).name
        self.images = _SingleChannelImages()
        self.rois = _FakeRois([1])
        self.analysis_set = AcqAnalysisSet(self.path)
        self._experimental_metadata = ExperimentMetadata(
            genotype="wt",
            age="P30",
            sex="F",
            branch_order=2,
            direction="upstream",
            depth=45.0,
            note="good quality",
        )

    def get_metadata_section(self, metadata_section_id: str) -> ExperimentMetadata:
        """Return experiment metadata for pool base-row tests."""
        if metadata_section_id == ExperimentMetadata.metadata_section_id:
            return self._experimental_metadata
        raise ValueError(f"Unknown metadata section_id: {metadata_section_id!r}")

    def get_schema_row(self) -> dict[str, object]:
        """Return file-list schema values used by the pool."""
        return {
            "name": self.name,
            "path": self.path,
            "parent": Path(self.path).parent.name,
            "grandparent": Path(self.path).parent.parent.name,
            "condition": "ctrl",
            "genotype": "wt",
            "num_channels": 1,
            "accept": True,
        }

    def get_default_channel(self) -> int:
        """Return default channel for AcqImageList tests."""
        return 0

    def get_default_roi(self) -> int:
        """Return default ROI for AcqImageList tests."""
        return 1

    def get_image_physical_units(self) -> tuple[float, float]:
        """Return fake physical pixel spacing."""
        return (0.01, 0.2)


def _event(peak_id: int, amplitude: float) -> PeakEvent:
    feature = EventFeature(value=amplitude, status="ok", reason="")
    return PeakEvent(
        peak_id=peak_id,
        status="ok",
        warnings=("warn-a",) if peak_id == 2 else (),
        onset_index=10 * peak_id,
        onset_time_sec=0.1 * peak_id,
        onset_value=1.0,
        peak_index=10 * peak_id + 2,
        peak_time_sec=0.1 * peak_id + 0.02,
        peak_value=1.0 + amplitude,
        peak_amplitude=amplitude,
        detection_method="derivative_threshold",
        baseline_mean=EventFeature(value=1.0),
        baseline_std=EventFeature(value=0.1),
        rise_10_90_sec=EventFeature(value=0.02),
        decay_90_10_sec=EventFeature(value=0.03),
        decay_time_sec=EventFeature(value=0.03),
        max_rise_slope=EventFeature(value=5.0),
        max_decay_slope=EventFeature(value=-4.0),
        auc=EventFeature(value=0.5),
        prominence=feature,
        level_crossings=(
            LevelCrossing(
                fraction=0.5,
                value=1.0 + amplitude / 2.0,
                left_index=10.5,
                right_index=15.5,
                width=5.0,
                width_sec=0.05,
                status="ok",
            ),
        ),
        onset_to_onset_interval_sec=None,
        peak_to_peak_interval_sec=None,
    )


def _analysis_with_events(events: tuple[PeakEvent, ...]) -> SumIntensityAnalysis:
    analysis = SumIntensityAnalysis(channel=0, roi_id=1)
    analysis.result.summary = {
        "analysis_date": "260629",
        "analysis_time": "12:00:00.000",
        "analysis_version": 1,
        "status": "ok",
        "num_timepoints": 20,
        "num_peaks": len(events),
        "num_space_pixels": 5,
        "seconds_per_line": 0.01,
        "f0_baseline": 1.0,
        "baseline_method": "percentile",
        "baseline_percentile": 20.0,
        "manual_f0_baseline": 1.0,
        "detrend_method": "none",
        "detection_method": "derivative_threshold",
        "detection_source": "df_f_signal",
        "peak_search_window_ms": 50.0,
        "width_search_window_ms": 750.0,
        "baseline_window_ms": 100.0,
        "peak_amplitude_mean": 2.0,
        "peak_amplitude_median": 2.0,
        "errors": ["error-a", "error-b"],
        "peak_events": [event.to_json_dict() for event in events],
    }
    return analysis


def test_pool_facing_sum_intensity_api_flattens_summary_and_peaks() -> None:
    """Pool-facing APIs should expose only scalar table-safe values."""
    analysis = _analysis_with_events((_event(1, 2.0),))

    summary_columns = SumIntensityAnalysis.get_pool_summary_columns()
    summary_values = analysis.get_pool_summary_values()
    peak_columns = SumIntensityAnalysis.get_pool_peak_columns()
    peak_rows = analysis.get_pool_peak_rows()

    assert "errors" not in summary_columns
    assert summary_values["error_count"] == 2
    assert summary_values["errors_text"] == "error-a; error-b"
    assert "prominence" in peak_columns
    assert "prominence_status" in peak_columns
    assert "width_50_sec" in peak_columns
    assert len(peak_rows) == 1
    assert peak_rows[0]["prominence"] == 2.0
    assert peak_rows[0]["width_50_sec"] == 0.05
    assert all(not isinstance(value, (dict, list, tuple)) for value in peak_rows[0].values())


def test_acq_image_list_owns_sum_intensity_pool(tmp_path: Path) -> None:
    """AcqImageList should attach an independent sum-intensity pool."""
    file_path = tmp_path / "sample.tif"
    file_path.write_text("")

    images = AcqImageList(str(file_path), file_factory=_PoolFakeAcqImage)

    assert hasattr(images, "sum_intensity_analysis_pool")
    df = images.sum_intensity_analysis_pool.get_dataframe()
    assert len(df) == 1
    assert df.loc[0, "peak_row_type"] == "not_analyzed"
    assert pd.isna(df.loc[0, "num_peaks"])


def test_sum_intensity_pool_creates_one_row_for_zero_peaks(tmp_path: Path) -> None:
    """Completed analysis with zero peaks should be visible as no_peaks."""
    file_path = tmp_path / "sample.tif"
    file_path.write_text("")
    images = AcqImageList(str(file_path), file_factory=_PoolFakeAcqImage)
    acq = images.get_file_by_index(0)
    acq.analysis_set.add(_analysis_with_events(()))

    images.sum_intensity_analysis_pool.refresh_rows(acq.file_id, channel=0, roi_id=1)
    df = images.sum_intensity_analysis_pool.get_dataframe()

    assert len(df) == 1
    assert df.loc[0, "peak_row_type"] == "no_peaks"
    assert df.loc[0, "num_peaks"] == 0
    assert pd.isna(df.loc[0, "peak_id"])


def test_sum_intensity_pool_creates_one_row_per_peak(tmp_path: Path) -> None:
    """Completed analysis with peaks should produce one row per peak."""
    file_path = tmp_path / "sample.tif"
    file_path.write_text("")
    images = AcqImageList(str(file_path), file_factory=_PoolFakeAcqImage)
    acq = images.get_file_by_index(0)
    acq.analysis_set.add(_analysis_with_events((_event(1, 2.0), _event(2, 3.0))))

    images.sum_intensity_analysis_pool.refresh_rows(acq.file_id, channel=0, roi_id=1)
    df = images.sum_intensity_analysis_pool.get_dataframe()

    assert len(df) == 2
    assert list(df["peak_row_type"]) == ["peak", "peak"]
    assert list(df["peak_id"]) == [1, 2]
    assert list(df["prominence"]) == [2.0, 3.0]
    assert df.loc[1, "peak_warning_count"] == 1
    assert df.loc[1, "peak_warnings_text"] == "warn-a"
    assert df["pool_row_id"].is_unique


def test_sum_intensity_pool_row_ids_for_selection_returns_peak_rows(tmp_path: Path) -> None:
    """row_ids_for_selection should return all peak row ids for one selection."""
    file_path = tmp_path / "sample.tif"
    file_path.write_text("")
    images = AcqImageList(str(file_path), file_factory=_PoolFakeAcqImage)
    acq = images.get_file_by_index(0)
    acq.analysis_set.add(_analysis_with_events((_event(1, 2.0), _event(2, 3.0))))
    pool = images.sum_intensity_analysis_pool
    pool.refresh_rows(acq.file_id, channel=0, roi_id=1)

    row_ids = pool.row_ids_for_selection(acq.file_id, channel=0, roi_id=1)

    assert len(row_ids) == 2
    assert len(set(row_ids)) == 2


def test_sum_intensity_pool_row_ids_for_selection_empty_for_no_peaks(tmp_path: Path) -> None:
    """row_ids_for_selection should return nothing when only sentinel rows exist."""
    file_path = tmp_path / "sample.tif"
    file_path.write_text("")
    images = AcqImageList(str(file_path), file_factory=_PoolFakeAcqImage)
    acq = images.get_file_by_index(0)
    acq.analysis_set.add(_analysis_with_events(()))
    pool = images.sum_intensity_analysis_pool
    pool.refresh_rows(acq.file_id, channel=0, roi_id=1)

    row_ids = pool.row_ids_for_selection(acq.file_id, channel=0, roi_id=1)

    assert row_ids == ()


def test_sum_intensity_pool_rejects_non_scalar_pool_values(tmp_path: Path) -> None:
    """Pool cells should fail fast when analysis APIs leak non-scalars."""
    file_path = tmp_path / "sample.tif"
    file_path.write_text("")
    images = AcqImageList(str(file_path), file_factory=_PoolFakeAcqImage)
    acq = images.get_file_by_index(0)
    analysis = _analysis_with_events(())
    analysis.result.summary["status"] = {"bad": "value"}
    acq.analysis_set.add(analysis)

    with pytest.raises(TypeError, match="status"):
        images.sum_intensity_analysis_pool.refresh_rows(acq.file_id, channel=0, roi_id=1)


def test_sum_intensity_pool_remove_and_refresh_file(tmp_path: Path) -> None:
    """Plural mutation helpers should maintain row numbering."""
    file_path = tmp_path / "sample.tif"
    file_path.write_text("")
    images = AcqImageList(str(file_path), file_factory=_PoolFakeAcqImage)
    acq = images.get_file_by_index(0)
    acq.analysis_set.add(_analysis_with_events((_event(1, 2.0), _event(2, 3.0))))
    images.sum_intensity_analysis_pool.refresh_file(acq.file_id)
    assert len(images.sum_intensity_analysis_pool.get_dataframe()) == 2

    images.sum_intensity_analysis_pool.remove_rows(acq.file_id, channel=0, roi_id=1)
    assert images.sum_intensity_analysis_pool.get_dataframe().empty

    images.sum_intensity_analysis_pool.refresh_file(acq.file_id)
    df = images.sum_intensity_analysis_pool.get_dataframe()
    assert len(df) == 2
    assert list(df["pool_row"]) == [0, 1]


def test_sum_intensity_pool_column_names_are_unique() -> None:
    """Pool columns should fail fast on schema collisions."""
    columns = SumIntensityAnalysisPool.pool_column_names()
    assert len(columns) == len(set(columns))
