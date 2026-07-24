"""Tests for collection-level acqstore analysis pools."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from acqstore.acq_image.acq_analysis_set import AcqAnalysisSet
from acqstore.acq_image.acq_image_list import AcqImageList
from acqstore.acq_image.metadata import ExperimentMetadata
from acqstore.acq_image.analysis.event_analysis.event_analysis import EventAnalysis
from acqstore.acq_image.analysis.heart_rate_analysis.heart_rate_analysis import HeartRateAnalysis
from acqstore.acq_image.analysis.velocity_analysis.radon_velocity_analysis import (
    RadonVelocityAnalysis,
)
from acqstore.analysis_pool.base_analysis_pool import AnalysisPool, pool_column_name
from acqstore.analysis_pool.velocity_analysis_pool import VelocityAnalysisPool


class _FakeImages:
    """Minimal image-loader stand-in exposing channel information."""

    num_channels = 2

    def channels(self) -> list[int]:
        """Return fake channel indices."""
        return [0, 1]


class _FakeRois:
    """Minimal ROI-set stand-in exposing ROI ids."""

    def __init__(self, roi_ids: list[int]) -> None:
        self._roi_ids = list(roi_ids)

    def get_roi_ids(self) -> list[int]:
        """Return fake ROI ids."""
        return list(self._roi_ids)


class _PoolFakeAcqImage:
    """AcqImage-like object for analysis-pool tests."""

    def __init__(self, path: str) -> None:
        self.path = str(Path(path).resolve())
        self.file_id = self.path
        self.name = Path(self.path).name
        self.images = _FakeImages()
        self.rois = _FakeRois([1])
        self.analysis_set = AcqAnalysisSet(self.path)
        self._experimental_metadata = ExperimentMetadata(
            genotype='wt',
            age='P30',
            sex='F',
            branch_order=2,
            direction='upstream',
            depth=45.0,
            note='good quality',
        )

    def get_metadata_section(self, metadata_section_id: str) -> ExperimentMetadata:
        """Return experiment metadata for pool base-row tests."""
        if metadata_section_id == ExperimentMetadata.metadata_section_id:
            return self._experimental_metadata
        raise ValueError(f'Unknown metadata section_id: {metadata_section_id!r}')

    def get_schema_row(self) -> dict[str, object]:
        """Return file-list schema values used by the pool."""
        parent = Path(self.path).parent.name
        grandparent = Path(self.path).parent.parent.name
        return {
            "name": self.name,
            "saved": True,
            "path": self.path,
            "parent": parent,
            "grandparent": grandparent,
            "condition": "ctrl",
            "genotype": "wt",
            "num_channels": 2,
            "dims": "",
            "num_rois": 1,
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


class _SingleChannelImages(_FakeImages):
    """Single-channel fake image-loader."""

    num_channels = 1

    def channels(self) -> list[int]:
        """Return the only fake channel index."""
        return [0]


class _SingleChannelFakeAcqImage(_PoolFakeAcqImage):
    """Single-channel variant for row-refresh tests."""

    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.images = _SingleChannelImages()

    def get_schema_row(self) -> dict[str, object]:
        """Return file-list schema values used by the pool."""
        row = super().get_schema_row()
        row["num_channels"] = 1
        return row


def test_pool_column_name_skips_existing_prefix() -> None:
    assert pool_column_name("velocity", "velocity_mean") == "velocity_mean"
    assert pool_column_name("velocity", "analysis_date") == "velocity_analysis_date"
    assert pool_column_name("hr", "lomb_bpm") == "hr_lomb_bpm"
    assert pool_column_name("event", "num_events") == "event_num_events"


def test_velocity_analysis_pool_column_names_are_unique() -> None:
    columns = VelocityAnalysisPool.pool_column_names()
    assert len(columns) == len(set(columns))


def test_acq_image_list_owns_velocity_analysis_pool(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.tif"
    file_path.write_text("")

    images = AcqImageList(str(file_path), file_factory=_SingleChannelFakeAcqImage)

    assert hasattr(images, "velocity_analysis_pool")
    df = images.velocity_analysis_pool.get_dataframe()
    assert list(df["channel"]) == [0]
    assert list(df["roi_id"]) == [1]
    assert df.loc[0, "pool_row_id"] == AnalysisPool.build_pool_row_id(
        str(file_path.resolve()),
        channel=0,
        roi_id=1,
    )


def test_velocity_analysis_pool_creates_seed_rows_without_analysis(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.tif"
    file_path.write_text("")

    images = AcqImageList(str(file_path), file_factory=_PoolFakeAcqImage)
    df = images.velocity_analysis_pool.get_dataframe()

    assert len(df) == 2
    assert list(df["pool_row"]) == [0, 1]
    assert set(df["pool_row_id"]) == {
        AnalysisPool.build_pool_row_id(str(file_path.resolve()), channel=0, roi_id=1),
        AnalysisPool.build_pool_row_id(str(file_path.resolve()), channel=1, roi_id=1),
    }
    assert "velocity_mean" in df.columns
    assert "velocity_analysis_date" in df.columns
    assert "velocity_analysis_time" in df.columns
    assert "velocity_analysis_version" in df.columns
    assert "hr_analysis_date" in df.columns
    assert "event_analysis_version" in df.columns
    assert "hr_status" in df.columns
    assert "event_num_events" in df.columns
    assert "velocity_velocity_mean" not in df.columns
    assert pd.isna(df.loc[0, "velocity_mean"])
    assert df.loc[0, "age"] == "P30"
    assert df.loc[0, "sex"] == "F"
    assert df.loc[0, "branch_order"] == 2
    assert df.loc[0, "direction"] == "upstream"
    assert df.loc[0, "depth"] == 45.0
    assert df.loc[0, "note"] == "good quality"


def test_velocity_analysis_pool_refreshes_one_row_from_summaries(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.tif"
    file_path.write_text("")
    images = AcqImageList(str(file_path), file_factory=_SingleChannelFakeAcqImage)
    acq = images.get_file_by_index(0)

    velocity = RadonVelocityAnalysis(channel=0, roi_id=1)
    velocity.result.summary = {
        "analysis_date": "260623",
        "analysis_time": "14:32:07.042",
        "analysis_version": 1,
        "num_windows": 3,
        "velocity_mean": 12.5,
        "velocity_median": 12.0,
        "velocity_cv": 0.1,
    }
    acq.analysis_set.add(velocity)

    hr = HeartRateAnalysis(channel=0, roi_id=1)
    hr.result.summary = {
        "analysis_date": "260623",
        "analysis_time": "14:32:07.042",
        "analysis_version": 1,
        "version": 1,
        "n_total": 10,
        "n_valid": 9,
        "valid_frac": 0.9,
        "status": "ok",
        "lomb": {"bpm": 300.0, "f_hz": 5.0, "status": "ok"},
        "welch": {"bpm": 306.0, "f_hz": 5.1, "status": "ok"},
        "agreement": {"delta_bpm": 6.0, "abs_delta_bpm": 6.0, "agree_ok": True},
    }
    acq.analysis_set.add(hr)

    event = EventAnalysis(channel=0, roi_id=1)
    event.result.summary = {
        "analysis_date": "260623",
        "analysis_time": "14:32:07.042",
        "analysis_version": 2,
        "version": 2,
        "parent_analysis_name": "radon_velocity",
        "events": [
            {"event_type": "user", "duration": 0.5},
            {"event_type": "rise", "duration": 1.5},
        ],
    }
    acq.analysis_set.add(event)

    images.velocity_analysis_pool.refresh_row(acq.file_id, channel=0, roi_id=1)
    df = images.velocity_analysis_pool.get_dataframe()

    assert df.loc[0, "velocity_num_windows"] == 3
    assert df.loc[0, "velocity_analysis_date"] == "260623"
    assert df.loc[0, "velocity_analysis_version"] == 1
    assert df.loc[0, "velocity_mean"] == 12.5
    assert df.loc[0, "hr_status"] == "ok"
    assert df.loc[0, "hr_lomb_bpm"] == 300.0
    assert df.loc[0, "hr_agreement_agree_ok"] is True
    assert df.loc[0, "event_num_events"] == 2
    assert df.loc[0, "event_user_events"] == 1
    assert df.loc[0, "event_rise_events"] == 1
    assert df.loc[0, "event_mean_duration"] == 1.0


def test_velocity_analysis_pool_remove_row_and_remove_roi(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.tif"
    file_path.write_text("")
    images = AcqImageList(str(file_path), file_factory=_PoolFakeAcqImage)

    images.velocity_analysis_pool.remove_row(str(file_path.resolve()), channel=0, roi_id=1)
    df = images.velocity_analysis_pool.get_dataframe()
    assert len(df) == 1
    assert list(df["channel"]) == [1]
    assert list(df["pool_row"]) == [0]

    images.velocity_analysis_pool.remove_roi(str(file_path.resolve()), roi_id=1)
    assert images.velocity_analysis_pool.get_dataframe().empty


def test_velocity_analysis_pool_to_csv(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.tif"
    file_path.write_text("")
    images = AcqImageList(str(file_path), file_factory=_SingleChannelFakeAcqImage)
    out_path = tmp_path / "exports" / "velocity_pool.csv"

    images.velocity_analysis_pool.to_csv(out_path)

    loaded = pd.read_csv(out_path)
    assert list(loaded.columns) == list(images.velocity_analysis_pool.get_dataframe().columns)
    assert loaded.loc[0, "roi_id"] == 1


class _StringlyMetadataFakeAcqImage(_PoolFakeAcqImage):
    """AcqImage-like object whose metadata mimics stringly typed sidecar JSON."""

    def __init__(self, path: str) -> None:
        super().__init__(path)
        self._experimental_metadata = ExperimentMetadata.from_dict(
            {
                "genotype": "wt",
                "age": "P30",
                "sex": "F",
                "branch_order": "2",
                "direction": "upstream",
                "depth": "45",
                "note": "good quality",
            }
        )


def test_velocity_analysis_pool_coerces_stringly_experiment_metadata_dtypes(
    tmp_path: Path,
) -> None:
    """Pool rebuild should expose nullable numeric dtypes for coerced metadata."""
    file_path = tmp_path / "sample.tif"
    file_path.write_text("")
    images = AcqImageList(str(file_path), file_factory=_StringlyMetadataFakeAcqImage)
    df = images.velocity_analysis_pool.get_dataframe()

    assert str(df["branch_order"].dtype) == "Int64"
    assert str(df["depth"].dtype) == "Float64"
    assert df.loc[0, "branch_order"] == 2
    assert df.loc[0, "depth"] == 45.0
