"""Tests for analysis CSV table persistence."""

from pathlib import Path

import pandas as pd
import pytest

from acqstore.acq_image.acq_analysis_set import AcqAnalysisSet
from acqstore.acq_image.analysis.velocity_analysis.radon_velocity_analysis import RadonVelocityAnalysis


def test_table_with_bookkeeping_adds_channel_and_roi() -> None:
    """table_with_bookkeeping should add channel and ROI columns."""
    analysis = RadonVelocityAnalysis(channel=3, roi_id=4)
    analysis.result.table = pd.DataFrame({"x": [1, 2]})

    table = analysis.table_with_bookkeeping()

    assert table is not None
    assert list(table.columns) == ["channel", "roi_id", "x"]
    assert table["channel"].tolist() == [3, 3]
    assert table["roi_id"].tolist() == [4, 4]


def test_table_with_bookkeeping_rejects_reserved_columns() -> None:
    """Analysis tables may not define reserved bookkeeping columns."""
    analysis = RadonVelocityAnalysis(channel=3, roi_id=4)
    analysis.result.table = pd.DataFrame({"channel": [1]})

    with pytest.raises(ValueError):
        analysis.table_with_bookkeeping()


def test_save_and_load_results_df(tmp_path: Path) -> None:
    """AcqAnalysisSet should save and load combined CSV tables."""
    source = tmp_path / "myfile.tif"
    source.write_text("dummy")

    analysis_set = AcqAnalysisSet(source)
    a = RadonVelocityAnalysis(channel=0, roi_id=1)
    a.result.table = pd.DataFrame({"time_s": [0.0], "radon_velocity": [1.0]})
    b = RadonVelocityAnalysis(channel=1, roi_id=2)
    b.result.table = pd.DataFrame({"time_s": [0.0], "radon_velocity": [2.0]})
    analysis_set.add(a)
    analysis_set.add(b)
    analysis_set.save_results_df(source)

    assert (tmp_path / "myfile.tif.radon_velocity.csv").exists()

    loaded = AcqAnalysisSet(source)
    loaded.load_json_analysis(analysis_set.serialize_json_analysis())
    loaded.load_all_results_dfs_from_csv(source)

    loaded_a = loaded.get_required(a.key)
    assert loaded_a.get_column("radon_velocity") == [1.0]


def test_save_results_df_deletes_stale_csv_when_no_rows_remain(tmp_path: Path) -> None:
    """save_results_df should delete stale CSVs when no tables remain."""
    source = tmp_path / "myfile.tif"
    source.write_text("dummy")
    stale = tmp_path / "myfile.tif.radon_velocity.csv"
    stale.write_text("channel,roi_id,x\\n")

    AcqAnalysisSet(source).save_results_df(source)

    assert not stale.exists()


def test_save_results_df_rejects_inconsistent_columns(tmp_path: Path) -> None:
    """Same analysis type should require same table columns."""
    source = tmp_path / "myfile.tif"
    source.write_text("dummy")

    analysis_set = AcqAnalysisSet(source)
    a = RadonVelocityAnalysis(channel=0, roi_id=1)
    a.result.table = pd.DataFrame({"time_s": [0.0], "radon_velocity": [1.0]})
    b = RadonVelocityAnalysis(channel=1, roi_id=2)
    b.result.table = pd.DataFrame({"time_s": [0.0], "speed": [2.0]})
    analysis_set.add(a)
    analysis_set.add(b)

    with pytest.raises(ValueError):
        analysis_set.save_results_df(source)
