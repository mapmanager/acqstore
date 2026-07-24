"""Tests for AcqAnalysisSet.results_csv_loaded semantics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from acqstore.acq_image.acq_analysis_set import AcqAnalysisSet
from acqstore.acq_image.analysis.model import AnalysisPlotData, AnalysisResult, BaseAnalysis
from acqstore.acq_image.analysis.registry import register_analysis_class


class DummyRadonAnalysis(BaseAnalysis):
    """Minimal Radon velocity analysis for CSV-loaded tests."""

    analysis_name = "radon_velocity"

    def run(self, data_provider, *, context=None, dependencies=None) -> AnalysisResult:
        return self.result

    def get_plot_data(self) -> AnalysisPlotData:
        return AnalysisPlotData(x=(0.0, 1.0), y=(1.0, 2.0), x_label="Time (s)", y_label="Velocity")


class SummaryOnlyAnalysis(DummyRadonAnalysis):
    """Analysis type with no CSV sidecar (like heart_rate)."""

    analysis_name = "summary_only_no_csv"


def test_results_csv_loaded_ignores_analyses_without_csv_sidecar(tmp_path: Path) -> None:
    """CSV-loaded state should not require tables for analyses with no sidecar file."""
    register_analysis_class(SummaryOnlyAnalysis)
    source = tmp_path / "example.tif"
    source.touch()
    radon_csv = source.with_name(f"{source.name}.radon_velocity.csv")
    radon_csv.write_text("channel,roi_id,value\n0,1,1.0\n")

    analysis_set = AcqAnalysisSet(source)
    radon = DummyRadonAnalysis(channel=0, roi_id=1)
    summary_only = SummaryOnlyAnalysis(channel=0, roi_id=1)
    summary_only.result = AnalysisResult(summary={"ok": True}, table=None)
    analysis_set.add(radon)
    analysis_set.add(summary_only)

    analysis_set.load_all_results_dfs_from_csv(source)

    assert radon.result.table is not None
    assert summary_only.result.table is None
    assert analysis_set.results_csv_loaded() is True


def test_results_csv_loaded_requires_table_when_csv_sidecar_exists(tmp_path: Path) -> None:
    """When a CSV sidecar exists, the matching analysis must have a loaded table."""
    source = tmp_path / "example.tif"
    source.touch()
    radon_csv = source.with_name(f"{source.name}.radon_velocity.csv")
    radon_csv.write_text("channel,roi_id,value\n0,1,1.0\n")

    analysis_set = AcqAnalysisSet(source)
    radon = DummyRadonAnalysis(channel=0, roi_id=1)
    analysis_set.add(radon)
    analysis_set._results_csv_loaded = True

    assert analysis_set.results_csv_loaded() is False
