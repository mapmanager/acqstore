"""Tests for paired sidecar loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from acqstore.common_analysis.dff0_diameter_analysis.analysis import (
    Dff0DiameterAnalysis,
)


@pytest.fixture(name="sample_paths")
def fixture_sample_paths() -> tuple[Path, Path, Path]:
    """Return uploaded example sidecar paths when available."""
    root = Path("/mnt/data")
    paths = (
        root / "220110n_0005.tif.diameter.csv",
        root / "220110n_0005.tif.sum_intensity.csv",
        root / "220110n_0005.tif.json",
    )
    if not all(path.exists() for path in paths):
        pytest.skip("Uploaded example sidecars are unavailable")
    return paths


def test_load_example_sidecars(
    sample_paths: tuple[Path, Path, Path],
) -> None:
    """Load and validate the uploaded one-file example."""
    diameter_csv, reporter_csv, analysis_json = sample_paths
    analysis = Dff0DiameterAnalysis.from_sidecars(
        diameter_csv=diameter_csv,
        reporter_csv=reporter_csv,
        analysis_json=analysis_json,
        channel=0,
        roi_id=1,
    )

    summary = analysis.get_alignment_summary()
    assert summary["num_points"] == 2500
    assert summary["num_reporter_events"] == 6
    assert summary["seconds_per_point"] == pytest.approx(0.0044284)
    assert analysis.get_reporter_events_dataframe()["onset_index"].tolist() == [
        51,
        494,
        940,
        1386,
        1830,
        2275,
    ]
