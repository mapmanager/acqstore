"""Sidecar integration tests using a compact generated fixture."""

from __future__ import annotations

import json

import pandas as pd

from acqstore.common_analysis.dff0_diameter_analysis.analysis import Dff0DiameterAnalysis
from acqstore.common_analysis.dff0_diameter_analysis.models import (
    SignalFilterMethod,
    TriggeredEventParams,
)


def test_sidecar_loader_runs_triggered_events(tmp_path) -> None:
    """Three sidecars should produce one event per JSON reporter onset."""
    count = 20
    time = [index * 0.1 for index in range(count)]
    diameter = pd.DataFrame({
        "channel": [0] * count,
        "roi_id": [1] * count,
        "center_row": list(range(count)),
        "time_s": time,
        "diameter_um": [10, 10, 10, 10, 10, 10, 9, 8, 9, 10, 10, 10, 10, 9, 8, 9, 10, 10, 10, 10],
    })
    reporter = pd.DataFrame({
        "channel": [0] * count,
        "roi_id": [1] * count,
        "time_index": list(range(count)),
        "time_sec": time,
        "df_f_signal": [0.0] * count,
    })
    events = []
    for peak_id, onset_index in enumerate((5, 12), start=1):
        events.append({
            "peak_id": peak_id,
            "status": "ok",
            "warnings": [],
            "detection_method": "derivative_threshold",
            "onset": {"index": onset_index, "time_sec": time[onset_index], "value": 0.0},
            "peak": {"index": onset_index + 1, "time_sec": time[onset_index + 1], "value": 1.0, "amplitude": 1.0},
            "features": {}, "intervals": {}, "level_crossings": [],
        })
    document = {
        "analysis": [
            {"analysis_name": "sum_intensity", "channel": 0, "roi_id": 1, "summary": {"peak_events": events}},
            {"analysis_name": "diameter", "channel": 0, "roi_id": 1, "summary": {}},
        ]
    }
    diameter_path = tmp_path / "sample.tif.diameter.csv"
    reporter_path = tmp_path / "sample.tif.sum_intensity.csv"
    json_path = tmp_path / "sample.tif.json"
    diameter.to_csv(diameter_path, index=False)
    reporter.to_csv(reporter_path, index=False)
    json_path.write_text(json.dumps(document), encoding="utf-8")

    analysis = Dff0DiameterAnalysis.from_sidecars(
        diameter_csv=diameter_path,
        reporter_csv=reporter_path,
        analysis_json=json_path,
        channel=0,
        roi_id=1,
        triggered_event_params=TriggeredEventParams(
            pre_points=3,
            post_points=8,
            post_search_window_points=5,
            baseline_start_offset_points=-3,
            baseline_stop_offset_points=0,
            filter_method=SignalFilterMethod.NONE,
        ),
    )

    assert len(analysis.triggered_events) == 2
    assert len(analysis.triggered_events_dataframe()) == 2
    assert analysis.triggered_events[0].truncated_by_next_seed
