"""Load and validate paired analysis sidecars."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .models import (
    AnalysisSelection,
    DiameterFilterParams,
    Dff0DiameterDataset,
    ReporterEvent,
)
from .preprocessing import filter_diameter


_REQUIRED_DIAMETER_COLUMNS = {
    "channel",
    "roi_id",
    "time_s",
    "center_row",
    "diameter_um",
}
_REQUIRED_REPORTER_COLUMNS = {
    "channel",
    "roi_id",
    "time_index",
    "time_sec",
    "df_f_signal",
}


def load_sidecars(
    *,
    diameter_csv: str | Path,
    reporter_csv: str | Path,
    analysis_json: str | Path,
    selection: AnalysisSelection,
    diameter_filter: DiameterFilterParams | None = None,
) -> Dff0DiameterDataset:
    """Load one paired reporter/diameter dataset from three sidecars.

    The JSON is local to one raw data file, so no raw-data absolute path is
    required. Channel and ROI remain mandatory because a sidecar may contain
    more than one analysis selection.

    Args:
        diameter_csv: Per-file diameter trace CSV.
        reporter_csv: Per-file sum-intensity trace CSV.
        analysis_json: Per-file structured analysis JSON.
        selection: Channel and ROI to load.
        diameter_filter: Local filtering applied to raw ``diameter_um``.

    Returns:
        Validated paired dataset.

    Raises:
        ValueError: If required columns, analyses, or time alignment are
            missing or ambiguous.
    """
    diameter_path = Path(diameter_csv)
    reporter_path = Path(reporter_csv)
    json_path = Path(analysis_json)
    params = diameter_filter or DiameterFilterParams()

    diameter_all = pd.read_csv(diameter_path)
    reporter_all = pd.read_csv(reporter_path)
    with json_path.open("r", encoding="utf-8") as handle:
        json_document: dict[str, Any] = json.load(handle)

    _require_columns(diameter_all, _REQUIRED_DIAMETER_COLUMNS, "diameter CSV")
    _require_columns(reporter_all, _REQUIRED_REPORTER_COLUMNS, "sum-intensity CSV")

    diameter = _select_rows(diameter_all, selection, "diameter CSV")
    reporter = _select_rows(reporter_all, selection, "sum-intensity CSV")
    diameter = diameter.sort_values("center_row").reset_index(drop=True)
    reporter = reporter.sort_values("time_index").reset_index(drop=True)

    diameter["diameter_um_raw"] = pd.to_numeric(
        diameter["diameter_um"], errors="coerce"
    )
    diameter["diameter_um_analysis"] = filter_diameter(
        diameter["diameter_um_raw"], params
    )

    seconds_per_point = _validate_trace_alignment(diameter, reporter)
    sum_analysis = _find_json_analysis(
        json_document,
        analysis_name="sum_intensity",
        selection=selection,
    )
    _find_json_analysis(
        json_document,
        analysis_name="diameter",
        selection=selection,
    )
    events = _load_reporter_events(sum_analysis, reporter, seconds_per_point)

    source_name = _infer_source_name(json_path)
    return Dff0DiameterDataset(
        source_name=source_name,
        selection=selection,
        seconds_per_point=seconds_per_point,
        reporter=reporter,
        diameter=diameter,
        events=events,
        analysis_json=json_document,
        diameter_csv_path=diameter_path,
        reporter_csv_path=reporter_path,
        analysis_json_path=json_path,
    )


def _require_columns(
    dataframe: pd.DataFrame,
    required: set[str],
    table_name: str,
) -> None:
    missing = sorted(required.difference(dataframe.columns))
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def _select_rows(
    dataframe: pd.DataFrame,
    selection: AnalysisSelection,
    table_name: str,
) -> pd.DataFrame:
    selected = dataframe.loc[
        (dataframe["channel"] == selection.channel)
        & (dataframe["roi_id"] == selection.roi_id)
    ].copy()
    if selected.empty:
        raise ValueError(
            f"{table_name} has no rows for channel={selection.channel}, "
            f"roi_id={selection.roi_id}"
        )
    return selected


def _validate_trace_alignment(
    diameter: pd.DataFrame,
    reporter: pd.DataFrame,
) -> float:
    if len(diameter) != len(reporter):
        raise ValueError(
            "Diameter and reporter traces have different sample counts: "
            f"{len(diameter)} != {len(reporter)}"
        )

    diameter_index = diameter["center_row"].to_numpy(dtype=int)
    reporter_index = reporter["time_index"].to_numpy(dtype=int)
    if not np.array_equal(diameter_index, reporter_index):
        mismatch = int(np.flatnonzero(diameter_index != reporter_index)[0])
        raise ValueError(
            "Diameter center_row and reporter time_index are not aligned at "
            f"row {mismatch}: {diameter_index[mismatch]} != {reporter_index[mismatch]}"
        )

    diameter_time = diameter["time_s"].to_numpy(dtype=float)
    reporter_time = reporter["time_sec"].to_numpy(dtype=float)
    if not np.allclose(diameter_time, reporter_time, rtol=0.0, atol=1e-9):
        max_error = float(np.nanmax(np.abs(diameter_time - reporter_time)))
        raise ValueError(
            "Diameter and reporter times are not aligned; maximum absolute "
            f"difference is {max_error:.12g} s"
        )

    if len(reporter_time) < 2:
        raise ValueError("At least two samples are required")
    intervals = np.diff(reporter_time)
    seconds_per_point = float(np.median(intervals))
    if seconds_per_point <= 0:
        raise ValueError("Sampling interval must be positive")
    if not np.allclose(intervals, seconds_per_point, rtol=0.0, atol=1e-9):
        max_error = float(np.max(np.abs(intervals - seconds_per_point)))
        raise ValueError(
            "Reporter sampling interval is not uniform; maximum deviation "
            f"from median is {max_error:.12g} s"
        )
    return seconds_per_point


def _find_json_analysis(
    document: dict[str, Any],
    *,
    analysis_name: str,
    selection: AnalysisSelection,
) -> dict[str, Any]:
    matches = [
        analysis
        for analysis in document.get("analysis", [])
        if analysis.get("analysis_name") == analysis_name
        and analysis.get("channel") == selection.channel
        and analysis.get("roi_id") == selection.roi_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {analysis_name!r} JSON analysis for "
            f"channel={selection.channel}, roi_id={selection.roi_id}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _load_reporter_events(
    analysis: dict[str, Any],
    reporter: pd.DataFrame,
    seconds_per_point: float,
) -> tuple[ReporterEvent, ...]:
    event_records = analysis.get("summary", {}).get("peak_events", [])
    events: list[ReporterEvent] = []
    sample_count = len(reporter)

    for raw_event in event_records:
        onset = raw_event.get("onset", {})
        peak = raw_event.get("peak", {})
        onset_index = int(onset["index"])
        peak_index = int(peak["index"])
        if not 0 <= onset_index < sample_count:
            raise ValueError(f"Reporter onset index out of bounds: {onset_index}")
        if not 0 <= peak_index < sample_count:
            raise ValueError(f"Reporter peak index out of bounds: {peak_index}")

        onset_time = float(onset["time_sec"])
        table_onset_time = float(reporter.iloc[onset_index]["time_sec"])
        tolerance = max(1e-9, seconds_per_point * 1e-6)
        if not np.isclose(onset_time, table_onset_time, rtol=0.0, atol=tolerance):
            raise ValueError(
                f"JSON onset time does not match reporter table at index "
                f"{onset_index}: {onset_time} != {table_onset_time}"
            )

        events.append(
            ReporterEvent(
                peak_id=int(raw_event["peak_id"]),
                onset_index=onset_index,
                onset_time_sec=onset_time,
                onset_value=float(onset["value"]),
                peak_index=peak_index,
                peak_time_sec=float(peak["time_sec"]),
                peak_value=float(peak["value"]),
                peak_amplitude=float(peak["amplitude"]),
                status=str(raw_event.get("status", "")),
                raw_event=dict(raw_event),
            )
        )

    return tuple(events)


def _infer_source_name(json_path: Path) -> str:
    suffix = ".json"
    return json_path.name[: -len(suffix)] if json_path.name.endswith(suffix) else json_path.name


def load_dataset_from_acq_image(
    *,
    acq_image: Any,
    channel: int,
    roi_id: int,
) -> Dff0DiameterDataset:
    """Load one paired dataset through public ``AcqImage`` analysis APIs.

    Args:
        acq_image: Loaded acquisition with analysis tables available.
        channel: Selected zero-based channel.
        roi_id: Selected ROI identifier.

    Returns:
        Validated paired reporter/diameter dataset.

    Raises:
        ValueError: If either required analysis or aligned table is unavailable.
    """
    try:
        sum_analysis = acq_image.analysis_set.get_analysis(
            "sum_intensity", channel=channel, roi_id=roi_id
        )
        diameter_analysis = acq_image.analysis_set.get_analysis(
            "diameter", channel=channel, roi_id=roi_id
        )
    except KeyError as exc:
        raise ValueError(
            "Both sum_intensity and diameter analyses are required for "
            f"channel={channel}, roi_id={roi_id}"
        ) from exc

    reporter = sum_analysis.result.table.copy()
    diameter = diameter_analysis.result.table.copy()
    if reporter.empty or diameter.empty:
        raise ValueError("Required analysis result table is empty")

    selection = AnalysisSelection(channel=channel, roi_id=roi_id)
    reporter = _select_in_memory_rows(reporter, selection, "sum-intensity table")
    diameter = _select_in_memory_rows(diameter, selection, "diameter table")
    reporter = _normalize_in_memory_reporter_table(reporter)
    diameter = _normalize_in_memory_diameter_table(diameter)
    seconds_per_point = _validate_trace_alignment(diameter, reporter)

    events = tuple(
        ReporterEvent(
            peak_id=int(event.peak_id),
            onset_index=int(event.onset_index),
            onset_time_sec=float(event.onset_time_sec),
            onset_value=float(event.onset_value),
            peak_index=int(event.peak_index),
            peak_time_sec=float(event.peak_time_sec),
            peak_value=float(event.peak_value),
            peak_amplitude=float(event.peak_amplitude),
            status=str(event.status),
            raw_event=event.to_json_dict(),
        )
        for event in sum_analysis.get_peak_events()
    )
    source_path = Path(str(acq_image.path))
    return Dff0DiameterDataset(
        source_name=source_path.name,
        selection=selection,
        seconds_per_point=seconds_per_point,
        reporter=reporter,
        diameter=diameter,
        events=events,
        analysis_json={},
        diameter_csv_path=Path(),
        reporter_csv_path=Path(),
        analysis_json_path=Path(),
    )


def _select_in_memory_rows(
    table: pd.DataFrame,
    selection: AnalysisSelection,
    table_name: str,
) -> pd.DataFrame:
    """Select one channel/ROI when those identity columns are present."""
    if {"channel", "roi_id"}.issubset(table.columns):
        table = table.loc[
            (table["channel"] == selection.channel)
            & (table["roi_id"] == selection.roi_id)
        ].copy()
    if table.empty:
        raise ValueError(
            f"{table_name} has no rows for channel={selection.channel}, "
            f"roi_id={selection.roi_id}"
        )
    return table.reset_index(drop=True)


def _normalize_in_memory_reporter_table(table: pd.DataFrame) -> pd.DataFrame:
    """Validate and order one in-memory sum-intensity trace table."""
    required = {"time_index", "time_sec", "df_f_signal"}
    _require_columns(table, required, "sum-intensity table")
    return table.sort_values("time_index").reset_index(drop=True)


def _normalize_in_memory_diameter_table(table: pd.DataFrame) -> pd.DataFrame:
    """Validate and order one in-memory diameter trace table."""
    required = {"center_row", "time_s", "diameter_um"}
    _require_columns(table, required, "diameter table")
    result = table.sort_values("center_row").reset_index(drop=True)
    result["diameter_um_raw"] = pd.to_numeric(result["diameter_um"], errors="coerce")
    return result
