"""Canonical per-instance analysis resources for native AcqStore OME-Zarr."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import pandas as pd

from acqstore.acq_image.analysis.model import AnalysisKey
from acqstore.acq_image.io.store_utils import (
    join_store_path,
    path_exists,
    read_dataframe_csv,
    write_dataframe_csv,
)

if TYPE_CHECKING:
    from acqstore.acq_image.acq_analysis_set import AcqAnalysisSet


def analysis_resource_id(analysis_name: str, *, channel: int, roi_id: int) -> str:
    """Return a filesystem-safe analysis-instance identifier.

    Args:
        analysis_name: Stable AcqStore analysis name.
        channel: Analysis channel index.
        roi_id: Analysis ROI identifier.

    Returns:
        Collision-safe identifier for one analysis instance.
    """
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "-", analysis_name).strip("-") or "analysis"
    return f"{safe_name}__c{channel}__r{roi_id}"


def write_native_analysis_resources(
    analysis_set: AcqAnalysisSet,
    store_root: str,
) -> list[dict[str, Any]]:
    """Write canonical per-instance resources and return manifest entries.

    Args:
        analysis_set: Analysis collection whose result resources are exported.
        store_root: Native OME-Zarr store root.

    Returns:
        Ordered native-manifest analysis entries.
    """
    entries: list[dict[str, Any]] = []
    analyses = sorted(
        analysis_set.as_list(),
        key=lambda item: (item.key.analysis_name, item.key.channel, item.key.roi_id),
    )
    for analysis in analyses:
        resource_id = analysis_resource_id(
            str(analysis.key.analysis_name),
            channel=int(analysis.key.channel),
            roi_id=int(analysis.key.roi_id),
        )
        table_path: str | None = None
        if analysis.result.table is not None:
            table_path = f"acqstore/analysis/{resource_id}.table.csv"
            write_dataframe_csv(
                join_store_path(store_root, *table_path.split("/")),
                analysis.result.table,
            )
        peaks_path: str | None = None
        rows_getter = getattr(analysis, "get_pool_peak_rows", None)
        columns_getter = getattr(analysis, "get_pool_peak_columns", None)
        if callable(rows_getter):
            rows = tuple(rows_getter())
            columns = tuple(columns_getter()) if callable(columns_getter) else ()
            peaks_path = f"acqstore/analysis/{resource_id}.peaks.csv"
            write_dataframe_csv(
                join_store_path(store_root, *peaks_path.split("/")),
                pd.DataFrame(list(rows), columns=list(columns) or None),
            )
        entries.append(
            {
                "id": resource_id,
                "analysis_name": str(analysis.key.analysis_name),
                "channel": int(analysis.key.channel),
                "roi_id": int(analysis.key.roi_id),
                "resources": {"table": table_path, "peaks": peaks_path},
            }
        )
    return entries


def load_native_analysis_resources(
    analysis_set: AcqAnalysisSet,
    store_root: str,
    entries: object,
) -> None:
    """Load manifest-declared tables into their exact analysis instances.

    Args:
        analysis_set: Hydrated analysis collection receiving result tables.
        store_root: Native OME-Zarr store root.
        entries: Untrusted ``analyses`` value from the native manifest.

    Raises:
        ValueError: If the resource index is malformed or references an unknown
            analysis instance.
        FileNotFoundError: If a declared table resource is missing.
    """
    if not isinstance(entries, list):
        raise ValueError("Native Zarr manifest field 'analyses' must be a list")
    analysis_set.unload_results_dfs()
    seen: set[AnalysisKey] = set()
    for index, raw in enumerate(entries):
        if not isinstance(raw, dict):
            raise ValueError(f"Native Zarr analyses[{index}] must be an object")
        try:
            key = AnalysisKey(
                str(raw["analysis_name"]),
                int(raw["channel"]),
                int(raw["roi_id"]),
            )
            resources = raw["resources"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Native Zarr analyses[{index}] is malformed") from exc
        if key in seen:
            raise ValueError(f"Duplicate native Zarr analysis resource identity: {key}")
        seen.add(key)
        analysis = analysis_set.get(key)
        if analysis is None:
            raise ValueError(f"Native Zarr manifest references unknown analysis: {key}")
        if not isinstance(resources, dict):
            raise ValueError(f"Native Zarr resources for {key} must be an object")
        table_path = resources.get("table")
        if table_path is None:
            continue
        if not isinstance(table_path, str) or not table_path:
            raise ValueError(f"Native Zarr table resource for {key} must be a path or null")
        if table_path.startswith("/") or ".." in table_path.split("/"):
            raise ValueError(f"Native Zarr table resource escapes the store: {table_path}")
        absolute = join_store_path(store_root, *table_path.split("/"))
        if not path_exists(absolute):
            raise FileNotFoundError(f"Native Zarr analysis table is missing: {absolute}")
        analysis.result.table = read_dataframe_csv(absolute)
    analysis_set._results_csv_loaded = True
    analysis_set.set_clean()
