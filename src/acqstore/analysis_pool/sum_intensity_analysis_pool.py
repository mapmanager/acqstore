"""Sum-intensity peak-event pool for AcqImageList collections.

The sum-intensity pool is a backend DataFrame cache owned by an
``AcqImageList``. It has one or more rows per acquisition image/channel/ROI:
one row for each detected peak, one explicit ``no_peaks`` row when analysis ran
successfully with zero detected peaks, and one ``not_analyzed`` seed row when no
sum-intensity analysis exists for that selection.

The pool intentionally does not know individual sum-intensity feature names. It
uses the pool-facing API on ``SumIntensityAnalysis`` to retrieve scalar summary
columns and scalar flattened peak rows, so future analysis features can be added
in the analysis core without changing this pool.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from acqstore.acq_image.analysis.model import AnalysisKey
from acqstore.acq_image.analysis.sum_intensity_analysis.sum_intensity_analysis import (
    SumIntensityAnalysis,
)
from acqstore.acq_image.metadata import EXPERIMENT_METADATA_SCHEMA, ExperimentMetadata
from acqstore.schema import ValueType

if TYPE_CHECKING:
    from acqstore.acq_image.acq_image import AcqImage
    from acqstore.acq_image.acq_image_list import AcqImageList


class SumIntensityAnalysisPool:
    """Flat peak-event table owned by an ``AcqImageList``.

    Args:
        acq_image_list: Collection that owns this pool.

    Raises:
        ValueError: If pool column names collide.
    """

    experiment_metadata_columns: tuple[str, ...] = (
        "age",
        "sex",
        "branch_order",
        "direction",
        "depth",
        "note",
    )
    base_columns: tuple[str, ...] = (
        "pool_row_id",
        "pool_row",
        "name",
        "path",
        "parent",
        "grandparent",
        "condition",
        "genotype",
        "age",
        "sex",
        "branch_order",
        "direction",
        "depth",
        "note",
        "accept",
        "channel",
        "roi_id",
        "step_y",
        "step_x",
    )
    row_type_column = "peak_row_type"

    def __init__(self, acq_image_list: AcqImageList) -> None:
        self._acq_image_list = acq_image_list
        self._validate_columns()
        self._df = pd.DataFrame(columns=self.columns)
        self.rebuild()

    @classmethod
    def pool_column_names(cls) -> tuple[str, ...]:
        """Return the complete pool column schema.

        Returns:
            Tuple containing base columns, scalar summary columns, row type, and
            flattened peak columns.

        Raises:
            ValueError: If any column names collide.
        """
        columns = (
            cls.base_columns
            + SumIntensityAnalysis.get_pool_summary_columns()
            + (cls.row_type_column,)
            + SumIntensityAnalysis.get_pool_peak_columns()
        )
        duplicates = _duplicates(columns)
        if duplicates:
            raise ValueError(
                "SumIntensityAnalysisPool columns must be unique; duplicate columns: "
                + ", ".join(duplicates)
            )
        return columns

    @classmethod
    def _validate_columns(cls) -> None:
        columns = cls.pool_column_names()
        if len(columns) != len(set(columns)):
            raise ValueError("SumIntensityAnalysisPool columns must be unique")

    @property
    def columns(self) -> tuple[str, ...]:
        """Return the complete pool column schema.

        Returns:
            Tuple of DataFrame column names.
        """
        return self.pool_column_names()

    @property
    def dataframe(self) -> pd.DataFrame:
        """Return the live pool DataFrame.

        Returns:
            Internal DataFrame object.
        """
        return self._df

    def get_dataframe(self, *, copy: bool = True) -> pd.DataFrame:
        """Return the pool DataFrame.

        Args:
            copy: When true, return a copy so caller mutations do not affect the
                pool. When false, return the live DataFrame.

        Returns:
            Pool DataFrame with one or more rows per file/channel/ROI.
        """
        if copy:
            return self._df.copy()
        return self._df

    def row_ids_for_selection(
        self,
        file_id: str,
        *,
        channel: int,
        roi_id: int,
        peak_row_types: Sequence[str] = ("peak",),
    ) -> tuple[str, ...]:
        """Return ``pool_row_id`` values for one file/channel/ROI selection.

        Args:
            file_id: Stable acquisition-file identifier.
            channel: Zero-based channel index.
            roi_id: ROI identifier.
            peak_row_types: Row types to include. Defaults to detected peak rows
                only; pass an empty sequence to include all row types for the
                selection.

        Returns:
            Matching ``pool_row_id`` strings in table order.
        """
        if self._df.empty:
            return ()
        mask = (
            (self._df["path"] == str(file_id))
            & (self._df["channel"] == int(channel))
            & (self._df["roi_id"] == int(roi_id))
        )
        if peak_row_types:
            mask = mask & self._df[self.row_type_column].isin(list(peak_row_types))
        if not mask.any():
            return ()
        return tuple(self._df.loc[mask, "pool_row_id"].astype(str).tolist())

    def rebuild(self) -> None:
        """Rebuild the entire pool from the current ``AcqImageList`` state.

        Returns:
            None.
        """
        rows: list[dict[str, object]] = []
        for acq_image in self._acq_image_list.get_files():
            for channel, roi_id in self._iter_selection_keys(acq_image):
                rows.extend(self._build_rows(acq_image, channel=channel, roi_id=roi_id))
        self._df = pd.DataFrame(rows, columns=self.columns)
        self._finish_table_update()

    def refresh_rows(self, file_id: str, *, channel: int, roi_id: int) -> None:
        """Create or replace all rows for one file/channel/ROI selection.

        Args:
            file_id: Stable acquisition-file identifier.
            channel: Zero-based channel index.
            roi_id: ROI identifier.

        Raises:
            KeyError: If ``file_id`` is not present in the owning list.
        """
        acq_image = self._get_required_acq_image(file_id)
        self.remove_rows(file_id, channel=channel, roi_id=roi_id)
        rows = self._build_rows(acq_image, channel=int(channel), roi_id=int(roi_id))
        if rows:
            self._df = pd.concat(
                [self._df, pd.DataFrame(rows, columns=self.columns)],
                ignore_index=True,
            )
        self._finish_table_update()

    def remove_rows(self, file_id: str, *, channel: int, roi_id: int) -> None:
        """Remove all peak rows for one file/channel/ROI selection.

        Args:
            file_id: Stable acquisition-file identifier.
            channel: Zero-based channel index.
            roi_id: ROI identifier.
        """
        if self._df.empty:
            return
        mask = (
            (self._df["path"] == str(file_id))
            & (self._df["channel"] == int(channel))
            & (self._df["roi_id"] == int(roi_id))
        )
        self._df = self._df.loc[~mask].reset_index(drop=True)
        self._reset_display_row_numbers()

    def refresh_file(self, file_id: str) -> None:
        """Refresh all rows for one acquisition file.

        Args:
            file_id: Stable acquisition-file identifier.

        Raises:
            KeyError: If ``file_id`` is not present in the owning list.
        """
        acq_image = self._get_required_acq_image(file_id)
        if not self._df.empty:
            self._df = self._df.loc[self._df["path"] != str(file_id)].reset_index(drop=True)
        rows: list[dict[str, object]] = []
        for channel, roi_id in self._iter_selection_keys(acq_image):
            rows.extend(self._build_rows(acq_image, channel=channel, roi_id=roi_id))
        if rows:
            self._df = pd.concat(
                [self._df, pd.DataFrame(rows, columns=self.columns)],
                ignore_index=True,
            )
        self._finish_table_update()

    def remove_roi(self, file_id: str, *, roi_id: int) -> None:
        """Remove all rows for one file/ROI across channels.

        Args:
            file_id: Stable acquisition-file identifier.
            roi_id: ROI identifier.
        """
        if self._df.empty:
            return
        mask = (self._df["path"] == str(file_id)) & (self._df["roi_id"] == int(roi_id))
        self._df = self._df.loc[~mask].reset_index(drop=True)
        self._reset_display_row_numbers()

    def to_csv(self, path: str | Path) -> None:
        """Write the current pool DataFrame to CSV.

        Args:
            path: Destination CSV path. Parent directories are created when
                needed.
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._df.to_csv(out_path, index=False)

    @classmethod
    def build_pool_row_id(
        cls,
        file_id: str,
        *,
        channel: int,
        roi_id: int,
        peak_row_type: str,
        peak_id: object = None,
    ) -> str:
        """Return the canonical unique pool row identifier.

        Args:
            file_id: Stable acquisition-file identifier.
            channel: Zero-based channel index.
            roi_id: ROI identifier.
            peak_row_type: Row type such as ``"peak"`` or ``"no_peaks"``.
            peak_id: Peak identifier for detected peak rows.

        Returns:
            Stable string suitable for GUI unique-row-id contracts.
        """
        peak_part = "none" if peak_id is None or peak_id is pd.NA else str(peak_id)
        return (
            f"{file_id}|channel={int(channel)}|roi_id={int(roi_id)}|"
            f"peak_row_type={peak_row_type}|peak_id={peak_part}"
        )

    def _get_required_acq_image(self, file_id: str) -> AcqImage:
        acq_image = self._acq_image_list.get_file_by_id(file_id)
        if acq_image is None:
            raise KeyError(f"file_id not found in AcqImageList: {file_id!r}")
        return acq_image

    def _iter_selection_keys(self, acq_image: AcqImage) -> list[tuple[int, int]]:
        keys: set[tuple[int, int]] = set()
        for channel in self._iter_channels(acq_image):
            for roi_id in self._iter_roi_ids(acq_image):
                keys.add((int(channel), int(roi_id)))
        analysis_set = getattr(acq_image, "analysis_set", None)
        if analysis_set is not None:
            for analysis in analysis_set.as_list():
                if analysis.key.analysis_name == SumIntensityAnalysis.analysis_name:
                    keys.add((int(analysis.key.channel), int(analysis.key.roi_id)))
        return sorted(keys)

    def _iter_channels(self, acq_image: AcqImage) -> Sequence[int]:
        images = getattr(acq_image, "images", None)
        channels = getattr(images, "channels", None)
        if callable(channels):
            return tuple(int(channel) for channel in channels())
        num_channels = getattr(images, "num_channels", None)
        if num_channels is not None:
            return tuple(range(int(num_channels)))
        try:
            schema_row = acq_image.get_schema_row()
            return tuple(range(int(schema_row.get("num_channels", 0))))
        except Exception:
            return ()

    def _iter_roi_ids(self, acq_image: AcqImage) -> Sequence[int]:
        rois = getattr(acq_image, "rois", None)
        get_roi_ids = getattr(rois, "get_roi_ids", None)
        if callable(get_roi_ids):
            return tuple(int(roi_id) for roi_id in get_roi_ids())
        return ()

    def _build_rows(self, acq_image: AcqImage, *, channel: int, roi_id: int) -> list[dict[str, object]]:
        base = self._build_base_row(acq_image, channel=channel, roi_id=roi_id)
        analysis = self._sum_intensity_analysis(acq_image, channel=channel, roi_id=roi_id)
        if analysis is None:
            return [self._complete_row(base, {}, {}, peak_row_type="not_analyzed")]
        summary = analysis.get_pool_summary_values()
        peak_rows = analysis.get_pool_peak_rows()
        if not peak_rows:
            return [self._complete_row(base, summary, {}, peak_row_type="no_peaks")]
        return [
            self._complete_row(base, summary, peak_row, peak_row_type="peak")
            for peak_row in peak_rows
        ]

    def _sum_intensity_analysis(
        self,
        acq_image: AcqImage,
        *,
        channel: int,
        roi_id: int,
    ) -> SumIntensityAnalysis | None:
        analysis_set = getattr(acq_image, "analysis_set", None)
        if analysis_set is None:
            return None
        key = AnalysisKey(
            analysis_name=SumIntensityAnalysis.analysis_name,
            channel=int(channel),
            roi_id=int(roi_id),
        )
        analysis = analysis_set.get(key)
        if analysis is None:
            return None
        if not isinstance(analysis, SumIntensityAnalysis):
            raise TypeError(
                "Expected SumIntensityAnalysis for sum-intensity pool, got "
                f"{type(analysis).__name__}"
            )
        return analysis

    def _complete_row(
        self,
        base: dict[str, object],
        summary: dict[str, object],
        peak_row: dict[str, object],
        *,
        peak_row_type: str,
    ) -> dict[str, object]:
        peak_id = peak_row.get("peak_id")
        row: dict[str, object] = {column: pd.NA for column in self.columns}
        row.update(base)
        row.update(summary)
        row[self.row_type_column] = peak_row_type
        row.update(peak_row)
        row["pool_row_id"] = self.build_pool_row_id(
            str(base["path"]),
            channel=int(base["channel"]),
            roi_id=int(base["roi_id"]),
            peak_row_type=peak_row_type,
            peak_id=None if peak_row_type != "peak" else peak_id,
        )
        self._validate_scalar_row(row)
        return {column: row.get(column, pd.NA) for column in self.columns}

    def _build_base_row(self, acq_image: AcqImage, *, channel: int, roi_id: int) -> dict[str, object]:
        schema_row = acq_image.get_schema_row()
        step_y, step_x = self._safe_physical_units(acq_image)
        file_id = str(getattr(acq_image, "file_id", schema_row.get("path", "")))
        exp_values = self._experiment_metadata_values(acq_image)
        row = {
            "pool_row_id": self.build_pool_row_id(
                file_id,
                channel=channel,
                roi_id=roi_id,
                peak_row_type="pending",
            ),
            "pool_row": 0,
            "name": schema_row.get("name", Path(file_id).name),
            "path": file_id,
            "parent": schema_row.get("parent", pd.NA),
            "grandparent": schema_row.get("grandparent", pd.NA),
            "condition": schema_row.get("condition", pd.NA),
            "genotype": schema_row.get("genotype", pd.NA),
            "accept": schema_row.get("accept", pd.NA),
            "channel": int(channel),
            "roi_id": int(roi_id),
            "step_y": step_y,
            "step_x": step_x,
        }
        for column in self.experiment_metadata_columns:
            row[column] = self._pool_value(exp_values.get(column, pd.NA))
        return row

    def _experiment_metadata_values(self, acq_image: AcqImage) -> dict[str, object]:
        section = acq_image.get_metadata_section(ExperimentMetadata.metadata_section_id)
        return section.get_values()

    @staticmethod
    def _pool_value(value: object) -> object:
        if value is None:
            return pd.NA
        return value

    def _finish_table_update(self) -> None:
        self._sort_rows()
        self._apply_experiment_metadata_column_dtypes(self._df)
        self._reset_display_row_numbers()

    def _apply_experiment_metadata_column_dtypes(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        fields_by_name = {fs.name: fs for fs in EXPERIMENT_METADATA_SCHEMA.fields}
        for column in self.experiment_metadata_columns:
            if column not in df.columns:
                continue
            field_schema = fields_by_name.get(column)
            if field_schema is None:
                continue
            if field_schema.value_type is ValueType.INT:
                df[column] = df[column].astype("Int64")
            elif field_schema.value_type is ValueType.FLOAT:
                df[column] = df[column].astype("Float64")

    def _safe_physical_units(self, acq_image: AcqImage) -> tuple[object, object]:
        getter = getattr(acq_image, "get_image_physical_units", None)
        if not callable(getter):
            return (pd.NA, pd.NA)
        try:
            step_y, step_x = getter()
        except Exception:
            return (pd.NA, pd.NA)
        return (step_y, step_x)

    def _sort_rows(self) -> None:
        if self._df.empty:
            return
        sort_columns = ["path", "channel", "roi_id", self.row_type_column, "peak_id"]
        self._df = self._df.sort_values(by=sort_columns, kind="stable").reset_index(drop=True)

    def _reset_display_row_numbers(self) -> None:
        if "pool_row" in self._df.columns:
            self._df["pool_row"] = range(len(self._df))

    @classmethod
    def _validate_scalar_row(cls, row: dict[str, object]) -> None:
        for column, value in row.items():
            if _is_scalar_cell(value):
                continue
            raise TypeError(
                f"SumIntensityAnalysisPool cell must be scalar; column {column!r} "
                f"has value type {type(value).__name__}"
            )


def _duplicates(values: tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


def _is_scalar_cell(value: object) -> bool:
    if value is None or value is pd.NA:
        return True
    if isinstance(value, (str, int, float, bool, np.integer, np.floating, np.bool_)):
        return True
    return False
