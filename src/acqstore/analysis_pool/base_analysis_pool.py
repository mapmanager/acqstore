"""Base DataFrame pool for collection-level analysis summaries.

Analysis pools live in ``acqstore`` because they are backend data-model objects,
not GUI widgets. They provide a scripting-friendly, flat pandas DataFrame with
one row per acquisition image/channel/ROI selection and one scalar value per
column. CloudScope can consume the same DataFrame at runtime without knowing
analysis-specific sidecar details.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import pandas as pd

from acqstore.acq_image.analysis.model import AnalysisKey, BaseAnalysis
from acqstore.acq_image.metadata import EXPERIMENT_METADATA_SCHEMA, ExperimentMetadata
from acqstore.schema import ValueType

if TYPE_CHECKING:
    from acqstore.acq_image.acq_image import AcqImage
    from acqstore.acq_image.acq_image_list import AcqImageList


def pool_column_name(prefix: str, summary_key: str) -> str:
    """Return a unique pool DataFrame column name for one summary key.

    Summary keys that already start with ``{prefix}_`` are used as-is so
    analysis-local metric names such as ``velocity_mean`` are not double-prefixed.
    All other keys receive the pool spec prefix, for example
    ``analysis_date`` becomes ``velocity_analysis_date``.

    Args:
        prefix: Pool analysis-spec prefix such as ``"velocity"`` or ``"hr"``.
        summary_key: Analysis-local summary column name.

    Returns:
        Stable pool column name.
    """
    head = f"{prefix}_"
    if summary_key.startswith(head):
        return summary_key
    return f"{head}{summary_key}"


class AnalysisPool:
    """Flat summary table owned by an ``AcqImageList``.

    ``AnalysisPool`` is the base class for concrete collection-level tables such
    as ``VelocityAnalysisPool``. It owns the DataFrame, row identity, row
    refresh/removal behavior, and CSV export. Derived classes define which
    analysis types contribute summary columns.

    The pool does not run analysis and is not a persistence source of truth. It
    reflects state already present in each ``AcqImage`` analysis set, especially
    small JSON summary payloads that are inexpensive to load for large file
    collections.

    Args:
        acq_image_list: Collection that owns this pool.
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
    analysis_specs: tuple[tuple[str, type[BaseAnalysis]], ...] = ()
    _analysis_column_specs: ClassVar[
        tuple[tuple[str, str, type[BaseAnalysis]], ...] | None
    ] = None

    def __init__(self, acq_image_list: AcqImageList) -> None:
        self._acq_image_list = acq_image_list
        self._df = pd.DataFrame(columns=self.columns)
        self.rebuild()

    @classmethod
    def get_analysis_column_specs(
        cls,
    ) -> tuple[tuple[str, str, type[BaseAnalysis]], ...]:
        """Return cached ``(pool_column, summary_key, analysis_cls)`` tuples.

        Column specs are computed once per concrete pool class because analysis
        summary schemas are fixed for the lifetime of a process.

        Returns:
            Tuple of pool column mappings for each configured analysis spec.
        """
        cached = cls._analysis_column_specs
        if cached is not None:
            return cached
        specs: list[tuple[str, str, type[BaseAnalysis]]] = []
        for prefix, analysis_cls in cls.analysis_specs:
            for summary_key in analysis_cls.get_summary_columns():
                specs.append(
                    (pool_column_name(prefix, summary_key), summary_key, analysis_cls)
                )
        cls._analysis_column_specs = tuple(specs)
        return cls._analysis_column_specs

    @classmethod
    def pool_column_names(cls) -> tuple[str, ...]:
        """Return the complete pool column schema for this pool class.

        Returns:
            Tuple containing base columns followed by analysis summary columns.
        """
        analysis_columns = tuple(
            pool_column for pool_column, _, _ in cls.get_analysis_column_specs()
        )
        return cls.base_columns + analysis_columns

    @property
    def columns(self) -> tuple[str, ...]:
        """Return the complete pool column schema.

        Returns:
            Tuple containing base columns followed by prefixed analysis summary
            columns.
        """
        return self.pool_column_names()

    @property
    def dataframe(self) -> pd.DataFrame:
        """Return the live pool DataFrame.

        Returns:
            The internal DataFrame object. Mutating it directly is possible but
            callers that need isolation should use :meth:`get_dataframe`.
        """
        return self._df

    def get_dataframe(self, *, copy: bool = True) -> pd.DataFrame:
        """Return the pool DataFrame.

        Args:
            copy: When true, return a copy so caller mutations do not affect the
                pool. When false, return the live DataFrame.

        Returns:
            Pool DataFrame with one row per ``(file, channel, roi_id)``.
        """
        if copy:
            return self._df.copy()
        return self._df

    def rebuild(self) -> None:
        """Rebuild the entire pool from the current ``AcqImageList`` state.

        Returns:
            None.
        """
        rows: list[dict[str, object]] = []
        for acq_image in self._acq_image_list.get_files():
            for channel, roi_id in self._iter_selection_keys(acq_image):
                rows.append(self._build_row(acq_image, channel=channel, roi_id=roi_id))
        self._df = pd.DataFrame(rows, columns=self.columns)
        self._apply_experiment_metadata_column_dtypes(self._df)
        self._reset_display_row_numbers()

    def refresh_row(self, file_id: str, *, channel: int, roi_id: int) -> None:
        """Create or replace one row from current ``AcqImage`` state.

        Args:
            file_id: Stable acquisition-file identifier.
            channel: Zero-based channel index.
            roi_id: ROI identifier.

        Raises:
            KeyError: If ``file_id`` is not present in the owning list.
        """
        acq_image = self._get_required_acq_image(file_id)
        row = self._build_row(acq_image, channel=int(channel), roi_id=int(roi_id))
        row_id = str(row["pool_row_id"])
        existing = self._df["pool_row_id"] == row_id if "pool_row_id" in self._df.columns else []
        if len(self._df) and bool(existing.any()):
            for column in self.columns:
                self._df.loc[existing, column] = row[column]
        else:
            self._df = pd.concat(
                [self._df, pd.DataFrame([row], columns=self.columns)],
                ignore_index=True,
            )
        self._sort_rows()
        self._apply_experiment_metadata_column_dtypes(self._df)
        self._reset_display_row_numbers()

    def remove_row(self, file_id: str, *, channel: int, roi_id: int) -> None:
        """Remove one row if present.

        Args:
            file_id: Stable acquisition-file identifier.
            channel: Zero-based channel index.
            roi_id: ROI identifier.
        """
        row_id = self.build_pool_row_id(file_id, channel=channel, roi_id=roi_id)
        if "pool_row_id" not in self._df.columns:
            return
        self._df = self._df.loc[self._df["pool_row_id"] != row_id].reset_index(drop=True)
        self._reset_display_row_numbers()

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
    def build_pool_row_id(cls, file_id: str, *, channel: int, roi_id: int) -> str:
        """Return the canonical unique pool row identifier.

        Args:
            file_id: Stable acquisition-file identifier, currently the resolved
                source path.
            channel: Zero-based channel index.
            roi_id: ROI identifier.

        Returns:
            Stable string suitable for GUI unique-row-id contracts.
        """
        return f"{file_id}|channel={int(channel)}|roi_id={int(roi_id)}"

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
                if self._is_pool_analysis(analysis):
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

    def _is_pool_analysis(self, analysis: BaseAnalysis) -> bool:
        return any(
            analysis.key.analysis_name == analysis_cls.analysis_name
            for _, analysis_cls in self.analysis_specs
        )

    def _build_row(self, acq_image: AcqImage, *, channel: int, roi_id: int) -> dict[str, object]:
        base = self._build_base_row(acq_image, channel=channel, roi_id=roi_id)
        row: dict[str, object] = {column: base.get(column, pd.NA) for column in self.base_columns}
        analysis_set = getattr(acq_image, "analysis_set", None)
        values_by_cls: dict[type[BaseAnalysis], dict[str, object]] = {}
        for pool_column, summary_key, analysis_cls in self.get_analysis_column_specs():
            values = values_by_cls.get(analysis_cls)
            if values is None:
                values = {}
                if analysis_set is not None:
                    key = AnalysisKey(
                        analysis_name=analysis_cls.analysis_name,
                        channel=int(channel),
                        roi_id=int(roi_id),
                    )
                    analysis = analysis_set.get(key)
                    if analysis is not None:
                        values = analysis.get_summary_values()
                values_by_cls[analysis_cls] = values
            row[pool_column] = values.get(summary_key, pd.NA)
        return {column: row.get(column, pd.NA) for column in self.columns}

    def _build_base_row(self, acq_image: AcqImage, *, channel: int, roi_id: int) -> dict[str, object]:
        schema_row = acq_image.get_schema_row()
        step_y, step_x = self._safe_physical_units(acq_image)
        file_id = str(getattr(acq_image, "file_id", schema_row.get("path", "")))
        exp_values = self._experiment_metadata_values(acq_image)
        row = {
            "pool_row_id": self.build_pool_row_id(file_id, channel=channel, roi_id=roi_id),
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
        """Return experiment-metadata values for pool base columns.

        Args:
            acq_image: Acquisition image whose experiment metadata should be read.

        Returns:
            Mapping from experiment-metadata field name to backend value.
        """
        section = acq_image.get_metadata_section(ExperimentMetadata.metadata_section_id)
        return section.get_values()

    @staticmethod
    def _pool_value(value: object) -> object:
        """Normalize one backend metadata value for pool DataFrame storage.

        Args:
            value: Raw backend metadata value.

        Returns:
            ``pandas.NA`` when ``value`` is ``None``; otherwise the value unchanged.
        """
        if value is None:
            return pd.NA
        return value

    def _apply_experiment_metadata_column_dtypes(self, df: pd.DataFrame) -> None:
        """Cast experiment-metadata pool columns to nullable pandas numeric dtypes.

        Args:
            df: Live pool DataFrame to update in place.

        Returns:
            None.
        """
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
        self._df = self._df.sort_values(
            by=["path", "channel", "roi_id"],
            kind="stable",
        ).reset_index(drop=True)

    def _reset_display_row_numbers(self) -> None:
        if "pool_row" in self._df.columns:
            self._df["pool_row"] = range(len(self._df))
