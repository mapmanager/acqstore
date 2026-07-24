"""Core analysis models and value objects.

This module defines the small public analysis framework used by velocity,
diameter, and future analyses. Analysis instances are identified by analysis
name, channel, and ROI; receive image data through an ``AnalysisDataProvider``;
report progress/cancellation through ``AnalysisRunContext``; and return
``AnalysisResult`` objects containing JSON-serializable summaries and optional
pandas tables.

These classes are part of the public API because they appear in generated docs,
analysis signatures, and scripting workflows.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from acqstore.acq_image.analysis.data_provider import AnalysisDataProvider

RUN_SUMMARY_METADATA_KEYS: tuple[str, ...] = (
    "analysis_date",
    "analysis_time",
    "analysis_version",
)


class AnalysisCancelled(RuntimeError):
    """Raised when an analysis run is cancelled."""


class AnalysisExclusionError(ValueError):
    """Raised when adding an analysis would violate an exclusive-group rule.

    Two analyses with the same non-empty ``exclusive_group`` may not coexist
    for the same ``(channel, roi_id)`` inside one ``AcqAnalysisSet``.
    """


class DetectionValueType(StrEnum):
    """Supported value types for detection parameters."""

    STR = "str"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    ENUM = "enum"


class DetectionParamCategory(StrEnum):
    """Scientific category for grouping detection parameters in frontends.

    Categories describe where a parameter belongs in the analysis pipeline.
    GUI code may use this metadata to add headings while preserving schema
    order. The category is intentionally scientific metadata, not layout
    metadata.
    """

    PREPROCESSING = "Preprocessing"
    PEAK_DETECTION = "Peak Detection"


@dataclass(frozen=True, slots=True)
class DetectionParamSchema:
    """Schema entry describing one detection parameter.

    Args:
        name: Stable parameter key stored in ``detection_params``.
        display_name: Human-readable name for UI rendering.
        value_type: Detection parameter value type.
        default: Default value used when missing from user-supplied params.
        description: Optional help text for UI rendering.
        visible: Whether parameter should be shown in UI.
        editable: Whether parameter should be editable in UI.
        category: Optional scientific grouping for frontend organization.
        choices: Allowed choices for enum-like parameters.
        unit: Optional unit string (for example, ``"px"``).
        methods: Optional tuple of detection-method names that use this
            parameter. ``None`` means the parameter applies to all methods and
            is always visible in method-aware UIs.
    """

    name: str
    display_name: str
    value_type: DetectionValueType
    default: object
    description: str = ""
    visible: bool = True
    editable: bool = True
    choices: tuple[object, ...] | None = None
    unit: str | None = None
    methods: tuple[str, ...] | None = None
    category: DetectionParamCategory | None = None


@dataclass(frozen=True, slots=True)
class AnalysisOverlayTraceData:
    """ROI-local physical coordinates for one raster trace overlay.

    Coordinates are relative to the analyzed ROI crop:
      - ``x`` is time in seconds from the ROI's first row.
      - ``y`` is space in microns from the ROI's first column.

    GUI adapters translate these values to full-image Plotly coordinates using
    the selected ROI bounds and raster grid spacing.

    Args:
        trace_id: Stable overlay identifier.
        x: Time axis values in seconds (ROI-local).
        y: Space axis values in microns (ROI-local).
        color: Optional rendering hint.
        name: Optional human-readable series name.
        visible: Whether the trace should be visible.
    """

    trace_id: str
    x: tuple[float, ...]
    y: tuple[float, ...]
    color: str | None = None
    name: str | None = None
    visible: bool = True

    def __post_init__(self) -> None:
        """Validate overlay trace lengths.

        Raises:
            ValueError: If x and y lengths differ.
        """
        if len(self.x) != len(self.y):
            raise ValueError(
                f"overlay x and y must have same length, got {len(self.x)} and {len(self.y)}"
            )


@dataclass(frozen=True, slots=True)
class AnalysisKey:
    """Unique identity for one analysis instance.

    Identity is ``(analysis_name, channel, roi_id)``. The parent ``AcqImage`` is
    not part of the key because one ``AcqAnalysisSet`` belongs to exactly one
    acquisition image.

    Args:
        analysis_name: Stable analysis type name.
        channel: Channel index.
        roi_id: ROI identifier.
    """

    analysis_name: str
    channel: int
    roi_id: int

    def csv_filename(self, source_name: str) -> str:
        """Return CSV filename for this analysis type and source file.

        Args:
            source_name: Source file basename including extension, such as
                ``"myfile.tif"``.

        Returns:
            Filename such as ``"myfile.tif.velocity.csv"``.
        """
        return f"{source_name}.{self.analysis_name}.csv"


@dataclass(slots=True)
class AnalysisRunContext:
    """Runtime context for progress reporting and cooperative cancellation.

    Analysis code receives this object from GUI controllers, batch runners, or
    scripts. Derived analyses should call :meth:`report_progress` at natural
    milestones and :meth:`raise_if_cancelled` or :meth:`is_cancelled` inside
    long loops.

    Args:
        progress_callback: Optional callback receiving ``(fraction, message)``.
            ``fraction`` should be between 0 and 1 when known, or ``None`` when
            progress is indeterminate.
        cancel_callback: Optional callback returning ``True`` when cancellation
            has been requested.
    """

    progress_callback: Callable[[float | None, str], None] | None = None
    cancel_callback: Callable[[], bool] | None = None

    def report_progress(self, fraction: float | None, message: str = "") -> None:
        """Report progress for a running analysis.

        Args:
            fraction: Fraction complete, or None if unknown.
            message: Human-readable progress message.

        Returns:
            None.
        """
        if self.progress_callback is not None:
            self.progress_callback(fraction, message)

    def is_cancelled(self) -> bool:
        """Return whether cancellation has been requested.

        Returns:
            True if cancellation has been requested.
        """
        return bool(self.cancel_callback is not None and self.cancel_callback())

    def raise_if_cancelled(self) -> None:
        """Raise if cancellation has been requested.

        Raises:
            AnalysisCancelled: If cancellation has been requested.
        """
        if self.is_cancelled():
            raise AnalysisCancelled("Analysis cancelled")




@dataclass(frozen=True, slots=True)
class AnalysisPlotData:
    """Display-ready x/y plot data for one analysis.

    GUI widgets and notebooks should be able to plot this object without knowing
    analysis-specific table column names such as ``time_s``, ``velocity``, or
    ``diameter_um``. Values are already in physical units when an analysis can
    provide them.

    Args:
        x: X-axis values.
        y: Y-axis values.
        x_label: Human-readable x-axis label including units when known.
        y_label: Human-readable y-axis label including units when known.
        series_name: Human-readable series name.
    """

    x: tuple[float, ...]
    y: tuple[float, ...]
    x_label: str
    y_label: str
    series_name: str = "analysis"

    def __post_init__(self) -> None:
        """Validate plot data lengths.

        Raises:
            ValueError: If x and y lengths differ.
        """
        if len(self.x) != len(self.y):
            raise ValueError(f"x and y must have same length, got {len(self.x)} and {len(self.y)}")


@dataclass(slots=True)
class AnalysisResult:
    """Outputs from one completed or loaded analysis.

    ``summary`` stores small JSON-serializable values that belong in an
    acquisition sidecar. ``table`` stores larger per-row results that can be
    saved to CSV and inspected in notebooks or GUI tables.

    Args:
        summary: Small JSON-serializable result dictionary.
        table: Optional large tabular output. Tables produced by derived
            analyses must not include reserved bookkeeping columns ``channel``
            or ``roi_id``; those are added by :meth:`BaseAnalysis.table_with_bookkeeping`.
    """

    summary: dict[str, Any] = field(default_factory=dict)
    table: pd.DataFrame | None = None


class BaseAnalysis(ABC):
    """Base class for one ROI/channel analysis instance.

    Derived classes define a stable ``analysis_name``, optional
    ``detection_schema``, and a concrete :meth:`run` method. The base class owns
    analysis identity, validated detection parameters, dirty state,
    serialization helpers, and common table utilities.

    Detection parameters describe scientific behavior and are serialized with
    the analysis. Runtime execution options, such as multiprocessing worker
    counts, should live on derived classes and should not be serialized unless
    they change scientific results.

    Args:
        channel: Zero-based channel index for this analysis.
        roi_id: ROI identifier for this analysis.
        detection_params: Optional detection parameter values. Missing values
            are filled from ``detection_schema`` defaults when available.
    """

    analysis_name: ClassVar[str]
    depends_on: ClassVar[tuple[str, ...]] = ()
    detection_schema: ClassVar[tuple[Any, ...]] = ()
    summary_columns: ClassVar[tuple[str, ...]] = ()
    analysis_version: ClassVar[int | float | str | None] = None
    exclusive_group: ClassVar[str | None] = None

    def __init__(
        self,
        *,
        channel: int,
        roi_id: int,
        detection_params: dict[str, Any] | None = None,
    ) -> None:
        self.key = AnalysisKey(
            analysis_name=self.analysis_name,
            channel=channel,
            roi_id=roi_id,
        )
        self.detection_params = self.get_default_detection_params()
        if detection_params is not None:
            self.validate_detection_params(detection_params)
            self.detection_params.update(detection_params)
        self.result = AnalysisResult()
        self._dirty = False

    def is_dirty(self) -> bool:
        """Return whether this analysis has unsaved changes.

        Returns:
            True if this analysis has unsaved changes.
        """
        return self._dirty

    def set_dirty(self) -> None:
        """Mark this analysis dirty.

        Returns:
            None.
        """
        self._dirty = True

    def set_clean(self) -> None:
        """Mark this analysis clean.

        Returns:
            None.
        """
        self._dirty = False

    @classmethod
    def get_detection_schema(cls) -> tuple[DetectionParamSchema, ...]:
        """Return detection parameter schema.

        Returns:
            Tuple of ``DetectionParamSchema`` entries.

        Raises:
            TypeError: If the class ``detection_schema`` contains non-schema
                entries.
        """
        schema: list[DetectionParamSchema] = []
        for entry in cls.detection_schema:
            if not isinstance(entry, DetectionParamSchema):
                raise TypeError(
                    f"{cls.__name__}.detection_schema must contain DetectionParamSchema "
                    f"entries, got: {type(entry).__name__}"
                )
            schema.append(entry)
        return tuple(schema)

    @classmethod
    def get_detection_schema_dataframe(cls) -> pd.DataFrame:
        """Return this analysis type's detection-parameter schema as a DataFrame.

        This is a scripting and documentation convenience that describes the
        full detection-parameter schema. It is available on every analysis type
        (for example ``RadonVelocityAnalysis``, ``DiameterAnalysis``,
        ``EventAnalysis``, ``HeartRateAnalysis``).

        Returns:
            DataFrame indexed by parameter ``name`` with one row per detection
            parameter. Columns are ``display_name``, ``type``, ``default``,
            ``choices``, ``unit``, ``editable``, ``visible``, ``methods``,
            ``category``, and ``description``. The DataFrame is empty (columns only) when the
            analysis declares no detection parameters.

        Raises:
            TypeError: If the class ``detection_schema`` contains non-schema
                entries.
        """
        columns = [
            "name",
            "display_name",
            "type",
            "default",
            "choices",
            "unit",
            "editable",
            "visible",
            "methods",
            "category",
            "description",
        ]
        rows = [
            {
                "name": entry.name,
                "display_name": entry.display_name,
                "type": entry.value_type.value,
                "default": entry.default,
                "choices": entry.choices,
                "unit": entry.unit,
                "editable": entry.editable,
                "visible": entry.visible,
                "methods": entry.methods,
                "category": entry.category.value if entry.category is not None else None,
                "description": entry.description,
            }
            for entry in cls.get_detection_schema()
        ]
        return pd.DataFrame(rows, columns=columns).set_index("name")

    def finalize_summary(self, summary: dict[str, Any]) -> dict[str, Any]:
        """Merge run metadata into a summary with metadata keys first.

        Args:
            summary: Analysis-local summary produced by the derived ``run``
                implementation before metadata is applied.

        Returns:
            Summary dictionary with ``analysis_date``, ``analysis_time``, and
            optional ``analysis_version`` prepended before remaining keys.
        """
        now = datetime.now()
        metadata: dict[str, Any] = {
            "analysis_date": now.strftime("%y%m%d"),
            "analysis_time": f"{now.strftime('%H:%M:%S')}.{now.microsecond // 1000:03d}",
        }
        if self.analysis_version is not None:
            metadata["analysis_version"] = self.analysis_version

        metadata_keys = set(RUN_SUMMARY_METADATA_KEYS)
        rest = {key: value for key, value in summary.items() if key not in metadata_keys}
        ordered: dict[str, Any] = {}
        ordered.update(metadata)
        ordered.update(rest)
        return ordered

    @classmethod
    def get_summary_columns(cls) -> tuple[str, ...]:
        """Return flat summary keys intended for analysis-pool tables.

        Derived analyses declare this class-level schema so collection-level
        pools can create stable DataFrame columns even when a given
        ``(channel, roi_id)`` has no completed analysis. Column names returned
        here are analysis-local names. Pool classes map them to unique pool
        column names, prepending the pool spec prefix unless a key already
        starts with that prefix.

        Returns:
            Tuple of stable analysis-local summary column names.
        """
        return tuple(str(column) for column in cls.summary_columns)

    def get_summary_values(self) -> dict[str, object]:
        """Return flat summary values for analysis-pool tables.

        The default implementation selects values from
        :attr:`AnalysisResult.summary` using :meth:`get_summary_columns` and
        fills missing keys with :data:`pandas.NA`. Analyses with nested summary
        dictionaries should override this method and still return exactly the
        keys declared by :meth:`get_summary_columns`.

        Returns:
            Mapping from analysis-local summary column name to scalar value.
        """
        return {
            key: self.result.summary.get(key, pd.NA)
            for key in self.get_summary_columns()
        }

    @classmethod
    def get_default_detection_params(cls) -> dict[str, Any]:
        """Return default detection parameters from ``detection_schema``.

        Returns:
            Mapping from parameter name to default value.

        Raises:
            ValueError: If the schema contains duplicate parameter names.
        """
        defaults: dict[str, Any] = {}
        for field_schema in cls.get_detection_schema():
            if field_schema.name in defaults:
                raise ValueError(
                    f"Duplicate detection param schema name: {field_schema.name!r}"
                )
            defaults[field_schema.name] = field_schema.default
        return defaults

    @classmethod
    def validate_detection_params(cls, params: dict[str, Any]) -> None:
        """Validate detection parameter mapping against schema.

        Args:
            params: Detection parameter mapping.

        Returns:
            None.

        Raises:
            KeyError: If any key is not present in the schema.
            TypeError: If any value has the wrong type.
            ValueError: If any enum value is not in ``choices``.
        """
        if not isinstance(params, dict):
            raise TypeError(f"detection_params must be dict, got: {type(params).__name__}")

        schema_by_name = {entry.name: entry for entry in cls.get_detection_schema()}
        for key, value in params.items():
            if key not in schema_by_name:
                raise KeyError(f"Unknown detection param: {key!r}")

            entry = schema_by_name[key]
            match entry.value_type:
                case DetectionValueType.INT:
                    if isinstance(value, bool) or not isinstance(value, int):
                        raise TypeError(f"{key!r} must be int, got: {type(value).__name__}")
                case DetectionValueType.FLOAT:
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        raise TypeError(
                            f"{key!r} must be float or int, got: {type(value).__name__}"
                        )
                case DetectionValueType.BOOL:
                    if not isinstance(value, bool):
                        raise TypeError(f"{key!r} must be bool, got: {type(value).__name__}")
                case DetectionValueType.STR:
                    if not isinstance(value, str):
                        raise TypeError(f"{key!r} must be str, got: {type(value).__name__}")
                case DetectionValueType.ENUM:
                    if entry.choices is None:
                        raise ValueError(f"{key!r} has value_type=ENUM but no choices")
                case _:
                    raise ValueError(f"Unsupported detection value type: {entry.value_type!r}")

            if entry.choices is not None and value not in entry.choices:
                raise ValueError(f"{key!r} must be one of {entry.choices!r}, got: {value!r}")

    @abstractmethod
    def run(
        self,
        data_provider: AnalysisDataProvider,
        *,
        context: AnalysisRunContext | None = None,
        dependencies: dict[str, BaseAnalysis] | None = None,
    ) -> AnalysisResult:
        """Run analysis using a narrow data-provider API.

        Derived implementations should read pixels and physical units only
        through ``data_provider``. They should use ``context`` for progress and
        cancellation, respect dependency analyses when declared, store results in
        ``self.result``, and call ``self.set_dirty()`` after mutating results.

        Args:
            data_provider: Minimal image/header access provider.
            context: Optional runtime context for progress and cancellation.
            dependencies: Dependency analyses keyed by analysis name.

        Returns:
            Analysis result. Implementations usually return ``self.result``.
        """
        raise NotImplementedError

    def get_plot_data(self) -> AnalysisPlotData | None:
        """Return display-ready plot data for this analysis.

        Derived analyses override this when they have a canonical x/y plot.

        Returns:
            Plot data, or None when the analysis has no canonical plot.
        """
        return None

    def get_overlay_traces(self) -> tuple[AnalysisOverlayTraceData, ...]:
        """Return ROI-local trace overlays for raster viewers.

        Derived analyses override this when they expose edge or path overlays.
        Coordinates are ROI-local physical units; GUI layers translate to
        full-image Plotly coordinates.

        Returns:
            Tuple of overlay trace data. Empty when no overlays exist.
        """
        return ()

    def has_table(self) -> bool:
        """Return whether this analysis has table output.

        Returns:
            True if result table exists.
        """
        return self.result.table is not None

    def get_table_columns(self) -> list[str]:
        """Return result table column names.

        Returns:
            Column names, or an empty list when no table exists.
        """
        if self.result.table is None:
            return []
        return list(self.result.table.columns)

    def get_column(self, name: str) -> list[Any]:
        """Return one result table column as a list.

        Args:
            name: Column name.

        Returns:
            Column values as a list.

        Raises:
            ValueError: If this analysis has no table.
            KeyError: If the column does not exist.
        """
        if self.result.table is None:
            raise ValueError(f"Analysis {self.analysis_name!r} has no table")
        if name not in self.result.table.columns:
            raise KeyError(f"Column not found: {name!r}")
        return self.result.table[name].tolist()

    def table_with_bookkeeping(self) -> pd.DataFrame | None:
        """Return table with channel and ROI bookkeeping columns.

        Returns:
            DataFrame with ``channel`` and ``roi_id`` columns added, or None if
            the analysis has no table output.

        Raises:
            ValueError: If result table already contains reserved bookkeeping
                columns.
        """
        if self.result.table is None:
            return None

        reserved = {"channel", "roi_id"}
        overlap = reserved.intersection(self.result.table.columns)
        if overlap:
            raise ValueError(
                f"Analysis table already has reserved columns: {sorted(overlap)}"
            )

        df = self.result.table.copy()
        df.insert(0, "roi_id", self.key.roi_id)
        df.insert(0, "channel", self.key.channel)
        return df

    def to_json_dict(self) -> dict[str, Any]:
        """Return JSON-serializable analysis record.

        Returns:
            Dictionary containing identity, detection params, and summary.
        """
        return {
            "analysis_name": self.key.analysis_name,
            "channel": self.key.channel,
            "roi_id": self.key.roi_id,
            "detection_params": dict(self.detection_params),
            "summary": dict(self.result.summary),
        }

    def load_json_dict(self, record: dict[str, Any]) -> None:
        """Load detection params and summary from a JSON record.

        Args:
            record: Analysis record from source sidecar JSON.

        Returns:
            None.
        """
        self.detection_params = dict(record.get("detection_params", {}))
        self.result.summary = dict(record.get("summary", {}))

    def save_record_json(self, path: str | Path) -> None:
        """Save this analysis record to a standalone JSON file.

        This helper is useful in tests. Normal AcqImage save code should use
        ``AcqAnalysisSet.serialize_json_analysis()`` and merge records into the
        source sidecar JSON.

        Args:
            path: Output JSON file path.

        Returns:
            None.
        """
        Path(path).write_text(json.dumps(self.to_json_dict(), indent=2))

    def load_record_json(self, path: str | Path) -> None:
        """Load this analysis record from a standalone JSON file.

        Args:
            path: Input JSON file path.

        Returns:
            None.
        """
        self.load_json_dict(json.loads(Path(path).read_text()))
