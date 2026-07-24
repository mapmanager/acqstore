"""Analysis collection owned by one AcqImage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from acqstore.acq_image.analysis.data_provider import AnalysisDataProvider
from acqstore.acq_image.analysis.model import (
    AnalysisExclusionError,
    AnalysisKey,
    AnalysisResult,
    AnalysisRunContext,
    BaseAnalysis,
)
from acqstore.acq_image.analysis.registry import get_analysis_class


class AcqAnalysisSet:
    """Collection and orchestrator for analyses owned by one AcqImage.

    Args:
        source_path: Source acquisition file path.
        data_provider: Thin provider used by analyses to access ROI image data
            and physical units.
    """

    def __init__(
        self,
        source_path: str | Path,
        data_provider: AnalysisDataProvider | None = None,
    ) -> None:
        self.source_path = str(source_path)
        self._data_provider = data_provider
        self._analyses: dict[AnalysisKey, BaseAnalysis] = {}
        self._results_csv_loaded = False
        self._dirty = False

    def is_dirty(self) -> bool:
        """Return whether this set or any child analysis is dirty.

        Returns:
            True if unsaved changes exist.
        """
        return self._dirty or any(analysis.is_dirty() for analysis in self._analyses.values())

    def set_dirty(self) -> None:
        """Mark this analysis set dirty.

        Returns:
            None.
        """
        self._dirty = True

    def set_clean(self) -> None:
        """Mark this set and all child analyses clean.

        Returns:
            None.
        """
        self._dirty = False
        for analysis in self._analyses.values():
            analysis.set_clean()

    def add(self, analysis: BaseAnalysis) -> None:
        """Add one analysis instance.

        Args:
            analysis: Analysis instance to store.

        Raises:
            ValueError: If an analysis already exists with the same key.
            AnalysisExclusionError: If another analysis in the same
                ``exclusive_group`` already exists for the same
                ``(channel, roi_id)``.
        """
        if analysis.key in self._analyses:
            raise ValueError(f"Analysis already exists: {analysis.key}")
        self._raise_if_exclusive_conflict(analysis)
        self.require_dependencies(analysis)
        self._analyses[analysis.key] = analysis
        self.set_dirty()

    def _raise_if_exclusive_conflict(self, analysis: BaseAnalysis) -> None:
        """Raise if another analysis in the same exclusive group is present.

        Args:
            analysis: Candidate analysis being added.

        Raises:
            AnalysisExclusionError: If another analysis with the same non-None
                ``exclusive_group`` already exists for the same
                ``(channel, roi_id)``.
        """
        group = analysis.exclusive_group
        if group is None:
            return
        channel = analysis.key.channel
        roi_id = analysis.key.roi_id
        analysis_name = analysis.key.analysis_name
        for existing in self._analyses.values():
            if existing.exclusive_group != group:
                continue
            if existing.key.channel != channel or existing.key.roi_id != roi_id:
                continue
            if existing.key.analysis_name == analysis_name:
                continue
            raise AnalysisExclusionError(
                f"Cannot add {analysis_name!r}: {existing.key.analysis_name!r} already "
                f"exists for channel={channel}, roi_id={roi_id} "
                f"(exclusive group {group!r})"
            )

    def get_primary_kymograph_analysis(
        self,
        *,
        channel: int,
        roi_id: int,
    ) -> BaseAnalysis | None:
        """Return the active ``primary_kymograph`` analysis for one selection.

        Args:
            channel: Channel index.
            roi_id: ROI identifier.

        Returns:
            The single analysis with ``exclusive_group == "primary_kymograph"``
            for the given ``(channel, roi_id)``, or None.
        """
        return self.get_exclusive_group_analysis(
            group="primary_kymograph",
            channel=channel,
            roi_id=roi_id,
        )

    def get_exclusive_group_analysis(
        self,
        *,
        group: str,
        channel: int,
        roi_id: int,
    ) -> BaseAnalysis | None:
        """Return the single analysis in an exclusive group for one selection.

        Args:
            group: Exclusive group name.
            channel: Channel index.
            roi_id: ROI identifier.

        Returns:
            Matching analysis, or None when no analysis is present.
        """
        for analysis in self._analyses.values():
            if analysis.exclusive_group != group:
                continue
            if analysis.key.channel != channel or analysis.key.roi_id != roi_id:
                continue
            return analysis
        return None

    def create(
        self,
        analysis_name: str,
        *,
        channel: int,
        roi_id: int,
        detection_params: dict[str, Any] | None = None,
    ) -> BaseAnalysis:
        """Create and add one analysis instance.

        Args:
            analysis_name: Registered analysis type name.
            channel: Channel index.
            roi_id: ROI identifier.
            detection_params: Optional detection parameter values.

        Returns:
            Newly created analysis instance.

        Raises:
            KeyError: If the analysis type is not registered.
            ValueError: If duplicate analysis identity already exists.
        """
        cls = get_analysis_class(analysis_name)
        analysis = cls(channel=channel, roi_id=roi_id, detection_params=detection_params)
        self.add(analysis)
        return analysis

    def get_or_create(
        self,
        analysis_name: str,
        *,
        channel: int,
        roi_id: int,
        detection_params: dict[str, Any] | None = None,
    ) -> BaseAnalysis:
        """Return an existing analysis or create it when missing.

        Args:
            analysis_name: Registered analysis type name.
            channel: Channel index.
            roi_id: ROI identifier.
            detection_params: Optional detection parameter values used only when
                creating a new analysis.

        Returns:
            Existing or newly created analysis.
        """
        key = AnalysisKey(
            analysis_name=analysis_name,
            channel=channel,
            roi_id=roi_id,
        )
        existing = self.get(key)
        if existing is not None:
            return existing
        return self.create(
            analysis_name,
            channel=channel,
            roi_id=roi_id,
            detection_params=detection_params,
        )

    @staticmethod
    def _resolve_analysis_name(analysis: str | type[BaseAnalysis]) -> str:
        """Resolve an analysis selector to its registered analysis name.

        Args:
            analysis: Registered analysis type name, or an analysis class whose
                ``analysis_name`` is registered.

        Returns:
            The registered analysis name.

        Raises:
            TypeError: If ``analysis`` is neither a string nor a ``BaseAnalysis``
                subclass.
        """
        if isinstance(analysis, type) and issubclass(analysis, BaseAnalysis):
            return analysis.analysis_name
        if isinstance(analysis, str):
            return analysis
        raise TypeError(
            f"analysis must be a str or BaseAnalysis subclass, "
            f"got: {type(analysis).__name__}"
        )

    def create_and_run(
        self,
        analysis: str | type[BaseAnalysis],
        *,
        channel: int,
        roi_id: int,
        detection_params: dict[str, Any] | None = None,
        replace_existing: bool = False,
        execution_options: dict[str, Any] | None = None,
        context: AnalysisRunContext | None = None,
    ) -> BaseAnalysis:
        """Create one analysis and run it in a single call.

        This is a scripting convenience over :meth:`create` and
        :meth:`run_analysis`. All inputs are validated before the analysis set
        is mutated, so a failure (for example invalid detection params, an
        unregistered analysis type, or a missing data provider) leaves the set
        unchanged.

        Args:
            analysis: Registered analysis type name, or an analysis class whose
                ``analysis_name`` is registered. Passing a class is a
                scripting convenience; the registered class for that name is
                always used to build the instance.
            channel: Channel index.
            roi_id: ROI identifier.
            detection_params: Optional detection parameter values. Missing
                values are filled from the analysis ``detection_schema``
                defaults.
            replace_existing: If True, remove any existing analysis with the
                same identity before creating the new one. If False, a
                duplicate identity raises ``ValueError``.
            execution_options: Optional runtime execution options forwarded to
                the analysis ``set_execution_options`` method (for example
                ``{"use_multiprocessing": False}`` for Radon velocity, which is
                recommended inside Jupyter). These are not detection parameters
                and are not serialized.
            context: Optional progress/cancellation context.

        Returns:
            The created analysis, with its result populated by the run.

        Raises:
            TypeError: If ``analysis`` is neither a string nor a
                ``BaseAnalysis`` subclass, or if ``execution_options`` is given
                for an analysis type that does not support execution options.
            RuntimeError: If no data provider was configured.
            KeyError: If the analysis type is not registered.
            ValueError: If a duplicate identity exists and ``replace_existing``
                is False, or if required dependencies are missing.
            AnalysisExclusionError: If an exclusive-group conflict exists.
        """
        analysis_name = self._resolve_analysis_name(analysis)

        if self._data_provider is None:
            raise RuntimeError("Cannot run analysis without a data provider")

        cls = get_analysis_class(analysis_name)
        candidate = cls(channel=channel, roi_id=roi_id, detection_params=detection_params)

        if execution_options:
            setter = getattr(candidate, "set_execution_options", None)
            if setter is None:
                raise TypeError(
                    f"{analysis_name!r} does not support execution options"
                )
            setter(**execution_options)

        if candidate.key in self._analyses:
            if not replace_existing:
                raise ValueError(f"Analysis already exists: {candidate.key}")
            self.remove(candidate.key)

        self.add(candidate)
        self.run_analysis(candidate.key, context=context)
        return candidate

    def get(self, key: AnalysisKey) -> BaseAnalysis | None:
        """Return analysis by key.

        Args:
            key: Analysis identity.

        Returns:
            Analysis instance, or None if missing.
        """
        return self._analyses.get(key)

    def get_required(self, key: AnalysisKey) -> BaseAnalysis:
        """Return analysis by key or raise.

        Args:
            key: Analysis identity.

        Returns:
            Analysis instance.

        Raises:
            KeyError: If no analysis exists for the key.
        """
        analysis = self.get(key)
        if analysis is None:
            raise KeyError(f"Analysis not found: {key}")
        return analysis

    def get_analysis(
        self,
        analysis: str | type[BaseAnalysis],
        *,
        channel: int,
        roi_id: int,
    ) -> BaseAnalysis:
        """Return an analysis by type, channel, and ROI, or raise.

        This is a scripting convenience over :meth:`get_required` that builds the
        :class:`AnalysisKey` for you, so callers do not need a previously created
        analysis instance to look one up (for example after reloading an
        ``AcqImage`` from disk).

        Args:
            analysis: Registered analysis type name, or an analysis class whose
                ``analysis_name`` is registered (for example
                ``RadonVelocityAnalysis``).
            channel: Channel index.
            roi_id: ROI identifier.

        Returns:
            The matching analysis instance.

        Raises:
            TypeError: If ``analysis`` is neither a string nor a ``BaseAnalysis``
                subclass.
            KeyError: If no analysis exists for the resolved identity.

        See Also:
            :meth:`find_analysis` for nullable lookup.
        """
        key = AnalysisKey(
            analysis_name=self._resolve_analysis_name(analysis),
            channel=channel,
            roi_id=roi_id,
        )
        return self.get_required(key)

    def find_analysis(
        self,
        analysis: str | type[BaseAnalysis],
        *,
        channel: int,
        roi_id: int,
    ) -> BaseAnalysis | None:
        """Return a matching analysis, or None when absent.

        Scripting convenience over :meth:`get` that builds the
        :class:`AnalysisKey` for you. Use :meth:`get_analysis` when absence
        should raise :class:`KeyError`.

        Args:
            analysis: Registered analysis type name, or an analysis class whose
                ``analysis_name`` is registered (for example
                ``RadonVelocityAnalysis``).
            channel: Channel index.
            roi_id: ROI identifier.

        Returns:
            The matching analysis instance, or None if no analysis exists for
            the resolved identity.

        Raises:
            TypeError: If ``analysis`` is neither a string nor a ``BaseAnalysis``
                subclass.
        """
        key = AnalysisKey(
            analysis_name=self._resolve_analysis_name(analysis),
            channel=channel,
            roi_id=roi_id,
        )
        return self.get(key)

    def remove(self, key: AnalysisKey) -> bool:
        """Remove one analysis by key.

        Args:
            key: Analysis identity to remove.

        Returns:
            True if an analysis was removed, False if no analysis existed for
            the key.
        """
        if key not in self._analyses:
            return False
        del self._analyses[key]
        self.set_dirty()
        return True

    def as_list(self) -> list[BaseAnalysis]:
        """Return analyses in insertion order.

        Returns:
            List of analyses.
        """
        return list(self._analyses.values())

    def require_dependencies(self, analysis: BaseAnalysis) -> dict[str, BaseAnalysis]:
        """Return dependency analyses required by one analysis.

        Args:
            analysis: Analysis whose dependencies should be resolved.

        Returns:
            Mapping from dependency analysis name to dependency instance.

        Raises:
            ValueError: If a required dependency is missing.
        """
        dependencies: dict[str, BaseAnalysis] = {}
        for dependency_name in analysis.depends_on:
            key = AnalysisKey(
                analysis_name=dependency_name,
                channel=analysis.key.channel,
                roi_id=analysis.key.roi_id,
            )
            dependency = self.get(key)
            if dependency is None:
                raise ValueError(
                    f"{analysis.key.analysis_name!r} requires {dependency_name!r} "
                    f"for channel={analysis.key.channel}, roi_id={analysis.key.roi_id}"
                )
            dependencies[dependency_name] = dependency
        return dependencies

    def run_analysis(
        self,
        key: AnalysisKey,
        *,
        context: AnalysisRunContext | None = None,
    ) -> AnalysisResult:
        """Run one analysis by key.

        Args:
            key: Analysis identity.
            context: Optional progress/cancellation context.

        Returns:
            Analysis result.

        Raises:
            RuntimeError: If no data provider was configured.
            KeyError: If the analysis does not exist.
            ValueError: If required dependencies are missing.
        """
        if self._data_provider is None:
            raise RuntimeError("Cannot run analysis without a data provider")

        analysis = self.get_required(key)
        dependencies = self.require_dependencies(analysis)
        result = analysis.run(
            self._data_provider,
            context=context,
            dependencies=dependencies,
        )
        self._results_csv_loaded = True
        self.set_dirty()
        return result

    def delete_roi(self, roi_id: int) -> int:
        """Delete analyses depending on one ROI.

        Args:
            roi_id: ROI identifier that was deleted.

        Returns:
            Number of analyses removed.
        """
        keys = [key for key in self._analyses if key.roi_id == roi_id]
        for key in keys:
            del self._analyses[key]
        if keys:
            self.set_dirty()
        return len(keys)

    def edit_roi(self, roi_id: int) -> int:
        """Delete analyses depending on one edited ROI.

        For v1, ROI edits invalidate dependent analyses and users can rerun
        analysis after the edit.

        Args:
            roi_id: ROI identifier that was edited.

        Returns:
            Number of analyses removed.
        """
        return self.delete_roi(roi_id)

    def serialize_json_analysis(self) -> list[dict[str, Any]]:
        """Return JSON-serializable records for all analyses.

        Returns:
            List of analysis records to store under the AcqImage sidecar JSON
            ``"analysis"`` key.
        """
        return [analysis.to_json_dict() for analysis in self._analyses.values()]

    def load_json_analysis(self, records: list[dict[str, Any]]) -> None:
        """Replace analyses from JSON records.

        Args:
            records: Analysis records loaded from the AcqImage sidecar JSON
                ``"analysis"`` key.

        Returns:
            None.

        Raises:
            KeyError: If any analysis class is not registered.
            ValueError: If duplicate records exist.
        """
        self._analyses.clear()
        self._results_csv_loaded = False

        for record in records:
            analysis_name = str(record["analysis_name"])
            channel = int(record["channel"])
            roi_id = int(record["roi_id"])
            cls = get_analysis_class(analysis_name)
            analysis = cls(channel=channel, roi_id=roi_id)
            analysis.load_json_dict(record)
            self.add(analysis)

        self.set_clean()

    def results_csv_loaded(self) -> bool:
        """Return whether required analysis CSV tables are loaded.

        Returns:
            True when there are no analyses, when lazy CSV loading has not been
            attempted yet (``_results_csv_loaded`` is false), or when every
            analysis whose sidecar CSV file exists on disk has a loaded result
            table. Analyses without a CSV sidecar (for example ``heart_rate``)
            do not block the loaded state.
        """
        if not self._analyses:
            return True
        if not self._results_csv_loaded:
            return False
        source = Path(self.source_path)
        for analysis in self._analyses.values():
            csv_path = self.analysis_csv_path(source, analysis.key.analysis_name)
            if not csv_path.exists():
                continue
            if analysis.result.table is None:
                return False
        return True

    def unload_results_dfs(self) -> None:
        """Drop loaded result DataFrames from every child analysis.

        JSON summaries, detection parameters, analysis identities, and dirty
        state are preserved. This supports CloudScope's lazy unload workflow
        without removing analysis rows from the file tree.
        """
        for analysis in self._analyses.values():
            analysis.result.table = None
        self._results_csv_loaded = False
        self.set_clean()

    def load_all_results_dfs_from_csv(self, source_path: str | Path) -> None:
        """Load CSV tables for all analyses with matching sidecar files.

        Args:
            source_path: Source acquisition file path.

        Returns:
            None.
        """
        source = Path(source_path)
        tables_by_name: dict[str, pd.DataFrame] = {}

        for analysis in self._analyses.values():
            analysis_name = analysis.key.analysis_name
            csv_path = self.analysis_csv_path(source, analysis_name)
            if csv_path.exists() and analysis_name not in tables_by_name:
                tables_by_name[analysis_name] = pd.read_csv(csv_path)

        self.unload_results_dfs()
        for analysis in self._analyses.values():
            table = tables_by_name.get(analysis.key.analysis_name)
            if table is None:
                continue

            if "channel" not in table.columns or "roi_id" not in table.columns:
                raise ValueError(
                    f"Analysis CSV for {analysis.key.analysis_name!r} is missing "
                    "required channel/roi_id columns"
                )

            mask = (
                (table["channel"] == analysis.key.channel)
                & (table["roi_id"] == analysis.key.roi_id)
            )
            sub = table.loc[mask].copy()
            if sub.empty:
                continue
            analysis.result.table = sub.drop(columns=["channel", "roi_id"])

        self._results_csv_loaded = True
        self.set_clean()


    def results_tables_by_name(self) -> dict[str, pd.DataFrame]:
        """Return combined result tables keyed by analysis name.

        Returns:
            Mapping from analysis type name to a DataFrame containing all table
            rows for that analysis type, with channel/ROI bookkeeping columns.

        Raises:
            ValueError: If one analysis type produces inconsistent table columns.
        """
        tables_by_name: dict[str, list[pd.DataFrame]] = {}
        for analysis in self._analyses.values():
            table = analysis.table_with_bookkeeping()
            if table is None:
                continue
            tables_by_name.setdefault(analysis.key.analysis_name, []).append(table)

        combined_by_name: dict[str, pd.DataFrame] = {}
        for analysis_name, tables in tables_by_name.items():
            self._validate_same_columns(analysis_name, tables)
            combined_by_name[analysis_name] = pd.concat(tables, ignore_index=True)
        return combined_by_name

    def save_results_tables_to_directory(self, directory: str | Path) -> None:
        """Save combined CSV tables under ``directory`` as ``<analysis_name>.csv``."""
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        combined_by_name = self.results_tables_by_name()
        existing_names = {path.stem for path in out_dir.glob("*.csv")}
        for analysis_name, combined in combined_by_name.items():
            combined.to_csv(out_dir / f"{analysis_name}.csv", index=False)
        for analysis_name in existing_names - set(combined_by_name.keys()):
            (out_dir / f"{analysis_name}.csv").unlink(missing_ok=True)

    def load_results_tables_by_name(self, tables_by_name: dict[str, pd.DataFrame]) -> None:
        """Hydrate child analysis result tables from combined tables by name."""
        self.unload_results_dfs()
        for analysis in self._analyses.values():
            table = tables_by_name.get(analysis.key.analysis_name)
            if table is None:
                continue
            if "channel" not in table.columns or "roi_id" not in table.columns:
                raise ValueError(
                    f"Analysis table for {analysis.key.analysis_name!r} is missing "
                    "required channel/roi_id columns"
                )
            mask = (
                (table["channel"] == analysis.key.channel)
                & (table["roi_id"] == analysis.key.roi_id)
            )
            sub = table.loc[mask].copy()
            if sub.empty:
                continue
            analysis.result.table = sub.drop(columns=["channel", "roi_id"])
        self._results_csv_loaded = True
        self.set_clean()

    def load_results_tables_from_directory(self, directory: str | Path) -> None:
        """Load combined CSV tables from ``directory`` into existing analyses."""
        src_dir = Path(directory)
        if not src_dir.is_dir():
            self.unload_results_dfs()
            return
        tables_by_name = {path.stem: pd.read_csv(path) for path in src_dir.glob("*.csv")}
        self.load_results_tables_by_name(tables_by_name)

    def save_results_df(self, source_path: str | Path) -> None:
        """Save combined CSV tables by analysis type.

        Args:
            source_path: Source acquisition file path.

        Returns:
            None.

        Raises:
            ValueError: If one analysis type produces inconsistent table columns.
        """
        if not self._results_csv_loaded and any(
            analysis.result.table is None for analysis in self._analyses.values()
        ):
            return

        source = Path(source_path)
        tables_by_name: dict[str, list[pd.DataFrame]] = {}

        for analysis in self._analyses.values():
            table = analysis.table_with_bookkeeping()
            if table is None:
                continue
            tables_by_name.setdefault(analysis.key.analysis_name, []).append(table)

        existing_names = {
            path.name.removeprefix(source.name + ".").removesuffix(".csv")
            for path in source.parent.glob(f"{source.name}.*.csv")
        }

        for analysis_name, tables in tables_by_name.items():
            self._validate_same_columns(analysis_name, tables)
            combined = pd.concat(tables, ignore_index=True)
            combined.to_csv(self.analysis_csv_path(source, analysis_name), index=False)

        for analysis_name in existing_names - set(tables_by_name.keys()):
            self.analysis_csv_path(source, analysis_name).unlink(missing_ok=True)

    @staticmethod
    def analysis_csv_path(source_path: str | Path, analysis_name: str) -> Path:
        """Return sidecar CSV path for one source file and analysis type.

        Args:
            source_path: Source acquisition file path.
            analysis_name: Analysis type name.

        Returns:
            Path such as ``myfile.tif.velocity.csv``.
        """
        source = Path(source_path)
        return source.with_name(f"{source.name}.{analysis_name}.csv")

    @staticmethod
    def _validate_same_columns(
        analysis_name: str,
        tables: list[pd.DataFrame],
    ) -> None:
        """Validate same table columns for one analysis type.

        Args:
            analysis_name: Analysis type name.
            tables: Tables for that analysis type.

        Raises:
            ValueError: If table columns differ.
        """
        if not tables:
            return

        expected = list(tables[0].columns)
        for table in tables[1:]:
            actual = list(table.columns)
            if actual != expected:
                raise ValueError(
                    f"Analysis {analysis_name!r} produced inconsistent table columns: "
                    f"expected {expected}, got {actual}"
                )
