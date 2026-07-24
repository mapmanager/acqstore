"""Collection loader for groups of acquisition files.

``AcqImageList`` is the public collection object for loading one file, a folder
of files, or a CSV-defined list of files. It keeps an ordered list of
``AcqImage`` objects and exposes display-ready schema/tree rows used by
CloudScope while remaining useful from scripts and notebooks.

The class supports cooperative progress reporting and cancellation for GUI load
and save workflows. Those same APIs are safe to use from Python code when a
caller wants structured warnings rather than exceptions for every failed file.
"""

from __future__ import annotations

import os
import csv
import logging
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from acqstore.schema import (
    ACQ_FILE_LIST_SCHEMA,
    SchemaDefinition,
    ValueType,
    validate_values_for_schema,
)

from .metadata import EXPERIMENT_METADATA_SCHEMA, ExperimentMetadata

from .supported_import_extensions import (
    get_allowed_import_extensions,
    path_has_allowed_import_extension,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from acqstore.acq_image.acq_image import AcqImage


def _build_file_list(path: str | Path, file_types: Sequence[str], *, folder_depth: int = 4) -> list[str]:
    """Build a list of files under ``path`` up to a bounded directory depth.

    Depth is 1-based from ``path``: depth ``1`` collects only files directly in
    ``path``; depth ``2`` also includes files in immediate child directories; and so
    on, up to ``folder_depth``.

    Args:
        path: Directory to traverse (must be a directory when called from
            :class:`AcqImageList`).
        file_types: Extensions to include (no leading dot).
        folder_depth: Maximum directory depth to visit (must be >= 1).

    Returns:
        Sorted list of absolute file paths.

    Raises:
        ValueError: If ``folder_depth`` is less than 1.
    """
    if folder_depth < 1:
        raise ValueError(f'folder_depth must be >= 1, got {folder_depth}')
    allowed_exts = {ext.lower().lstrip(".") for ext in file_types}
    result: list[str] = []
    root = Path(path).resolve()
    queue: deque[tuple[Path, int]] = deque()
    queue.append((root, 1))
    while queue:
        current, depth = queue.popleft()
        if not current.is_dir():
            continue
        try:
            entries = sorted(current.iterdir(), key=lambda p: p.name.lower())
        except OSError:
            continue
        for p in entries:
            if p.is_file():
                if path_has_allowed_import_extension(p) and _extension_is_allowed(p, allowed_exts):
                    result.append(str(p.resolve()))
            elif p.is_dir():
                if path_has_allowed_import_extension(p) and _extension_is_allowed(p, allowed_exts):
                    result.append(str(p.resolve()))
                    continue
                if depth < folder_depth:
                    queue.append((p, depth + 1))
    return sorted(result)


def _extension_is_allowed(path: str | Path, allowed_exts: set[str]) -> bool:
    """Return whether ``path`` has an allowed normalized import extension."""
    from .supported_import_extensions import normalize_import_extension_for_path

    return normalize_import_extension_for_path(path) in allowed_exts


class SaveEvent(StrEnum):
    """Save lifecycle event type for list-level save iteration."""

    SAVING = 'saving'
    SAVED = 'saved'
    CANCELLED = 'cancelled'


class LoadErrorType(StrEnum):
    """Structured non-fatal load warning categories."""

    MISSING_FILE = 'missing_file'
    UNSUPPORTED_FILE_TYPE = 'unsupported_file_type'
    LOADER_ERROR = 'loader_error'
    CSV_ERROR = 'csv_error'


class PathKind(StrEnum):
    """Supported input path kinds for file discovery/loading."""

    FILE = 'file'
    FOLDER = 'folder'
    CSV = 'csv'


@dataclass(frozen=True, slots=True)
class SaveProgress:
    """Progress event emitted while iterating list saves.

    Attributes:
        event: Save lifecycle event type.
        completed: Number of files saved so far.
        total: Number of dirty files planned for save.
        file_id: File identifier associated with this event.
    """

    event: SaveEvent
    completed: int
    total: int
    file_id: str | None


@dataclass(frozen=True, slots=True)
class LoadWarning:
    """Non-fatal warning collected during safe loading.

    Attributes:
        message: Human-readable warning message.
        path: User-facing path associated with the warning, when available.
        row_index: One-based CSV row index including the header row, when the
            warning came from a manifest row.
        error_type: Structured warning category for GUI and script reporting.
        rel_path: Manifest-relative path value associated with the warning.
        resolved_path: Absolute path resolved from ``rel_path`` and the
            manifest root, when available.
    """

    message: str
    path: str | None = None
    row_index: int | None = None
    error_type: LoadErrorType | None = None
    rel_path: str | None = None
    resolved_path: str | None = None


@dataclass(frozen=True, slots=True)
class LoadResult:
    """Result payload for safe load operations."""

    acq_image_list: AcqImageList
    warnings: tuple[LoadWarning, ...]
    discovered_count: int = 0


class LoadCancelled(RuntimeError):
    """Raised when a cooperative load operation is cancelled."""


class AcqImageList:
    """Ordered collection of loaded acquisition files.

    ``AcqImageList`` is the preferred entry point when a workflow operates on
    more than one acquisition. It can load a single file, discover supported
    files under a folder, or load file paths from a CSV. Files are stored in
    stable display order and can be accessed by file identifier, index, or
    iteration.

    The constructor is strict and raises when loading fails. Use
    :meth:`load_safe` for GUI-style workflows that should return partial results
    and non-fatal warnings instead of failing the entire load.

    Examples:
        Load a folder and iterate files::

            from acqstore.acq_image.acq_image_list import AcqImageList

            images = AcqImageList("/path/to/data")
            for acq in images:
                print(acq.name, acq.file_id)

        Safely load a folder and inspect warnings::

            result = AcqImageList.load_safe(
                "/path/to/data",
                kind="folder",
                folder_depth=4,
            )
            images = result.acq_image_list
            for warning in result.warnings:
                print(warning.message, warning.path)

    Args:
        path: File, folder, or CSV path.
        file_factory: Optional factory for creating ``AcqImage``-like objects.
        folder_depth: Maximum directory depth used for folder discovery.
        path_kind: Optional explicit ``PathKind``. When omitted, the kind is
            inferred from the path.
        load_images: Passed to default ``AcqImage`` construction. Ignored when a
            custom ``file_factory`` is supplied.
        load_analysis_csv: Passed to default ``AcqImage`` construction. Ignored
            when a custom ``file_factory`` is supplied.
        root_path: Optional manifest root for CSV loads. When omitted, CSV
            ``_rel_path`` values are resolved relative to the CSV parent.
    """

    def __init__(
        self,
        path: str,
        *,
        file_factory: Callable[[str], AcqImage] | None = None,
        folder_depth: int = 4,
        path_kind: PathKind | str | None = None,
        load_images: bool = True,
        load_analysis_csv: bool = True,
        root_path: str | Path | None = None,
    ):
        """Load one file, a folder of files, or a CSV file list.

        Args:
            path: Filesystem path to one file, directory, or CSV file.
            file_factory: Optional factory for creating file objects. Defaults to
                ``AcqImage`` and is mainly useful for tests.
            folder_depth: When ``path`` is a directory, maximum directory depth to
                search (>= 1). Depth ``1`` is only the given folder; each increment
                includes one more level of child directories. Ignored when ``path`` is
                a file.
            path_kind: Optional explicit source type (``file``, ``folder``, or
                ``csv``). When omitted, the constructor infers kind from path
                suffix and filesystem checks.
            load_images: When true, default ``AcqImage`` construction eagerly
                loads primary pixels.
            load_analysis_csv: When true, default ``AcqImage`` construction
                eagerly loads analysis CSV result tables.
            root_path: Optional manifest root for CSV loads. When omitted, CSV
                ``_rel_path`` values are resolved relative to the CSV parent.

        Raises:
            ValueError: If ``folder_depth`` is less than one or strict CSV
                parsing fails.
            Exception: Propagates file-loader exceptions from ``AcqImage`` when
                any discovered file cannot be loaded.
        """
        self.path = str(path)
        self.source_root_path: str | None = None
        if folder_depth < 1:
            raise ValueError(f'folder_depth must be >= 1, got {folder_depth}')

        detected_kind: PathKind | str | None = path_kind
        if detected_kind is None:
            path_obj = Path(path)
            if path_obj.suffix.lower() == '.csv':
                detected_kind = PathKind.CSV
            elif os.path.isdir(path):
                detected_kind = PathKind.FOLDER
            else:
                detected_kind = PathKind.FILE

        if detected_kind == PathKind.FOLDER:
            self.source_root_path = str(Path(path).expanduser().resolve(strict=False))
            self.file_list = _build_file_list(path, get_allowed_import_extensions(), folder_depth=folder_depth)
        elif detected_kind == PathKind.CSV:
            csv_root = Path(root_path).expanduser() if root_path is not None else Path(path).expanduser().parent
            self.source_root_path = str(csv_root.resolve(strict=False))
            self.file_list = self._build_file_list_from_csv(path, root_path=csv_root)
        else:
            file_path = Path(path).expanduser().resolve(strict=False)
            self.source_root_path = str(file_path.parent)
            self.file_list = [str(file_path)]

        if file_factory is None:
            from acqstore.acq_image.acq_image import AcqImage

            def file_factory(file_path: str) -> AcqImage:
                return AcqImage(
                    file_path,
                    load_images=load_images,
                    load_analysis_csv=load_analysis_csv,
                )
        self._files = [file_factory(file_path) for file_path in self.file_list]
        self._files_by_id = {acq_file.file_id: acq_file for acq_file in self._files}
        self._attach_analysis_pools()

    @classmethod
    def load_safe(
        cls,
        path: str,
        *,
        kind: PathKind | str,
        file_factory: Callable[[str], AcqImage] | None = None,
        folder_depth: int = 4,
        progress_callback: Callable[[int, int, str], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
        load_images: bool = True,
        load_analysis_csv: bool = True,
        root_path: str | Path | None = None,
    ) -> LoadResult:
        """Load acquisition files while collecting non-fatal warnings.

        This is the preferred loading API for GUI, notebook, and batch workflows
        that should keep usable files even when one file fails. Missing files,
        bad CSV rows, and individual loader errors are returned as
        :class:`LoadWarning` records. Cancellation is cooperative and checked
        between file loads.

        Args:
            path: Input path supplied by the user or caller.
            kind: Explicit source kind (``file``, ``folder``, or ``csv``).
            file_factory: Optional file-construction callback for tests or
                dependency injection.
            folder_depth: Maximum folder traversal depth for folder loads.
            progress_callback: Optional callback called as
                ``progress_callback(completed, total, message)`` after discovery
                and after each attempted file load.
            should_cancel: Optional callback checked between file loads. Return
                ``True`` to cancel loading.
            load_images: When true, default ``AcqImage`` construction eagerly
                loads primary pixels.
            load_analysis_csv: When true, default ``AcqImage`` construction
                eagerly loads analysis CSV result tables.
            root_path: Optional manifest root used for CSV loads. When omitted,
                CSV ``_rel_path`` values are resolved relative to the CSV parent.

        Returns:
            :class:`LoadResult` containing an ``AcqImageList`` and collected
            warnings. The list may be empty.

        Raises:
            LoadCancelled: If ``should_cancel`` requests cancellation.
        """
        warnings: list[LoadWarning] = []
        path_obj = Path(path).expanduser()
        base_path = str(path_obj.resolve(strict=False))
        if isinstance(kind, str):
            try:
                kind = PathKind(kind)
            except ValueError:
                warnings.append(LoadWarning(message=f'Unsupported load kind: {kind}', path=base_path, error_type=LoadErrorType.CSV_ERROR))
                obj = cls.__new__(cls)
                obj.path = base_path
                obj.source_root_path = None
                obj.file_list = []
                obj._files = []
                obj._files_by_id = {}
                obj._attach_analysis_pools()
                return LoadResult(acq_image_list=obj, warnings=tuple(warnings), discovered_count=0)

        candidate_paths: list[str] = []
        if kind == PathKind.FOLDER:
            if not path_obj.exists() or not path_obj.is_dir():
                warnings.append(LoadWarning(message='Folder does not exist or is not a directory', path=base_path, error_type=LoadErrorType.MISSING_FILE))
            else:
                candidate_paths = _build_file_list(path_obj, get_allowed_import_extensions(), folder_depth=folder_depth)
        elif kind == PathKind.FILE:
            if not path_obj.exists() or not (path_obj.is_file() or path_has_allowed_import_extension(path_obj)):
                warnings.append(LoadWarning(message='File does not exist or is not a supported file/store', path=base_path, error_type=LoadErrorType.MISSING_FILE))
            else:
                candidate_paths = [str(path_obj.resolve())]
        elif kind == PathKind.CSV:
            csv_paths, csv_warnings = cls._build_file_list_from_csv_safe(path_obj, root_path=root_path)
            candidate_paths = csv_paths
            warnings.extend(csv_warnings)
        else:
            warnings.append(LoadWarning(message=f'Unsupported load kind: {kind}', path=base_path, error_type=LoadErrorType.CSV_ERROR))

        files: list[AcqImage] = []
        total = len(candidate_paths)
        if progress_callback is not None:
            progress_callback(0, total, f'Discovered {total} file(s)')
        for candidate in candidate_paths:
            if should_cancel is not None and should_cancel():
                raise LoadCancelled('Load cancelled')
            try:
                if file_factory is None:
                    from acqstore.acq_image.acq_image import AcqImage

                    built = AcqImage(
                        candidate,
                        load_images=load_images,
                        load_analysis_csv=load_analysis_csv,
                    )
                else:
                    built = file_factory(candidate)
                files.append(built)
            except Exception as exc:
                resolved_candidate = str(Path(candidate).resolve(strict=False))
                message = f'Failed to load file: {exc}'
                logger.error('%s: %s', message, resolved_candidate)
                warnings.append(LoadWarning(message=message, path=resolved_candidate, error_type=LoadErrorType.LOADER_ERROR, resolved_path=resolved_candidate))
            if progress_callback is not None:
                progress_callback(len(files), total, f'Loaded {len(files)}/{total}')

        obj = cls.__new__(cls)
        obj.path = base_path
        if kind == PathKind.FOLDER:
            obj.source_root_path = str(path_obj.resolve(strict=False))
        elif kind == PathKind.CSV:
            csv_root = Path(root_path).expanduser() if root_path is not None else path_obj.parent
            obj.source_root_path = str(csv_root.resolve(strict=False))
        elif kind == PathKind.FILE:
            obj.source_root_path = str(path_obj.resolve(strict=False).parent)
        else:
            obj.source_root_path = None
        obj.file_list = [str(Path(file.path).resolve(strict=False)) if hasattr(file, 'path') else file.file_id for file in files]
        obj._files = files
        obj._files_by_id = {acq_file.file_id: acq_file for acq_file in files}
        obj._attach_analysis_pools()
        return LoadResult(acq_image_list=obj, warnings=tuple(warnings), discovered_count=total)

    @classmethod
    def from_manifest_csv(
        cls,
        csv_path: str | Path,
        *,
        root_path: str | Path | None = None,
        file_factory: Callable[[str], AcqImage] | None = None,
        load_images: bool = True,
        load_analysis_csv: bool = True,
    ) -> LoadResult:
        """Safely load an acquisition list from a manifest CSV.

        Args:
            csv_path: CSV file containing a required ``_rel_path`` column.
            root_path: Optional manifest root. When omitted, paths are resolved
                relative to the CSV parent directory.
            file_factory: Optional file-construction callback for tests or
                dependency injection.
            load_images: When true, default ``AcqImage`` construction eagerly
                loads primary pixels.
            load_analysis_csv: When true, default ``AcqImage`` construction
                eagerly loads analysis CSV result tables.

        Returns:
            Structured load result containing the loaded list and warnings.

        Raises:
            LoadCancelled: Never raised by this wrapper because no cancellation
                callback is accepted.
        """
        return cls.load_safe(
            str(csv_path),
            kind=PathKind.CSV,
            file_factory=file_factory,
            load_images=load_images,
            load_analysis_csv=load_analysis_csv,
            root_path=root_path,
        )

    @staticmethod
    def _build_file_list_from_csv_safe(
        csv_path: Path,
        *,
        root_path: str | Path | None = None,
    ) -> tuple[list[str], list[LoadWarning]]:
        """Parse CSV ``_rel_path`` rows with per-row warnings.

        Args:
            csv_path: Manifest CSV path.
            root_path: Optional root directory for resolving ``_rel_path``.

        Returns:
            Tuple of candidate absolute paths and non-fatal warnings.
        """
        warnings: list[LoadWarning] = []
        result: list[str] = []
        csv_resolved = csv_path.expanduser().resolve(strict=False)
        if not csv_path.exists() or not csv_path.is_file():
            warning = LoadWarning(
                message='CSV file does not exist',
                path=str(csv_resolved),
                error_type=LoadErrorType.MISSING_FILE,
            )
            logger.error('%s: %s', warning.message, warning.path)
            return ([], [warning])

        manifest_root = (
            Path(root_path).expanduser().resolve(strict=False)
            if root_path is not None
            else csv_path.parent.resolve(strict=False)
        )
        if root_path is not None and (not manifest_root.exists() or not manifest_root.is_dir()):
            warning = LoadWarning(
                message='Manifest root_path does not exist or is not a directory',
                path=str(manifest_root),
                error_type=LoadErrorType.MISSING_FILE,
            )
            logger.error('%s: %s', warning.message, warning.path)
            return ([], [warning])

        try:
            with csv_path.open('r', encoding='utf-8', newline='') as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None or '_rel_path' not in reader.fieldnames:
                    warnings.append(
                        LoadWarning(
                            message='CSV is missing required column "_rel_path"',
                            path=str(csv_resolved),
                            error_type=LoadErrorType.CSV_ERROR,
                        )
                    )
                    return ([], warnings)
                seen_rel_paths: set[str] = set()
                for index, row in enumerate(reader, start=2):
                    raw_value = row.get('_rel_path')
                    if raw_value is None or not str(raw_value).strip():
                        warnings.append(
                            LoadWarning(
                                message='CSV row has blank _rel_path',
                                path=str(csv_resolved),
                                row_index=index,
                                error_type=LoadErrorType.CSV_ERROR,
                            )
                        )
                        continue
                    rel_value = str(raw_value).strip()
                    rel_path = Path(rel_value)
                    if rel_path.is_absolute():
                        warnings.append(
                            LoadWarning(
                                message='CSV _rel_path must be relative',
                                path=str(csv_resolved),
                                row_index=index,
                                error_type=LoadErrorType.CSV_ERROR,
                                rel_path=rel_value,
                            )
                        )
                        continue
                    if rel_value in seen_rel_paths:
                        warnings.append(
                            LoadWarning(
                                message='CSV contains duplicate _rel_path',
                                path=str(csv_resolved),
                                row_index=index,
                                error_type=LoadErrorType.CSV_ERROR,
                                rel_path=rel_value,
                            )
                        )
                        continue
                    seen_rel_paths.add(rel_value)
                    candidate = (manifest_root / rel_path).resolve(strict=False)
                    try:
                        candidate.relative_to(manifest_root)
                    except ValueError:
                        warnings.append(
                            LoadWarning(
                                message='CSV _rel_path escapes manifest root',
                                path=str(csv_resolved),
                                row_index=index,
                                error_type=LoadErrorType.CSV_ERROR,
                                rel_path=rel_value,
                                resolved_path=str(candidate),
                            )
                        )
                        continue
                    if not candidate.exists() or not (candidate.is_file() or path_has_allowed_import_extension(candidate)):
                        warning = LoadWarning(
                            message='CSV _rel_path target does not exist',
                            path=str(candidate),
                            row_index=index,
                            error_type=LoadErrorType.MISSING_FILE,
                            rel_path=rel_value,
                            resolved_path=str(candidate),
                        )
                        logger.error('%s: %s', warning.message, warning.path)
                        warnings.append(warning)
                        continue
                    if not path_has_allowed_import_extension(candidate):
                        warning = LoadWarning(
                            message='CSV _rel_path target is not a supported file/store',
                            path=str(candidate),
                            row_index=index,
                            error_type=LoadErrorType.UNSUPPORTED_FILE_TYPE,
                            rel_path=rel_value,
                            resolved_path=str(candidate),
                        )
                        logger.error('%s: %s', warning.message, warning.path)
                        warnings.append(warning)
                        continue
                    result.append(str(candidate))
        except Exception as exc:
            warnings.append(
                LoadWarning(
                    message=f'Failed to parse CSV: {exc}',
                    path=str(csv_resolved),
                    error_type=LoadErrorType.CSV_ERROR,
                )
            )
            return ([], warnings)

        return (result, warnings)

    def _build_file_list_from_csv(
        self,
        path: str | Path,
        *,
        root_path: str | Path | None = None,
    ) -> list[str]:
        """Strict CSV parser used by constructor path-kind csv mode.

        Args:
            path: Manifest CSV path.
            root_path: Optional root directory for resolving ``_rel_path``.

        Returns:
            Absolute candidate file paths.

        Raises:
            ValueError: If manifest parsing produces any warning.
        """
        paths, warnings = self._build_file_list_from_csv_safe(Path(path), root_path=root_path)
        if warnings:
            first = warnings[0]
            raise ValueError(first.message)
        return paths

    def to_manifest_csv(
        self,
        csv_path: str | Path,
        *,
        root_path: str | Path | None = None,
    ) -> Path:
        """Write all files in this list to a manifest CSV.

        Args:
            csv_path: Destination CSV path.
            root_path: Optional root used to compute ``_rel_path`` values.

        Returns:
            Resolved destination path.

        Raises:
            ValueError: If a file path cannot be represented relative to the
                effective root.
            OSError: If writing fails.
        """
        from acqstore.acq_image.acq_image_manifest import AcqImageListManifest

        return AcqImageListManifest(self).write_manifest_csv(csv_path, root_path=root_path)

    def to_randomized_manifest_master_csv(
        self,
        csv_path: str | Path,
        *,
        groupby_column: str,
        random_seed: int | None = None,
        root_path: str | Path | None = None,
    ) -> Path:
        """Write the full deterministic randomized manifest for this list.

        Args:
            csv_path: Destination CSV path.
            groupby_column: Schema row column used to define groups.
            random_seed: Optional seed for deterministic shuffling.
            root_path: Optional root used to compute ``_rel_path`` values.

        Returns:
            Resolved destination path.

        Raises:
            KeyError: If ``groupby_column`` is unknown.
            ValueError: If grouping or relative-path validation fails.
            OSError: If writing fails.
        """
        from acqstore.acq_image.acq_image_manifest import AcqImageListManifest

        return AcqImageListManifest(self).write_randomized_manifest_master_csv(
            csv_path,
            groupby_column=groupby_column,
            random_seed=random_seed,
            root_path=root_path,
        )

    def to_randomized_manifest_csv(
        self,
        csv_path: str | Path,
        *,
        master_csv_path: str | Path,
        n_per_group: int,
        allow_unbalanced: bool = False,
    ) -> Path:
        """Write a sampled randomized manifest from a master CSV.

        Args:
            csv_path: Destination CSV path.
            master_csv_path: Source randomized master CSV path.
            n_per_group: Number of rows to keep from each randomized group.
            allow_unbalanced: When false, every group must have at least
                ``n_per_group`` files.

        Returns:
            Resolved destination path.

        Raises:
            ValueError: If sampling validation fails.
            OSError: If the master CSV cannot be read or the output file cannot
                be written.
        """
        from acqstore.acq_image.acq_image_manifest import AcqImageListManifest

        return AcqImageListManifest(self).write_randomized_manifest_csv(
            csv_path,
            master_csv_path=master_csv_path,
            n_per_group=n_per_group,
            allow_unbalanced=allow_unbalanced,
        )

    def _attach_analysis_pools(self) -> None:
        """Create collection-level analysis pools owned by this list."""
        from acqstore.analysis_pool.sum_intensity_analysis_pool import (
            SumIntensityAnalysisPool,
        )
        from acqstore.analysis_pool.velocity_analysis_pool import VelocityAnalysisPool

        self.velocity_analysis_pool = VelocityAnalysisPool(self)
        self.sum_intensity_analysis_pool = SumIntensityAnalysisPool(self)

    def __len__(self) -> int:
        """Return number of files in the collection."""
        return len(self._files)

    def __iter__(self) -> Iterator[AcqImage]:
        """Iterate files in stable display order."""
        return iter(self._files)

    def get_files(self) -> Sequence[AcqImage]:
        """Return files in stable display order."""
        return tuple(self._files)

    def get_file_by_id(self, file_id: str) -> AcqImage | None:
        """Return one file by stable identifier.

        Args:
            file_id: Stable file identifier.

        Returns:
            Matching file object, or ``None`` when not found.
        """
        return self._files_by_id.get(file_id)

    def get_file_by_index(self, index: int) -> AcqImage:
        """Return one file by stable display index.

        Args:
            index: Zero-based display index.

        Returns:
            Matching file object.

        Raises:
            IndexError: If the display index is out of range.
        """
        return self._files[index]

    def has_file_id(self, file_id: str) -> bool:
        """Return whether the file identifier exists in the collection."""
        return file_id in self._files_by_id

    def get_default_file_id(self) -> str | None:
        """Return the default file identifier in stable display order.

        Returns first file in list."""
        if not self._files:
            return None
        return self._files[0].file_id

    def get_default_selection(self) -> tuple[str | None, int | None, int | None]:
        """Return default primary selection for initial app state.

        Returns:
            Tuple of (file_id, channel, roi) using backend-native values.
            Any tuple member may be None when no explicit default exists.
        """
        default_file_id = self.get_default_file_id()
        if default_file_id is None:
            return (None, None, None)
        acq_file = self._files_by_id[default_file_id]
        return (
            default_file_id,
            acq_file.get_default_channel(),
            acq_file.get_default_roi(),
        )

    def get_dirty_files(self) -> Sequence[AcqImage]:
        """Return dirty files in stable display order."""
        return tuple(acq_file for acq_file in self._files if acq_file.is_dirty)

    def has_dirty_files(self) -> bool:
        """Return whether any file in the collection is dirty."""
        return any(acq_file.is_dirty for acq_file in self._files)

    def iter_save_all(
        self,
        *,
        should_cancel: Callable[[], bool] | None = None,
    ) -> Iterator[SaveProgress]:
        """Persist dirty files while yielding progress events.

        Args:
            should_cancel: Optional callback checked between file saves. Return
                True to stop iteration early.

        Yields:
            Save progress events for save start/finish and cancellation.
        """
        dirty_files = list(self.get_dirty_files())
        total = len(dirty_files)
        completed = 0

        for acq_file in dirty_files:
            if should_cancel is not None and should_cancel():
                yield SaveProgress(
                    event=SaveEvent.CANCELLED,
                    completed=completed,
                    total=total,
                    file_id=acq_file.file_id,
                )
                return

            yield SaveProgress(
                event=SaveEvent.SAVING,
                completed=completed,
                total=total,
                file_id=acq_file.file_id,
            )
            acq_file.save()
            completed += 1
            yield SaveProgress(
                event=SaveEvent.SAVED,
                completed=completed,
                total=total,
                file_id=acq_file.file_id,
            )

    def save_all(self, *, should_cancel: Callable[[], bool] | None = None) -> None:
        """Persist all dirty files in the collection.

        Args:
            should_cancel: Optional callback checked between file saves.
        """
        for _event in self.iter_save_all(should_cancel=should_cancel):
            continue

    def load_lazy_data(
        self,
        *,
        load_images: bool = True,
        load_analysis_csv: bool = True,
    ) -> None:
        """Load selected lazy data categories for every acquisition in the list.

        Args:
            load_images: Load primary image pixels when true.
            load_analysis_csv: Load analysis CSV result tables when true.
        """
        for acq_file in self._files:
            acq_file.load_lazy_data(
                load_images=load_images,
                load_analysis_csv=load_analysis_csv,
            )

    def unload_lazy_data(
        self,
        *,
        unload_images: bool = True,
        unload_analysis_csv: bool = True,
    ) -> None:
        """Unload selected lazy data categories for every acquisition in the list.

        Args:
            unload_images: Unload primary image pixels when true.
            unload_analysis_csv: Unload analysis CSV result tables when true.
        """
        for acq_file in self._files:
            acq_file.unload_lazy_data(
                unload_images=unload_images,
                unload_analysis_csv=unload_analysis_csv,
            )

    def get_unique_metadata_values(self, field_name: str) -> list[str]:
        """Return sorted unique non-empty values for one experiment metadata field.

        Values are collected from every loaded :class:`AcqImage` in this list.
        Only string fields declared by ``EXPERIMENT_METADATA_SCHEMA`` are
        supported.

        Args:
            field_name: Experiment metadata schema field name (e.g. ``species``).

        Returns:
            Sorted list of unique non-empty string values.

        Raises:
            ValueError: If ``field_name`` is unknown or not a string field.
        """
        fields_by_name = {field.name: field for field in EXPERIMENT_METADATA_SCHEMA.fields}
        if field_name not in fields_by_name:
            raise ValueError(f'Unknown experiment_metadata field: {field_name!r}')
        field_schema = fields_by_name[field_name]
        if field_schema.value_type is not ValueType.STR:
            raise ValueError(
                f'Field {field_name!r} is not a string field; '
                f'got value_type={field_schema.value_type!r}'
            )

        values: set[str] = set()
        section_id = ExperimentMetadata.metadata_section_id
        for acq_file in self._files:
            section = acq_file.get_metadata_section(section_id)
            raw = section.get_values().get(field_name)
            if raw is None:
                continue
            text = str(raw).strip()
            if text:
                values.add(text)
        return sorted(values)

    def get_schema(self) -> SchemaDefinition:
        """Return schema definition for rows in this list."""
        return ACQ_FILE_LIST_SCHEMA

    def get_schema_rows(self) -> list[dict[str, object]]:
        """Return schema-keyed rows for all files in stable display order.

        Returns:
            List of row dictionaries keyed by schema field name.

        Raises:
            KeyError: If a required schema field is missing from any row.
            ValueError: If any row has keys not declared by the schema.
        """
        schema = self.get_schema()
        rows = [acq_file.get_schema_row() for acq_file in self.get_files()]
        for row in rows:
            validate_values_for_schema(schema, row)
        return rows

    def get_tree_rows(self) -> list[dict[str, object]]:
        """Return tree rows for all files in stable display order.

        Returns:
            Flat row list containing each file row followed by its analysis
            child rows.
        """
        rows: list[dict[str, object]] = []
        for acq_file in self.get_files():
            rows.extend(acq_file.get_tree_rows())
        return rows
