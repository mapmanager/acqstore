"""Manifest CSV helpers for acquisition file lists.

This backend module owns CSV manifest writing and deterministic randomized
sampling for :class:`acqstore.acq_image.acq_image_list.AcqImageList`. It keeps
manifest-specific path, grouping, and randomization rules out of the collection
class while preserving thin scripting wrappers on ``AcqImageList``.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from acqstore.schema import ACQ_FILE_LIST_SCHEMA, ValueType

MANIFEST_REL_PATH_COLUMN = '_rel_path'
MANIFEST_GROUP_COLUMN = '_group'
MANIFEST_RANDOM_ORDER_COLUMN = '_random_order'
MANIFEST_SOURCE_INDEX_COLUMN = '_source_index'

_CATEGORICAL_VALUE_TYPES = {ValueType.STR, ValueType.BOOL, ValueType.ENUM}


@dataclass(frozen=True, slots=True)
class ManifestRow:
    """One row prepared for manifest CSV writing.

    Attributes:
        rel_path: Relative path from the selected manifest root to the raw data
            file.
        group: Optional group value used for randomized manifests.
        random_order: Optional zero-based order after deterministic shuffling
            within one group.
        source_index: Optional zero-based index in the source ``AcqImageList``.
    """

    rel_path: str
    group: str | None = None
    random_order: int | None = None
    source_index: int | None = None


class AcqImageListManifest:
    """Build manifest CSV files from an acquisition image list.

    The helper is intentionally small and stateless. Public scripts should
    usually call the thin ``AcqImageList`` wrapper methods rather than construct
    this class directly.
    """

    def __init__(self, acq_image_list: object) -> None:
        """Create a manifest helper for one acquisition list.

        Args:
            acq_image_list: Object exposing ``get_schema_rows()`` and optional
                ``source_root_path`` attributes compatible with ``AcqImageList``.
        """
        self._acq_image_list = acq_image_list

    def write_manifest_csv(
        self,
        csv_path: str | Path,
        *,
        root_path: str | Path | None = None,
    ) -> Path:
        """Write a manifest CSV containing all files in list order.

        Args:
            csv_path: Destination CSV path.
            root_path: Optional root used to compute ``_rel_path`` values. When
                omitted, the list source root is used, falling back to the CSV
                parent directory.

        Returns:
            Resolved destination path.

        Raises:
            ValueError: If any source file cannot be represented relative to the
                selected root.
            OSError: If the output file cannot be written.
        """
        destination = Path(csv_path).expanduser().resolve(strict=False)
        root = self._resolve_output_root(destination, root_path)
        rows = [ManifestRow(rel_path=rel_path) for rel_path in self._relative_paths(root)]
        self._write_rows(destination, rows, include_random_columns=False)
        return destination

    def write_randomized_manifest_master_csv(
        self,
        csv_path: str | Path,
        *,
        groupby_column: str,
        random_seed: int | None = None,
        root_path: str | Path | None = None,
    ) -> Path:
        """Write all files in deterministic randomized order within each group.

        Args:
            csv_path: Destination CSV path.
            groupby_column: Schema row column used to define groups.
            random_seed: Optional seed for deterministic shuffling.
            root_path: Optional root used to compute ``_rel_path`` values.

        Returns:
            Resolved destination path.

        Raises:
            KeyError: If ``groupby_column`` is not in the list schema.
            ValueError: If the grouping column is not categorical-like, contains
                empty values, or any file cannot be made relative to the root.
            OSError: If the output file cannot be written.
        """
        destination = Path(csv_path).expanduser().resolve(strict=False)
        root = self._resolve_output_root(destination, root_path)
        rows = self._randomized_rows(
            groupby_column=groupby_column,
            random_seed=random_seed,
            root_path=root,
        )
        self._write_rows(destination, rows, include_random_columns=True)
        return destination

    def write_randomized_manifest_csv(
        self,
        csv_path: str | Path,
        *,
        master_csv_path: str | Path,
        n_per_group: int,
        allow_unbalanced: bool = False,
    ) -> Path:
        """Write a sampled randomized manifest CSV from a master manifest.

        Args:
            csv_path: Destination CSV path.
            master_csv_path: Source randomized master CSV path.
            n_per_group: Number of rows to keep from each randomized group.
            allow_unbalanced: When false, every group must contain at least
                ``n_per_group`` files. When true, smaller groups contribute all
                available files.

        Returns:
            Resolved destination path.

        Raises:
            ValueError: If ``n_per_group`` is less than one, the master CSV is
                not a randomized manifest, or a group is too small while
                ``allow_unbalanced`` is false.
            OSError: If the master CSV cannot be read or the output file cannot
                be written.
        """
        if n_per_group < 1:
            raise ValueError(f'n_per_group must be >= 1, got {n_per_group}')
        destination = Path(csv_path).expanduser().resolve(strict=False)
        rows = self._read_randomized_master_rows(master_csv_path)
        groups: dict[str, list[ManifestRow]] = defaultdict(list)
        for row in rows:
            assert row.group is not None
            groups[row.group].append(row)
        if not allow_unbalanced:
            too_small = sorted(group for group, group_rows in groups.items() if len(group_rows) < n_per_group)
            if too_small:
                raise ValueError(
                    f'Group(s) have fewer than n_per_group={n_per_group} rows: {too_small}'
                )
        sampled: list[ManifestRow] = []
        for group in sorted(groups):
            sampled.extend(groups[group][:n_per_group])
        self._write_rows(destination, sampled, include_random_columns=True)
        return destination

    def _resolve_output_root(self, csv_path: Path, root_path: str | Path | None) -> Path:
        """Return the effective root used for output relative paths."""
        if root_path is not None:
            return Path(root_path).expanduser().resolve(strict=False)
        source_root_path = getattr(self._acq_image_list, 'source_root_path', None)
        if source_root_path:
            return Path(str(source_root_path)).expanduser().resolve(strict=False)
        return csv_path.parent.resolve(strict=False)

    def _relative_paths(self, root_path: Path) -> list[str]:
        """Return source file paths relative to ``root_path``."""
        rel_paths: list[str] = []
        for row in self._schema_rows():
            source_path = Path(str(row['path'])).expanduser().resolve(strict=False)
            try:
                rel_path = source_path.relative_to(root_path)
            except ValueError as exc:
                raise ValueError(
                    f'File path is not under manifest root: path={source_path}, root={root_path}'
                ) from exc
            rel_paths.append(rel_path.as_posix())
        return rel_paths

    def _schema_rows(self) -> list[dict[str, object]]:
        """Return schema rows from the wrapped list."""
        get_schema_rows = getattr(self._acq_image_list, 'get_schema_rows')
        rows = get_schema_rows()
        return list(rows)

    def _randomized_rows(
        self,
        *,
        groupby_column: str,
        random_seed: int | None,
        root_path: Path,
    ) -> list[ManifestRow]:
        """Return manifest rows randomized within each group."""
        self._validate_groupby_column(groupby_column)
        rows = self._schema_rows()
        rel_paths = self._relative_paths(root_path)
        groups: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for index, row in enumerate(rows):
            raw_group = row.get(groupby_column)
            group = '' if raw_group is None else str(raw_group).strip()
            if not group:
                raise ValueError(f'Grouping column {groupby_column!r} has an empty value at source index {index}')
            groups[group].append((index, rel_paths[index]))
        rng = random.Random(random_seed)
        output: list[ManifestRow] = []
        for group in sorted(groups):
            items = list(groups[group])
            rng.shuffle(items)
            for random_order, (source_index, rel_path) in enumerate(items):
                output.append(
                    ManifestRow(
                        rel_path=rel_path,
                        group=group,
                        random_order=random_order,
                        source_index=source_index,
                    )
                )
        return output

    def _validate_groupby_column(self, groupby_column: str) -> None:
        """Validate that a schema column is suitable for grouping."""
        fields = {field.name: field for field in ACQ_FILE_LIST_SCHEMA.fields}
        if groupby_column not in fields:
            raise KeyError(f'Unknown AcqImageList schema column: {groupby_column!r}')
        field = fields[groupby_column]
        if field.value_type not in _CATEGORICAL_VALUE_TYPES:
            raise ValueError(
                f'Column {groupby_column!r} is not categorical-like; '
                f'got value_type={field.value_type.value!r}'
            )

    @staticmethod
    def _read_randomized_master_rows(master_csv_path: str | Path) -> list[ManifestRow]:
        """Read rows from a randomized master manifest CSV."""
        required_columns = {
            MANIFEST_REL_PATH_COLUMN,
            MANIFEST_GROUP_COLUMN,
            MANIFEST_RANDOM_ORDER_COLUMN,
            MANIFEST_SOURCE_INDEX_COLUMN,
        }
        source = Path(master_csv_path).expanduser().resolve(strict=False)
        with source.open('r', encoding='utf-8', newline='') as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])
            missing = sorted(required_columns - fieldnames)
            if missing:
                raise ValueError(f'Randomized master CSV is missing required column(s): {missing}')
            rows: list[ManifestRow] = []
            for index, row in enumerate(reader, start=2):
                rel_path = AcqImageListManifest._csv_cell(row, MANIFEST_REL_PATH_COLUMN)
                group = AcqImageListManifest._csv_cell(row, MANIFEST_GROUP_COLUMN)
                random_order = AcqImageListManifest._parse_int_column(
                    row,
                    MANIFEST_RANDOM_ORDER_COLUMN,
                    row_index=index,
                )
                source_index = AcqImageListManifest._parse_int_column(
                    row,
                    MANIFEST_SOURCE_INDEX_COLUMN,
                    row_index=index,
                )
                if not rel_path:
                    raise ValueError(f'Randomized master CSV row {index} has blank {MANIFEST_REL_PATH_COLUMN}')
                if not group:
                    raise ValueError(f'Randomized master CSV row {index} has blank {MANIFEST_GROUP_COLUMN}')
                rows.append(
                    ManifestRow(
                        rel_path=rel_path,
                        group=group,
                        random_order=random_order,
                        source_index=source_index,
                    )
                )
        return rows

    @staticmethod
    def _csv_cell(row: dict[str, str | None], column: str) -> str:
        """Return a stripped CSV cell value, treating missing cells as blank."""
        value = row.get(column)
        return '' if value is None else str(value).strip()

    @staticmethod
    def _parse_int_column(row: dict[str, str | None], column: str, *, row_index: int) -> int:
        """Return a required integer column from a CSV row."""
        value = AcqImageListManifest._csv_cell(row, column)
        if not value:
            raise ValueError(f'Randomized master CSV row {row_index} has blank {column}')
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(
                f'Randomized master CSV row {row_index} has non-integer {column}: {value!r}'
            ) from exc

    @staticmethod
    def _write_rows(csv_path: Path, rows: Sequence[ManifestRow], *, include_random_columns: bool) -> None:
        """Write manifest rows to disk."""
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [MANIFEST_REL_PATH_COLUMN]
        if include_random_columns:
            fieldnames.extend([
                MANIFEST_GROUP_COLUMN,
                MANIFEST_RANDOM_ORDER_COLUMN,
                MANIFEST_SOURCE_INDEX_COLUMN,
            ])
        with csv_path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                payload: dict[str, object] = {MANIFEST_REL_PATH_COLUMN: row.rel_path}
                if include_random_columns:
                    payload[MANIFEST_GROUP_COLUMN] = row.group
                    payload[MANIFEST_RANDOM_ORDER_COLUMN] = row.random_order
                    payload[MANIFEST_SOURCE_INDEX_COLUMN] = row.source_index
                writer.writerow(payload)
