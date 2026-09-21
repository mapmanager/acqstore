"""Object-oriented AcqStore OME-Zarr Collection v1 export.

The exporter writes only the additive AcqStore v1 JSON and CSV contract around
independently valid OME-Zarr images. Pixel and pyramid writing is delegated to
the existing public :class:`AcqImage` OME-Zarr methods without changing their
OME-NGFF behavior.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from acqstore.acq_image.roi import LineROI, RectROI

from .validator import get_schema_path, validate_collection

if TYPE_CHECKING:
    from acqstore.acq_image.acq_image import AcqImage
    from acqstore.acq_image.acq_image_list import AcqImageList
    from acqstore.acq_image.analysis.model import BaseAnalysis


class AcqStoreOmeZarrImageExporter:
    """Export one ``AcqImage`` into an in-progress v1 collection.

    Args:
        collection_root: Staging root for the complete collection.
        image_id: Newly allocated opaque v1 member identifier.
        zarr_format: Zarr format passed unchanged to public OME-Zarr writers.
    """

    def __init__(
        self,
        collection_root: str | Path,
        *,
        image_id: str,
        zarr_format: int = 3,
    ) -> None:
        """Create an exporter for one v1 member.

        Args:
            collection_root: Staging root for the complete collection.
            image_id: Newly allocated opaque v1 member identifier.
            zarr_format: Zarr format passed to public OME-Zarr writers.
        """
        self._root = Path(collection_root)
        self._image_id = image_id
        self._zarr_format = zarr_format
        self._roi_ids: set[int] = set()

    def export(self, acq_image: AcqImage) -> dict[str, Any]:
        """Write one primary image and its declared AcqStore resources.

        Args:
            acq_image: Fully loaded source acquisition image.

        Returns:
            Collection member descriptor for ``collection.json``.

        Raises:
            ValueError: If required pixels or analysis tables are not loaded,
                an ROI is unsupported, or source values cannot satisfy v1.
        """
        if not acq_image.images_loaded:
            raise ValueError(f'AcqImage pixels are not loaded: {acq_image.name}')
        if not acq_image.analysis_csv_loaded:
            raise ValueError(f'AcqImage analysis tables are not loaded: {acq_image.name}')

        primary_path = Path('images') / self._image_id
        acq_image.save_as_ome_zarr(
            self._root / primary_path,
            overwrite=False,
            zarr_format=self._zarr_format,
        )

        metadata_dir = self._root / 'metadata' / self._image_id
        metadata_dir.mkdir(parents=True, exist_ok=False)
        rois = self._build_rois(acq_image)
        acqimage_relative = Path('metadata') / self._image_id / 'acqimage.json'
        self._write_json(
            self._root / acqimage_relative,
            self._build_acqimage_document(acq_image, rois),
        )

        resources: dict[str, str] = {'acqimage': acqimage_relative.as_posix()}
        analyses = self._export_analyses(acq_image)
        if analyses:
            analyses_relative = Path('metadata') / self._image_id / 'analyses.json'
            self._write_json(
                self._root / analyses_relative,
                {
                    'format': 'acqstore-analyses',
                    'version': 1,
                    'image_id': self._image_id,
                    'analyses': analyses,
                },
            )
            resources['analyses'] = analyses_relative.as_posix()

        member: dict[str, Any] = {
            'id': self._image_id,
            'name': str(acq_image.name),
            'ome_zarr': primary_path.as_posix(),
            'resources': resources,
            'summary': self._build_summary(acq_image, analyses),
        }
        reference = self._export_reference(acq_image)
        if reference is not None:
            member['reference_image'] = reference
        return member

    def _build_rois(self, acq_image: AcqImage) -> list[dict[str, Any]]:
        """Build v1 ROI documents from existing public ROI objects.

        Args:
            acq_image: Source acquisition image.

        Returns:
            Schema-ready ROI objects.

        Raises:
            TypeError: If the image contains an unsupported ROI class.
        """
        output: list[dict[str, Any]] = []
        for roi in acq_image.rois:
            roi_id = int(roi.roi_id)
            if roi_id in self._roi_ids:
                raise ValueError(f'Duplicate AcqStore ROI ID: {roi_id}')
            self._roi_ids.add(roi_id)
            common: dict[str, Any] = {
                'id': roi_id,
                'name': str(roi.name) or f'ROI {roi.roi_id}',
                'coordinate_space': 'primary-image-full-resolution-pixels',
            }
            if roi.note:
                common['metadata'] = {'note': str(roi.note)}
            if isinstance(roi, RectROI):
                output.append(
                    {
                        **common,
                        'type': 'rectangle',
                        'start': [int(roi.bounds.dim1_start), int(roi.bounds.dim0_start)],
                        'stop': [int(roi.bounds.dim1_stop), int(roi.bounds.dim0_stop)],
                    }
                )
            elif isinstance(roi, LineROI):
                output.append(
                    {
                        **common,
                        'type': 'line',
                        'start': [int(roi.endpoints.col0), int(roi.endpoints.row0)],
                        'stop': [int(roi.endpoints.col1), int(roi.endpoints.row1)],
                    }
                )
            else:
                raise TypeError(f'Unsupported v1 ROI class: {type(roi).__name__}')
        return output

    def _build_acqimage_document(
        self,
        acq_image: AcqImage,
        rois: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build one schema-ready ``acqimage.json`` object.

        Args:
            acq_image: Source acquisition image.
            rois: Previously allocated v1 ROI documents.

        Returns:
            Complete v1 AcqImage document.
        """
        document: dict[str, Any] = {
            'format': 'acqstore-acqimage',
            'version': 1,
            'image_id': self._image_id,
            'accepted': bool(acq_image.get_schema_row()['accept']),
            'rois': rois,
            'experiment_metadata': self._json_value(acq_image.get_metadata_section('experiment_metadata').get_values()),
            'image_metadata': self._json_value(acq_image.get_metadata_section('acq_image_header').get_values()),
        }
        axis_display = self._build_axis_display(acq_image)
        if axis_display:
            document['axis_display'] = axis_display
        return document

    def _build_axis_display(self, acq_image: AcqImage) -> dict[str, Any]:
        """Build sparse display overrides for exceptional raster axes.

        Args:
            acq_image: Source acquisition image.

        Returns:
            Empty object for ordinary images or a sparse axis mapping.
        """
        header = acq_image.pixels.header
        overrides: dict[str, Any] = {}
        temporal_units = {
            's': 'second',
            'sec': 'second',
            'second': 'second',
            'seconds': 'second',
            'ms': 'millisecond',
            'millisecond': 'millisecond',
            'milliseconds': 'millisecond',
            'us': 'microsecond',
            'µs': 'microsecond',
            'microsecond': 'microsecond',
            'microseconds': 'microsecond',
        }
        for index, axis in enumerate(header.dims):
            raw_unit = str(header.physical_units_labels[index]).strip().lower()
            if str(axis).upper() != 'Y' or raw_unit not in temporal_units:
                continue
            overrides['y'] = {
                'type': 'time',
                'unit': temporal_units[raw_unit],
                'scale': float(header.physical_units[index]),
            }
        return overrides

    def _export_analyses(self, acq_image: AcqImage) -> list[dict[str, Any]]:
        """Write analysis CSVs and build analysis envelopes.

        Args:
            acq_image: Source acquisition image.

        Returns:
            Analysis documents for instances with summaries and/or tables.

        Raises:
            ValueError: If an analysis references an unknown ROI.
        """
        output: list[dict[str, Any]] = []
        analysis_dir = self._root / 'analysis' / self._image_id
        for analysis in acq_image.analysis_set.as_list():
            if not analysis.has_results():
                continue
            source_roi_id = int(analysis.key.roi_id)
            if source_roi_id not in self._roi_ids:
                raise ValueError(f'Analysis {analysis.key.analysis_name!r} references unknown ROI {source_roi_id}')
            analysis_id = str(uuid.uuid4())
            table = analysis.table_with_bookkeeping()
            csv_path: str | None = None
            if table is not None:
                analysis_dir.mkdir(parents=True, exist_ok=True)
                csv_relative = Path('analysis') / self._image_id / f'{analysis_id}.csv'
                table.to_csv(self._root / csv_relative, index=False)
                csv_path = csv_relative.as_posix()
            output.append(
                self._build_analysis_document(
                    analysis,
                    analysis_id=analysis_id,
                    roi_id=source_roi_id,
                    csv_path=csv_path,
                )
            )
        return output

    def _build_analysis_document(
        self,
        analysis: BaseAnalysis,
        *,
        analysis_id: str,
        roi_id: int,
        csv_path: str | None,
    ) -> dict[str, Any]:
        """Build one typed analysis envelope.

        Args:
            analysis: Source analysis instance.
            analysis_id: Newly allocated opaque analysis identifier.
            roi_id: Native AcqStore ROI identifier.
            csv_path: Optional collection-root-relative result table path.

        Returns:
            Complete schema-ready analysis object.
        """
        document: dict[str, Any] = {
            'id': analysis_id,
            'type': str(analysis.key.analysis_name),
            'roi_id': roi_id,
            'channel': int(analysis.key.channel),
            'parameters': self._json_value(dict(analysis.detection_params)),
            'summary': self._json_value(dict(analysis.result.summary)),
        }
        if csv_path is not None:
            document['resources'] = [
                {
                    'id': 'table',
                    'media_type': 'text/csv',
                    'path': csv_path,
                }
            ]
        return document

    def _export_reference(self, acq_image: AcqImage) -> dict[str, str] | None:
        """Write an optional reference image and metadata document.

        Args:
            acq_image: Source acquisition image.

        Returns:
            Reference-image link for ``collection.json``, or ``None``.

        Raises:
            ValueError: If scan-path points are fractional or malformed.
        """
        if not acq_image.images.has_reference_image:
            return None
        reference_relative = Path('images') / f'{self._image_id}-reference'
        acq_image.save_reference_as_ome_zarr(
            self._root / reference_relative,
            overwrite=False,
            zarr_format=self._zarr_format,
        )
        values = self._json_value(acq_image.get_metadata_section('reference_image_metadata').get_values())
        x_values = values.pop('scan_path_x_pixels', [])
        y_values = values.pop('scan_path_y_pixels', [])
        has_scan_path = bool(values.pop('has_scan_path', False))
        values.pop('scan_path_num_points', None)
        metadata_relative = Path('metadata') / self._image_id / 'reference-image.json'
        document: dict[str, Any] = {
            'format': 'acqstore-reference-image',
            'version': 1,
            'image_id': self._image_id,
            'metadata': values,
        }
        if has_scan_path:
            if len(x_values) != len(y_values) or len(x_values) < 2:
                raise ValueError('Reference scan path must contain matching X/Y arrays with at least two points')
            points = [[self._integer_coordinate(x), self._integer_coordinate(y)] for x, y in zip(x_values, y_values, strict=True)]
            document['scan_path'] = {
                'coordinate_space': 'reference-image-full-resolution-pixels',
                'points': points,
            }
        self._write_json(self._root / metadata_relative, document)
        return {
            'ome_zarr': reference_relative.as_posix(),
            'metadata': metadata_relative.as_posix(),
        }

    def _build_summary(
        self,
        acq_image: AcqImage,
        analyses: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build deliberately small non-authoritative discovery metadata.

        Args:
            acq_image: Source acquisition image.
            analyses: Exported analysis envelopes.

        Returns:
            Untyped summary object for collection discovery.
        """
        header = acq_image.pixels.header
        return {
            'shape': [int(value) for value in header.shape],
            'dims': [str(dim).lower() for dim in header.dims],
            'dtype': str(header.dtype),
            'num_channels': int(header.num_channels),
            'num_rois': int(acq_image.rois.num_rois),
            'analysis_types': sorted({str(item['type']) for item in analyses}),
            'accepted': bool(acq_image.get_schema_row()['accept']),
            'has_reference_image': bool(acq_image.images.has_reference_image),
        }

    @staticmethod
    def _integer_coordinate(value: object) -> int:
        """Return an integer-valued pixel coordinate without rounding.

        Args:
            value: Numeric coordinate supplied by the public metadata API.

        Returns:
            Exact integer coordinate.

        Raises:
            ValueError: If the value is negative, non-finite, or fractional.
        """
        numeric = float(value)
        if not np.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
            raise ValueError(f'v1 scan-path coordinate must be a non-negative integer, got {value!r}')
        return int(numeric)

    @classmethod
    def _json_value(cls, value: Any) -> Any:
        """Convert common scientific scalar/container values to JSON values.

        Args:
            value: Value returned by an existing public AcqStore API.

        Returns:
            Recursively converted JSON-compatible value.
        """
        if isinstance(value, dict):
            return {str(key): cls._json_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._json_value(item) for item in value]
        if isinstance(value, np.ndarray):
            return cls._json_value(value.tolist())
        if isinstance(value, np.generic):
            return value.item()
        if value is pd.NA:
            return None
        return value

    @staticmethod
    def _write_json(path: Path, document: dict[str, Any]) -> None:
        """Write one deterministic UTF-8 JSON document.

        Args:
            path: Destination file path.
            document: JSON-compatible object.

        Returns:
            None.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document, indent=2) + '\n', encoding='utf-8')


class AcqStoreOmeZarrCollectionExporter:
    """Export an ``AcqImageList`` as a staged v1 collection.

    Args:
        destination: New local directory ending in ``.ome.zarr``.
        name: Optional human-readable collection name.
        overwrite: Whether an existing destination may be replaced.
        zarr_format: Zarr format passed unchanged to image writers.
    """

    def __init__(
        self,
        destination: str | Path,
        *,
        name: str | None = None,
        overwrite: bool = False,
        zarr_format: int = 3,
    ) -> None:
        """Configure one collection export operation.

        Args:
            destination: New local directory ending in ``.ome.zarr``.
            name: Optional human-readable collection name.
            overwrite: Whether an existing destination may be replaced.
            zarr_format: Zarr format passed unchanged to image writers.

        Raises:
            ValueError: If the destination is remote, has the wrong suffix, or
                ``zarr_format`` is unsupported.
        """
        raw_destination = str(destination)
        if '://' in raw_destination:
            raise ValueError('v1 collection export supports local paths only')
        self._destination = Path(destination).expanduser().resolve(strict=False)
        if not self._destination.name.lower().endswith('.ome.zarr'):
            raise ValueError("Collection destination must end in '.ome.zarr'")
        if zarr_format not in {2, 3}:
            raise ValueError(f'zarr_format must be 2 or 3, got {zarr_format!r}')
        self._name = name
        self._overwrite = overwrite
        self._zarr_format = zarr_format

    def export(self, acq_image_list: AcqImageList) -> Path:
        """Export a non-empty acquisition list as one v1 collection.

        Args:
            acq_image_list: Fully loaded source collection.

        Returns:
            Resolved completed collection path.

        Raises:
            FileExistsError: If the destination exists and overwrite is false.
            ValueError: If the collection is empty or fails conformance.
        """
        members = tuple(acq_image_list)
        if not members:
            raise ValueError('Cannot export an empty AcqImageList')
        if self._destination.exists() and not self._overwrite:
            raise FileExistsError(f'Destination already exists: {self._destination}')

        self._destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f'.{self._destination.name}.staging-',
            dir=self._destination.parent,
        ) as temporary_directory:
            staged = Path(temporary_directory) / self._destination.name
            staged.mkdir()
            self._build(acq_image_list, members, staged)
            validate_collection(staged, get_schema_path())
            self._validate_ome_images(staged)
            self._install(staged)
        return self._destination

    def _build(
        self,
        acq_image_list: AcqImageList,
        members: tuple[AcqImage, ...],
        staged: Path,
    ) -> None:
        """Build a complete collection in a staging directory.

        Args:
            acq_image_list: Source collection owning analysis pools.
            members: Ordered source image snapshot.
            staged: Empty staging root.

        Returns:
            None.
        """
        results: list[dict[str, Any]] = []
        table_identity: dict[tuple[str, int], str] = {}
        for acq_image in members:
            image_id = str(uuid.uuid4())
            image_exporter = AcqStoreOmeZarrImageExporter(
                staged,
                image_id=image_id,
                zarr_format=self._zarr_format,
            )
            results.append(image_exporter.export(acq_image))
            for source_roi_id in image_exporter._roi_ids:
                for source_identity in {str(acq_image.path), str(acq_image.file_id)}:
                    key = (source_identity, source_roi_id)
                    if key in table_identity:
                        raise ValueError(f'Duplicate analysis-table identity: {key!r}')
                    table_identity[key] = image_id
        table_resources = self._export_collection_tables(
            acq_image_list,
            staged,
            table_identity,
        )
        document: dict[str, Any] = {
            'format': 'acqstore-ome-zarr-collection',
            'version': 1,
            'id': str(uuid.uuid4()),
            'name': self._name or self._default_name(acq_image_list),
            'created': datetime.now(UTC).replace(microsecond=0).isoformat().replace('+00:00', 'Z'),
            'producer': {'name': 'acqstore'},
            'members': results,
        }
        if table_resources:
            document['resources'] = {'tables': table_resources}
        AcqStoreOmeZarrImageExporter._write_json(
            staged / 'acqstore' / 'collection.json',
            document,
        )

    def _export_collection_tables(
        self,
        acq_image_list: AcqImageList,
        staged: Path,
        table_identity: dict[tuple[str, int], str],
    ) -> list[dict[str, str]]:
        """Write non-empty generic collection-level CSV resources.

        Args:
            acq_image_list: Source collection owning current analysis pools.
            staged: Collection staging root.

        Returns:
            Generic CSV descriptors for ``collection.json``.
        """
        resources: list[dict[str, str]] = []
        velocity_pool = acq_image_list.velocity_analysis_pool
        velocity = velocity_pool.get_dataframe()
        velocity_columns = tuple(column for column, _, _ in velocity_pool.get_analysis_column_specs())
        sum_pool = acq_image_list.sum_intensity_analysis_pool
        sum_intensity = sum_pool.get_dataframe()
        pools = (
            (
                'velocity',
                velocity,
                bool(velocity_columns) and bool(velocity[list(velocity_columns)].notna().any(axis=None)),
            ),
            (
                'sum_intensity',
                sum_intensity,
                sum_pool.row_type_column in sum_intensity.columns
                and bool(sum_intensity[sum_pool.row_type_column].notna().any())
                and bool(sum_intensity[sum_pool.row_type_column].astype(str).ne('not_analyzed').any()),
            ),
        )
        for resource_id, dataframe, has_results in pools:
            if not has_results:
                continue
            dataframe = self._link_collection_table(dataframe, table_identity)
            relative = Path('tables') / f'{resource_id}.csv'
            (staged / relative).parent.mkdir(parents=True, exist_ok=True)
            dataframe.to_csv(staged / relative, index=False)
            resources.append(
                {
                    'id': resource_id,
                    'media_type': 'text/csv',
                    'path': relative.as_posix(),
                }
            )
        return resources

    @staticmethod
    def _link_collection_table(
        dataframe: pd.DataFrame,
        table_identity: dict[tuple[str, int], str],
    ) -> pd.DataFrame:
        """Return an export copy linked to its Collection v1 member."""
        required = {'pool_row_id', 'path', 'roi_id'}
        missing = required - set(dataframe.columns)
        if missing:
            raise ValueError(f'Analysis table is missing required columns: {", ".join(sorted(missing))}')
        table = dataframe.copy()
        image_ids: list[str] = []
        for path, source_roi_id in zip(table['path'], table['roi_id'], strict=True):
            key = (str(path), int(source_roi_id))
            image_id = table_identity.get(key)
            if image_id is None:
                raise ValueError(f'Analysis table row does not reference an exported ROI: {key!r}')
            image_ids.append(image_id)
        table.insert(1, 'acq_image_id', image_ids)
        return table

    def _install(self, staged: Path) -> None:
        """Atomically install a validated staging directory.

        Args:
            staged: Validated staging collection.

        Returns:
            None.
        """
        if self._destination.exists():
            backup = self._destination.with_name(f'.{self._destination.name}.backup-{uuid.uuid4()}')
            os.replace(self._destination, backup)
            try:
                os.replace(staged, self._destination)
            except Exception:
                os.replace(backup, self._destination)
                raise
            shutil.rmtree(backup)
        else:
            os.replace(staged, self._destination)

    @staticmethod
    def _validate_ome_images(staged: Path) -> None:
        """Open every primary and reference image independently as OME-Zarr.

        Args:
            staged: Schema-valid collection staging root.

        Returns:
            None.

        Raises:
            ValueError: If a declared image cannot be opened as OME-Zarr.
        """
        from acqstore.acq_image.io.ome_zarr import read_acq_pixels_ome_zarr

        collection = json.loads((staged / 'acqstore' / 'collection.json').read_text(encoding='utf-8'))
        for member in collection['members']:
            read_acq_pixels_ome_zarr(staged / member['ome_zarr'], lazy=True)
            reference = member.get('reference_image')
            if reference is not None:
                read_acq_pixels_ome_zarr(staged / reference['ome_zarr'], lazy=True)

    def _default_name(self, acq_image_list: AcqImageList) -> str:
        """Return a human-readable name without assigning identity semantics.

        Args:
            acq_image_list: Source collection.

        Returns:
            Source-root name when available, otherwise destination name.
        """
        source_root = getattr(acq_image_list, 'source_root_path', None)
        if source_root:
            name = Path(str(source_root)).expanduser().name
            if name:
                return name
        return self._destination.name
