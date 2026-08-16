"""Persistence backends for AcqImage sidecar state and analysis result tables.

Pixel loading is intentionally not handled here. Primary image pixels remain the
responsibility of :class:`acqstore.acq_image.file_loaders.base_file_loader.BaseFileLoader`
and its concrete subclasses. These persistence backends only abstract where an
AcqImage's JSON sidecar state and tabular analysis results are read from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from acqstore.acq_image.acq_analysis_set import AcqAnalysisSet
    from acqstore.acq_image.acq_image import AcqImage


class AcqPersistenceBackend:
    """Base interface for loading persisted non-pixel AcqImage state."""

    @property
    def supports_source_save(self) -> bool:
        """Return whether :meth:`AcqImage.save` may write back to this source.

        Returns:
            ``True`` when the existing AcqImage ``save`` workflow is valid for
            this source backend.
        """
        return True

    def load_sidecar(self, acq_image: AcqImage) -> None:
        """Load persisted JSON state into one AcqImage when available.

        Args:
            acq_image: Destination AcqImage instance.

        Returns:
            None.
        """
        raise NotImplementedError

    def load_analysis_tables(self, analysis_set: AcqAnalysisSet) -> None:
        """Load persisted result tables into one analysis set.

        Args:
            analysis_set: Destination AcqAnalysisSet whose analysis identities
                have already been hydrated from JSON.

        Returns:
            None.
        """
        raise NotImplementedError


@dataclass(slots=True)
class FileSidecarPersistence(AcqPersistenceBackend):
    """Persistence backend for ordinary acquisition files and sidecar files.

    Args:
        source_path: Acquisition path used to locate ``.json`` and analysis CSV
            sidecars.
    """

    source_path: str

    def load_sidecar(self, acq_image: AcqImage) -> None:
        """Load the traditional ``<source>.json`` sidecar.

        Args:
            acq_image: Destination AcqImage instance.

        Returns:
            None.
        """
        acq_image.load_sidecar_json()

    def load_analysis_tables(self, analysis_set: AcqAnalysisSet) -> None:
        """Load traditional per-analysis CSV sidecars.

        Args:
            analysis_set: Destination AcqAnalysisSet.

        Returns:
            None.
        """
        analysis_set.load_all_results_dfs_from_csv(self.source_path)


@dataclass(slots=True)
class NativeZarrPersistence(AcqPersistenceBackend):
    """Persistence backend for AcqStore-native ``.cs.ome.zarr`` stores.

    Args:
        source_path: Native Zarr store path.
    """

    source_path: str

    def load_sidecar(self, acq_image: AcqImage) -> None:
        """Load embedded AcqStore JSON from a native Zarr store.

        Args:
            acq_image: Destination AcqImage instance.

        Returns:
            None.
        """
        acq_image.load_native_zarr_sidecar_json()

    def load_analysis_tables(self, analysis_set: AcqAnalysisSet) -> None:
        """Load embedded native-Zarr analysis CSV tables.

        Args:
            analysis_set: Destination AcqAnalysisSet.

        Returns:
            None.
        """
        from .io.store_utils import join_store_path

        analysis_set.load_results_tables_from_directory(
            join_store_path(self.source_path, "acqstore", "analysis")
        )


@dataclass(slots=True)
class NwbPersistence(AcqPersistenceBackend):
    """Read-only persistence backend for one logical AcqImage inside an NWB file.

    NWB import is intentionally read/import-only in this implementation. Runtime
    mutation remains supported in memory, but callers must explicitly export
    with :func:`acqstore.nwb_io.save_nwb` or
    :func:`acqstore.nwb_io.save_nwb_collection` rather than writing in place.

    Args:
        nwb_path: Physical local NWB file path.
        member_id: Logical AcqStore member identifier within the NWB file.
        sidecar_payload: Optional AcqImage JSON payload embedded in an AcqStore
            manifest. Stock NWB members have no payload.
        analysis_tables: Mapping from AcqStore analysis name to NWB DynamicTable
            name for this member.
    """

    nwb_path: str
    member_id: str
    sidecar_payload: dict[str, object] | None
    analysis_tables: dict[str, str]

    @property
    def supports_source_save(self) -> bool:
        """Return false because in-place NWB mutation is not implemented.

        Returns:
            Always ``False``.
        """
        return False

    def load_sidecar(self, acq_image: AcqImage) -> None:
        """Hydrate the existing embedded AcqImage JSON payload.

        Args:
            acq_image: Destination AcqImage instance.

        Returns:
            None.
        """
        if self.sidecar_payload is not None:
            acq_image._load_sidecar_payload(
                self.sidecar_payload,
                source=f"{self.nwb_path}#{self.member_id}",
            )

    def load_analysis_tables(self, analysis_set: AcqAnalysisSet) -> None:
        """Load only this member's NWB DynamicTables into pandas DataFrames.

        Args:
            analysis_set: Destination AcqAnalysisSet.

        Returns:
            None.

        Raises:
            ImportError: If the optional NWB dependencies are not installed.
            ValueError: If a manifest-referenced DynamicTable is missing.
        """
        api = _require_nwb_runtime()
        tables_by_name: dict[str, Any] = {}

        # Keep the file open only while converting the requested tables. No HDF5
        # objects escape this method, so unload_analysis_csv() can release all
        # resulting DataFrames normally.
        with api.NWBHDF5IO(path=self.nwb_path, mode="r", load_namespaces=True) as io:
            nwbfile = io.read()
            for analysis_name, table_name in self.analysis_tables.items():
                table = nwbfile.analysis.get(table_name)
                if table is None:
                    raise ValueError(
                        f"NWB analysis table {table_name!r} is missing for "
                        f"member {self.member_id!r}"
                    )
                tables_by_name[analysis_name] = table.to_dataframe().reset_index(drop=True)

        analysis_set.load_results_tables_by_name(tables_by_name)


@dataclass(frozen=True, slots=True)
class _NwbRuntimeApi:
    """Container for lazily imported PyNWB runtime classes.

    Args:
        NWBHDF5IO: PyNWB HDF5 I/O class.
    """

    NWBHDF5IO: Any


def _require_nwb_runtime() -> _NwbRuntimeApi:
    """Import the optional PyNWB runtime dependency lazily.

    Returns:
        Runtime classes needed for lazy NWB table loading.

    Raises:
        ImportError: If AcqStore's optional ``nwb`` extra is not installed.
    """
    try:
        from pynwb import NWBHDF5IO
    except ImportError as exc:
        raise ImportError(
            "NWB support requires AcqStore's optional 'nwb' dependencies. "
            "Install them with: uv sync --extra nwb"
        ) from exc
    return _NwbRuntimeApi(NWBHDF5IO=NWBHDF5IO)


def create_persistence_backend(
    path: str,
    *,
    is_memory_backed: bool,
    file_loader: Any | None = None,
) -> AcqPersistenceBackend | None:
    """Create the default persistence backend for an existing AcqImage source.

    Args:
        path: Acquisition source path.
        is_memory_backed: Whether the AcqImage has no filesystem persistence source.
        file_loader: Optional source loader carrying container member identity.

    Returns:
        Matching persistence backend, or ``None`` for in-memory acquisitions.
    """
    if is_memory_backed:
        return None
    from .file_loaders.nwb_file_loader import NwbFileLoader

    if isinstance(file_loader, NwbFileLoader):
        member = file_loader.member
        return NwbPersistence(
            nwb_path=path,
            member_id=member.member_id,
            sidecar_payload=member.sidecar_payload,
            analysis_tables=dict(member.analysis_tables),
        )
    lower = path.lower()
    if lower.endswith((".cs.ome.zarr", ".cs.ome.zarr.zip")):
        return NativeZarrPersistence(path)
    return FileSidecarPersistence(path)
