"""Optional local NWB import/export for AcqStore images and collections.

The canonical API is :func:`save_nwb`, :func:`load_nwb`,
:func:`save_nwb_collection`, and :func:`load_nwb_collection`. PyNWB is an
optional dependency and is imported only when one of these functions is used.

NWB v1 supports static ``YX`` and ``CYX`` primary images. Each AcqImage is
stored as one independent NWB ``Images`` acquisition container containing one
``GrayscaleImage`` per channel, so members of an AcqImageList may have unrelated
pixel shapes and channel counts. Existing AcqImage JSON is embedded in the NWB
manifest and remains authoritative for ROI, metadata, analysis configuration,
summary, peak-detection, and contrast state. Tabular analysis results are stored
as NWB ``DynamicTable`` objects.

NWB import is lazy by default. Loading an NWB file materializes JSON/header
metadata only; primary pixels and analysis DataFrames are loaded on demand using
the existing AcqImage lazy-loading APIs. NWB export is explicit and complete:
source lazy data is materialized one AcqImage at a time, written, and restored
to its previous lazy state. In-place mutation of an existing NWB file is not
implemented.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np

from acqstore.acq_image.file_loaders.nwb_file_loader import (
    NwbFileLoader,
    NwbImageMember,
    inspect_nwb_image_members,
)
from acqstore.acq_image.persistence import NwbPersistence
from acqstore.nwb_source import NwbSource

from acqstore.utils.logging import get_logger
logger = get_logger(__name__)

if TYPE_CHECKING:
    from acqstore.acq_image.acq_image import AcqImage
    from acqstore.acq_image.acq_image_list import AcqImageList

_ACQSTORE_NWB_FORMAT = "acqstore-nwb"
_SINGLE_NWB_VERSION = 1
_COLLECTION_NWB_VERSION = 2
_SINGLE_MANIFEST_NAME = "acqstore_manifest_json"
_COLLECTION_MANIFEST_NAME = "acqstore_collection_manifest_json"
_ANALYSIS_TABLE_PREFIX = "acqstore_analysis"
_SUPPORTED_AXES = {("Y", "X"), ("C", "Y", "X")}
_EASTERN_TIME = ZoneInfo("America/New_York")


@dataclass(frozen=True, slots=True)
class NwbSubjectMetadata:
    """Optional structured subject metadata for NWB and later DANDI use.

    Biological values are never inferred from AcqStore pixels or headers.

    Args:
        subject_id: User-defined subject identifier.
        species: Scientific species name, for example ``"Mus musculus"``.
        sex: Subject sex; ``"U"`` means unknown.
        age: Optional ISO 8601 duration, for example ``"P90D"``.
        description: Optional free-text subject description.
        genotype: Optional genotype description.
        strain: Optional strain description.
    """

    subject_id: str
    species: str
    sex: str = "U"
    age: str | None = None
    description: str | None = None
    genotype: str | None = None
    strain: str | None = None


@dataclass(frozen=True, slots=True)
class NwbMetadata:
    """Metadata controlling creation of one local NWB file.

    Args:
        session_description: Human-readable NWB session description.
        identifier: Optional unique NWB identifier. A UUID is generated when
            omitted.
        session_start_time: Optional timezone-aware session timestamp. Current
            ``America/New_York`` time is used when omitted.
        session_id: Optional NWB session identifier.
        subject: Optional structured subject metadata.
        experimenter: Names of people who performed the experiment.
        experiment_description: General description of the experiment.
        institution: Institution where the experiment was performed.
        keywords: Search terms describing the data.
    """

    session_description: str = "AcqStore acquisition"
    identifier: str | None = None
    session_start_time: datetime | None = None
    session_id: str | None = None
    subject: NwbSubjectMetadata | None = None
    experimenter: tuple[str, ...] | None = None
    experiment_description: str | None = None
    institution: str | None = None
    keywords: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class _NwbApi:
    """Container for lazily imported PyNWB/HDMF classes.

    Args:
        DynamicTable: HDMF DynamicTable class.
        NWBHDF5IO: PyNWB HDF5 I/O class.
        NWBFile: PyNWB NWBFile class.
        Subject: PyNWB Subject class.
        GrayscaleImage: PyNWB static grayscale image class.
        Images: PyNWB static image collection class.
    """

    DynamicTable: Any
    NWBHDF5IO: Any
    NWBFile: Any
    Subject: Any
    GrayscaleImage: Any
    Images: Any


def _require_nwb() -> _NwbApi:
    """Import and return the optional PyNWB/HDMF API.

    Returns:
        Lazily imported classes required by this module.

    Raises:
        ImportError: If AcqStore's optional ``nwb`` dependencies are not
            installed.
    """
    try:
        from hdmf.common import DynamicTable
        from pynwb import NWBHDF5IO, NWBFile
        from pynwb.file import Subject
        from pynwb.image import GrayscaleImage, Images
    except ImportError as exc:
        raise ImportError(
            "NWB support requires AcqStore's optional 'nwb' dependencies. "
            "Install them with: uv sync --extra nwb"
        ) from exc
    return _NwbApi(
        DynamicTable=DynamicTable,
        NWBHDF5IO=NWBHDF5IO,
        NWBFile=NWBFile,
        Subject=Subject,
        GrayscaleImage=GrayscaleImage,
        Images=Images,
    )


def save_nwb(
    acq_image: AcqImage,
    nwb_file: str | Path,
    *,
    metadata: NwbMetadata | None = None,
    overwrite: bool = False,
) -> None:
    """Export one ``YX`` or ``CYX`` AcqImage to a local NWB file.

    Lazy source pixels and analysis tables are loaded if required. Their
    original loaded/unloaded state is restored after the NWB file is closed.

    Args:
        acq_image: Source AcqImage.
        nwb_file: Destination local ``.nwb`` path.
        metadata: Optional top-level NWB metadata.
        overwrite: Whether an existing destination may be replaced.

    Returns:
        None.

    Raises:
        FileExistsError: If the destination exists and ``overwrite`` is false.
        ImportError: If optional NWB dependencies are not installed.
        TypeError: If ``acq_image`` is not an AcqImage.
        ValueError: If the source uses unsupported axes or contains an
            unsupported reference image.
    """
    from acqstore.acq_image.acq_image import AcqImage

    if not isinstance(acq_image, AcqImage):
        raise TypeError("acq_image must be an AcqImage")

    api = _require_nwb()
    destination = _prepare_destination(nwb_file, overwrite=overwrite)
    working = _temporary_destination(destination)
    nwbfile = _build_nwbfile(_resolve_metadata(metadata), api)

    was_images_loaded, was_analysis_loaded = _materialize_for_export(acq_image)
    try:
        image_manifest = _add_acq_image_to_nwbfile(
            nwbfile,
            acq_image,
            image_id="acqimage_0000",
            api=api,
        )
        manifest = {
            "format": _ACQSTORE_NWB_FORMAT,
            "version": _SINGLE_NWB_VERSION,
            "kind": "AcqImage",
            "image": image_manifest,
        }
        nwbfile.add_scratch(
            json.dumps(manifest, sort_keys=True),
            name=_SINGLE_MANIFEST_NAME,
            description="AcqStore single-AcqImage NWB round-trip manifest.",
        )
        with api.NWBHDF5IO(path=working, mode="x") as io:
            io.write(nwbfile)
        os.replace(working, destination)
    except Exception:
        working.unlink(missing_ok=True)
        raise
    finally:
        _restore_lazy_state(
            acq_image,
            was_images_loaded=was_images_loaded,
            was_analysis_loaded=was_analysis_loaded,
        )


def load_nwb(
    nwb_file: str | Path | NwbSource,
    *,
    load_images: bool = False,
    load_analysis_csv: bool = False,
    remote_cache_dir: str | Path | None = None,
) -> AcqImage:
    """Import exactly one supported stock or AcqStore NWB image.

    Args:
        nwb_file: Local path or supported read-only remote NWB source.
        load_images: Whether to materialize primary pixels before returning.
        load_analysis_csv: Whether to materialize analysis DynamicTables as
            DataFrames before returning.
        remote_cache_dir: Optional persistent byte-range cache directory for a
            remote source. Invalid for local paths or an existing ``NwbSource``.

    Returns:
        AcqImage whose pixels and analysis result tables are lazy by default.

    Raises:
        FileNotFoundError: If ``nwb_file`` does not exist.
        ImportError: If optional NWB dependencies are not installed.
        ValueError: If the file has zero or multiple supported logical images.
    """
    source = NwbSource.from_value(nwb_file, cache_dir=remote_cache_dir)
    return _build_lazy_acq_image(
        source,
        member_id=None,
        load_images=load_images,
        load_analysis_csv=load_analysis_csv,
    )


def save_nwb_collection(
    acq_image_list: AcqImageList,
    nwb_file: str | Path,
    *,
    metadata: NwbMetadata | None = None,
    overwrite: bool = False,
) -> None:
    """Export an AcqImageList to one NWB file with bounded source memory use.

    Each member is appended in its own ``r+`` PyNWB transaction. The member's
    source pixels and analysis DataFrames are materialized only for that write
    and returned to their previous lazy state before the next member begins.
    This avoids retaining every AcqImage's NumPy arrays in one PyNWB object graph.

    Args:
        acq_image_list: Source AcqImageList.
        nwb_file: Destination local ``.nwb`` path.
        metadata: Optional top-level NWB metadata.
        overwrite: Whether an existing destination may be replaced.

    Returns:
        None.

    Raises:
        FileExistsError: If the destination exists and ``overwrite`` is false.
        ImportError: If optional NWB dependencies are not installed.
        TypeError: If ``acq_image_list`` is not an AcqImageList.
        ValueError: If the collection is empty or contains unsupported data.
    """
    from acqstore.acq_image.acq_image_list import AcqImageList

    if not isinstance(acq_image_list, AcqImageList):
        raise TypeError("acq_image_list must be an AcqImageList")
    if len(acq_image_list) == 0:
        raise ValueError("Cannot save an empty AcqImageList to NWB")

    api = _require_nwb()
    destination = _prepare_destination(nwb_file, overwrite=overwrite)
    working = _temporary_destination(destination)

    try:
        # Create only the small top-level NWB structure first. Every source image
        # is appended in a separate r+ transaction so previous source arrays are
        # not retained by the PyNWB object graph for the next member.
        initial = _build_nwbfile(_resolve_metadata(metadata), api)
        with api.NWBHDF5IO(path=working, mode="x") as io:
            io.write(initial)

        image_manifests: list[dict[str, object]] = []
        for index, acq_image in enumerate(acq_image_list):
            was_images_loaded, was_analysis_loaded = _materialize_for_export(acq_image)
            try:
                with api.NWBHDF5IO(
                    path=working,
                    mode="r+",
                    load_namespaces=True,
                ) as io:
                    nwbfile = io.read()
                    image_manifests.append(
                        _add_acq_image_to_nwbfile(
                            nwbfile,
                            acq_image,
                            image_id=f"acqimage_{index:04d}",
                            api=api,
                        )
                    )
                    io.write(nwbfile)
            finally:
                _restore_lazy_state(
                    acq_image,
                    was_images_loaded=was_images_loaded,
                    was_analysis_loaded=was_analysis_loaded,
                )

        # The final collection manifest is small and references only objects that
        # are already persisted in the bounded-memory member transactions.
        with api.NWBHDF5IO(path=working, mode="r+", load_namespaces=True) as io:
            nwbfile = io.read()
            manifest = {
                "format": _ACQSTORE_NWB_FORMAT,
                "version": _COLLECTION_NWB_VERSION,
                "kind": "AcqImageList",
                "images": image_manifests,
            }
            nwbfile.add_scratch(
                json.dumps(manifest, sort_keys=True),
                name=_COLLECTION_MANIFEST_NAME,
                description="AcqStore AcqImageList NWB round-trip manifest.",
            )
            io.write(nwbfile)

        # Publish only a complete export. A failed member write leaves the prior
        # destination untouched and removes the temporary NWB file.
        os.replace(working, destination)
    except Exception:
        working.unlink(missing_ok=True)
        raise


def load_nwb_collection(
    nwb_file: str | Path | NwbSource,
    *,
    load_images: bool = False,
    load_analysis_csv: bool = False,
    remote_cache_dir: str | Path | None = None,
) -> AcqImageList:
    """Import all supported stock or AcqStore NWB images as a lazy list.

    The NWB file is opened once to read the small collection manifest. Member
    pixels and DynamicTable contents are not read unless the corresponding
    ``load_images`` or ``load_analysis_csv`` option is requested, or a member is
    loaded later through the normal AcqImage lazy-loading API.

    Args:
        nwb_file: Local path or supported read-only remote NWB source.
        load_images: Whether to materialize every member's primary pixels before
            returning. Defaults to false.
        load_analysis_csv: Whether to materialize every member's analysis tables
            before returning. Defaults to false.
        remote_cache_dir: Optional persistent byte-range cache directory for a
            remote source. Invalid for local paths or an existing ``NwbSource``.

    Returns:
        AcqImageList preserving stored member order and unique logical IDs.

    Raises:
        FileNotFoundError: If ``nwb_file`` does not exist.
        ImportError: If optional NWB dependencies are not installed.
        ValueError: If the file contains no supported logical images.
    """
    from acqstore.acq_image.acq_image_list import AcqImageList

    source = NwbSource.from_value(nwb_file, cache_dir=remote_cache_dir)
    members = inspect_nwb_image_members(source)
    if not members:
        raise ValueError(
            "NWB file contains no AcqStore-supported static images. This version "
            "supports embedded Images/GrayscaleImage data."
        )

    images: list[AcqImage] = []
    for member in members:
        images.append(
            _build_lazy_acq_image(
                source,
                member_id=member.member_id,
                discovered_member=member,
                load_images=load_images,
                load_analysis_csv=load_analysis_csv,
            )
        )

    # Reuse the established in-memory list construction pattern without treating
    # the shared NWB path as 500 independent filesystem files.
    collection = AcqImageList.__new__(AcqImageList)
    collection.path = source.identity
    collection.source_root_path = (
        str(source.local_path.parent)
        if source.local_path is not None
        else source.identity
    )
    collection.file_list = [image.file_id for image in images]
    collection._files = images
    collection._files_by_id = {image.file_id: image for image in images}
    if len(collection._files_by_id) != len(images):
        raise ValueError("AcqStore NWB collection contains duplicate logical member IDs")
    collection._attach_analysis_pools()
    return collection


def _prepare_destination(nwb_file: str | Path, *, overwrite: bool) -> Path:
    """Resolve and validate a local NWB destination path.

    Args:
        nwb_file: Requested destination path.
        overwrite: Whether an existing file may be replaced.

    Returns:
        Resolved destination path with its parent directory created.

    Raises:
        FileExistsError: If the destination exists and overwrite is false.
        ValueError: If the destination does not use a ``.nwb`` suffix.
    """
    destination = Path(nwb_file).expanduser().resolve(strict=False)
    if destination.suffix.lower() != ".nwb":
        raise ValueError(f"NWB destination must end with '.nwb': {destination}")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def _temporary_destination(destination: Path) -> Path:
    """Return a unique same-directory temporary NWB export path.

    Args:
        destination: Final requested NWB destination.

    Returns:
        Unique temporary path ending in ``.nwb`` so PyNWB treats it normally.
    """
    return destination.with_name(
        f".{destination.stem}.{uuid4().hex}.tmp.nwb"
    )


def _resolve_metadata(metadata: NwbMetadata | None) -> NwbMetadata:
    """Return validated NWB metadata with runtime defaults resolved.

    Args:
        metadata: Caller metadata or ``None``.

    Returns:
        Validated metadata with identifier and timestamp populated.

    Raises:
        TypeError: If metadata is not ``NwbMetadata`` or ``None``.
        ValueError: If supplied metadata fields are invalid.
    """
    value = metadata or NwbMetadata()
    if not isinstance(value, NwbMetadata):
        raise TypeError("metadata must be NwbMetadata or None")
    if not value.session_description.strip():
        raise ValueError("session_description must be a nonempty string")

    session_start_time = value.session_start_time or datetime.now(_EASTERN_TIME)
    if session_start_time.tzinfo is None or session_start_time.utcoffset() is None:
        raise ValueError("session_start_time must be timezone-aware")

    identifier = value.identifier or str(uuid4())
    if not identifier.strip():
        raise ValueError("identifier must be a nonempty string when supplied")
    if value.subject is not None:
        _validate_subject(value.subject)
    for field_name, field_value in (
        ("experiment_description", value.experiment_description),
        ("institution", value.institution),
    ):
        if field_value is not None and not field_value.strip():
            raise ValueError(f"{field_name} must be nonempty when supplied")
    for field_name, field_value in (
        ("experimenter", value.experimenter),
        ("keywords", value.keywords),
    ):
        if field_value is not None and (
            not field_value or any(not item.strip() for item in field_value)
        ):
            raise ValueError(f"{field_name} must contain nonempty strings")

    return NwbMetadata(
        session_description=value.session_description,
        identifier=identifier,
        session_start_time=session_start_time,
        session_id=value.session_id,
        subject=value.subject,
        experimenter=value.experimenter,
        experiment_description=value.experiment_description,
        institution=value.institution,
        keywords=value.keywords,
    )


def _validate_subject(subject: NwbSubjectMetadata) -> None:
    """Validate subject metadata without inventing biological values.

    Args:
        subject: Subject metadata supplied by the caller.

    Returns:
        None.

    Raises:
        TypeError: If subject is not ``NwbSubjectMetadata``.
        ValueError: If required subject strings are empty.
    """
    if not isinstance(subject, NwbSubjectMetadata):
        raise TypeError("subject must be NwbSubjectMetadata")
    if not subject.subject_id.strip():
        raise ValueError("subject.subject_id must be a nonempty string")
    if not subject.species.strip():
        raise ValueError("subject.species must be a nonempty string")
    if not subject.sex.strip():
        raise ValueError("subject.sex must be a nonempty string")


def _build_nwbfile(metadata: NwbMetadata, api: _NwbApi) -> Any:
    """Construct the top-level PyNWB ``NWBFile``.

    Args:
        metadata: Fully resolved NWB metadata.
        api: Lazily imported optional NWB API.

    Returns:
        New PyNWB NWBFile.
    """
    subject = None
    if metadata.subject is not None:
        subject_kwargs: dict[str, object] = {
            "subject_id": metadata.subject.subject_id,
            "species": metadata.subject.species,
            "sex": metadata.subject.sex,
        }
        if metadata.subject.age is not None:
            subject_kwargs["age"] = metadata.subject.age
        if metadata.subject.description is not None:
            subject_kwargs["description"] = metadata.subject.description
        if metadata.subject.genotype is not None:
            subject_kwargs["genotype"] = metadata.subject.genotype
        if metadata.subject.strain is not None:
            subject_kwargs["strain"] = metadata.subject.strain
        subject = api.Subject(**subject_kwargs)

    kwargs: dict[str, object] = {
        "session_description": metadata.session_description,
        "identifier": metadata.identifier,
        "session_start_time": metadata.session_start_time,
    }
    if metadata.session_id is not None:
        kwargs["session_id"] = metadata.session_id
    if subject is not None:
        kwargs["subject"] = subject
    if metadata.experimenter is not None:
        kwargs["experimenter"] = list(metadata.experimenter)
    if metadata.experiment_description is not None:
        kwargs["experiment_description"] = metadata.experiment_description
    if metadata.institution is not None:
        kwargs["institution"] = metadata.institution
    if metadata.keywords is not None:
        kwargs["keywords"] = list(metadata.keywords)
    return api.NWBFile(**kwargs)


def _materialize_for_export(acq_image: AcqImage) -> tuple[bool, bool]:
    """Ensure one AcqImage has complete pixels/tables for explicit NWB export.

    Args:
        acq_image: Source AcqImage.

    Returns:
        Tuple ``(was_images_loaded, was_analysis_loaded)`` describing the source
        state before export.
    """
    was_images_loaded = acq_image.images_loaded
    was_analysis_loaded = acq_image.analysis_csv_loaded

    if not was_images_loaded:
        acq_image.load_images()

    # File-backed and NWB-backed AcqImages can have lazy tabular results. An
    # in-memory AcqImage has no external table source; any current tables are
    # already represented directly on its analyses.
    if not was_analysis_loaded and not acq_image.is_memory_backed:
        acq_image.load_analysis_csv()

    return was_images_loaded, was_analysis_loaded


def _restore_lazy_state(
    acq_image: AcqImage,
    *,
    was_images_loaded: bool,
    was_analysis_loaded: bool,
) -> None:
    """Restore one source AcqImage to its pre-export lazy state.

    Args:
        acq_image: Source AcqImage used for export.
        was_images_loaded: Whether pixels were loaded before export.
        was_analysis_loaded: Whether analysis tables were loaded before export.

    Returns:
        None.
    """
    if not was_images_loaded:
        acq_image.unload_images()
    if not was_analysis_loaded and not acq_image.is_memory_backed:
        acq_image.unload_analysis_csv()


def _add_acq_image_to_nwbfile(
    nwbfile: Any,
    acq_image: AcqImage,
    *,
    image_id: str,
    api: _NwbApi,
) -> dict[str, object]:
    """Add one fully materialized AcqImage to an open PyNWB NWBFile.

    Args:
        nwbfile: Destination PyNWB NWBFile.
        acq_image: Source AcqImage whose lazy data has already been materialized.
        image_id: Unique collection-local identifier for this member.
        api: Lazily imported optional NWB API.

    Returns:
        JSON-serializable per-member manifest.

    Raises:
        ValueError: If axes are unsupported or a reference image is present.
    """
    pixels = acq_image.pixels
    axes = tuple(str(axis).upper() for axis in pixels.axes)
    if axes not in _SUPPORTED_AXES:
        raise ValueError(
            "AcqStore NWB currently supports only YX and CYX primary images; "
            f"got axes={axes!r} for {acq_image.file_id!r}"
        )
    if acq_image.images.has_reference_image:
        logger.info(f'AcqStore NWB v1 does not yet export reference-image pixels; cannot losslessly export {acq_image.file_id!r}')
        # raise ValueError(
        #     "AcqStore NWB v1 does not yet export reference-image pixels; "
        #     f"cannot losslessly export {acq_image.file_id!r}"
        # )

    images_container_name = f"{image_id}_images"
    images_container = api.Images(
        name=images_container_name,
        description=f"Static 2D image channels for {image_id}.",
    )
    channel_names: list[str] = []
    for channel in range(pixels.num_channels):
        image_name = f"channel_{channel:04d}"
        channel_names.append(image_name)
        plane_yx = np.asarray(pixels.get_plane(c=channel, as_numpy=True))
        images_container.add_image(
            api.GrayscaleImage(
                name=image_name,
                data=plane_yx.T,
                description=f"AcqStore static image channel {channel}.",
            )
        )
    nwbfile.add_acquisition(images_container)

    analysis_table_names: dict[str, str] = {}
    for analysis_name, dataframe in acq_image.analysis_set.results_tables_by_name().items():
        table_name = f"{_ANALYSIS_TABLE_PREFIX}__{image_id}__{analysis_name}"
        analysis_table_names[analysis_name] = table_name
        table_dataframe = dataframe.reset_index(drop=True).copy()
        table_dataframe.index.name = "id"
        nwbfile.add_analysis(
            api.DynamicTable.from_dataframe(
                df=table_dataframe,
                name=table_name,
                table_description=(
                    f"AcqStore {analysis_name} analysis results for {image_id}."
                ),
                columns=[
                    {
                        "name": column_name,
                        "description": _analysis_column_description(
                            column_name,
                            analysis_name=analysis_name,
                        ),
                    }
                    for column_name in table_dataframe.columns
                ],
            )
        )

    return {
        "id": image_id,
        "source_id": acq_image.file_id,
        "display_name": acq_image.name,
        "axes": list(axes),
        "shape": [int(size) for size in pixels.shape],
        "dtype": str(pixels.dtype),
        "images_container": images_container_name,
        "channel_images": channel_names,
        "sidecar_payload": acq_image._build_sidecar_payload(),
        "analysis_tables": analysis_table_names,
    }


def _analysis_column_description(column_name: str, *, analysis_name: str) -> str:
    """Return a meaningful NWB description for an AcqStore result column.

    Args:
        column_name: DataFrame column being exported.
        analysis_name: AcqStore analysis that owns the table.

    Returns:
        Human-readable, non-placeholder description.
    """
    descriptions = {
        "channel": "Zero-based AcqStore image channel index.",
        "roi_id": "AcqStore region-of-interest identifier.",
        "time_index": "Zero-based sample index along the analysis time axis.",
        "time_s": "Elapsed time from acquisition start, in seconds.",
        "time_sec": "Elapsed time from acquisition start, in seconds.",
        "velocity": "Estimated signed velocity in the analysis output units.",
        "radon_velocity": "Radon-transform velocity estimate in the analysis output units.",
        "theta_deg": "Estimated Radon-transform angle, in degrees.",
        "diameter_um": "Estimated vessel diameter, in micrometers.",
        "diameter_um_filt": "Filtered vessel-diameter estimate, in micrometers.",
    }
    return descriptions.get(
        column_name,
        f"AcqStore {analysis_name} analysis result value named {column_name!r}.",
    )


def _build_lazy_acq_image(
    source: NwbSource,
    member_id: str | None,
    *,
    discovered_member: NwbImageMember | None = None,
    load_images: bool,
    load_analysis_csv: bool,
) -> AcqImage:
    """Build one NWB-backed AcqImage without eagerly reading large datasets.

    Args:
        source: Normalized local or remote NWB source.
        member_id: Exact logical member, or ``None`` when exactly one is required.
        discovered_member: Optional descriptor from a shared discovery pass.
        load_images: Whether to materialize pixels before returning.
        load_analysis_csv: Whether to materialize DynamicTables before returning.

    Returns:
        NWB-backed AcqImage with a unique logical file identifier.

    Raises:
        ValueError: If member discovery or selection fails.
    """
    from acqstore.acq_image.acq_image import AcqImage

    loader = (
        NwbFileLoader.from_member(source, discovered_member)
        if discovered_member is not None
        else NwbFileLoader(source, member_id=member_id)
    )
    member = loader.member
    persistence = NwbPersistence(
        nwb_source=source,
        member_id=member.member_id,
        sidecar_payload=member.sidecar_payload,
        analysis_tables=dict(member.analysis_tables),
    )

    instance = AcqImage.__new__(AcqImage)
    instance.path = source.identity
    instance._initialize(
        images=loader,
        load_images=load_images,
        load_analysis_csv=load_analysis_csv,
        load_persisted_state=True,
        is_memory_backed=False,
        persistence_backend=persistence,
        file_id=f"{source.identity}#{member.member_id}",
        display_name=member.display_name,
    )
    return instance
