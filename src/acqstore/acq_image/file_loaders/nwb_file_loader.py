"""Lazy discovery and pixel loading for static images stored in NWB files.

Generic NWB support is deliberately conservative. A stock ``GrayscaleImage``
is one logical YX AcqImage; AcqStore never guesses that several equal-shaped
images are channels or Z planes. AcqStore-authored manifests may explicitly
group several grayscale planes into one CYX member for lossless round trips.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .base_file_loader import BaseFileLoader, ImageHeader, format_file_size

ACQSTORE_NWB_FORMAT = "acqstore-nwb"
SINGLE_NWB_VERSION = 1
COLLECTION_NWB_VERSION = 2
SINGLE_MANIFEST_NAME = "acqstore_manifest_json"
COLLECTION_MANIFEST_NAME = "acqstore_collection_manifest_json"
SUPPORTED_NWB_AXES = {("Y", "X"), ("C", "Y", "X")}


@dataclass(frozen=True, slots=True)
class NwbImageMember:
    """Metadata-only description of one logical static image in an NWB file."""

    member_id: str
    images_container: str
    channel_images: tuple[str, ...]
    axes: tuple[str, ...]
    header: ImageHeader
    display_name: str
    sidecar_payload: dict[str, object] | None = None
    analysis_tables: tuple[tuple[str, str], ...] = ()
    is_acqstore_native: bool = False


@dataclass(frozen=True, slots=True)
class _NwbLoaderApi:
    """Container for lazily imported PyNWB runtime classes."""

    NWBHDF5IO: Any
    Images: Any
    GrayscaleImage: Any


def _require_nwb_loader_api() -> _NwbLoaderApi:
    """Import the optional PyNWB API lazily.

    Returns:
        Runtime API required to inspect and read local NWB files.

    Raises:
        ImportError: If AcqStore's optional ``nwb`` extra is not installed.
    """
    try:
        from pynwb import NWBHDF5IO
        from pynwb.image import GrayscaleImage, Images
    except ImportError as exc:
        raise ImportError(
            "NWB support requires AcqStore's optional 'nwb' dependencies. "
            "Install them with: uv sync --extra nwb"
        ) from exc
    return _NwbLoaderApi(
        NWBHDF5IO=NWBHDF5IO,
        Images=Images,
        GrayscaleImage=GrayscaleImage,
    )


def inspect_nwb_image_members(path: str | Path) -> tuple[NwbImageMember, ...]:
    """Inspect supported static image members without loading their pixels.

    AcqStore manifests explicitly define channel grouping and round-trip state.
    Without a manifest, every stock ``GrayscaleImage`` is an independent YX
    member because standard NWB does not assign channel or Z meaning to an
    ``Images`` collection.

    Args:
        path: Local HDF5-backed NWB file.

    Returns:
        Supported logical image members in stable order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ImportError: If optional NWB dependencies are unavailable.
        ValueError: If native metadata is malformed or contradicts datasets.
    """
    source = Path(path).expanduser().resolve(strict=True)
    api = _require_nwb_loader_api()
    with api.NWBHDF5IO(path=source, mode="r", load_namespaces=True) as io:
        nwbfile = io.read()
        native_entries = _read_native_manifest_entries(nwbfile)
        if native_entries is not None:
            return tuple(
                _native_member_from_manifest(source, nwbfile, entry, api)
                for entry in native_entries
            )
        return _discover_stock_members(source, nwbfile, api)


def _read_native_manifest_entries(nwbfile: Any) -> list[dict[str, object]] | None:
    """Return native manifest entries, or ``None`` for a stock NWB file."""
    single = _read_optional_json_scratch(nwbfile, SINGLE_MANIFEST_NAME)
    collection = _read_optional_json_scratch(nwbfile, COLLECTION_MANIFEST_NAME)
    if single is not None and collection is not None:
        raise ValueError("NWB file contains conflicting AcqStore manifests")
    if single is not None:
        _validate_manifest_root(single, SINGLE_NWB_VERSION, "AcqImage")
        image = single.get("image")
        if not isinstance(image, dict):
            raise ValueError("AcqStore NWB single-image manifest is missing 'image'")
        return [image]
    if collection is not None:
        _validate_manifest_root(collection, COLLECTION_NWB_VERSION, "AcqImageList")
        images = collection.get("images")
        if not isinstance(images, list) or not images:
            raise ValueError("AcqStore NWB collection manifest must contain images")
        if not all(isinstance(image, dict) for image in images):
            raise ValueError("AcqStore NWB collection image entries must be objects")
        return images
    return None


def _read_optional_json_scratch(
    nwbfile: Any,
    scratch_name: str,
) -> dict[str, object] | None:
    """Read an optional JSON object from NWB scratch."""
    scratch = nwbfile.scratch.get(scratch_name)
    if scratch is None:
        return None
    raw = scratch.data
    if hasattr(raw, "shape") and getattr(raw, "shape", None) == () and hasattr(raw, "__getitem__"):
        raw = raw[()]
    if isinstance(raw, np.ndarray) and raw.shape == ():
        raw = raw.item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if not isinstance(raw, str):
        raise ValueError(f"AcqStore NWB scratch {scratch_name!r} is not a JSON string")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"AcqStore NWB scratch {scratch_name!r} contains invalid JSON"
        ) from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"AcqStore NWB scratch {scratch_name!r} must contain an object")
    return parsed


def _validate_manifest_root(
    manifest: dict[str, object],
    version: int,
    kind: str,
) -> None:
    """Validate the identity and version of an AcqStore root manifest."""
    if manifest.get("format") != ACQSTORE_NWB_FORMAT:
        raise ValueError("NWB scratch uses an unsupported AcqStore format marker")
    if manifest.get("version") != version:
        raise ValueError(
            f"Unsupported AcqStore NWB version {manifest.get('version')!r}; "
            f"expected {version}"
        )
    if manifest.get("kind") != kind:
        raise ValueError(
            f"AcqStore NWB kind is {manifest.get('kind')!r}; expected {kind!r}"
        )


def _native_member_from_manifest(
    source: Path,
    nwbfile: Any,
    manifest: dict[str, object],
    api: _NwbLoaderApi,
) -> NwbImageMember:
    """Build and validate one manifest-defined native member."""
    member_id = _required_string(manifest, "id")
    display_name = _required_string(manifest, "display_name")
    container_name = _required_string(manifest, "images_container")
    channel_names = _required_string_list(manifest, "channel_images")
    raw_axes = manifest.get("axes")
    if not isinstance(raw_axes, list) or not all(isinstance(axis, str) for axis in raw_axes):
        raise ValueError(f"NWB member {member_id!r} has invalid axes")
    axes = tuple(axis.upper() for axis in raw_axes)
    if axes not in SUPPORTED_NWB_AXES:
        raise ValueError(f"AcqStore NWB does not support axes={axes!r}")

    container = nwbfile.acquisition.get(container_name)
    if not isinstance(container, api.Images):
        raise ValueError(
            f"NWB Images container {container_name!r} is missing for member {member_id!r}"
        )
    planes = []
    for image_name in channel_names:
        image = container.images.get(image_name)
        if not isinstance(image, api.GrayscaleImage):
            raise ValueError(
                f"NWB GrayscaleImage {image_name!r} is missing for member {member_id!r}"
            )
        planes.append(image)
    if not planes:
        raise ValueError(f"NWB member {member_id!r} contains no grayscale planes")

    sidecar_payload = manifest.get("sidecar_payload")
    if not isinstance(sidecar_payload, dict):
        raise ValueError(f"NWB member {member_id!r} is missing AcqImage JSON")
    analysis_raw = manifest.get("analysis_tables", {})
    if not isinstance(analysis_raw, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in analysis_raw.items()
    ):
        raise ValueError(f"NWB member {member_id!r} has invalid analysis_tables")

    plane_shape_xy, dtype = _validate_matching_planes(planes, member_id)
    expected_shape = _manifest_shape(manifest, axes)
    actual_shape = (
        (plane_shape_xy[1], plane_shape_xy[0])
        if axes == ("Y", "X")
        else (len(planes), plane_shape_xy[1], plane_shape_xy[0])
    )
    if actual_shape != expected_shape:
        raise ValueError(
            f"NWB member {member_id!r} manifest shape {expected_shape!r} "
            f"does not match stored pixels {actual_shape!r}"
        )
    if axes == ("Y", "X") and len(planes) != 1:
        raise ValueError(f"YX NWB member {member_id!r} must contain one image")

    header = _native_header(source, manifest, axes, dtype, sidecar_payload)
    return NwbImageMember(
        member_id=member_id,
        images_container=container_name,
        channel_images=channel_names,
        axes=axes,
        header=header,
        display_name=display_name,
        sidecar_payload=sidecar_payload,
        analysis_tables=tuple(analysis_raw.items()),
        is_acqstore_native=True,
    )


def _discover_stock_members(
    source: Path,
    nwbfile: Any,
    api: _NwbLoaderApi,
) -> tuple[NwbImageMember, ...]:
    """Discover each stock ``GrayscaleImage`` as an independent YX member."""
    members: list[NwbImageMember] = []
    for container_name in sorted(nwbfile.acquisition):
        container = nwbfile.acquisition[container_name]
        if not isinstance(container, api.Images):
            continue
        for image_name in sorted(container.images):
            image = container.images[image_name]
            if not isinstance(image, api.GrayscaleImage):
                continue
            shape_xy = tuple(int(value) for value in image.data.shape)
            if len(shape_xy) != 2 or any(value <= 0 for value in shape_xy):
                continue
            member_id = f"{container_name}/{image_name}"
            members.append(
                NwbImageMember(
                    member_id=member_id,
                    images_container=container_name,
                    channel_images=(image_name,),
                    axes=("Y", "X"),
                    header=_stock_header(source, nwbfile, image, shape_xy),
                    display_name=member_id,
                )
            )
    return tuple(members)


def _stock_header(
    source: Path,
    nwbfile: Any,
    image: Any,
    shape_xy: tuple[int, int],
) -> ImageHeader:
    """Create a YX header from unambiguous standard NWB image metadata."""
    shape_yx = (shape_xy[1], shape_xy[0])
    units = (1.0, 1.0)
    labels = ("Pixels", "Pixels")
    resolution = getattr(image, "resolution", None)
    try:
        pixels_per_cm = float(resolution)
    except (TypeError, ValueError):
        pixels_per_cm = math.nan
    if math.isfinite(pixels_per_cm) and pixels_per_cm > 0:
        units = (1.0 / pixels_per_cm, 1.0 / pixels_per_cm)
        labels = ("cm", "cm")

    session_start = getattr(nwbfile, "session_start_time", None)
    date = session_start.strftime("%Y%m%d") if session_start is not None else ""
    time = session_start.strftime("%H:%M:%S") if session_start is not None else ""
    return ImageHeader(
        path=str(source),
        shape=shape_yx,
        dims=("Y", "X"),
        sizes={"Y": shape_yx[0], "X": shape_yx[1]},
        dtype=np.dtype(image.data.dtype),
        num_channels=1,
        num_scenes=1,
        physical_units=units,
        physical_units_labels=labels,
        date=date,
        time=time,
        file_size=format_file_size(source),
    )


def _native_header(
    source: Path,
    manifest: dict[str, object],
    axes: tuple[str, ...],
    dtype: np.dtype[Any],
    sidecar_payload: dict[str, object],
) -> ImageHeader:
    """Create a header from explicit AcqStore-native metadata."""
    shape = _manifest_shape(manifest, axes)
    header_payload = sidecar_payload.get("image_header_metadata")
    if not isinstance(header_payload, dict):
        raise ValueError("AcqStore NWB JSON is missing image_header_metadata")
    units, labels = ImageHeader.default_physical_for_dims(axes)
    units_list = list(units)
    labels_list = list(labels)
    for dim, unit_key, label_key in (
        ("Y", "physical_unit_y", "physical_label_y"),
        ("X", "physical_unit_x", "physical_label_x"),
    ):
        index = axes.index(dim)
        units_list[index] = float(header_payload[unit_key])
        labels_list[index] = str(header_payload[label_key])
    return ImageHeader(
        path=str(source),
        shape=shape,
        dims=axes,
        sizes=dict(zip(axes, shape, strict=True)),
        dtype=dtype,
        num_channels=shape[axes.index("C")] if "C" in axes else 1,
        num_scenes=1,
        physical_units=tuple(units_list),
        physical_units_labels=tuple(labels_list),
        date=str(header_payload.get("date", "")),
        time=str(header_payload.get("time", "")),
        file_size=format_file_size(source),
    )


def _manifest_shape(
    manifest: dict[str, object],
    axes: tuple[str, ...],
) -> tuple[int, ...]:
    """Return a validated positive shape matching ``axes``."""
    raw = manifest.get("shape")
    if not isinstance(raw, list) or len(raw) != len(axes):
        raise ValueError("AcqStore NWB member shape does not match its axes")
    try:
        shape = tuple(int(value) for value in raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("AcqStore NWB member shape must contain integers") from exc
    if any(value <= 0 for value in shape):
        raise ValueError("AcqStore NWB member shape values must be positive")
    return shape


def _validate_matching_planes(
    planes: list[Any],
    member_id: str,
) -> tuple[tuple[int, int], np.dtype[Any]]:
    """Validate native channel plane shape and dtype consistency."""
    shapes = {tuple(int(value) for value in plane.data.shape) for plane in planes}
    if len(shapes) != 1:
        raise ValueError(
            f"NWB member {member_id!r} contains channel images with different shapes"
        )
    shape = next(iter(shapes))
    if len(shape) != 2 or any(value <= 0 for value in shape):
        raise ValueError(f"NWB member {member_id!r} contains a non-2D grayscale image")
    dtypes = {np.dtype(plane.data.dtype) for plane in planes}
    if len(dtypes) != 1:
        raise ValueError(f"NWB member {member_id!r} contains mixed channel dtypes")
    return (shape[0], shape[1]), next(iter(dtypes))


def _required_string(manifest: dict[str, object], key: str) -> str:
    """Return a required nonempty manifest string."""
    value = manifest.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"AcqStore NWB member field {key!r} must be a string")
    return value


def _required_string_list(
    manifest: dict[str, object],
    key: str,
) -> tuple[str, ...]:
    """Return a required nonempty list of nonempty strings."""
    value = manifest.get(key)
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(f"AcqStore NWB member field {key!r} must be a string list")
    return tuple(value)


class NwbFileLoader(BaseFileLoader):
    """Lazy pixel loader for one supported logical image in a local NWB file."""

    def __init__(self, path: str, *, member_id: str | None = None) -> None:
        """Inspect metadata, resolve one member, and leave pixels unloaded."""
        members = inspect_nwb_image_members(path)
        if member_id is None:
            if not members:
                raise ValueError(
                    "NWB file contains no AcqStore-supported static images. "
                    "This version supports embedded Images/GrayscaleImage data."
                )
            if len(members) > 1:
                available = ", ".join(member.member_id for member in members)
                raise ValueError(
                    "NWB file contains multiple supported logical images; use "
                    "load_nwb_collection() or AcqImageList.from_nwb(). "
                    f"Available member IDs: {available}"
                )
            selected = members[0]
        else:
            selected = next(
                (member for member in members if member.member_id == member_id),
                None,
            )
            if selected is None:
                available = ", ".join(member.member_id for member in members) or "none"
                raise ValueError(
                    f"NWB member {member_id!r} was not found; available members: {available}"
                )

        self.member = selected
        self.member_id = selected.member_id
        self.images_container = selected.images_container
        self.channel_images = selected.channel_images
        self.axes = selected.axes
        resolved = str(Path(path).expanduser().resolve(strict=True))
        super().__init__(resolved, header=selected.header)

    @classmethod
    def from_member(
        cls,
        path: str | Path,
        member: NwbImageMember,
    ) -> NwbFileLoader:
        """Construct a loader from a member returned by one discovery pass.

        Args:
            path: Physical NWB file inspected to create ``member``.
            member: Previously discovered logical member descriptor.

        Returns:
            Lazy loader for exactly ``member`` without reopening NWB metadata.

        Raises:
            ValueError: If the member header belongs to another physical file.
        """
        resolved = str(Path(path).expanduser().resolve(strict=True))
        if str(Path(member.header.path).resolve(strict=False)) != resolved:
            raise ValueError("NWB member descriptor belongs to a different file")
        instance = cls.__new__(cls)
        instance.member = member
        instance.member_id = member.member_id
        instance.images_container = member.images_container
        instance.channel_images = member.channel_images
        instance.axes = member.axes
        BaseFileLoader.__init__(instance, resolved, header=member.header)
        return instance

    def read_header(self) -> ImageHeader:
        """Reject direct header reads because construction performs discovery."""
        raise RuntimeError("NwbFileLoader resolves its header during NWB discovery")

    def _load_full_image_array(self) -> np.ndarray:
        """Materialize only the selected member in AcqStore axis order."""
        api = _require_nwb_loader_api()
        channels: list[np.ndarray] = []
        with api.NWBHDF5IO(path=self.path, mode="r", load_namespaces=True) as io:
            nwbfile = io.read()
            images = nwbfile.acquisition.get(self.images_container)
            if not isinstance(images, api.Images):
                raise ValueError(
                    f"NWB Images container {self.images_container!r} is missing "
                    f"for member {self.member_id!r}"
                )
            for image_name in self.channel_images:
                image = images.images.get(image_name)
                if not isinstance(image, api.GrayscaleImage):
                    raise ValueError(
                        f"NWB GrayscaleImage {image_name!r} is missing for "
                        f"member {self.member_id!r}"
                    )
                channels.append(np.asarray(image.data[:]).T)

        if self.axes == ("Y", "X"):
            if len(channels) != 1:
                raise ValueError(f"YX member {self.member_id!r} must contain one image")
            result = channels[0]
        elif self.axes == ("C", "Y", "X"):
            result = np.stack(channels, axis=0)
        else:
            raise ValueError(f"Unsupported NWB member axes: {self.axes!r}")
        if result.shape != self.header.shape:
            raise ValueError(
                f"NWB member {self.member_id!r} loaded shape {result.shape!r} "
                f"does not match discovered shape {self.header.shape!r}"
            )
        return result
