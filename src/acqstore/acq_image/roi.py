"""ROI models and ROI set management for CloudScope/acqstore.

This module defines a strict ROI schema for 2D image ROIs. All ROI
coordinates are pixel coordinates relative to an external 2D numpy array using
``array[row, col]`` convention.

The schema supports:
- rectangular ROIs
- line-segment ROIs
- runtime -> dict/JSON -> runtime round trips
- clamping to known image bounds
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, ClassVar

import numpy as np


class RoiType(StrEnum):
    """Supported ROI shape types.

    Values are serialized to disk and should remain stable.
    """

    RECTROI = "rectroi"
    LINESEGMENTROI = "linesegmentroi"


@dataclass(frozen=True)
class ImageBounds:
    """Image bounds for a 2D image and optional slice count.

    Args:
        width: Image width in pixels. This is dimension 1, columns.
        height: Image height in pixels. This is dimension 0, rows.
        num_slices: Number of slices or planes. Use 1 for 2D images.

    Raises:
        ValueError: If width, height, or num_slices are invalid.
    """

    width: int
    height: int
    num_slices: int = 1

    def __post_init__(self) -> None:
        """Validate image bounds after construction.

        Raises:
            ValueError: If width, height, or num_slices are invalid.
        """
        if self.width <= 0:
            raise ValueError(f"ImageBounds.width must be > 0, got {self.width}")
        if self.height <= 0:
            raise ValueError(f"ImageBounds.height must be > 0, got {self.height}")
        if self.num_slices <= 0:
            raise ValueError(f"ImageBounds.num_slices must be > 0, got {self.num_slices}")


@dataclass(frozen=True)
class RectRoiBounds:
    """Rectangular ROI bounds in image pixel coordinates.

    Coordinates use numpy-style dimension naming:
    - dim0 = rows / height
    - dim1 = columns / width

    Stop coordinates are exclusive, matching numpy slicing.

    Args:
        dim0_start: Starting row, inclusive.
        dim0_stop: Stopping row, exclusive.
        dim1_start: Starting column, inclusive.
        dim1_stop: Stopping column, exclusive.
    """

    dim0_start: int
    dim0_stop: int
    dim1_start: int
    dim1_stop: int

    @classmethod
    def from_image_bounds(cls, image_bounds: ImageBounds) -> RectRoiBounds:
        """Create bounds covering the full image.

        Args:
            image_bounds: Image bounds to cover.

        Returns:
            Rectangular ROI bounds spanning the full image.
        """
        return cls(
            dim0_start=0,
            dim0_stop=image_bounds.height,
            dim1_start=0,
            dim1_stop=image_bounds.width,
        )

    def clamped_to(self, image_bounds: ImageBounds) -> RectRoiBounds:
        """Return bounds clamped to image size with inverted edges fixed.

        Args:
            image_bounds: Image bounds used for clamping.

        Returns:
            Clamped rectangular ROI bounds.
        """
        dim0_start = max(0, min(self.dim0_start, image_bounds.height))
        dim0_stop = max(0, min(self.dim0_stop, image_bounds.height))
        dim1_start = max(0, min(self.dim1_start, image_bounds.width))
        dim1_stop = max(0, min(self.dim1_stop, image_bounds.width))

        if dim0_start > dim0_stop:
            dim0_start, dim0_stop = dim0_stop, dim0_start
        if dim1_start > dim1_stop:
            dim1_start, dim1_stop = dim1_stop, dim1_start

        return RectRoiBounds(
            dim0_start=dim0_start,
            dim0_stop=dim0_stop,
            dim1_start=dim1_start,
            dim1_stop=dim1_stop,
        )


@dataclass(frozen=True)
class LineEndpoints:
    """Line-segment ROI endpoints in image pixel coordinates.

    Endpoint coordinates use numpy-style ``array[row, col]`` convention.
    Unlike rectangular stop coordinates, line endpoints identify actual pixels
    and are therefore clamped to ``0..height-1`` and ``0..width-1``.

    Args:
        row0: Row coordinate of the first endpoint.
        col0: Column coordinate of the first endpoint.
        row1: Row coordinate of the second endpoint.
        col1: Column coordinate of the second endpoint.
    """

    row0: int
    col0: int
    row1: int
    col1: int

    def clamped_to(self, image_bounds: ImageBounds) -> LineEndpoints:
        """Return endpoints clamped to valid image pixel indices.

        Args:
            image_bounds: Image bounds used for clamping.

        Returns:
            Clamped line endpoints.
        """
        max_row = image_bounds.height - 1
        max_col = image_bounds.width - 1

        return LineEndpoints(
            row0=max(0, min(self.row0, max_row)),
            col0=max(0, min(self.col0, max_col)),
            row1=max(0, min(self.row1, max_row)),
            col1=max(0, min(self.col1, max_col)),
        )


@dataclass
class BaseROI:
    """Base ROI schema shared by all ROI shapes.

    Args:
        roi_id: Stable integer ROI identifier assigned by ``RoiSet``.
        name: Human-readable display name.
        note: Optional human-readable note.
    """

    roi_id: int
    name: str = ""
    note: str = ""

    roi_type: ClassVar[RoiType]
    schema_version: ClassVar[str] = "1.0"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable v1 envelope.

        Returns:
            Serialized ROI dictionary.

        Raises:
            NotImplementedError: Always for the base class.
        """
        raise NotImplementedError


@dataclass
class RectROI(BaseROI):
    """Axis-aligned rectangular ROI in full-image pixel coordinates.

    Args:
        roi_id: Stable integer ROI identifier assigned by ``RoiSet``.
        name: Human-readable display name.
        note: Optional human-readable note.
        bounds: Rectangular ROI bounds.
    """

    bounds: RectRoiBounds = field(
        default_factory=lambda: RectRoiBounds(
            dim0_start=0,
            dim0_stop=0,
            dim1_start=0,
            dim1_stop=0,
        )
    )

    roi_type: ClassVar[RoiType] = RoiType.RECTROI
    schema_version: ClassVar[str] = "1.0"

    def clamp_to_image(self, img: np.ndarray) -> None:
        """Clamp ROI bounds to be fully inside a 2D image.

        Args:
            img: 2D numpy array.

        Raises:
            ValueError: If img is not 2D.
        """
        self.bounds = clamp_rect_bounds_to_image(self.bounds, img)

    def clamp_to_bounds(self, image_bounds: ImageBounds) -> None:
        """Clamp ROI bounds to image bounds.

        Args:
            image_bounds: Bounds used to clamp the ROI.
        """
        self.bounds = self.bounds.clamped_to(image_bounds)

    def get_time_bounds(self, voxels: list[float] | None) -> tuple[float, float] | None:
        """Return ROI dim0 range in physical time units.

        For kymographs, dim0 is commonly the time dimension. This method
        converts the ROI dim0 bounds from pixel coordinates to physical units
        using ``voxels[0]``.

        Args:
            voxels: Voxel sizes. ``voxels[0]`` is the physical size of dim0.

        Returns:
            Tuple ``(time_min, time_max)`` if voxels are available, otherwise
            None.
        """
        if voxels is None or len(voxels) < 1:
            return None

        time_min = float(self.bounds.dim0_start * voxels[0])
        time_max = float(self.bounds.dim0_stop * voxels[0])
        return (time_min, time_max)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable v1 envelope.

        Returns:
            Serialized RectROI dictionary.
        """
        return {
            "roi_id": int(self.roi_id),
            "roi_type": self.roi_type.value,
            "version": self.schema_version,
            "name": self.name,
            "note": self.note,
            "data": {
                "row_start": int(self.bounds.dim0_start),
                "row_stop": int(self.bounds.dim0_stop),
                "col_start": int(self.bounds.dim1_start),
                "col_stop": int(self.bounds.dim1_stop),
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RectROI:
        """Create a RectROI from a strict v1 serialized dictionary.

        Args:
            data: Serialized ROI dictionary produced by ``to_dict``.

        Returns:
            RectROI instance.

        Raises:
            ValueError: If required fields are missing or invalid.
            NotImplementedError: If the ROI type is not ``rectroi``.
        """
        _require_exact_keys(
            data=data,
            required={"roi_id", "roi_type", "version", "name", "note", "data"},
            context="RectROI",
        )

        roi_type = _parse_roi_type(data["roi_type"])
        if roi_type is not cls.roi_type:
            raise NotImplementedError(
                f"RectROI.from_dict only supports roi_type={cls.roi_type.value!r}; "
                f"got {roi_type.value!r}"
            )

        version = str(data["version"])
        if version != cls.schema_version:
            raise ValueError(f"Unsupported RectROI version {version!r}")

        data_block = data["data"]
        if not isinstance(data_block, dict):
            raise ValueError("Serialized RectROI field 'data' must be a dict")

        _require_exact_keys(
            data=data_block,
            required={"row_start", "row_stop", "col_start", "col_stop"},
            context="RectROI data",
        )

        return cls(
            roi_id=int(data["roi_id"]),
            name=str(data["name"]),
            note=str(data["note"]),
            bounds=RectRoiBounds(
                dim0_start=int(data_block["row_start"]),
                dim0_stop=int(data_block["row_stop"]),
                dim1_start=int(data_block["col_start"]),
                dim1_stop=int(data_block["col_stop"]),
            ),
        )


@dataclass
class LineROI(BaseROI):
    """Line-segment ROI with two endpoints in full-image pixel coordinates.

    Args:
        roi_id: Stable integer ROI identifier assigned by ``RoiSet``.
        name: Human-readable display name.
        note: Optional human-readable note.
        endpoints: Line endpoints.
    """

    endpoints: LineEndpoints = field(
        default_factory=lambda: LineEndpoints(row0=0, col0=0, row1=0, col1=0)
    )

    roi_type: ClassVar[RoiType] = RoiType.LINESEGMENTROI
    schema_version: ClassVar[str] = "1.0"

    def clamp_to_image(self, img: np.ndarray) -> None:
        """Clamp line endpoints to valid pixels in a 2D image.

        Args:
            img: 2D numpy array.

        Raises:
            ValueError: If img is not 2D.
        """
        self.endpoints = clamp_line_endpoints_to_image(self.endpoints, img)

    def clamp_to_bounds(self, image_bounds: ImageBounds) -> None:
        """Clamp line endpoints to image bounds.

        Args:
            image_bounds: Bounds used to clamp the ROI.
        """
        self.endpoints = self.endpoints.clamped_to(image_bounds)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable v1 envelope.

        Returns:
            Serialized LineROI dictionary.
        """
        return {
            "roi_id": int(self.roi_id),
            "roi_type": self.roi_type.value,
            "version": self.schema_version,
            "name": self.name,
            "note": self.note,
            "data": {
                "row0": int(self.endpoints.row0),
                "col0": int(self.endpoints.col0),
                "row1": int(self.endpoints.row1),
                "col1": int(self.endpoints.col1),
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LineROI:
        """Create a LineROI from a strict v1 serialized dictionary.

        Args:
            data: Serialized ROI dictionary produced by ``to_dict``.

        Returns:
            LineROI instance.

        Raises:
            ValueError: If required fields are missing or invalid.
            NotImplementedError: If the ROI type is not ``linesegmentroi``.
        """
        _require_exact_keys(
            data=data,
            required={"roi_id", "roi_type", "version", "name", "note", "data"},
            context="LineROI",
        )

        roi_type = _parse_roi_type(data["roi_type"])
        if roi_type is not cls.roi_type:
            raise NotImplementedError(
                f"LineROI.from_dict only supports roi_type={cls.roi_type.value!r}; "
                f"got {roi_type.value!r}"
            )

        version = str(data["version"])
        if version != cls.schema_version:
            raise ValueError(f"Unsupported LineROI version {version!r}")

        data_block = data["data"]
        if not isinstance(data_block, dict):
            raise ValueError("Serialized LineROI field 'data' must be a dict")

        _require_exact_keys(
            data=data_block,
            required={"row0", "col0", "row1", "col1"},
            context="LineROI data",
        )

        return cls(
            roi_id=int(data["roi_id"]),
            name=str(data["name"]),
            note=str(data["note"]),
            endpoints=LineEndpoints(
                row0=int(data_block["row0"]),
                col0=int(data_block["col0"]),
                row1=int(data_block["row1"]),
                col1=int(data_block["col1"]),
            ),
        )


class RoiSet:
    """Container and manager for multiple ROI instances.

    This class owns the ROIs, assigns unique integer IDs, and preserves
    creation order via an internal dictionary.

    Args:
        image_bounds: Bounds used to clamp ROI coordinates.
    """

    def __init__(self, image_bounds: ImageBounds) -> None:
        self.image_bounds = image_bounds
        self._rois: dict[int, BaseROI] = {}
        self._next_id: int = 1
        self._is_dirty: bool = False

    def is_dirty(self) -> bool:
        """Return whether the ROI set has unsaved changes.

        Returns:
            True if the ROI set has unsaved changes.
        """
        return self._is_dirty

    def set_dirty(self) -> None:
        """Set the dirty flag to True."""
        self._is_dirty = True

    def set_clean(self) -> None:
        """Set the dirty flag to False."""
        self._is_dirty = False

    def create_rect_roi(
        self,
        bounds: RectRoiBounds | None = None,
        name: str = "",
        note: str = "",
    ) -> RectROI:
        """Create and add a rectangular ROI.

        Args:
            bounds: Desired rectangular bounds. If None, defaults to full image.
            name: Display name.
            note: Optional notes.

        Returns:
            Newly created RectROI instance.
        """
        if bounds is None:
            bounds = RectRoiBounds.from_image_bounds(self.image_bounds)

        roi = RectROI(
            roi_id=self._allocate_roi_id(),
            name=name,
            note=note,
            bounds=bounds.clamped_to(self.image_bounds),
        )
        self._rois[roi.roi_id] = roi
        self.set_dirty()
        return roi

    def create_line_roi(
        self,
        endpoints: LineEndpoints,
        name: str = "",
        note: str = "",
    ) -> LineROI:
        """Create and add a line-segment ROI.

        Args:
            endpoints: Desired line endpoints.
            name: Display name.
            note: Optional notes.

        Returns:
            Newly created LineROI instance.
        """
        roi = LineROI(
            roi_id=self._allocate_roi_id(),
            name=name,
            note=note,
            endpoints=endpoints.clamped_to(self.image_bounds),
        )
        self._rois[roi.roi_id] = roi
        self.set_dirty()
        return roi

    def edit_rect_roi(
        self,
        roi_id: int,
        *,
        bounds: RectRoiBounds | None = None,
        name: str | None = None,
        note: str | None = None,
    ) -> RectROI:
        """Edit a rectangular ROI.

        Args:
            roi_id: Identifier of the ROI to edit.
            bounds: New rectangular bounds. If None, bounds are unchanged.
            name: New name. If None, name is unchanged.
            note: New note. If None, note is unchanged.

        Returns:
            Edited RectROI instance.

        Raises:
            ValueError: If ROI is not found.
            TypeError: If ROI exists but is not a RectROI.
        """
        roi = self._get_required_roi(roi_id)
        if not isinstance(roi, RectROI):
            raise TypeError(f"ROI {roi_id} is not a RectROI")

        changed = False

        if bounds is not None:
            new_bounds = bounds.clamped_to(self.image_bounds)
            if roi.bounds != new_bounds:
                roi.bounds = new_bounds
                changed = True

        changed = self._update_common_fields(roi, name=name, note=note) or changed

        if changed:
            self.set_dirty()

        return roi

    def edit_line_roi(
        self,
        roi_id: int,
        *,
        endpoints: LineEndpoints | None = None,
        name: str | None = None,
        note: str | None = None,
    ) -> LineROI:
        """Edit a line-segment ROI.

        Args:
            roi_id: Identifier of the ROI to edit.
            endpoints: New line endpoints. If None, endpoints are unchanged.
            name: New name. If None, name is unchanged.
            note: New note. If None, note is unchanged.

        Returns:
            Edited LineROI instance.

        Raises:
            ValueError: If ROI is not found.
            TypeError: If ROI exists but is not a LineROI.
        """
        roi = self._get_required_roi(roi_id)
        if not isinstance(roi, LineROI):
            raise TypeError(f"ROI {roi_id} is not a LineROI")

        changed = False

        if endpoints is not None:
            new_endpoints = endpoints.clamped_to(self.image_bounds)
            if roi.endpoints != new_endpoints:
                roi.endpoints = new_endpoints
                changed = True

        changed = self._update_common_fields(roi, name=name, note=note) or changed

        if changed:
            self.set_dirty()

        return roi

    def delete(self, roi_id: int) -> None:
        """Delete a ROI by ID.

        Args:
            roi_id: Identifier of the ROI to delete.

        Raises:
            ValueError: If ROI is not found.
        """
        self._get_required_roi(roi_id)
        del self._rois[roi_id]
        self.set_dirty()

    def clear(self) -> int:
        """Delete all ROIs and reset the internal ID counter.

        Returns:
            Number of ROIs deleted.
        """
        count = len(self._rois)
        self._rois.clear()
        self._next_id = 1

        if count > 0:
            self.set_dirty()

        return count

    def get(self, roi_id: int) -> BaseROI | None:
        """Return a ROI by ID.

        Args:
            roi_id: Identifier of the ROI to retrieve.

        Returns:
            ROI instance, or None if not present.
        """
        return self._rois.get(roi_id)

    def has_roi(self, roi_id: int) -> bool:
        """Return whether a ROI exists.

        Args:
            roi_id: Identifier of the ROI to check.

        Returns:
            True if ``roi_id`` exists in this ROI set.
        """
        return roi_id in self._rois

    def get_roi_ids(self) -> list[int]:
        """Return ROI IDs in creation order.

        Returns:
            List of ROI IDs.
        """
        return list(self._rois.keys())

    @property
    def num_rois(self) -> int:
        """Return the number of ROIs in the set.

        Returns:
            Number of ROIs.
        """
        return len(self._rois)

    def as_list(self) -> list[BaseROI]:
        """Return ROIs in creation order.

        Returns:
            List of ROI instances.
        """
        return list(self._rois.values())

    def to_list(self) -> list[dict[str, Any]]:
        """Serialize all ROIs to v1 dictionaries.

        Returns:
            List of serialized ROI dictionaries.
        """
        return [roi.to_dict() for roi in self._rois.values()]

    def from_list(self, data: list[dict[str, Any]]) -> RoiSet:
        """Replace this ROI set from serialized ROI dictionaries.

        Args:
            data: List of dictionaries, each produced by ROI ``to_dict``.

        Returns:
            This RoiSet instance.

        Raises:
            ValueError: If duplicate ROI IDs are present.
            NotImplementedError: If an unsupported ROI type is present.
        """
        self._rois.clear()
        self._next_id = 1

        for item in data:
            roi = roi_from_dict(item)

            if roi.roi_id in self._rois:
                raise ValueError(f"Duplicate ROI ID in serialized data: {roi.roi_id}")

            self._clamp_roi(roi)
            self._rois[roi.roi_id] = roi
            self._next_id = max(self._next_id, roi.roi_id + 1)

        self.set_clean()
        return self

    def __iter__(self) -> Iterator[BaseROI]:
        """Iterate over ROIs in creation order.

        Returns:
            Iterator over ROI instances.
        """
        return iter(self._rois.values())

    def _allocate_roi_id(self) -> int:
        """Allocate and return the next ROI ID.

        Returns:
            Newly allocated ROI ID.
        """
        roi_id = self._next_id
        self._next_id += 1
        return roi_id

    def _get_required_roi(self, roi_id: int) -> BaseROI:
        """Return a ROI or raise if it does not exist.

        Args:
            roi_id: Identifier of the ROI.

        Returns:
            ROI instance.

        Raises:
            ValueError: If ROI is not found.
        """
        roi = self._rois.get(roi_id)
        if roi is None:
            raise ValueError(f"ROI {roi_id} not found")
        return roi

    def _update_common_fields(
        self,
        roi: BaseROI,
        *,
        name: str | None,
        note: str | None,
    ) -> bool:
        """Update fields shared by all ROI types.

        Args:
            roi: ROI to update.
            name: New name, or None to leave unchanged.
            note: New note, or None to leave unchanged.

        Returns:
            True if any field changed, False otherwise.
        """
        changed = False

        if name is not None and roi.name != name:
            roi.name = name
            changed = True

        if note is not None and roi.note != note:
            roi.note = note
            changed = True

        return changed

    def _clamp_roi(self, roi: BaseROI) -> None:
        """Clamp a ROI to this set's image bounds.

        Args:
            roi: ROI to clamp.

        Raises:
            NotImplementedError: If ROI type is unsupported.
        """
        if isinstance(roi, RectROI):
            roi.clamp_to_bounds(self.image_bounds)
            return

        if isinstance(roi, LineROI):
            roi.clamp_to_bounds(self.image_bounds)
            return

        raise NotImplementedError(f"Unsupported ROI class: {type(roi).__name__}")


def roi_from_dict(data: dict[str, Any]) -> BaseROI:
    """Deserialize a ROI from a strict v1 dictionary.

    Args:
        data: Serialized ROI dictionary.

    Returns:
        ROI instance.

    Raises:
        ValueError: If required fields are missing or invalid.
        NotImplementedError: If the ROI type is unsupported.
    """
    if "roi_type" not in data:
        raise ValueError("Serialized ROI is missing required key 'roi_type'")

    roi_type = _parse_roi_type(data["roi_type"])

    if roi_type is RoiType.RECTROI:
        return RectROI.from_dict(data)

    if roi_type is RoiType.LINESEGMENTROI:
        return LineROI.from_dict(data)

    raise NotImplementedError(f"Unsupported ROI type: {roi_type.value!r}")


def clamp_rect_bounds_to_image(
    bounds: RectRoiBounds,
    img: np.ndarray,
) -> RectRoiBounds:
    """Clamp rectangular bounds to a 2D image shape.

    Args:
        bounds: Rectangular ROI bounds to clamp.
        img: 2D numpy array.

    Returns:
        Clamped rectangular ROI bounds.

    Raises:
        ValueError: If img is not 2D.
    """
    if img.ndim != 2:
        raise ValueError(f"Expected a 2D image, got ndim {img.ndim}")

    height, width = img.shape
    return bounds.clamped_to(ImageBounds(width=width, height=height))


def clamp_line_endpoints_to_image(
    endpoints: LineEndpoints,
    img: np.ndarray,
) -> LineEndpoints:
    """Clamp line endpoints to a 2D image shape.

    Args:
        endpoints: Line endpoints to clamp.
        img: 2D numpy array.

    Returns:
        Clamped line endpoints.

    Raises:
        ValueError: If img is not 2D.
    """
    if img.ndim != 2:
        raise ValueError(f"Expected a 2D image, got ndim {img.ndim}")

    height, width = img.shape
    return endpoints.clamped_to(ImageBounds(width=width, height=height))


def _parse_roi_type(value: object) -> RoiType:
    """Parse a serialized ROI type value.

    Args:
        value: Serialized ROI type value.

    Returns:
        Parsed ROI type.

    Raises:
        ValueError: If value does not match a supported ROI type.
    """
    try:
        return RoiType(str(value).lower())
    except ValueError:
        raise ValueError(f"Unsupported ROI type: {value!r}") from None


def _require_exact_keys(
    data: dict[str, Any],
    required: set[str],
    context: str,
) -> None:
    """Validate that a dictionary contains exactly the required keys.

    Args:
        data: Dictionary to validate.
        required: Required key set.
        context: Human-readable context for error messages.

    Raises:
        ValueError: If required keys are missing or unexpected keys are present.
    """
    actual = set(data.keys())
    missing = sorted(required - actual)
    unexpected = sorted(actual - required)

    if missing:
        raise ValueError(f"{context} is missing required keys: {missing}")

    if unexpected:
        raise ValueError(f"{context} has unexpected keys: {unexpected}")
