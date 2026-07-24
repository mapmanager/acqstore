"""CZI file loader for CloudScope acquisition images.

This module contains the CloudScope/acqstore CZI adapter. Pixel data are loaded
from scene ``0`` lazily through :mod:`czifile`. Reference images are read from
CZI attachments using the same attachment rules explored in the standalone CZI
reference-image investigation scripts.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from typing import Any, BinaryIO
import xml.etree.ElementTree as ET

import czifile
import numpy as np

from .base_file_loader import BaseFileLoader, ImageHeader, ReferenceImage
from .oir_file_loader import _image_header_from_scene
from acqstore.utils.logging import get_logger

logger = get_logger(__name__)


class CziFileLoader(BaseFileLoader):
    """Lazy-loading CZI reader for scene ``0`` only."""

    def __init__(self, path: str, header: ImageHeader | None = None) -> None:
        self._has_reference_image = False
        self._cached_reference_array: np.ndarray | None = None
        super().__init__(path, header)

    @contextmanager
    def _open_czi(self) -> Iterator[Any]:
        """Yield a ``czifile.CziFile`` opened from :attr:`path` or :attr:`_stream`.

        Yields:
            Open ``czifile.CziFile`` instance.
        """
        if self._stream is not None:
            self._stream.seek(0)
            with czifile.CziFile(self._stream) as czi:
                yield czi
        else:
            with czifile.CziFile(self.path) as czi:
                yield czi

    @classmethod
    def read_header_from_stream(cls, stream: BinaryIO, filename: str) -> ImageHeader:
        """Read CZI header from a stream without retaining the loader.

        Args:
            stream: Binary stream positioned at a CZI file.
            filename: Logical filename to store on the returned header.

        Returns:
            Header for the CZI file.
        """
        return cls.from_stream(stream, filename).header

    @property
    def has_reference_image(self) -> bool:
        """Return whether a CZI reference attachment was found during header read.

        Returns:
            ``True`` when a decoded reference attachment matched the current
            channel-first ``(C, Y, X)`` heuristic.
        """
        return self._has_reference_image

    @property
    def reference_image(self) -> ReferenceImage | None:
        """Return a CZI reference/overview image snapshot when one exists.

        CZI line-scan files can store the Zeiss frame/reference image as a
        decoded attachment named ``Image`` with content type ``ZISRAW`` and
        channel-first shape ``(C, Y, X)``. This property converts that attachment
        into acqstore's shared :class:`ReferenceImage` object so CloudScope views
        can display it without CZI-specific logic.

        Returns:
            Immutable :class:`ReferenceImage`, or ``None`` when the CZI has no
            matching reference attachment.
        """
        if self._referenceImage is not None:
            return self._referenceImage

        with self._open_czi() as czi_file:
            self._referenceImage = _reference_snapshot_from_czi(
                czi_file,
                reference_array=self._cached_reference_array,
            )

        return self._referenceImage

    def read_header(self) -> ImageHeader:
        """Read CZI image header metadata.

        Returns:
            Header for scene ``0``.
        """
        return self._read_czi_header()

    def _physical_units_for_header(self, scene: Any) -> tuple[tuple[Any, ...], tuple[str, ...]]:
        """Return per-axis physical calibration from a czifile scene.

        Args:
            scene: czifile scene-like object.

        Returns:
            Tuple of ``(physical_units, physical_units_labels)`` aligned to the
            scene coordinates.
        """
        czi_scene = scene
        xarr = czi_scene.asxarray()
        n = len(xarr.coords)
        _physical_units: list[Any] = [None] * n
        _physical_units_labels = [""] * n
        for idx, coord_str in enumerate(xarr.coords):
            if xarr[coord_str] is None or len(xarr[coord_str]) < 2:
                _physical_units[idx] = None
                _physical_units_labels[idx] = "unknown"
                continue
            value0 = xarr[coord_str][1] - xarr[coord_str][0]
            value: Any = value0.item()
            if coord_str in ("X", "Y"):
                value = float(value) * 1e6
            elif coord_str == "C":
                value = float("nan")
            _physical_units[idx] = value
            if coord_str in ("X", "Y"):
                _physical_units_labels[idx] = "um"
            elif coord_str == "T":
                _physical_units_labels[idx] = "seconds"
            else:
                _physical_units_labels[idx] = "unknown"

        return tuple(_physical_units), tuple(_physical_units_labels)

    def _read_czi_header(self) -> ImageHeader:
        """Read header information from the first scene of a CZI file.

        Common dimension patterns include ``('C','T','X')`` (line-scan),
        ``('C','T','Y','X')`` (frames), and ``('C','Y','X')`` (2D).

        Returns:
            Header for scene ``0`` with CloudScope-compatible line-scan dims.
        """
        logical = self.path
        with self._open_czi() as czi_file:
            num_scenes = len(czi_file.scenes)
            scene = czi_file.scenes[0]
            header = _image_header_from_scene(logical, scene, num_scenes=num_scenes)
            ref_array = _find_czi_reference_array(czi_file)
            self._has_reference_image = ref_array is not None
            self._cached_reference_array = ref_array

        dims = header.dims
        # CZI line-scan kymographs can report ('C', 'T', 'X') with no 'Y' axis.
        # CloudScope expects 2D image planes as (Y, X) after optional C/T/Z selection.
        # For this CZI subset the slow scan axis is labeled 'T'; treat it as 'Y'.
        # Skip when 'Y' is already present, e.g. ('C', 'T', 'Y', 'X') frame stacks.
        if 'Y' not in dims and 'T' in dims and 'X' in dims:
            logger.warning(
                "CZI header dims %r at %r: remapping 'T' axis to 'Y' for CloudScope (Y, X) convention",
                dims,
                logical,
            )
            new_dims = tuple('Y' if dim == 'T' else dim for dim in dims)
            new_sizes = dict(header.sizes)
            new_sizes['Y'] = int(new_sizes.pop('T'))
            new_labels = list(header.physical_units_labels)
            t_idx = dims.index('T')
            if t_idx < len(new_labels) and new_labels[t_idx] == 'T':
                new_labels[t_idx] = 'Y'
            header = replace(
                header,
                dims=new_dims,
                sizes=new_sizes,
                physical_units_labels=tuple(new_labels),
            )

        return header

    def _load_full_image_array(self) -> np.ndarray:
        """Load full pixel data from CZI scene ``0``.

        Returns:
            NumPy array containing scene ``0`` pixels.
        """
        logger.info('')
        with self._open_czi() as czi_file:
            return np.asarray(czi_file.scenes[0].asarray())


def _reference_snapshot_from_czi(
    czi_file: Any,
    reference_array: np.ndarray | None = None,
) -> ReferenceImage | None:
    """Build a :class:`ReferenceImage` from a CZI reference attachment.

    Args:
        czi_file: Open ``czifile.CziFile``-like object.
        reference_array: Optional pre-decoded ``(C, Y, X)`` reference array.
            When omitted, the first matching attachment is loaded from
            ``czi_file``.

    Returns:
        Immutable reference-image snapshot, or ``None`` if no reference image
        attachment matches the current CZI heuristic.
    """
    if reference_array is None:
        reference_array = _find_czi_reference_array(czi_file)
    if reference_array is None:
        return None
    reference_array = np.asarray(reference_array)
    reference_array.setflags(write=False)
    scaling = _reference_pixel_size_um_from_czi(czi_file)
    if scaling is None:
        coord_scales: tuple[tuple[str, float], ...] = ()
    else:
        x_um_per_pixel, y_um_per_pixel = scaling
        coord_scales = (
            ('X', x_um_per_pixel),
            ('Y', y_um_per_pixel),
        )

    scan_path = _find_czi_scan_path_plot_array(czi_file, reference_array, scaling)

    return ReferenceImage(
        array=reference_array,
        dims=('C', 'Y', 'X'),
        num_channels=int(reference_array.shape[0]),
        line_roi=None,
        coord_units=(('X', 'um'), ('Y', 'um')),
        coord_scales=coord_scales,
        coords=(),
        scan_path=scan_path,
    )




def _find_czi_scan_path_plot_array(
    czi_file: Any,
    reference_array: np.ndarray,
    scaling_um: tuple[float, float] | None,
) -> np.ndarray | None:
    """Return a plot-ready CZI scan path in reference-image pixel coordinates.

    Args:
        czi_file: Open ``czifile.CziFile``-like object.
        reference_array: Reference image array shaped as ``(C, Y, X)``.
        scaling_um: Reference-image pixel size as ``(x_um_per_pixel,
            y_um_per_pixel)``.

    Returns:
        Read-only ``(2, N)`` array where row ``0`` contains X pixel
        coordinates and row ``1`` contains Y pixel coordinates, or ``None``
        when no candidate CZI scan path is found or scaling is unavailable.
    """
    if scaling_um is None:
        return None
    raw_scan_path = _find_czi_scan_path_array(czi_file)
    if raw_scan_path is None:
        return None

    x_um_per_pixel, y_um_per_pixel = scaling_um
    x_m_per_pixel = x_um_per_pixel * 1e-6
    y_m_per_pixel = y_um_per_pixel * 1e-6
    if x_m_per_pixel <= 0.0 or y_m_per_pixel <= 0.0:
        return None

    _channels, height, width = reference_array.shape
    raw = np.asarray(raw_scan_path, dtype=float)

    # Current best-supported CZI trajectory hypothesis: xy_center. The raw
    # attachment rows are interpreted as physical X/Y scanner coordinates in
    # meters, scaled to pixels, then centered on the reference image. Keep this
    # transform isolated so future CZI work can test alternatives such as:
    #   xy_origin: x = raw[0] / sx,          y = raw[1] / sy
    #   yx_origin: x = raw[1] / sx,          y = raw[0] / sy
    #   yx_center: x = raw[1] / sx + w / 2,  y = raw[0] / sy + h / 2
    x_pixels = raw[0] / x_m_per_pixel + (width / 2.0)
    y_pixels = raw[1] / y_m_per_pixel + (height / 2.0)

    scan_path = np.asarray([x_pixels, y_pixels], dtype=float)
    scan_path.setflags(write=False)
    return scan_path


def _find_czi_scan_path_array(czi_file: Any) -> np.ndarray | None:
    """Return the raw CZI candidate scan-path attachment.

    Args:
        czi_file: Open ``czifile.CziFile``-like object.

    Returns:
        Read-only raw ``(2, N)`` floating-point path array in the physical
        coordinate values stored by Zeiss, or ``None`` when no attachment passes
        the current scan-path heuristic.
    """
    for attachment in czi_file.attachments():
        entry = attachment.attachment_entry
        if not _is_czi_reference_attachment(entry):
            continue
        try:
            data = attachment.data()
        except Exception as exc:  # pragma: no cover - defensive around czifile decoding.
            logger.warning('Could not decode CZI attachment %r: %s', getattr(entry, 'filename', ''), exc)
            continue
        if not isinstance(data, np.ndarray):
            continue
        array = np.asarray(data)
        if not _is_czi_scan_path_array(array):
            continue
        array.setflags(write=False)
        return array
    return None


def _is_czi_scan_path_array(array: np.ndarray) -> bool:
    """Return whether an array matches the current CZI scan-path heuristic.

    Args:
        array: Decoded CZI attachment array.

    Returns:
        ``True`` for floating-point arrays shaped as ``(2, N)`` with enough
        samples to represent a dense scanner trajectory.
    """
    return (
        array.ndim == 2
        and array.shape[0] == 2
        and array.shape[1] > 100
        and np.issubdtype(array.dtype, np.floating)
    )


def _find_czi_reference_array(czi_file: Any) -> np.ndarray | None:
    """Return the first CZI attachment matching the reference-image heuristic.

    Args:
        czi_file: Open ``czifile.CziFile``-like object.

    Returns:
        Read-only ``(C, Y, X)`` reference image array, or ``None``.
    """
    for attachment in czi_file.attachments():
        entry = attachment.attachment_entry
        if not _is_czi_reference_attachment(entry):
            continue
        try:
            data = attachment.data()
        except Exception as exc:  # pragma: no cover - defensive around czifile decoding.
            logger.warning('Could not decode CZI attachment %r: %s', getattr(entry, 'filename', ''), exc)
            continue
        if not isinstance(data, np.ndarray):
            continue
        array = np.asarray(data)
        if not _is_channel_first_reference_array(array):
            continue
        array.setflags(write=False)
        return array
    return None


def _is_czi_reference_attachment(entry: Any) -> bool:
    """Return whether a CZI attachment entry has reference-image metadata.

    Args:
        entry: ``CziAttachmentEntry``-like object from ``attachment.attachment_entry``.

    Returns:
        ``True`` when the entry name and content type match the current
        reference-image rule.
    """
    return getattr(entry, 'name', None) == 'Image' and _content_type_name(
        getattr(entry, 'content_file_type', None)
    ) == 'ZISRAW'


def _is_channel_first_reference_array(array: np.ndarray) -> bool:
    """Return whether an array looks like a CZI channel-first reference image.

    Args:
        array: Decoded attachment array.

    Returns:
        ``True`` for arrays shaped like ``(C, Y, X)``.
    """
    return (
        array.ndim == 3
        and array.shape[0] >= 1
        and array.shape[0] <= 16
        and array.shape[1] > 32
        and array.shape[2] > 32
    )


def _reference_pixel_size_um_from_czi(czi_file: Any) -> tuple[float, float] | None:
    """Return CZI reference pixel size from XML ``ScalingX`` and ``ScalingY``.

    Args:
        czi_file: Open ``czifile.CziFile``-like object.

    Returns:
        ``(x_um_per_pixel, y_um_per_pixel)``, or ``None`` if either scaling tag
        is missing or invalid.
    """
    root = _xml_root_from_czi(czi_file)
    if root is None:
        return None
    scaling_x_m = _first_float_text_by_tag(root, 'ScalingX')
    scaling_y_m = _first_float_text_by_tag(root, 'ScalingY')
    if scaling_x_m is None or scaling_y_m is None:
        return None
    return (scaling_x_m * 1e6, scaling_y_m * 1e6)


def _xml_root_from_czi(czi_file: Any) -> ET.Element | None:
    """Return parsed CZI XML metadata root using real ``czifile`` APIs.

    Args:
        czi_file: Open ``czifile.CziFile``-like object.

    Returns:
        XML root element, or ``None`` when metadata cannot be parsed.
    """
    root = getattr(czi_file, 'xml_element', None)
    if root is not None:
        return root
    try:
        metadata = czi_file.metadata()
    except Exception:  # pragma: no cover - defensive around czifile metadata.
        return None
    if not metadata:
        return None
    try:
        return ET.fromstring(metadata)
    except ET.ParseError:
        return None


def _first_float_text_by_tag(root: ET.Element, tag_name: str) -> float | None:
    """Return the first parseable float text for an XML tag.

    Args:
        root: XML metadata root.
        tag_name: Namespace-stripped tag name to find.

    Returns:
        Parsed float, or ``None``.
    """
    for element in root.iter():
        if _strip_namespace(str(element.tag)) != tag_name:
            continue
        text = (element.text or '').strip()
        if not text:
            continue
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _strip_namespace(tag: str) -> str:
    """Return an XML tag name without an ElementTree namespace prefix.

    Args:
        tag: XML element tag, possibly namespace-qualified.

    Returns:
        Namespace-stripped tag name.
    """
    return tag.rsplit('}', 1)[-1] if '}' in tag else tag


def _content_type_name(content_type: Any) -> str:
    """Return a stable CZI attachment content-type name.

    Args:
        content_type: CZI content type enum or string.

    Returns:
        Enum value when present, otherwise ``str(content_type)``.
    """
    value = getattr(content_type, 'value', content_type)
    return str(value)
