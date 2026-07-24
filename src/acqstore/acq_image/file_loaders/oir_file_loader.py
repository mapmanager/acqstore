from contextlib import contextmanager
from collections.abc import Iterator
from typing import Any
from datetime import datetime
import xml.etree.ElementTree as ET

import numpy as np
import oirfile
from oirfile import METADATA

from .base_file_loader import BaseFileLoader, ImageHeader, ReferenceImage
from acqstore.utils.logging import get_logger

logger = get_logger(__name__)

def _oir_reference_spatial_coord_scales(
    ref: Any,
    *,
    pixel_length_x: float | None,
    pixel_length_y: float | None,
) -> dict[str, float]:
    """Return reference ``coord_scales`` with spatial µm/px steps for ``Y`` and ``X``.

    ``oirfile.reference.coord_scales`` are often coordinate-array deltas
    (``pixel_length / n_pixels``). Reference/overview images are spatial 2-D
    planes; use the parent ``OirFile`` pixel lengths as µm/px, matching
    Olympus TXT ``[Reference Image]`` unit-converted size.

    Args:
        ref: Open ``oirfile`` reference object.
        pixel_length_x: Micrometers per pixel along ``X``, from parent ``OirFile``.
        pixel_length_y: Micrometers per pixel along ``Y``; defaults to
            ``pixel_length_x`` when omitted.

    Returns:
        ``coord_scales`` mapping with ``Y``/``X`` overridden when lengths are
        available; otherwise the reference object's scales are returned unchanged.
    """
    scales = dict(ref.coord_scales)
    step_x = pixel_length_x
    step_y = pixel_length_y if pixel_length_y is not None else pixel_length_x
    dims = tuple(str(d) for d in ref.dims)
    if step_y is not None and np.isfinite(step_y) and step_y > 0 and "Y" in dims:
        scales["Y"] = float(step_y)
    if step_x is not None and np.isfinite(step_x) and step_x > 0 and "X" in dims:
        scales["X"] = float(step_x)
    return scales


def _reference_snapshot_from_oir_reference(
    ref: Any,
    *,
    pixel_length_x: float | None = None,
    pixel_length_y: float | None = None,
) -> ReferenceImage:
    """Build a :class:`ReferenceImage` while ``oirfile.OirFile`` is still open.

    Args:
        ref: Open ``oirfile`` reference attachment.
        pixel_length_x: Parent-file micrometers per pixel along ``X``.
        pixel_length_y: Parent-file micrometers per pixel along ``Y``.
    """
    raw = np.asarray(ref.asarray())
    raw.setflags(write=False)
    dims = tuple(str(d) for d in ref.dims)
    sizes: dict[str, int] = dict(ref.sizes)
    num_channels = int(sizes["C"]) if "C" in sizes else 1
    line_roi = ref.line_roi
    scan_path: np.ndarray | None = None
    if line_roi is not None:
        line_roi = tuple(float(x) for x in line_roi)
        x0, y0, x1, y1 = line_roi
        scan_path = np.asarray([[x0, x1], [y0, y1]], dtype=float)
        scan_path.setflags(write=False)
    units_t = tuple(sorted((str(k), str(v)) for k, v in ref.coord_units.items()))
    scales_raw = _oir_reference_spatial_coord_scales(
        ref,
        pixel_length_x=pixel_length_x,
        pixel_length_y=pixel_length_y,
    )
    scales_t = tuple(sorted((str(k), float(v)) for k, v in scales_raw.items()))
    coord_items: list[tuple[str, np.ndarray]] = []
    for key in sorted(ref.coords.keys(), key=str):
        c = np.asarray(ref.coords[key])
        c.setflags(write=False)
        coord_items.append((str(key), c))
    return ReferenceImage(
        array=raw,
        dims=dims,
        num_channels=num_channels,
        line_roi=line_roi,
        coord_units=units_t,
        coord_scales=scales_t,
        coords=tuple(coord_items),
        scan_path=scan_path,
    )

def _step_from_coord(coord: Any) -> float | None:
    """Spacing between the first two samples of a 1D numeric coordinate array.

    Args:
        coord: One-dimensional coordinate array-like, or ``None``.

    Returns:
        Difference as ``float``, or ``None`` if fewer than two points or the
        values are not numeric (for example channel-name labels on ``C``).
    """
    if coord is None or len(coord) < 2:
        return None

    arr = np.asarray(coord)
    if not np.issubdtype(arr.dtype, np.number):
        return None

    value = arr[1] - arr[0]
    if hasattr(value, "item"):
        value = value.item()
    return float(value)

def _iso8601_datetime_str_to_yyyymmdd_hhmmss(s: str) -> tuple[str, str]:
    """Parse oirfile-style ISO 8601 string; return ``(YYYYMMDD, HH:MM:SS)`` or empty strings."""
    t = s.strip()
    if not t:
        return ("", "")
    if t.endswith("Z"):
        t = t[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(t)
    except ValueError:
        return ("", "")
    return (dt.strftime("%Y%m%d"), dt.strftime("%H:%M:%S"))

def _strip_xml_tag(tag: str) -> str:
    """Return an XML element tag without an ElementTree namespace prefix."""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _enabled_axes_from_lsmimage_xml(xml: str) -> dict[str, dict[str, Any]]:
    """Parse enabled axis definitions from OIR LSMIMAGE metadata XML.

    Args:
        xml: Raw LSMIMAGE metadata XML string from ``oirfile``.

    Returns:
        Mapping of axis type (for example ``TIMELAPSE``) to parsed axis fields.
    """
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return {}

    axes: dict[str, dict[str, Any]] = {}
    for elem in root.iter():
        if _strip_xml_tag(str(elem.tag)) != "axis":
            continue
        enable_attr = elem.get("enable")
        if enable_attr is None:
            continue

        axis_type = ""
        info: dict[str, Any] = {"enable": enable_attr == "true"}
        for child in elem:
            ctag = _strip_xml_tag(str(child.tag))
            text = (child.text or "").strip()
            if not text:
                continue
            if ctag == "axis":
                axis_type = text
            elif ctag == "startPosition":
                info["start"] = float(text)
            elif ctag == "endPosition":
                info["end"] = float(text)
            elif ctag == "step":
                info["step"] = float(text)
            elif ctag == "maxSize":
                info["maxSize"] = int(text)

        if axis_type and info.get("enable", False):
            axes[axis_type] = info
    return axes


def _timelapse_axis_from_scene(scene: Any) -> dict[str, Any] | None:
    """Return enabled TIMELAPSE axis metadata from an open ``oirfile.OirFile``.

    Args:
        scene: Open ``oirfile.OirFile`` instance.

    Returns:
        Parsed TIMELAPSE axis info, or ``None`` when absent or disabled.
    """
    xml_metadata = scene.xml_metadata
    lsm_xmls = xml_metadata.get(METADATA.LSMIMAGE, [])
    if not lsm_xmls:
        return None
    axes = _enabled_axes_from_lsmimage_xml(lsm_xmls[0])
    return axes.get("TIMELAPSE")


def _is_y_timelapse_line_scan_axis(scene: Any) -> bool:
    """Return whether ``Y`` is the slow scan time axis for a line-scan kymograph.

    Line-scan OIR files can expose timelapse acquisition on ``Y`` without a
    separate ``T`` dimension. ``oirfile.coord_units`` still reports ``Y`` as
    micrometers; LSMIMAGE axis metadata is the source of truth for relabeling.

    Args:
        scene: Open ``oirfile.OirFile`` instance.

    Returns:
        ``True`` when TIMELAPSE is enabled and its ``maxSize`` matches ``Y``.
    """
    dims = tuple(str(d) for d in scene.dims)
    if "T" in dims or "Y" not in dims:
        return False
    timelapse = _timelapse_axis_from_scene(scene)
    if timelapse is None:
        return False
    max_size = timelapse.get("maxSize")
    y_size = scene.sizes.get("Y")
    if max_size is None or y_size is None:
        return False
    return int(max_size) == int(y_size)


def _series_interval_ms_values_from_lsmimage_xml(xml: str) -> list[float]:
    """Return all ``seriesInterval`` values (milliseconds) from LSMIMAGE XML.

    Args:
        xml: Raw LSMIMAGE metadata XML string from ``oirfile``.

    Returns:
        Parsed ``seriesInterval`` floats in document order.
    """
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return []

    values: list[float] = []
    for elem in root.iter():
        if _strip_xml_tag(str(elem.tag)) != "seriesInterval":
            continue
        text = (elem.text or "").strip()
        if not text:
            continue
        try:
            values.append(float(text))
        except ValueError:
            continue
    return values


def _line_scan_y_step_seconds_from_scene(scene: Any) -> float | None:
    """Return seconds-per-line for TIMELAPSE-on-Y line scans from LSMIMAGE XML.

    ``oirfile.coord_scales['Y']`` is often a spatial coordinate delta, not line
    period. Olympus exports and LSMIMAGE ``seriesInterval`` (milliseconds)
    match the TIFF+TXT sidecar ``secondsPerLine`` field.

    Args:
        scene: Open ``oirfile.OirFile`` instance.

    Returns:
        Positive seconds per line, or ``None`` when no usable interval is found.
    """
    xml_metadata = scene.xml_metadata
    lsm_xmls = xml_metadata.get(METADATA.LSMIMAGE, [])
    if not lsm_xmls:
        return None

    ms_values: list[float] = []
    for xml in lsm_xmls:
        ms_values.extend(_series_interval_ms_values_from_lsmimage_xml(xml))
    # Line-scan intervals are single-digit ms; exclude unrelated frame-rate values.
    candidates = [v for v in ms_values if v > 0.0 and v < 30.0]
    if not candidates:
        return None
    return min(candidates) / 1000.0


def _line_scan_x_step_um_from_scene(scene: Any) -> float | None:
    """Return micrometers-per-pixel for line-scan ``X`` from oirfile pixel length.

    ``oirfile.coord_scales['X']`` is often ``pixel_length / nX``. Olympus TXT
    exports report ``umPerPixel`` as the full per-pixel step (``_pixel_length_x``).

    Args:
        scene: Open ``oirfile.OirFile`` instance.

    Returns:
        Positive micrometers per pixel, or ``None`` when unavailable.
    """
    raw = getattr(scene, "_pixel_length_x", None)
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value) or value <= 0.0:
        return None
    return value


def _physical_units_for_oir_header(scene: Any) -> tuple[tuple[Any, ...], tuple[str, ...]]:
    """Return per-axis calibration for an open ``oirfile.OirFile``.

    Labels default to ``coord_units`` from ``oirfile``. Line-scan kymographs
    whose ``Y`` size matches an enabled TIMELAPSE axis are labeled ``seconds``
    and use LSMIMAGE ``seriesInterval`` (time) plus ``_pixel_length_x`` (space).

    Categorical channel axes (``C`` / ``S``) have no spatial step: ``oirfile``
    stores channel names in ``coords`` and omits them from ``coord_scales``.

    Args:
        scene: Open ``oirfile.OirFile`` instance.

    Returns:
        Tuple of ``(physical_units, physical_units_labels)`` aligned to
        ``scene.dims``. Channel dims use ``None`` step and empty label.
    """
    dims = tuple(str(d) for d in scene.dims)
    coord_units: dict[str, str] = dict(scene.coord_units)
    coord_scales: dict[str, float] = dict(scene.coord_scales)
    coords: Any | None = None
    y_is_timelapse_line_scan = _is_y_timelapse_line_scan_axis(scene)
    line_scan_y_step_s: float | None = None
    line_scan_x_step_um: float | None = None
    if y_is_timelapse_line_scan:
        line_scan_y_step_s = _line_scan_y_step_seconds_from_scene(scene)
        line_scan_x_step_um = _line_scan_x_step_um_from_scene(scene)
        if line_scan_y_step_s is None:
            logger.warning(
                'Line-scan OIR missing usable seriesInterval; falling back to coord_scales for Y'
            )
        if line_scan_x_step_um is None:
            logger.warning(
                'Line-scan OIR missing _pixel_length_x; falling back to coord_scales for X'
            )

    physical_units: list[Any] = []
    physical_units_labels: list[str] = []
    for dim in dims:
        if dim in ("C", "S"):
            physical_units.append(None)
            physical_units_labels.append("")
            continue

        if dim == "Y" and y_is_timelapse_line_scan and line_scan_y_step_s is not None:
            step: Any = line_scan_y_step_s
        elif dim == "X" and y_is_timelapse_line_scan and line_scan_x_step_um is not None:
            step = line_scan_x_step_um
        else:
            step = coord_scales.get(dim)
            if step is None:
                if coords is None:
                    coords = scene.coords
                step = _step_from_coord(coords.get(dim))
        physical_units.append(step)

        if dim == "Y" and y_is_timelapse_line_scan:
            label = "seconds"
        else:
            label = coord_units.get(dim, "")
            # TODO: possibly map to more meaningful display names
        physical_units_labels.append(label)

    return tuple(physical_units), tuple(physical_units_labels)


def _physical_units_for_header(scene: Any) -> tuple[tuple[Any, ...], tuple[str, ...]]:
    oir = scene
    coords = oir.coords
    n = len(coords)
    physical_units: list[Any] = [None] * n
    physical_units_labels = [""] * n
    for idx, coord_str in enumerate(coords):
        physical_units[idx] = _step_from_coord(coords[coord_str])
        physical_units_labels[idx] = coord_str
    return tuple(physical_units), tuple(physical_units_labels)

def _image_header_from_scene(
    path: str,
    scene: Any,
    num_scenes: int,
) -> ImageHeader:
    """Build an :class:`ImageHeader` from a czifile/oirfile scene-like object.

    Args:
        path: Absolute or resolved path stored on the header.
        scene: Object with ``shape``, ``dims``, ``sizes``, and ``dtype``.
        num_scenes: Total scene count for the file (OIR uses ``1``).

    Returns:
        Frozen :class:`ImageHeader` instance.
    """
    shape = tuple(int(v) for v in scene.shape)
    dims = tuple(str(d) for d in scene.dims)
    sizes = {str(k): int(v) for k, v in scene.sizes.items()}
    dtype = np.dtype(scene.dtype)
    num_channels = int(sizes["C"]) if "C" in sizes else 1
    physical_units, physical_units_labels = _physical_units_for_header(scene)
    date_s, time_s = _date_time_for_header(scene)
    return ImageHeader(
        path=path,
        shape=shape,
        dims=dims,
        sizes=sizes,
        dtype=dtype,
        num_channels=num_channels,
        num_scenes=num_scenes,
        physical_units=physical_units,
        physical_units_labels=physical_units_labels,
        date=date_s,
        time=time_s,
    )

def _date_time_for_header(scene: Any) -> tuple[str, str]:
    oir = scene
    raw = getattr(oir, "datetime", None)
    if raw is None:
        return ("", "")
    return _iso8601_datetime_str_to_yyyymmdd_hhmmss(str(raw))


def _image_header_from_oir_scene(
    path: str,
    scene: Any,
    num_scenes: int,
) -> ImageHeader:
    """Build an :class:`ImageHeader` from an open ``oirfile.OirFile``.

    Args:
        path: Absolute or resolved path stored on the header.
        scene: Open ``oirfile.OirFile`` instance.
        num_scenes: Total scene count for the file (OIR uses ``1``).

    Returns:
        Frozen :class:`ImageHeader` instance.
    """
    shape = tuple(int(v) for v in scene.shape)
    dims = tuple(str(d) for d in scene.dims)
    sizes = {str(k): int(v) for k, v in scene.sizes.items()}
    dtype = np.dtype(scene.dtype)
    num_channels = int(sizes["C"]) if "C" in sizes else 1
    physical_units, physical_units_labels = _physical_units_for_oir_header(scene)
    date_s, time_s = _date_time_for_header(scene)
    return ImageHeader(
        path=path,
        shape=shape,
        dims=dims,
        sizes=sizes,
        dtype=dtype,
        num_channels=num_channels,
        num_scenes=num_scenes,
        physical_units=physical_units,
        physical_units_labels=physical_units_labels,
        date=date_s,
        time=time_s,
    )


class OirFileLoader(BaseFileLoader):
    """Lazy-loading OIR reader aligned with :class:`MyCziImage`.

    Pixel data loads on demand. Supported OIR layouts follow the same CYX
    constraints as the helper functions in this module.
    """

    def __init__(self, path: str, header: ImageHeader | None = None) -> None:
        self._has_reference_image = False
        super().__init__(path, header)

    @contextmanager
    def _open_oir(self) -> Iterator[Any]:
        """Yield an ``oirfile.OirFile`` opened from :attr:`path` or :attr:`_stream`."""
        if self._stream is not None:
            self._stream.seek(0)
            with oirfile.OirFile(self._stream) as oir:
                yield oir
        else:
            with oirfile.OirFile(self.path) as oir:
                yield oir

    @property
    def has_reference_image(self) -> bool:
        """Return whether the OIR file has a reference attachment without decoding it.

        Returns:
            ``True`` when reference metadata was present during header read.
        """
        return self._has_reference_image

    @property
    def reference_image(self) -> ReferenceImage | None:
        """Return a frozen snapshot of the Olympus linescan reference image, if any.

        The snapshot is built while the OIR file is open and is safe to keep after close.

        Returns:
            :class:`ReferenceImageSnapshot`, or ``None`` when the file has no reference.
        """
        if self._referenceImage:
            return self._referenceImage

        with self._open_oir() as oir:
            ref = oir.reference
            if ref is None:
                self._referenceImage = None
            else:
                pixel_length_x = getattr(oir, "_pixel_length_x", None)
                pixel_length_y = getattr(oir, "_pixel_length_y", None)
                self._referenceImage = _reference_snapshot_from_oir_reference(
                    ref,
                    pixel_length_x=float(pixel_length_x) if pixel_length_x is not None else None,
                    pixel_length_y=float(pixel_length_y) if pixel_length_y is not None else None,
                )

        return self._referenceImage

    def read_header(self) -> ImageHeader:
        return self._read_oir_header()

    def _read_oir_header(self) -> ImageHeader:
        logical = self.path
        with self._open_oir() as oir_file:
            self._has_reference_image = oir_file.reference is not None
            return _image_header_from_oir_scene(logical, oir_file, num_scenes=1)

    def _load_full_image_array(self) -> np.ndarray:
        # logger.info('')
        with self._open_oir() as oir:
            return np.asarray(oir.asarray())
