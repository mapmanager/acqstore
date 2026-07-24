from __future__ import annotations  # treat all type hints as strings

from contextlib import contextmanager
from collections.abc import Iterator
from dataclasses import replace
from typing import Any, BinaryIO
import numpy as np
import tifffile
from typing import Self

from .base_file_loader import BaseFileLoader, ImageHeader
from .read_olympus_txt import read_olympus_txt_dict, _image_header_from_olympus_dict
from acqstore.utils.logging import get_logger

logger = get_logger(__name__)


def _tif_dims_from_ndim(ndim: int) -> tuple[str, ...]:
    """Map TIFF array rank to axis labels when tifffile axes are unavailable."""
    if ndim == 2:
        return ("Y", "X")
    if ndim == 3:
        return ("Z", "Y", "X")
    if ndim == 4:
        return ("C", "Z", "Y", "X")
    if ndim == 5:
        return ("T", "C", "Z", "Y", "X")
    raise ValueError(f"Unsupported TIFF ndim {ndim}; expected 2, 3, 4, or 5")


def _tif_dims_from_series_axes(axes: str, shape: tuple[int, ...]) -> tuple[str, ...]:
    """Return CloudScope axis labels from a tifffile series axes string.

    Args:
        axes: Axis string reported by ``tifffile.TiffPageSeries.axes``.
        shape: Series shape reported by ``tifffile.TiffPageSeries.shape``.

    Returns:
        Dimension labels aligned to ``shape``. Known microscopy axes are used
        directly. Sample axes (``S``) are mapped to CloudScope channels
        (``C``). Unknown or mismatched axes fall back to the historical
        rank-based TIFF policy.
    """
    axes_tuple = tuple(str(a) for a in axes)
    allowed = {"T", "C", "Z", "Y", "X"}

    if len(axes_tuple) == len(shape) and all(a in allowed for a in axes_tuple):
        return axes_tuple

    if len(axes_tuple) == len(shape) and "S" in axes_tuple:
        mapped = tuple("C" if a == "S" else a for a in axes_tuple)
        if all(a in allowed for a in mapped):
            return mapped

    return _tif_dims_from_ndim(len(shape))


def _ordered_olympus_split_channel_paths(
    tif_channel_paths: dict[int, Any],
    num_channels: int,
) -> tuple[str, ...] | None:
    """Return ordered on-disk paths when every Olympus channel file is present.

    Args:
        tif_channel_paths: 1-based channel index to path mapping from
            :func:`~read_olympus_txt.read_olympus_txt_dict`.
        num_channels: Channel count reported by the Olympus sidecar.

    Returns:
        ``(path_ch1, path_ch2, ...)`` when ``num_channels > 1`` and every
        expected path exists; otherwise ``None``.
    """
    if num_channels <= 1:
        return None
    ordered: list[str] = []
    for channel_number in range(1, num_channels + 1):
        path = tif_channel_paths.get(channel_number)
        if path is None:
            return None
        ordered.append(str(path))
    return tuple(ordered)


def _header_for_olympus_split_channels(
    base: ImageHeader,
    num_channels: int,
) -> ImageHeader:
    """Upgrade a kymograph Olympus header to ``(C, Y, X)`` for split-channel TIFFs."""
    y_size = int(base.sizes["Y"])
    x_size = int(base.sizes["X"])
    step_y = float(base.physical_units[base.dims.index("Y")])
    step_x = float(base.physical_units[base.dims.index("X")])
    label_y = str(base.physical_units_labels[base.dims.index("Y")])
    label_x = str(base.physical_units_labels[base.dims.index("X")])
    return replace(
        base,
        shape=(num_channels, y_size, x_size),
        dims=("C", "Y", "X"),
        sizes={"C": num_channels, "Y": y_size, "X": x_size},
        num_channels=num_channels,
        physical_units=(1.0, step_y, step_x),
        physical_units_labels=("Pixels", label_y, label_x),
    )


class TiffFileLoader(BaseFileLoader):
    """Lazy-loading TIFF reader using ``tifffile`` metadata and pixels.

    Header construction reads Olympus sidecar metadata when available, otherwise
    it inspects ``tifffile.TiffFile(...).series[0]``. This path reads TIFF
    metadata only and does not materialize pixel arrays. Pixel data are loaded
    later through :meth:`BaseFileLoader.load_image_data`.

    When an Olympus sidecar reports multiple channels stored as separate
    ``_C001T…`` / ``_C002T…`` TIFF siblings, this loader merges them into one
    ``(C, Y, X)`` volume on first pixel load.

    Args:
        path: Filesystem path to the TIFF.
        header: Optional pre-built header (e.g. from catalog Olympus parse).
        load_olympus_header: When true (default), look for Olympus ``.txt`` next
            to the TIFF on disk. Ignored when ``header`` is provided.
    """

    def __init__(
        self,
        path: str,
        header: ImageHeader | None = None,
        *,
        load_olympus_header: bool = True,
    ) -> None:
        self._load_olympus_header = load_olympus_header
        self._olympus_channel_paths: tuple[str, ...] | None = None
        super().__init__(path, header)

    @contextmanager
    def _open_tif(self) -> Iterator[BinaryIO | str]:
        """Yield :attr:`path` or rewound :attr:`_stream` for tifffile readers."""
        if self._stream is not None:
            self._stream.seek(0)
            yield self._stream
        else:
            yield self.path

    @classmethod
    def from_stream(
        cls,
        stream: BinaryIO,
        filename: str,
        header: ImageHeader | None = None,
        *,
        load_olympus_header: bool = False,
    ) -> Self:
        """Create a TIFF loader from a binary stream.

        Args:
            stream: Binary stream positioned at or containing a TIFF file.
            filename: Logical filename to keep in the header.
            header: Optional pre-built image header.
            load_olympus_header: Ignored for streams by default because an
                Olympus sidecar path is not available.

        Returns:
            TIFF loader with header metadata initialized and pixels unloaded.
        """
        inst = object.__new__(cls)
        inst.path = filename
        inst._stream = stream
        inst._load_olympus_header = load_olympus_header
        inst._olympus_channel_paths = None
        inst._post_init(header)
        return inst

    def read_header(self) -> ImageHeader:
        """Read TIFF header metadata without loading pixel data."""
        return self._read_tif_header()

    def _read_tif_array(self) -> np.ndarray:
        """Load the full TIFF pixel array."""
        with self._open_tif() as src:
            return np.asarray(tifffile.imread(src))

    def _read_olympus_split_channel_array(self) -> np.ndarray:
        """Load and stack Olympus split-channel sibling TIFFs into ``(C, Y, X)``."""
        if self._olympus_channel_paths is None:
            raise RuntimeError("Olympus split-channel paths are not initialized")
        planes: list[np.ndarray] = []
        y_size: int | None = None
        x_size: int | None = None
        for index, channel_path in enumerate(self._olympus_channel_paths):
            plane = np.asarray(tifffile.imread(channel_path))
            if plane.ndim != 2:
                raise ValueError(
                    f"Olympus split-channel TIFF must be 2D (Y, X); "
                    f"channel {index} at {channel_path!r} has shape {plane.shape}"
                )
            if y_size is None:
                y_size = int(plane.shape[0])
                x_size = int(plane.shape[1])
            elif plane.shape != (y_size, x_size):
                raise ValueError(
                    "Olympus split-channel TIFF shape mismatch: "
                    f"channel 0 is {(y_size, x_size)}, channel {index} at "
                    f"{channel_path!r} is {tuple(int(x) for x in plane.shape)}"
                )
            planes.append(plane)
        return np.stack(planes, axis=0)

    def _read_tif_header(self) -> ImageHeader:
        """Read TIFF header metadata using sidecar or tifffile series metadata."""
        self._olympus_channel_paths = None
        if self._stream is None and self._load_olympus_header:
            odict = read_olympus_txt_dict(self.path)
            if odict is not None:
                try:
                    base = _image_header_from_olympus_dict(self.path, odict)
                    num_channels = int(odict.get("numChannels", base.num_channels))
                    tif_channel_paths = odict.get("tifChannelPaths") or {}
                    ordered = _ordered_olympus_split_channel_paths(
                        tif_channel_paths,
                        num_channels,
                    )
                    if ordered is not None:
                        self._olympus_channel_paths = ordered
                        return _header_for_olympus_split_channels(
                            base,
                            len(ordered),
                        )
                    return base
                except (ValueError, TypeError, KeyError) as exc:
                    logger.warning(
                        "Olympus txt present but ImageHeader build failed, "
                        "falling back to tifffile metadata: %s",
                        exc,
                    )
        return self._read_tif_header_from_series()

    def _read_tif_header_from_series(self) -> ImageHeader:
        """Build an :class:`ImageHeader` from ``tifffile`` series metadata only."""
        with self._open_tif() as src:
            with tifffile.TiffFile(src) as tif:
                if not tif.series:
                    raise ValueError(f"TIFF file has no image series: {self.path}")
                series = tif.series[0]
                shape = tuple(int(x) for x in series.shape)
                dtype = np.dtype(series.dtype)
                axes = str(series.axes)
                num_scenes = int(len(tif.series))

        dims = _tif_dims_from_series_axes(axes, shape)
        sizes = {dims[i]: shape[i] for i in range(len(shape))}
        num_channels = int(sizes["C"]) if "C" in sizes else 1
        physical_units, physical_units_labels = ImageHeader.default_physical_for_dims(dims)
        return ImageHeader(
            path=self.path,
            shape=shape,
            dims=dims,
            sizes=sizes,
            dtype=dtype,
            num_channels=num_channels,
            num_scenes=num_scenes,
            physical_units=physical_units,
            physical_units_labels=physical_units_labels,
            date="",
            time="",
        )

    def _load_full_image_array(self) -> np.ndarray:
        """Load TIFF pixels and validate loaded shape against the cached header."""
        if self._img_data is not None:
            return self._img_data
        if self._olympus_channel_paths is not None:
            arr = self._read_olympus_split_channel_array()
        else:
            arr = self._read_tif_array()
        hdr = self._header
        if hdr is not None:
            expected = tuple(int(x) for x in hdr.shape)
            got = tuple(int(x) for x in arr.shape)
            if expected != got:
                raise ValueError(
                    f"TIFF array shape {got} does not match header shape {expected}"
                )
        return arr
