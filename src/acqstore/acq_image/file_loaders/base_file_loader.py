from typing import Any
from dataclasses import dataclass, fields, replace
import math
from pathlib import Path
import numpy as np
from typing import BinaryIO, Self

from ..roi import ImageBounds, RectRoiBounds

@dataclass(frozen=True)
class ImageHeader:
    """Header information for an imported image file.

    Attributes:
        path: Filesystem path when loading from disk, or the upload **filename**
            when loading from a binary stream (not necessarily on disk).
        shape: Raw array shape in file/native order.
        dims: Dimension labels in file/native order.
        sizes: Mapping from dimension label to number of elements.
        dtype: NumPy dtype of the pixel data.
        num_channels: Number of channels in the image.
        num_scenes: Number of scenes in the file (OIR uses ``1``).
        physical_units: Per-axis calibration steps or placeholders from the file.
        physical_units_labels: Human-readable labels aligned with ``physical_units``.
        date: Acquisition calendar date as ``YYYYMMDD``, or ``""`` if unknown.
        time: Acquisition time-of-day as ``HH:MM:SS`` (24-hour), or ``""`` if unknown.
        file_size: Finder-style decimal file size string, such as ``"842 KB"``
            or ``"4.8 MB"``. Empty when the source is not an on-disk file.
    """

    path: str
    shape: tuple[int, ...]
    dims: tuple[str, ...]
    sizes: dict[str, int]
    dtype: np.dtype
    num_channels: int
    num_scenes: int
    physical_units: tuple[Any, ...]
    physical_units_labels: tuple[str, ...]
    date: str = ""
    time: str = ""
    file_size: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Return the header as a plain dictionary."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def as_json_dict(self) -> dict[str, Any]:
        """Return a dict suitable for :func:`json.dumps` (dtype as string, NaN/inf → None)."""
        d = self.as_dict()
        fixed_units: list[Any] = []
        for x in d["physical_units"]:
            if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
                fixed_units.append(None)
            else:
                fixed_units.append(x)
        return {
            "path": d["path"],
            "shape": list(d["shape"]),
            "dims": list(d["dims"]),
            "sizes": dict(d["sizes"]),
            "dtype": str(d["dtype"]),
            "num_channels": d["num_channels"],
            # "num_scenes": d["num_scenes"],
            "physical_units": fixed_units,
            "physical_units_labels": list(d["physical_units_labels"]),
            "date": d["date"],
            "time": d["time"],
            "file_size": d["file_size"],
        }

    def format_dims_display(self) -> str:
        """Return compact axis sizes for display, e.g. ``C:2 Y:100 X:200``.

        Returns:
            Space-separated ``dim:size`` tokens in :attr:`dims` order.
        """
        return ' '.join(f'{dim}:{self.sizes[dim]}' for dim in self.dims if dim in self.sizes)

    @staticmethod
    def default_physical_for_dims(
        dims: tuple[str, ...],
    ) -> tuple[tuple[float, ...], tuple[str, ...]]:
        """Default per-axis calibration when unknown (e.g. TIFF).

        Args:
            dims: Axis labels; one value per axis.

        Returns:
            ``physical_units`` all ``1.0`` and ``physical_units_labels`` all ``\"Pixels\"``.
        """
        n = len(dims)
        units = tuple(1.0 for _ in range(n))
        labels = tuple("Pixels" for _ in range(n))
        return units, labels

    def with_coerced_physical_calibration(self) -> "ImageHeader":
        """Return a copy with physical units and labels aligned to :attr:`dims`.

        This is the canonical place where invalid or missing per-axis calibration
        from the file is mapped to defaults (finite ``> 0`` steps and non-empty
        labels). Callers such as :class:`BaseFileLoader` apply this so GUI and
        other layers do not duplicate silent coercion.

        Rules:
            - ``physical_units`` has length ``len(dims)``. Missing indices are
              treated as invalid and become ``1.0``.
            - Each step is coerced to ``float``; non-finite values or ``<= 0``
              become ``1.0``.
            - ``physical_units_labels`` has length ``len(dims)``; empty strings
              become ``\"Pixels\"``.

        Returns:
            ``self`` if already normalized; otherwise a new :class:`ImageHeader`.
        """
        n = len(self.dims)
        raw_u = list(self.physical_units)
        raw_l = list(self.physical_units_labels)
        while len(raw_u) < n:
            raw_u.append(None)
        while len(raw_l) < n:
            raw_l.append("")
        new_units: list[float] = []
        new_labels: list[str] = []
        for i in range(n):
            u = raw_u[i]
            try:
                v = float(u)
                if math.isnan(v) or math.isinf(v) or v <= 0.0:
                    v = 1.0
            except (TypeError, ValueError):
                v = 1.0
            new_units.append(v)
            lab = raw_l[i] if i < len(raw_l) else ""
            new_labels.append(str(lab) if lab else "Pixels")
        new_units_t = tuple(new_units)
        new_labels_t = tuple(new_labels)
        if (
            len(self.physical_units) == n
            and len(self.physical_units_labels) == n
            and new_units_t == self.physical_units
            and new_labels_t == self.physical_units_labels
        ):
            return self
        return replace(
            self,
            physical_units=new_units_t,
            physical_units_labels=new_labels_t,
        )

    def _physical_step_for_dim(self, dim: str) -> float | None:
        """Return a finite positive calibration step for ``dim``, or ``None`` if unknown."""
        if dim not in self.dims:
            return None
        i = self.dims.index(dim)
        if i >= len(self.physical_units):
            return None
        u = self.physical_units[i]
        if u is None:
            return None
        if isinstance(u, float) and (math.isnan(u) or math.isinf(u)):
            return None
        try:
            v = float(u)
        except (TypeError, ValueError):
            return None
        if v <= 0.0 or math.isnan(v) or math.isinf(v):
            return None
        return v

    def _physical_label_for_dim(self, dim: str) -> str:
        if dim not in self.dims:
            return dim
        i = self.dims.index(dim)
        if i < len(self.physical_units_labels):
            lab = self.physical_units_labels[i]
            if lab:
                return str(lab)
        return dim



def format_file_size(path: str | Path) -> str:
    """Return a Finder-style decimal file size string for an on-disk path.

    Args:
        path: File or directory path. Directory sizes are intentionally not
            expanded recursively; stores such as ``.cs.ome.zarr`` return an
            empty string unless the platform reports a direct size.

    Returns:
        Display string such as ``"842 KB"`` or ``"4.8 MB"``. Returns ``""``
        if size cannot be read.
    """
    try:
        size = Path(path).stat().st_size
    except OSError:
        return ""
    if size < 0:
        return ""
    units = ("bytes", "KB", "MB", "GB", "TB")
    value = float(size)
    unit_index = 0
    while value >= 1000.0 and unit_index < len(units) - 1:
        value /= 1000.0
        unit_index += 1
    if unit_index == 0:
        suffix = "byte" if size == 1 else "bytes"
        return f"{size} {suffix}"
    if value >= 10.0:
        return f"{value:.0f} {units[unit_index]}"
    return f"{value:.1f} {units[unit_index]}"

@dataclass(frozen=True, eq=False)
class ReferenceImagePlane:
    """Display-ready two-dimensional reference image plane.

    The plane is intentionally independent of GUI frameworks. Viewers can use
    ``array`` directly and scale the row/column axes with ``dx`` and ``dy``.
    As in :class:`RasterGridSpec`, ``dx`` is the physical step for row/Y
    indices and ``dy`` is the physical step for column/X indices.

    Args:
        array: Two-dimensional image plane in ``(Y, X)`` order.
        dx: Physical step per row/Y index.
        dy: Physical step per column/X index.
        x_unit: Unit label for the row/Y axis.
        y_unit: Unit label for the column/X axis.
    """

    array: np.ndarray
    dx: float
    dy: float
    x_unit: str
    y_unit: str

    def __post_init__(self) -> None:
        """Validate the reference plane.

        Raises:
            ValueError: If ``array`` is not two-dimensional or spacing is not
                positive and finite.
        """
        if self.array.ndim != 2:
            raise ValueError(f"ReferenceImagePlane.array must be 2D, got shape={self.array.shape}")
        for name, value in (("dx", self.dx), ("dy", self.dy)):
            v = float(value)
            if v <= 0.0 or math.isnan(v) or math.isinf(v):
                raise ValueError(f"{name} must be positive and finite, got {value!r}")


@dataclass(frozen=True, eq=False)
class ReferenceImage:
    """Immutable snapshot of a microscopy reference/overview image.

    ``array`` and every ndarray in ``coords`` are **read-only views** where
    possible; callers must not mutate them. ``eq=False`` avoids elementwise
    array equality on ``==``.
    """

    array: np.ndarray
    dims: tuple[str, ...]
    num_channels: int
    line_roi: tuple[float, float, float, float] | None
    coord_units: tuple[tuple[str, str], ...]
    coord_scales: tuple[tuple[str, float], ...]
    coords: tuple[tuple[str, np.ndarray], ...]
    scan_path: np.ndarray | None = None

    def has_scan_path(self) -> bool:
        """Return whether this reference image has a plot-ready scan path.

        Returns:
            ``True`` when :attr:`scan_path` contains a two-row path that can be
            plotted over reference-image planes.
        """
        return self.scan_path is not None

    def get_scan_path(self) -> np.ndarray | None:
        """Return the plot-ready scan path associated with this reference image.

        The returned array uses shape ``(2, N)`` where row ``0`` contains X
        pixel coordinates and row ``1`` contains Y pixel coordinates in the
        reference-image plotting coordinate system. For formats with sparse
        line metadata, such as OIR, ``N`` can be ``2``. For CZI spline-like
        scanner trajectories, ``N`` can be much larger.

        Returns:
            Read-only ``(2, N)`` scan path array, or ``None`` when no scan path
            metadata is available.
        """
        if self.scan_path is None:
            return None
        scan_path = np.asarray(self.scan_path)
        scan_path.setflags(write=False)
        return scan_path

    def get_scan_path_plot(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return scan path coordinates for direct Matplotlib plotting.

        Returns:
            ``(x_pixels, y_pixels)`` arrays, or ``None`` when no scan path is
            available.

        Raises:
            ValueError: If :attr:`scan_path` is not shaped as ``(2, N)``.
        """
        scan_path = self.get_scan_path()
        if scan_path is None:
            return None
        if scan_path.ndim != 2 or scan_path.shape[0] != 2:
            raise ValueError(f"scan_path must have shape (2, N), got shape={scan_path.shape}")
        x_pixels = np.asarray(scan_path[0])
        y_pixels = np.asarray(scan_path[1])
        x_pixels.setflags(write=False)
        y_pixels.setflags(write=False)
        return x_pixels, y_pixels

    def get_plane(self, channel: int | None = None) -> ReferenceImagePlane:
        """Return a display-ready 2D reference-image plane.

        Args:
            channel: Optional channel index. When the reference image has a
                channel dimension and ``channel`` is ``None``, channel ``0`` is
                used. When there is no channel dimension, ``channel`` may be
                ``None`` or ``0``.

        Returns:
            Display-ready reference image plane in ``(Y, X)`` order.

        Raises:
            ValueError: If ``channel`` is invalid.
            NotImplementedError: If the reference image layout cannot be mapped
                to one two-dimensional ``(Y, X)`` plane.
        """
        data, dims = self._select_channel(channel)
        plane = self._yx_plane_from_array(data, dims)
        dx = self._scale_for_dim("Y")
        dy = self._scale_for_dim("X")
        x_unit = self._unit_for_dim("Y")
        y_unit = self._unit_for_dim("X")
        return ReferenceImagePlane(array=plane, dx=dx, dy=dy, x_unit=x_unit, y_unit=y_unit)

    def _select_channel(self, channel: int | None) -> tuple[np.ndarray, tuple[str, ...]]:
        """Return ``(array, dims)`` after applying channel selection."""
        if "C" not in self.dims:
            if channel not in (None, 0):
                raise ValueError(f"Reference image has no channel axis; got channel={channel!r}")
            return self.array, self.dims

        c_axis = self.dims.index("C")
        n_channels = int(self.array.shape[c_axis])
        selected = 0 if channel is None else int(channel)
        if selected < 0 or selected >= n_channels:
            raise ValueError(
                f"Reference image channel {selected} is out of range for {n_channels} channels"
            )
        data = np.take(self.array, selected, axis=c_axis)
        dims = tuple(dim for dim in self.dims if dim != "C")
        return data, dims

    def _yx_plane_from_array(self, data: np.ndarray, dims: tuple[str, ...]) -> np.ndarray:
        """Return a read-only ``(Y, X)`` plane from ``data`` and ``dims``."""
        if "Y" not in dims or "X" not in dims:
            raise NotImplementedError(f"Reference image dims must include Y and X, got dims={dims!r}")
        if data.ndim != len(dims):
            raise NotImplementedError(
                f"Reference image data rank {data.ndim} does not match dims={dims!r}"
            )

        # Allow extra singleton dimensions such as Z/T of length 1, but fail
        # fast on non-singleton axes because choosing those planes is backend
        # domain logic that should be explicitly implemented when needed.
        slicer: list[object] = []
        kept_dims: list[str] = []
        for axis, dim in enumerate(dims):
            if dim in ("Y", "X"):
                slicer.append(slice(None))
                kept_dims.append(dim)
            elif data.shape[axis] == 1:
                slicer.append(0)
            else:
                raise NotImplementedError(
                    f"Unsupported non-singleton reference image axis {dim!r} with size {data.shape[axis]}"
                )
        plane = np.asarray(data[tuple(slicer)])
        if tuple(kept_dims) != ("Y", "X"):
            axes = [kept_dims.index("Y"), kept_dims.index("X")]
            plane = np.moveaxis(plane, axes, [0, 1])
        if plane.ndim != 2:
            raise NotImplementedError(f"Expected 2D reference plane, got shape={plane.shape}")
        plane.setflags(write=False)
        return plane

    def _scale_for_dim(self, dim: str) -> float:
        """Return physical scale for ``dim`` or pixel scale when unavailable."""
        mapping = dict(self.coord_scales)
        value = mapping.get(dim)
        if value is None:
            return 1.0
        scale = float(value)
        if scale <= 0.0 or math.isnan(scale) or math.isinf(scale):
            return 1.0
        return scale

    def _unit_for_dim(self, dim: str) -> str:
        """Return unit label for ``dim`` or ``Pixels`` when unavailable."""
        unit = dict(self.coord_units).get(dim)
        if unit:
            return str(unit)
        return "Pixels"

    def get_line_roi(self) -> tuple[float, float, float, float] | None:
        """Return the reference line ROI when available.

        Returns:
            ``(x0, y0, x1, y1)`` tuple, or None when the reference image has no
            line ROI metadata.
        """
        return self.line_roi

class BaseFileLoader:
    """Base class for lazy-loading image readers with a shared header and pixel API.
    """

    def __init__(self, path: str, header: ImageHeader | None = None) -> None:
        """Load metadata from disk unless a precomputed header is supplied.

        Args:
            path: Filesystem path to the image file.
            header: Optional pre-built header; if omitted, :meth:`read_header` is used.
        """
        self.path = path
        self._stream: BinaryIO | None = None
        self._post_init(header)

    @classmethod
    def from_stream(
        cls,
        stream: BinaryIO,
        filename: str,
        header: ImageHeader | None = None,
    ) -> Self:  # Use 'Self' for the return type (from typing)
        """Generic factory to create any loader from a stream."""
        inst = object.__new__(cls)
        inst.path = filename
        inst._stream = stream
        inst._post_init(header)    
        return inst

    def _post_init(self, header: ImageHeader | None) -> None:
        """Centralized state initialization for both __init__ and from_stream."""
        self._img_data = None
        self._referenceImage = None
        
        # Look up '_squeeze' on the class or instance; default to True
        self._squeeze = getattr(self, "_squeeze", True)
        
        # Ensure the header is always initialized and physical calibration coerced.
        if header is None:
            self._header = self.read_header().with_coerced_physical_calibration()
        else:
            self._header = header.with_coerced_physical_calibration()
        if not self._header.file_size and self._stream is None:
            self._header = replace(self._header, file_size=format_file_size(self.path))

    def replace_header(self, header: ImageHeader) -> None:
        """Replace the loader header at runtime (e.g. edited calibration).

        The only header fields expected to change after load are
        ``physical_units`` and ``physical_units_labels``. The replacement is
        normalized via :meth:`ImageHeader.with_coerced_physical_calibration`.

        Args:
            header: New header; must refer to the same logical file as this loader.

        Raises:
            ValueError: If ``header.path`` does not match this loader's ``path``.
        """
        if header.path != self.path:
            raise ValueError(
                f"Header path {header.path!r} does not match loader path {self.path!r}"
            )
        self._header = header.with_coerced_physical_calibration()

    @property
    def header(self) -> ImageHeader:
        """Header read at construction or injected."""
        return self._header

    @property
    def has_reference_image(self) -> bool:
        """Return whether this file has a reference image.

        Subclasses may override with a cheaper metadata-only probe. The default
        loads or inspects :attr:`reference_image`.

        Returns:
            ``True`` when a reference image is available.
        """
        return self.reference_image is not None

    @property
    def reference_image(self) -> ReferenceImage | None:
        """Return a reference image.
        """
        return self._referenceImage

    def read_header(self) -> ImageHeader:
        """Read header metadata for this loader's path.

        Returns:
            Parsed :class:`ImageHeader`.

        Raises:
            NotImplementedError: If the concrete subclass does not implement this.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement read_header()")

    @classmethod
    def read_header_from_path(cls, path: str) -> ImageHeader:
        """Read header from disk without keeping the loader instance.

        Args:
            path: Filesystem path to the image file.

        Returns:
            Parsed :class:`ImageHeader`.
        """
        return cls(path).header

    @classmethod
    def read_header_from_stream(cls, stream: BinaryIO, filename: str) -> ImageHeader:
        """Read header from a stream without retaining the loader instance."""
        return cls.from_stream(stream, filename).header

    def _load_full_image_array(self) -> np.ndarray:
        """Load the full pixel array for this loader's active scene(s).

        Returns:
            NumPy array in native dimension order.

        Raises:
            NotImplementedError: If the subclass does not implement loading.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _load_full_image_array()"
        )

    def pixels_loaded(self) -> bool:
        """Return whether the full pixel volume is cached in memory.

        Returns:
            ``True`` after :meth:`load_image_data` has completed at least once
            without a subsequent :meth:`unload_image_data`.
        """
        return self._img_data is not None

    def load_image_data(self) -> np.ndarray:
        """Load and cache the full image array (lazy, idempotent).

        Returns:
            The full pixel array for this file/scene.
        """
        if self._img_data is None:
            self._img_data = self._load_full_image_array()
        return self._img_data

    def load_pixels(self):
        """Return an :class:`AcqPixels` wrapper for this loader.

        The default implementation uses the current eager NumPy loading path.
        Directory-backed OME-Zarr loaders can override this to keep returned
        pixels backed by lazy Zarr/Dask-compatible arrays.
        """
        from ..acq_pixels import AcqPixels

        return AcqPixels(
            data=self.load_image_data(),
            header=self._header,
            source_path=self.path,
            acquisition_metadata={},
            source_metadata={"loader": type(self).__name__},
        )

    def unload_image_data(self) -> None:
        """Drop cached primary pixel data; header unchanged."""
        self._img_data = None

    def unload_reference_data(self) -> None:
        """Drop reference-image caches."""
        self._referenceImage = None

    def unload_all_cached_data(self) -> None:
        """Clear primary pixels and reference caches.

        Notes:
            CloudScope lazy image unload calls :meth:`unload_image_data` directly
            so reference images remain available after primary pixels are
            unloaded. Use this method only when all loader caches should be
            discarded.
        """
        self.unload_image_data()
        self.unload_reference_data()

    @staticmethod
    def _format_missing_channel_axis_error(
        *,
        channel: int,
        source_path: str | None,
        num_channels: int,
        dims: tuple[str, ...],
        array_shape: tuple[int, ...],
    ) -> str:
        """Build a compact error when the header claims channels but volume has no ``C`` axis."""
        path_part = f" from {source_path!r}" if source_path else ""
        return (
            f"Cannot slice ch {channel}{path_part}: num_channels={num_channels}, "
            f"dims={dims}, loaded shape {array_shape}, no C axis "
            f"(Olympus split-channel siblings may be missing)."
        )

    @staticmethod
    def yx_slice_from_volume(
        arr: np.ndarray,
        dims: tuple[str, ...],
        num_channels: int,
        channel: int,
        *,
        z: int = 0,
        t: int = 0,
        source_path: str | None = None,
        array_shape: tuple[int, ...] | None = None,
    ) -> np.ndarray:
        """Select ``T`` / ``Z`` / ``C`` and return a 2D ``(Y, X)`` view or copy of ``arr``."""
        if channel < 0 or channel >= num_channels:
            raise IndexError(
                f"Channel index {channel} out of range for {num_channels} channels"
            )
        dims_list = list(dims)
        if "Y" not in dims_list or "X" not in dims_list:
            raise ValueError(f"Expected dims to include Y and X; got dims={dims}")
        if "C" not in dims_list:
            if num_channels != 1:
                if source_path is not None or array_shape is not None:
                    shape = (
                        tuple(int(x) for x in array_shape)
                        if array_shape is not None
                        else tuple(int(x) for x in arr.shape)
                    )
                    raise ValueError(
                        BaseFileLoader._format_missing_channel_axis_error(
                            channel=channel,
                            source_path=source_path,
                            num_channels=num_channels,
                            dims=dims,
                            array_shape=shape,
                        )
                    )
                raise ValueError(
                    f"No C axis in dims but num_channels={num_channels}"
                )
            if channel != 0:
                raise IndexError(f"No C axis; only channel 0 is valid, got {channel}")

        arr_work = arr

        def _take_and_drop(dim_label: str, index: int) -> None:
            nonlocal arr_work, dims_list
            if dim_label not in dims_list:
                return
            axis = dims_list.index(dim_label)
            size = arr_work.shape[axis]
            if index < 0 or index >= size:
                raise IndexError(
                    f"{dim_label} index {index} out of range for size {size} (dims={tuple(dims_list)})"
                )
            arr_work = np.take(arr_work, indices=index, axis=axis)
            dims_list.pop(axis)

        _take_and_drop("T", t)
        _take_and_drop("Z", z)
        if "C" in dims_list:
            _take_and_drop("C", channel)

        if tuple(dims_list) != ("Y", "X"):
            raise ValueError(
                f"After selecting T/Z/C, expected remaining dims (Y, X); got {tuple(dims_list)} "
                f"with shape {arr_work.shape}"
            )

        return arr_work

    def get_slice_data_loaded(self, channel: int, z: int = 0, t: int = 0) -> np.ndarray:
        """Return a 2D ``(Y, X)`` slice from already-loaded pixel data.

        Unlike :meth:`get_slice_data`, this method does not read from disk.

        Args:
            channel: Channel index along ``C`` (or ``0`` when there is no ``C`` axis).
            z: Index along ``Z`` if present; ignored if ``Z`` is absent.
            t: Index along ``T`` if present; ignored if ``T`` is absent.

        Returns:
            Two-dimensional array with dimensions ``(Y, X)``.

        Raises:
            RuntimeError: If pixel data has not been loaded yet.
            IndexError: If any index is out of range.
            ValueError: If dims cannot be reduced to ``(Y, X)``.
        """
        if self._img_data is None:
            raise RuntimeError(
                'Image pixels are not loaded; call load_image_data() first.'
            )
        return self.yx_slice_from_volume(
            self._img_data,
            self._header.dims,
            self._header.num_channels,
            channel,
            z=z,
            t=t,
            source_path=self.path,
            array_shape=tuple(int(x) for x in self._img_data.shape),
        )

    def get_slice_data(self, channel: int, z: int = 0, t: int = 0) -> np.ndarray:
        """Return a single 2D ``(Y, X)`` slice after optional ``T``/``Z``/``C`` selection.

        Loads pixel data on first use.

        Args:
            channel: Channel index along ``C`` (or ``0`` when there is no ``C`` axis).
            z: Index along ``Z`` if present; ignored if ``Z`` is absent.
            t: Index along ``T`` if present; ignored if ``T`` is absent.

        Returns:
            Two-dimensional array with dimensions ``(Y, X)``.

        Raises:
            IndexError: If any index is out of range.
            ValueError: If dims cannot be reduced to ``(Y, X)``.
        """
        self.load_image_data()
        assert self._img_data is not None
        return self.yx_slice_from_volume(
            self._img_data,
            self._header.dims,
            self._header.num_channels,
            channel,
            z=z,
            t=t,
            source_path=self.path,
            array_shape=tuple(int(x) for x in self._img_data.shape),
        )

    def get_image_physical_units(self) -> tuple[float, float]:
        """Return per-pixel physical step along ``Y`` then ``X`` for 2D ``(Y, X)`` arrays.

        Values are taken from :attr:`header` after calibration coercion (invalid
        steps become ``1.0``). The tuple matches the axis order of
        :meth:`get_slice_data` (row spacing, then column spacing).

        Returns:
            ``(step_y, step_x)`` for the image plane.

        Raises:
            ValueError: If ``Y`` or ``X`` is missing from :attr:`header.dims`.
        """
        dims = self._header.dims
        if 'Y' not in dims or 'X' not in dims:
            raise ValueError(
                f'Expected header dims to include Y and X for 2D plane calibration; got dims={dims!r}'
            )
        i_y = dims.index('Y')
        i_x = dims.index('X')
        uy = float(self._header.physical_units[i_y])
        ux = float(self._header.physical_units[i_x])
        return (uy, ux)

    def get_roi_rect_image(
        self,
        channel: int,
        bounds: RectRoiBounds,
        *,
        z: int = 0,
        t: int = 0,
    ) -> np.ndarray:
        """Return a 2D ``(Y, X)`` crop of one slice for ``channel`` using rectangular bounds.

        Loads pixel data on first use. ``bounds`` are clamped to the slice shape.

        Args:
            channel: Channel index along ``C`` (or ``0`` when there is no ``C`` axis).
            bounds: ROI bounds in row/column indices (exclusive stops).
            z: Index along ``Z`` if present; ignored if ``Z`` is absent.
            t: Index along ``T`` if present; ignored if ``T`` is absent.

        Returns:
            Two-dimensional crop with dimensions ``(Y, X)``.

        Raises:
            IndexError: If ``channel``, ``z``, or ``t`` is out of range.
            ValueError: If dims cannot be reduced to ``(Y, X)``.
        """
        slice_2d = self.get_slice_data(channel, z=z, t=t)
        img_bounds = ImageBounds(
            width=int(slice_2d.shape[1]),
            height=int(slice_2d.shape[0]),
            num_slices=1,
        )
        b = bounds.clamped_to(img_bounds)
        return slice_2d[b.dim0_start : b.dim0_stop, b.dim1_start : b.dim1_stop]

    def get_channel_data(self, channel: int) -> np.ndarray:
        """Return the full array for one channel (``C`` axis removed).

        Loads pixel data on first use. Remaining dimensions stay in file order
        (e.g. ``T, Y, X`` or ``Z, Y, X``).

        Args:
            channel: Index along the ``C`` dimension, or ``0`` when there is no ``C`` axis.

        Returns:
            NumPy array without the channel axis.

        Raises:
            IndexError: If ``channel`` is invalid.
            ValueError: If the header implies multiple channels but there is no ``C`` dim.
        """
        self.load_image_data()
        if channel < 0 or channel >= self._header.num_channels:
            raise IndexError(
                f"Channel index {channel} out of range for {self._header.num_channels} channels"
            )

        if "C" not in self._header.dims:
            if self._header.num_channels != 1:
                assert self._img_data is not None
                raise ValueError(
                    self._format_missing_channel_axis_error(
                        channel=channel,
                        source_path=self.path,
                        num_channels=self._header.num_channels,
                        dims=self._header.dims,
                        array_shape=tuple(int(x) for x in self._img_data.shape),
                    )
                )
            if channel != 0:
                raise IndexError(f"No C axis; only channel 0 is valid, got {channel}")
            assert self._img_data is not None
            return self._img_data

        assert self._img_data is not None
        c_axis = self._header.dims.index("C")
        return np.take(self._img_data, indices=channel, axis=c_axis)

    # --- AcqImagesProtocol Implementation ---

    @property
    def num_channels(self) -> int:
        """Conforms to AcqImagesProtocol."""
        return self._header.num_channels

    @property
    def channel_indices(self) -> list[int]:
        """Conforms to AcqImagesProtocol."""
        return list(range(self.num_channels))

    @property
    def default_channel(self) -> int | None:
        """Conforms to AcqImagesProtocol. 
        Defaults to 0 if channels exist, otherwise None.
        """
        return 0 if self.num_channels > 0 else None

    def get_channel_image(self, channel: int) -> np.ndarray:
        """Conforms to AcqImagesProtocol using existing backend logic."""
        try:
            # Re-uses your existing, validated channel loader
            return self.get_channel_data(channel)
        except IndexError as e:
            raise KeyError(f"Channel {channel} not found") from e

    def get_reference_image(self, channel: int | None = None) -> np.ndarray | None:
        """Return a display-ready reference-image plane array when available.

        Args:
            channel: Optional reference image channel.

        Returns:
            Two-dimensional reference image plane, or None when no reference
            image exists.
        """
        ref = self.reference_image
        if ref is None:
            return None
        return ref.get_plane(channel).array
