"""Tests for lazy Nikon ND2 loading through :class:`Nd2FileLoader`."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pytest

from acqstore.acq_image.file_loaders.nd2_file_loader import Nd2FileLoader


@dataclass(frozen=True)
class _FakeVoxelSize:
    """Minimal voxel-size object matching the attributes used by the loader."""

    x: float
    y: float
    z: float


class _FakeND2File:
    """Context-manager fake for ``nd2.ND2File``."""

    def __init__(
        self,
        _path: str,
        *,
        sizes: dict[str, int] | None = None,
        pixels: np.ndarray | None = None,
    ) -> None:
        self.sizes = sizes or {"Z": 19, "C": 2, "Y": 512, "X": 512}
        self.shape = tuple(self.sizes.values())
        self.dtype = np.dtype("uint16")
        self._pixels = pixels
        self.asarray_positions: list[int | None] = []

    def __enter__(self) -> _FakeND2File:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def voxel_size(self) -> _FakeVoxelSize:
        """Return physical calibration for X/Y/Z axes."""
        return _FakeVoxelSize(x=0.12429611388044776, y=0.12429611388044776, z=0.25)

    def asarray(self, position: int | None = None) -> np.ndarray:
        """Return fake pixels and record requested position."""
        self.asarray_positions.append(position)
        if self._pixels is not None:
            return self._pixels
        return np.zeros(tuple(self.sizes.values()), dtype=self.dtype)


def _patch_nd2_file(
    monkeypatch: pytest.MonkeyPatch,
    fake: _FakeND2File,
) -> _FakeND2File:
    """Patch ``nd2.ND2File`` to return ``fake`` for all opens."""
    from acqstore.acq_image.file_loaders import nd2_file_loader

    monkeypatch.setattr(nd2_file_loader.nd2, "ND2File", lambda _path: fake)
    return fake


def test_nd2_loader_reads_header_without_loading_pixels(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _patch_nd2_file(monkeypatch, _FakeND2File("/tmp/sample.nd2"))

    loader = Nd2FileLoader("/tmp/sample.nd2")
    header = loader.header

    assert header.shape == (19, 2, 512, 512)
    assert header.dims == ("Z", "C", "Y", "X")
    assert header.sizes == {"Z": 19, "C": 2, "Y": 512, "X": 512}
    assert header.dtype == np.dtype("uint16")
    assert header.num_channels == 2
    assert header.num_scenes == 1
    assert header.physical_units == pytest.approx((0.25, 1.0, 0.12429611388044776, 0.12429611388044776))
    assert header.physical_units_labels == ("um", "Pixels", "um", "um")
    assert fake.asarray_positions == []
    assert loader.pixels_loaded() is False


def test_nd2_loader_loads_pixels_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    pixels = np.zeros((19, 2, 512, 512), dtype=np.uint16)
    fake = _patch_nd2_file(monkeypatch, _FakeND2File("/tmp/sample.nd2", pixels=pixels))
    loader = Nd2FileLoader("/tmp/sample.nd2")

    loaded = loader.load_image_data()

    assert loaded is pixels
    assert fake.asarray_positions == [None]
    assert loader.pixels_loaded() is True


def test_nd2_loader_uses_first_position_for_multi_position_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sizes = {"P": 3, "Z": 19, "C": 2, "Y": 512, "X": 512}
    pixels = np.zeros((19, 2, 512, 512), dtype=np.uint16)
    fake = _patch_nd2_file(
        monkeypatch,
        _FakeND2File("/tmp/multiposition.nd2", sizes=sizes, pixels=pixels),
    )
    loader = Nd2FileLoader("/tmp/multiposition.nd2")

    assert loader.header.shape == (19, 2, 512, 512)
    assert loader.header.dims == ("Z", "C", "Y", "X")
    assert loader.header.num_scenes == 3
    loaded = loader.load_image_data()

    assert loaded.shape == (19, 2, 512, 512)
    assert fake.asarray_positions == [0]


def test_nd2_loader_raises_on_loaded_shape_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _patch_nd2_file(
        monkeypatch,
        _FakeND2File(
            "/tmp/bad.nd2",
            sizes={"Z": 19, "C": 2, "Y": 512, "X": 512},
            pixels=np.zeros((18, 2, 512, 512), dtype=np.uint16),
        ),
    )
    loader = Nd2FileLoader("/tmp/bad.nd2")

    with pytest.raises(ValueError, match="loaded pixel shape does not match header"):
        loader.load_image_data()
    assert fake.asarray_positions == [None]


def test_nd2_loader_rejects_inconsistent_shape_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeND2File("/tmp/bad-header.nd2")
    fake.shape = (19, 2, 512)
    _patch_nd2_file(monkeypatch, fake)

    with pytest.raises(ValueError, match="shape does not match sizes metadata"):
        Nd2FileLoader("/tmp/bad-header.nd2")
