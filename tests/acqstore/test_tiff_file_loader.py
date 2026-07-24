"""Tests for lazy TIFF loading through :class:`TiffFileLoader`."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import tifffile

from acqstore.acq_image.acq_image import AcqImage
from acqstore.acq_image.file_loaders.tiff_file_loader import TiffFileLoader


def _write_synthetic_tif(
    path: Path,
    shape: tuple[int, ...] = (12, 8),
    *,
    dtype: np.dtype | type[np.generic] = np.uint16,
    axes: str | None = None,
) -> np.ndarray:
    """Write a small deterministic TIFF and return the source array."""
    arr = np.arange(int(np.prod(shape)), dtype=dtype).reshape(shape)
    metadata = {"axes": axes} if axes is not None else None
    tifffile.imwrite(path, arr, metadata=metadata, photometric="minisblack")
    return arr


def test_tiff_loader_header_does_not_load_pixels(tmp_path: Path) -> None:
    """Constructing a TIFF loader should read metadata without caching pixels."""
    path = tmp_path / "lazy_header.tif"
    _write_synthetic_tif(path, shape=(30, 18), axes="YX")

    loader = TiffFileLoader(str(path), load_olympus_header=False)

    assert loader.pixels_loaded() is False
    assert loader.header.shape == (30, 18)
    assert loader.header.dims == ("Y", "X")
    assert loader.header.sizes == {"Y": 30, "X": 18}
    assert loader.header.dtype == np.dtype(np.uint16)
    assert loader.header.num_channels == 1
    assert loader.header.num_scenes == 1


def test_tiff_loader_lazy_loads_pixels_on_load_image_data(tmp_path: Path) -> None:
    """TIFF pixels should materialize only through load_image_data."""
    path = tmp_path / "load_pixels_later.tif"
    expected = _write_synthetic_tif(path, shape=(10, 6), axes="YX")
    loader = TiffFileLoader(str(path), load_olympus_header=False)

    assert loader.pixels_loaded() is False
    actual = loader.load_image_data()

    assert loader.pixels_loaded() is True
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("axes", "shape", "expected_dims", "expected_channels"),
    [
        ("YX", (20, 10), ("Y", "X"), 1),
        ("CYX", (2, 20, 10), ("C", "Y", "X"), 2),
        ("TZYX", (3, 4, 20, 10), ("T", "Z", "Y", "X"), 1),
        ("TCZYX", (2, 3, 4, 20, 10), ("T", "C", "Z", "Y", "X"), 3),
    ],
)
def test_tiff_loader_header_from_tifffile_series_axes(
    tmp_path: Path,
    axes: str,
    shape: tuple[int, ...],
    expected_dims: tuple[str, ...],
    expected_channels: int,
) -> None:
    """Synthetic TIFF axis metadata should round-trip into ImageHeader dims."""
    path = tmp_path / f"axes_{axes}.tif"
    _write_synthetic_tif(path, shape=shape, axes=axes)

    loader = TiffFileLoader(str(path), load_olympus_header=False)

    assert loader.pixels_loaded() is False
    assert loader.header.shape == shape
    assert loader.header.dims == expected_dims
    assert loader.header.sizes == {
        expected_dims[index]: int(shape[index]) for index in range(len(shape))
    }
    assert loader.header.num_channels == expected_channels


def test_tiff_loader_from_stream_header_does_not_load_pixels(tmp_path: Path) -> None:
    """Stream-backed TIFF loaders should also initialize from metadata only."""
    path = tmp_path / "stream_source.tif"
    expected = _write_synthetic_tif(path, shape=(7, 5), axes="YX")
    stream = BytesIO(path.read_bytes())

    loader = TiffFileLoader.from_stream(stream, "stream_source.tif")

    assert loader.pixels_loaded() is False
    assert loader.header.path == "stream_source.tif"
    assert loader.header.shape == (7, 5)
    actual = loader.load_image_data()
    np.testing.assert_array_equal(actual, expected)


def _write_minimal_olympus_txt(
    path: Path,
    *,
    num_channels: int = 1,
    num_lines: int = 10,
    pixels_per_line: int = 8,
) -> None:
    """Write a minimal Olympus sidecar matching the parser's expectations."""
    path.write_text(
        "\n".join(
            [
                f'"Channel Dimension"\t"{num_channels} [Ch]"',
                '"X Dimension" 0 0 0 0 0 0.125',
                f'"Image Size" "{pixels_per_line}" x "{num_lines}" y',
                '"T Dimension" 0 0 0 100.0',
                '"Date"\t"10/30/2025 02:54:36.454 PM"',
                '"Bits/Pixel" "16"',
            ]
        ),
        encoding="utf-8",
    )


def _write_olympus_split_channel_fixture(
    folder: Path,
    *,
    num_lines: int = 10,
    pixels_per_line: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Write shared ``.txt`` plus ``_C001T001`` / ``_C002T001`` sibling TIFFs."""
    ch0 = np.arange(num_lines * pixels_per_line, dtype=np.uint16).reshape(
        num_lines,
        pixels_per_line,
    )
    ch1 = ch0 + 1000
    _write_minimal_olympus_txt(
        folder / "cell 05.txt",
        num_channels=2,
        num_lines=num_lines,
        pixels_per_line=pixels_per_line,
    )
    tifffile.imwrite(folder / "cell 05_C001T001.tif", ch0, photometric="minisblack")
    tifffile.imwrite(folder / "cell 05_C002T001.tif", ch1, photometric="minisblack")
    return ch0, ch1


def test_tiff_loader_olympus_split_channels_merge_from_c001(tmp_path: Path) -> None:
    """Olympus split-channel siblings should merge into one (C, Y, X) volume."""
    ch0, ch1 = _write_olympus_split_channel_fixture(tmp_path)
    loader = TiffFileLoader(str(tmp_path / "cell 05_C001T001.tif"))

    assert loader.pixels_loaded() is False
    assert loader.header.dims == ("C", "Y", "X")
    assert loader.header.shape == (2, 10, 8)
    assert loader.header.num_channels == 2

    merged = loader.load_image_data()
    assert merged.shape == (2, 10, 8)
    np.testing.assert_array_equal(merged[0], ch0)
    np.testing.assert_array_equal(merged[1], ch1)
    np.testing.assert_array_equal(loader.get_slice_data(0), ch0)
    np.testing.assert_array_equal(loader.get_slice_data(1), ch1)


def test_tiff_loader_olympus_split_channels_merge_from_c002(tmp_path: Path) -> None:
    """Opening the second channel TIFF should produce the same merged volume."""
    ch0, ch1 = _write_olympus_split_channel_fixture(tmp_path)
    loader = TiffFileLoader(str(tmp_path / "cell 05_C002T001.tif"))

    assert loader.header.dims == ("C", "Y", "X")
    assert loader.header.shape == (2, 10, 8)

    merged = loader.load_image_data()
    np.testing.assert_array_equal(merged[0], ch0)
    np.testing.assert_array_equal(merged[1], ch1)


def test_tiff_loader_single_channel_olympus_sidecar_unchanged(tmp_path: Path) -> None:
    """Single-channel Olympus kymographs should remain (Y, X)."""
    path = tmp_path / "sample.tif"
    expected = _write_synthetic_tif(path, shape=(12, 8), axes="YX")
    _write_minimal_olympus_txt(
        tmp_path / "sample.txt",
        num_channels=1,
        num_lines=12,
        pixels_per_line=8,
    )

    loader = TiffFileLoader(str(path))

    assert loader.header.dims == ("Y", "X")
    assert loader.header.num_channels == 1
    np.testing.assert_array_equal(loader.load_image_data(), expected)


def test_acq_image_tif_load_images_false_stays_unloaded(tmp_path: Path) -> None:
    """AcqImage should keep TIFF pixels unloaded when load_images is false."""
    path = tmp_path / "acq_image_lazy.tif"
    expected = _write_synthetic_tif(path, shape=(9, 4), axes="YX")

    acq = AcqImage(str(path), load_images=False, load_analysis_csv=False)

    assert acq.images_loaded is False
    assert acq.pixels_loaded() is False
    assert acq.images.header.shape == (9, 4)
    acq.load_images()
    assert acq.images_loaded is True
    np.testing.assert_array_equal(acq.pixels.data, expected)
