"""Tests for AcqStore reference-image display plane API."""

from __future__ import annotations

import numpy as np
import pytest

from acqstore.acq_image.file_loaders.base_file_loader import ReferenceImage, ReferenceImagePlane


def _reference_image(
    array: np.ndarray,
    dims: tuple[str, ...],
    *,
    num_channels: int = 1,
    coord_scales: tuple[tuple[str, float], ...] = (("Y", 0.5), ("X", 0.25)),
    coord_units: tuple[tuple[str, str], ...] = (("Y", "um"), ("X", "um")),
) -> ReferenceImage:
    """Create a ReferenceImage for tests.

    Args:
        array: Reference image array.
        dims: Dimension labels aligned to ``array``.
        num_channels: Number of channels reported by the reference image.
        coord_scales: Coordinate scales keyed by dimension name.
        coord_units: Coordinate units keyed by dimension name.

    Returns:
        Reference image instance.
    """
    return ReferenceImage(
        array=array,
        dims=dims,
        num_channels=num_channels,
        line_roi=None,
        coord_units=coord_units,
        coord_scales=coord_scales,
        coords=(),
    )


def test_reference_image_plane_validates_2d_array() -> None:
    """ReferenceImagePlane requires a two-dimensional array."""
    with pytest.raises(ValueError, match="2D"):
        ReferenceImagePlane(np.zeros((1, 2, 3)), dx=1.0, dy=1.0, x_unit="px", y_unit="px")


def test_reference_image_get_plane_2d_yx() -> None:
    """A two-dimensional YX reference image returns directly as a plane."""
    arr = np.arange(6, dtype=np.uint16).reshape(2, 3)
    ref = _reference_image(arr, ("Y", "X"))

    plane = ref.get_plane()

    np.testing.assert_array_equal(plane.array, arr)
    assert plane.dx == 0.5
    assert plane.dy == 0.25
    assert plane.x_unit == "um"
    assert plane.y_unit == "um"
    assert not plane.array.flags.writeable


def test_reference_image_get_plane_multichannel_cyx() -> None:
    """A CYX reference image selects the requested channel."""
    arr = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    ref = _reference_image(arr, ("C", "Y", "X"), num_channels=2)

    plane = ref.get_plane(1)

    np.testing.assert_array_equal(plane.array, arr[1])


def test_reference_image_get_plane_reorders_xy_to_yx() -> None:
    """ReferenceImage.get_plane returns YX order even when source dims are XY."""
    arr = np.arange(6, dtype=np.uint16).reshape(3, 2)
    ref = _reference_image(arr, ("X", "Y"))

    plane = ref.get_plane()

    np.testing.assert_array_equal(plane.array, arr.T)


def test_reference_image_get_plane_rejects_invalid_channel() -> None:
    """Invalid reference image channels fail fast."""
    arr = np.zeros((2, 3, 4), dtype=np.uint8)
    ref = _reference_image(arr, ("C", "Y", "X"), num_channels=2)

    with pytest.raises(ValueError, match="out of range"):
        ref.get_plane(2)


def test_reference_image_get_plane_rejects_unsupported_non_singleton_axis() -> None:
    """Non-singleton non-YX axes require explicit backend support."""
    arr = np.zeros((2, 3, 4), dtype=np.uint8)
    ref = _reference_image(arr, ("Z", "Y", "X"))

    with pytest.raises(NotImplementedError, match="non-singleton"):
        ref.get_plane()


def test_reference_image_get_scan_path_returns_none_when_missing() -> None:
    """Reference images without scan-path metadata report no scan path."""
    ref = _reference_image(np.zeros((2, 3), dtype=np.uint8), ("Y", "X"))

    assert ref.has_scan_path() is False
    assert ref.get_scan_path() is None
    assert ref.get_scan_path_plot() is None


def test_reference_image_get_scan_path_plot_returns_xy_rows() -> None:
    """ReferenceImage exposes stored ``(2, N)`` scan paths as X/Y arrays."""
    scan_path = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    ref = ReferenceImage(
        array=np.zeros((2, 3), dtype=np.uint8),
        dims=("Y", "X"),
        num_channels=1,
        line_roi=None,
        coord_units=(("Y", "um"), ("X", "um")),
        coord_scales=(("Y", 0.5), ("X", 0.25)),
        coords=(),
        scan_path=scan_path,
    )

    assert ref.has_scan_path() is True
    np.testing.assert_array_equal(ref.get_scan_path(), scan_path)
    x_pixels, y_pixels = ref.get_scan_path_plot()
    np.testing.assert_array_equal(x_pixels, np.asarray([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(y_pixels, np.asarray([4.0, 5.0, 6.0]))
    assert x_pixels.flags.writeable is False
    assert y_pixels.flags.writeable is False


def test_reference_image_get_scan_path_plot_rejects_invalid_shape() -> None:
    """ReferenceImage scan paths must use two rows for X and Y coordinates."""
    ref = ReferenceImage(
        array=np.zeros((2, 3), dtype=np.uint8),
        dims=("Y", "X"),
        num_channels=1,
        line_roi=None,
        coord_units=(("Y", "um"), ("X", "um")),
        coord_scales=(("Y", 0.5), ("X", 0.25)),
        coords=(),
        scan_path=np.zeros((3, 4), dtype=float),
    )

    with pytest.raises(ValueError, match="shape"):
        ref.get_scan_path_plot()
