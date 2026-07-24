"""Tests for ``ImageContrast``, ``contrast_clip_min_max``, and AcqImage wiring."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from acqstore.acq_image.acq_image import AcqImage
from acqstore.acq_image.image_contrast import ImageContrast, contrast_clip_min_max


def _write_tif(path: Path, shape: tuple[int, int] = (10, 20), dtype=np.uint8) -> None:
    tifffile.imwrite(path, np.zeros(shape, dtype=dtype))


def test_contrast_clip_min_max_returns_ints_for_uint8() -> None:
    img = np.array([[0, 50, 100, 200, 255]], dtype=np.uint8)
    lo, hi = contrast_clip_min_max(img, percentile_low=0.0, percentile_high=100.0)
    assert isinstance(lo, int)
    assert isinstance(hi, int)
    assert (lo, hi) == (0, 255)


def test_contrast_clip_min_max_returns_ints_for_uint16() -> None:
    img = np.arange(1000, dtype=np.uint16).reshape(20, 50)
    lo, hi = contrast_clip_min_max(img, percentile_low=1.0, percentile_high=99.0)
    assert isinstance(lo, int)
    assert isinstance(hi, int)
    assert 0 <= lo < hi <= 999


def test_contrast_clip_min_max_returns_ints_for_float() -> None:
    img = np.linspace(0.0, 1.0, 100, dtype=np.float32).reshape(10, 10)
    lo, hi = contrast_clip_min_max(img)
    assert isinstance(lo, int)
    assert isinstance(hi, int)
    assert lo <= hi


def test_contrast_clip_swaps_inverted_percentiles() -> None:
    img = np.arange(100, dtype=np.uint16).reshape(10, 10)
    a, b = contrast_clip_min_max(img, percentile_low=99.0, percentile_high=1.0)
    c, d = contrast_clip_min_max(img, percentile_low=1.0, percentile_high=99.0)
    assert (a, b) == (c, d)


def test_contrast_clip_clamps_percentiles_outside_range() -> None:
    img = np.arange(100, dtype=np.uint16).reshape(10, 10)
    lo, hi = contrast_clip_min_max(img, percentile_low=-10.0, percentile_high=150.0)
    assert (lo, hi) == (0, 99)


def test_contrast_clip_raises_on_empty_image() -> None:
    img = np.array([], dtype=np.uint8).reshape(0, 0)
    with pytest.raises(ValueError, match='non-empty'):
        contrast_clip_min_max(img)


def test_image_contrast_copy_is_independent() -> None:
    a = ImageContrast(color_lut='Gray', value_min=10, value_max=200, img_min=0, img_max=255)
    b = a.copy()
    b.color_lut = 'Plasma'
    assert a.color_lut == 'Gray'
    assert b.color_lut == 'Plasma'


def test_acq_image_no_contrast_entries_by_default(tmp_path: Path) -> None:
    p = tmp_path / 'sample.tif'
    _write_tif(p)
    acq = AcqImage(str(p))
    assert acq.get_image_contrast(0) is None
    assert acq.is_dirty is False


def test_ensure_image_contrast_from_plane_seeds_and_is_idempotent(tmp_path: Path) -> None:
    p = tmp_path / 'sample.tif'
    _write_tif(p)
    acq = AcqImage(str(p))
    plane = np.array([[0, 50, 100, 200, 255]] * 4, dtype=np.uint8)

    first = acq.ensure_image_contrast_from_plane(
        0,
        plane,
        default_color_lut='Green',
        percentile_low=1.0,
        percentile_high=99.0,
    )
    second = acq.ensure_image_contrast_from_plane(
        0,
        plane,
        default_color_lut='SHOULD_BE_IGNORED',
        percentile_low=1.0,
        percentile_high=99.0,
    )
    assert first is second
    assert first.color_lut == 'Green'
    assert first.img_min == 0
    assert first.img_max == 255


def test_ensure_image_contrast_from_plane_does_not_mark_dirty(tmp_path: Path) -> None:
    """Default seeding must not flip is_dirty (no unsolicited save prompts)."""
    p = tmp_path / 'sample.tif'
    _write_tif(p)
    acq = AcqImage(str(p))
    assert acq.is_dirty is False
    acq.ensure_image_contrast_from_plane(
        0,
        np.array([[0, 9]], dtype=np.uint8),
        default_color_lut='Gray',
        percentile_low=1.0,
        percentile_high=99.0,
    )
    assert acq.is_dirty is False


def test_set_image_contrast_marks_dirty(tmp_path: Path) -> None:
    """User-driven contrast updates must mark the file dirty."""
    p = tmp_path / 'sample.tif'
    _write_tif(p)
    acq = AcqImage(str(p))
    contrast = ImageContrast(
        color_lut='Plasma', value_min=10, value_max=200, img_min=0, img_max=255
    )
    acq.set_image_contrast(0, contrast)
    assert acq.is_dirty is True
    stored = acq.get_image_contrast(0)
    assert stored is not None
    assert stored.color_lut == 'Plasma'
    assert stored is not contrast  # set_image_contrast stores a copy


def test_acq_image_does_not_call_loader_for_contrast(tmp_path: Path) -> None:
    """Contract: AcqImage.image_contrast paths MUST NOT invoke ``get_slice_data``."""
    p = tmp_path / 'sample.tif'
    _write_tif(p)
    acq = AcqImage(str(p))

    calls: list[tuple] = []
    real = acq.images.get_slice_data

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    acq.images.get_slice_data = spy  # type: ignore[method-assign]

    plane = np.array([[1, 9, 17, 25]] * 3, dtype=np.uint8)
    acq.ensure_image_contrast_from_plane(
        0, plane, default_color_lut='Gray', percentile_low=1.0, percentile_high=99.0
    )
    acq.get_image_contrast(0)
    acq.set_image_contrast(
        0, ImageContrast(color_lut='Gray', value_min=1, value_max=9, img_min=0, img_max=255)
    )
    assert calls == []
