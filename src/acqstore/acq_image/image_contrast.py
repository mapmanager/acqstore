"""Per-channel image-contrast model and helper for one acquisition image.

``ImageContrast`` is the AcqImage-owned source of truth for the displayed color
LUT and intensity clipping window for one channel. The model does NOT decode
slice data; callers must supply the 2D plane to
:func:`AcqImage.ensure_image_contrast_from_plane`. This keeps
``PrimaryImageView`` the single decoder of slice data per selection.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass
class ImageContrast:
    """Display contrast state for one channel of an acquisition image.

    Attributes:
        color_lut: Widget LUT identifier; see
            ``nicewidgets.contrast_widget.colorscales.COLORSCALE_OPTIONS``.
        value_min: Current minimum intensity displayed (intensity clip min).
        value_max: Current maximum intensity displayed (intensity clip max).
        img_min: Minimum intensity of the underlying 2D plane at seed time.
        img_max: Maximum intensity of the underlying 2D plane at seed time.
    """

    color_lut: str
    value_min: int
    value_max: int
    img_min: int
    img_max: int

    def copy(self) -> ImageContrast:
        """Return a shallow copy of this contrast snapshot.

        Returns:
            New :class:`ImageContrast` with identical fields.
        """
        return replace(self)


def contrast_clip_min_max(
    img: np.ndarray,
    *,
    percentile_low: float = 1.0,
    percentile_high: float = 99.5,
) -> tuple[int, int]:
    """Return integer ``(min, max)`` intensity clip values for a 2D image.

    Uses :func:`numpy.percentile` so outliers do not dominate the auto window.
    Both percentiles are clamped to ``[0.0, 100.0]`` and swapped if
    ``percentile_low > percentile_high``.

    Args:
        img: 2D ndarray (any numeric dtype). Empty arrays raise ``ValueError``.
        percentile_low: Lower percentile (default 1.0).
        percentile_high: Upper percentile (default 99.5).

    Returns:
        ``(min, max)`` as integers; ``min`` is guaranteed not to exceed ``max``.

    Raises:
        ValueError: If ``img`` is empty.
    """
    if img.size == 0:
        raise ValueError('contrast_clip_min_max requires a non-empty image')
    low = max(0.0, min(100.0, float(percentile_low)))
    high = max(0.0, min(100.0, float(percentile_high)))
    if low > high:
        low, high = high, low
    lo, hi = np.percentile(img, [low, high])
    lo_i = int(lo)
    hi_i = int(hi)
    if lo_i > hi_i:
        lo_i, hi_i = hi_i, lo_i
    return lo_i, hi_i


__all__ = ['ImageContrast', 'contrast_clip_min_max']
