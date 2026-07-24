"""Data-provider interfaces used by analysis classes.

Analyses should not depend directly on ``AcqImage``. They receive an
``AnalysisDataProvider`` so the analysis framework has a small, testable API for
ROI image data and physical calibration. The concrete provider in this module
adapts one ``AcqImage`` to that interface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from acqstore.acq_image.acq_image import AcqImage


class AnalysisDataProvider(Protocol):
    """Minimal data access surface needed by analyses.

    Analysis implementations use this protocol instead of reaching into
    ``AcqImage`` internals. The contract is intentionally small: ROI-local image
    data and image physical spacing. This keeps analysis code reusable from GUI,
    batch, test, and notebook workflows.
    """

    def get_roi_image(self, channel: int, roi_id: int) -> np.ndarray:
        """Return image data for one channel cropped to one ROI.

        Args:
            channel: Channel index.
            roi_id: ROI identifier.

        Returns:
            Two-dimensional ROI image data.
        """
        ...

    def get_image_physical_units(self) -> tuple[float, float]:
        """Return physical units for the 2D image plane.

        Returns:
            Per-pixel ``(step_y, step_x)`` for ``(Y, X)`` image data.
        """
        ...


class AcqImageAnalysisDataProvider:
    """Analysis data provider backed by one ``AcqImage``.

    The provider delegates ROI cropping and physical-unit lookup to the parent
    acquisition image. It does not cache pixel data; each call reflects the
    current ROI and file-loader state.

    Args:
        acq_image: Parent acquisition image.
    """

    def __init__(self, acq_image: AcqImage) -> None:
        self._acq_image = acq_image

    def get_roi_image(self, channel: int, roi_id: int) -> np.ndarray:
        """Return image data for one channel cropped to one ROI.

        Args:
            channel: Channel index.
            roi_id: ROI identifier.

        Returns:
            Two-dimensional ROI image data.
        """
        return self._acq_image.get_roi_image(channel=channel, roi_id=roi_id)

    def get_image_physical_units(self) -> tuple[float, float]:
        """Return physical units for the 2D image plane.

        Returns:
            Per-pixel ``(step_y, step_x)`` for ``(Y, X)`` image data.
        """
        return self._acq_image.get_image_physical_units()
