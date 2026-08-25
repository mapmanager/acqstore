"""Fixtures for AcqStore OME-Zarr Collection v1 exporter tests."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from acqstore.acq_image.acq_image import AcqImage
from acqstore.acq_image.acq_image_list import AcqImageList


@pytest.fixture
def make_acq_image() -> Callable[..., AcqImage]:
    """Return a factory for small loaded in-memory acquisition images.

    Returns:
        Callable producing calibrated ``AcqImage`` instances.
    """

    def _make_acq_image(
        name: str = 'image.tif',
        *,
        shape: tuple[int, ...] = (32, 24),
        axes: tuple[str, ...] = ('Y', 'X'),
        axis_units: dict[str, str] | None = None,
        axis_spacing: dict[str, float] | None = None,
    ) -> AcqImage:
        """Build one loaded image fixture.

        Args:
            name: In-memory source identifier and display name.
            shape: Pixel-array shape.
            axes: Axis labels aligned with ``shape``.
            axis_units: Optional physical-unit labels by axis.
            axis_spacing: Optional physical scale by axis.

        Returns:
            Loaded in-memory acquisition image.
        """
        data = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
        units = axis_units or {axis: 'micrometer' for axis in axes}
        spacing = axis_spacing or {axis: 1.0 for axis in axes}
        return AcqImage.from_array(
            data,
            axes=axes,
            source_id=name,
            axis_units=units,
            axis_spacing=spacing,
        )

    return _make_acq_image


@pytest.fixture
def make_acq_image_list() -> Callable[..., AcqImageList]:
    """Return a test-only factory around explicit acquisition images.

    Returns:
        Callable producing an ``AcqImageList`` with analysis pools attached.
    """

    def _make_acq_image_list(*images: AcqImage) -> AcqImageList:
        """Build a collection without filesystem discovery.

        Args:
            images: Explicit source acquisition images.

        Returns:
            In-memory acquisition collection.
        """
        collection = AcqImageList.__new__(AcqImageList)
        collection.path = 'memory://collection'
        collection.source_root_path = None
        collection.file_list = [image.file_id for image in images]
        collection._files = list(images)
        collection._files_by_id = {image.file_id: image for image in images}
        collection._attach_analysis_pools()
        return collection

    return _make_acq_image_list
