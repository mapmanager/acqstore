"""Fixtures and marks for acqstore OME-Zarr persistence tests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

pytest.importorskip('zarr')
pytest.importorskip('bioio_ome_zarr')

from acqstore.acq_image.acq_pixels import AcqPixels
from acqstore.acq_image.file_loaders.base_file_loader import ImageHeader

requires_s3 = pytest.mark.skipif(
    os.environ.get('CLOUDSCOPE_RUN_S3_TESTS') != '1',
    reason='S3 tests require AWS credentials and a bioio-ome-zarr version with working direct S3 writes.',
)


@pytest.fixture
def make_pixels() -> Callable[..., AcqPixels]:
    """Return a factory for calibrated :class:`AcqPixels` test objects.

    Returns:
        Callable that builds small pixel objects with explicit, known header
        calibration independent of vendor file loaders.
    """

    def _make_pixels(
        path: Path,
        *,
        data: np.ndarray | None = None,
        dims: tuple[str, ...] = ('Y', 'X'),
        physical_units: tuple[float, ...] = (0.0005, 0.01),
        physical_units_labels: tuple[str, ...] = ('seconds', 'micrometer'),
    ) -> AcqPixels:
        if data is None:
            data = np.arange(6 * 4, dtype=np.uint16).reshape(6, 4)
        if len(dims) != data.ndim:
            raise ValueError('dims must match data rank')
        header = ImageHeader(
            path=str(path),
            shape=tuple(int(x) for x in data.shape),
            dims=dims,
            sizes={dims[i]: int(data.shape[i]) for i in range(len(dims))},
            dtype=data.dtype,
            num_channels=int(data.shape[dims.index('C')]) if 'C' in dims else 1,
            num_scenes=1,
            physical_units=physical_units,
            physical_units_labels=physical_units_labels,
            date='20260709',
            time='12:34:56',
            file_size='24 bytes',
        )
        return AcqPixels(data=data, header=header, source_path=str(path))

    return _make_pixels