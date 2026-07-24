"""Tests for :meth:`ImageHeader.format_dims_display`."""

from __future__ import annotations

import numpy as np

from acqstore.acq_image.file_loaders.base_file_loader import ImageHeader


def test_format_dims_display_orders_by_header_dims() -> None:
    header = ImageHeader(
        path='/tmp/sample.czi',
        shape=(2, 10000, 1024),
        dims=('C', 'Y', 'X'),
        sizes={'C': 2, 'Y': 10000, 'X': 1024},
        dtype=np.dtype('uint16'),
        num_channels=2,
        num_scenes=1,
        physical_units=(1.0, 1.0, 1.0),
        physical_units_labels=('Pixels', 'Pixels', 'Pixels'),
    )

    assert header.format_dims_display() == 'C:2 Y:10000 X:1024'
