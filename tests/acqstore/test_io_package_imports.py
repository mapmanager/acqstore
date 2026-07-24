"""Smoke tests for the acq_image.io package layout."""

from __future__ import annotations


def test_io_package_exports_expected_callables() -> None:
    """Moved IO helpers should import from acqstore.acq_image.io.* paths."""
    from acqstore.acq_image.io.ome_zarr import (
        read_acq_pixels_ome_zarr,
        write_acq_pixels_ome_zarr,
    )
    from acqstore.acq_image.io.store_utils import is_s3_path, join_store_path
    from acqstore.acq_image.io.tiff import save_pixels_as_tif

    assert callable(save_pixels_as_tif)
    assert callable(read_acq_pixels_ome_zarr)
    assert callable(write_acq_pixels_ome_zarr)
    assert is_s3_path('s3://bucket/key') is True
    assert join_store_path('s3://bucket', 'a', 'b') == 's3://bucket/a/b'
