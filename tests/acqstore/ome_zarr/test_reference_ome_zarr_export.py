"""OME-Zarr coverage for conventional spatial reference images."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from acqstore.acq_image.acq_image import AcqImage
from acqstore.acq_image.file_loaders.base_file_loader import ReferenceImage
from acqstore.acq_image.web_export import export_acq_image


def test_web_export_writes_conventional_spatial_reference_image(tmp_path: Path) -> None:
    """Web export retains spatial reference-image calibration."""
    data = np.arange(2 * 6 * 4, dtype=np.uint16).reshape(2, 6, 4)
    reference = ReferenceImage(
        array=data,
        dims=('C', 'Y', 'X'),
        num_channels=2,
        line_roi=None,
        coord_units=(('Y', 'micrometer'), ('X', 'micrometer')),
        coord_scales=(('Y', 0.5), ('X', 0.25)),
        coords=(),
    )
    acq = AcqImage.from_array(
        np.arange(32 * 20, dtype=np.uint16).reshape(32, 20),
        axes=('Y', 'X'),
        source_id='source.oir',
        axis_spacing={'Y': 0.002, 'X': 0.4},
        axis_units={'Y': 'seconds', 'X': 'micrometer'},
    )
    acq._images._referenceImage = reference
    destination = tmp_path / 'image-package'

    export_acq_image(acq, destination)

    import zarr

    group = zarr.open_group(str(destination / 'reference.ome.zarr'), mode='r')
    multiscale = dict(group.attrs)['ome']['multiscales'][0]
    assert multiscale['axes'] == [
        {'name': 'c', 'type': 'channel'},
        {'name': 'y', 'type': 'space', 'unit': 'micrometer'},
        {'name': 'x', 'type': 'space', 'unit': 'micrometer'},
    ]
    assert multiscale['datasets'][0]['coordinateTransformations'][0]['scale'] == [1.0, 0.5, 0.25]
    np.testing.assert_array_equal(np.asarray(group['0']), data)
