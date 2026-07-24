"""Exercise AcqImage TIFF, OME-Zarr, and native CS OME-Zarr exports.

This is a development script with hard-coded paths by design. Edit the constants
near the top, then run from the repository root with::

    uv run python scripts/ome_zarr/try_ome_zarr.py

The script loads either a folder of acquisition files or one acquisition file,
selects one ``AcqImage``, exports it to:

- standard TIFF using tifffile, with optional ImageJ/Fiji metadata;
- pure ``.ome.zarr`` without acqstore sidecars;
- native ``.cs.ome.zarr`` with acqstore metadata, ROI, contrast, and analysis
  sidecars;
- optional ZIP snapshots of the OME-Zarr stores.

It then reloads the saved Zarr stores through the normal ``AcqImage`` path and
prints shape, axes, metadata, ROI, and analysis summaries.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from acqstore.acq_image.acq_image import AcqImage
from acqstore.acq_image.acq_image_list import AcqImageList
from acqstore.acq_image.supported_import_extensions import path_has_allowed_import_extension

# Edit these paths for your machine. SOURCE_PATH may be one OIR/CZI/TIF/OME-Zarr
# file or a folder containing supported acquisitions plus sidecar JSON/CSV files.
SOURCE_PATH = Path('/Users/cudmore/Sites/cloudscope-data/demo-velocity/20251030')
SOURCE_PATH = Path('/Users/cudmore/Sites/cloudscope-data/threed-examples/S7SSC1R3.czi')

OUTPUT_DIR = Path('/Users/cudmore/Desktop/cloudscope_ome_zarr_try')
OVERWRITE = True
WRITE_ZIP_SNAPSHOTS = True
ZARR_FORMAT = 3  # 3 = OME-NGFF 0.5 / Zarr v3, 2 = OME-NGFF 0.4 / Zarr v2.


def _load_one_acq_image(path: Path) -> AcqImage:
    """Load one AcqImage from a file/store or the first item in a folder."""
    if path.is_dir() and not path_has_allowed_import_extension(path):
        acq_list = AcqImageList(str(path))
        if len(acq_list) == 0:
            raise RuntimeError(f'No supported files found in {path}')
        return acq_list.get_file_by_index(0)
    return AcqImage(str(path))


def _print_summary(label: str, acq: AcqImage) -> None:
    """Print a compact summary for one acquisition."""
    pixels = acq.pixels
    print(f'\n{label}')
    print('-' * len(label))
    print(f'path: {acq.path}')
    print(f'name: {acq.name}')
    print(f'axes: {pixels.axes}')
    print(f'shape: {pixels.shape}')
    print(f'dtype: {pixels.dtype}')
    print(f'channels: {pixels.num_channels}')
    print(f'pyramid_levels: {1 + len(pixels.levels)}')
    print(f'physical_units: {pixels.header.physical_units}')
    print(f'physical_units_labels: {pixels.header.physical_units_labels}')
    print(f'roi_count: {acq.rois.num_rois}')
    print(f'analysis_count: {len(acq.analysis_set.as_list())}')
    for section in acq.get_metadata_sections():
        print(f'metadata[{section.metadata_section_id}]: {section.get_values()}')


def _remove_path(path: Path) -> None:
    """Remove an existing local file or directory for repeatable script runs."""
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def main() -> None:
    """Run the local export/reload exercise."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    acq = _load_one_acq_image(SOURCE_PATH.expanduser())
    _print_summary('Loaded source AcqImage', acq)

    tif_path = OUTPUT_DIR / 'try_export.tif'
    ome_zarr_path = OUTPUT_DIR / 'try_export.ome.zarr'
    native_zarr_path = OUTPUT_DIR / 'try_export.cs.ome.zarr'
    ome_zarr_zip_path = OUTPUT_DIR / 'try_export.ome.zarr.zip'
    native_zarr_zip_path = OUTPUT_DIR / 'try_export.cs.ome.zarr.zip'

    if OVERWRITE:
        for path in (tif_path, ome_zarr_path, native_zarr_path, ome_zarr_zip_path, native_zarr_zip_path):
            _remove_path(path)

    acq.save_as_tif(tif_path, imagej_metadata=True, overwrite=OVERWRITE)
    print(f'\nSaved TIFF: {tif_path}')

    acq.save_as_ome_zarr(ome_zarr_path, overwrite=OVERWRITE, zarr_format=ZARR_FORMAT)
    print(f'Saved pure OME-Zarr: {ome_zarr_path}')
    reloaded_ome = AcqImage(str(ome_zarr_path))
    _print_summary('Reloaded pure OME-Zarr', reloaded_ome)

    acq.save_native_zarr(native_zarr_path, overwrite=OVERWRITE, zarr_format=ZARR_FORMAT)
    print(f'Saved native CS OME-Zarr: {native_zarr_path}')
    reloaded_native = AcqImage(str(native_zarr_path))
    _print_summary('Reloaded native CS OME-Zarr', reloaded_native)

    if WRITE_ZIP_SNAPSHOTS:
        acq.save_as_ome_zarr(ome_zarr_zip_path, overwrite=OVERWRITE, zarr_format=ZARR_FORMAT)
        print(f'Saved pure OME-Zarr ZIP snapshot: {ome_zarr_zip_path}')
        reloaded_ome_zip = AcqImage(str(ome_zarr_zip_path))
        _print_summary('Reloaded pure OME-Zarr ZIP', reloaded_ome_zip)

        acq.save_native_zarr(native_zarr_zip_path, overwrite=OVERWRITE, zarr_format=ZARR_FORMAT)
        print(f'Saved native CS OME-Zarr ZIP snapshot: {native_zarr_zip_path}')
        reloaded_native_zip = AcqImage(str(native_zarr_zip_path))
        _print_summary('Reloaded native CS OME-Zarr ZIP', reloaded_native_zip)


if __name__ == '__main__':
    main()
