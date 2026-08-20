"""Export a folder-backed AcqImageList as one multi-image OME-Zarr store.

Run with a source folder and destination:

    uv run python scripts/ome_zarr/try_ome_zarr_collection_export.py SOURCE OUTPUT.ome.zarr
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from acqstore.acq_image.acq_image_list import AcqImageList
from acqstore.acq_image.io.ome_zarr_collection import export_acq_image_list_ome_zarr


def main() -> None:
    """Load the configured folder and export its images as one collection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='Folder loaded as an AcqImageList')
    parser.add_argument('output', type=Path, help='Destination ending in .ome.zarr')
    parser.add_argument('--overwrite', action='store_true', help='Replace an existing output')
    args = parser.parse_args()

    # args.source = Path('')
    # args.output = Path('')
    # args.overwrite = True

    images = AcqImageList(
        str(args.source),
        load_images=True,
        load_analysis_csv=True,
    )
    print(f'Loaded {len(images)} acquisition images from {args.source}')
    destination = export_acq_image_list_ome_zarr(
        images,
        args.output,
        overwrite=args.overwrite,
    )
    manifest = json.loads((destination / 'acqstore' / 'acq_image_collection.json').read_text(encoding='utf-8'))
    print(f'Exported collection: {destination}')
    for entry in manifest['acq_images']:
        print(f'  {entry["id"]}: {entry["ome_zarr_path"]}')


if __name__ == '__main__':
    main()
