"""Export an acquisition path as AcqStore OME-Zarr Collection v1."""

from __future__ import annotations

import argparse
from pathlib import Path

from acqstore.acq_image.acq_image_list import AcqImageList
from acqstore.acq_image.io.ome_zarr_collection_v1.exporter import (
    AcqStoreOmeZarrCollectionExporter,
)


def main() -> int:
    """Load acquisition images and export a v1 collection.

    Returns:
        Process exit status. Zero indicates successful export.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='Acquisition file, folder, or manifest CSV.')
    parser.add_argument('destination', type=Path, help='Destination ending in .ome.zarr.')
    parser.add_argument('--name', help='Optional human-readable collection name.')
    parser.add_argument('--overwrite', action='store_true', help='Replace an existing destination.')
    args = parser.parse_args()

    images = AcqImageList(
        str(args.source),
        load_images=True,
        load_analysis_csv=True,
    )
    destination = AcqStoreOmeZarrCollectionExporter(
        args.destination,
        name=args.name,
        overwrite=args.overwrite,
    ).export(images)
    print(destination)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
