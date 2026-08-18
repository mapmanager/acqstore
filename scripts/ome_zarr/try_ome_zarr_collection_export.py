"""Export a folder-backed AcqImageList as one multi-image OME-Zarr store.

Edit ``SOURCE_FOLDER`` and ``OUTPUT_PATH`` before running:

    uv run python scripts/ome_zarr/try_ome_zarr_collection_export.py
"""

from __future__ import annotations

import json
from pathlib import Path

from acqstore.acq_image.acq_image_list import AcqImageList
from acqstore.acq_image.io.ome_zarr_collection import export_acq_image_list_ome_zarr


# diameter has both diameter and sum intensity anaysis, src is tif with no reference image
SOURCE_FOLDER = Path("/Users/cudmore/Sites/cs_project/cloudscope-data/data-samples/diameter-sample-data")
OUTPUT_PATH = Path("/Users/cudmore/Desktop/tmp/ome-zarr-collections/diameter_sample_data.ome.zarr")

# velocity has velocity analysis, src is oir and have reference images
SOURCE_FOLDER = Path("/Users/cudmore/Sites/cs_project/cloudscope-data/data-samples/velocity-sample-data")
OUTPUT_PATH = Path("/Users/cudmore/Desktop/tmp/ome-zarr-collections/velocity-sample-data.ome.zarr")

OVERWRITE = True


def main() -> None:
    """Load the configured folder and export its images as one collection."""
    images = AcqImageList(
        str(SOURCE_FOLDER),
        load_images=True,
        load_analysis_csv=True,
    )
    print(f"Loaded {len(images)} acquisition images from {SOURCE_FOLDER}")
    destination = export_acq_image_list_ome_zarr(
        images,
        OUTPUT_PATH,
        overwrite=OVERWRITE,
    )
    manifest = json.loads(
        (destination / "acqstore" / "manifest.json").read_text(encoding="utf-8")
    )
    print(f"Exported collection: {destination}")
    for entry in manifest["images"]:
        print(f"  {entry['id']}: {entry['path']}")


if __name__ == "__main__":
    main()
