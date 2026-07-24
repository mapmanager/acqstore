"""Try lazy TIFF loading against a local folder of TIFF example files.

This script intentionally takes no command-line arguments. Edit ``folder`` below
when probing a different local dataset.
"""

from __future__ import annotations

from pathlib import Path

from acqstore.acq_image.acq_image_list import AcqImageList


def main() -> None:
    """Load TIFF examples lazily, print headers, then load one image."""
    folder = Path("/Users/cudmore/Sites/cloudscope-data/tif-examples")
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    acq_images = AcqImageList(
        str(folder),
        load_images=False,
        load_analysis_csv=False,
    )

    print(f"folder: {folder}")
    print(f"num images: {len(acq_images)}")
    print("")

    for acq_image in acq_images:
        header = acq_image.header
        print("=" * 80)
        print(f"name: {acq_image.name}")
        print(f"path: {acq_image.path}")
        print(f"shape: {header.shape}")
        print(f"dims: {header.dims}")
        print(f"sizes: {header.sizes}")
        print(f"dtype: {header.dtype}")
        print(f"num_channels: {header.num_channels}")
        print(f"num_scenes: {header.num_scenes}")
        print(f"physical_units: {header.physical_units}")
        print(f"physical_units_labels: {header.physical_units_labels}")
        print(f"images_loaded: {acq_image.images_loaded}")
        print(f"analysis_csv_loaded: {acq_image.analysis_csv_loaded}")

    if len(acq_images) == 0:
        return

    first = acq_images.get_files()[0]
    print("")
    print("=" * 80)
    print(f"Loading first image pixels: {first.name}")
    print(f"before images_loaded: {first.images_loaded}")
    first.load_images()
    print(f"after images_loaded: {first.images_loaded}")
    print(f"loaded shape: {first.images.data.shape}")
    print(f"loaded dtype: {first.images.data.dtype}")


if __name__ == "__main__":
    main()
