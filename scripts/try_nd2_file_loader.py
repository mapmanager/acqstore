"""Try the acqstore ND2 file loader against one local ND2 file.

This script intentionally takes no command-line arguments. Edit ``path`` below
when probing a different local dataset.
"""

from __future__ import annotations

from pathlib import Path

from acqstore.acq_image.acq_image import AcqImage


def main() -> None:
    """Load one ND2 file lazily, print header diagnostics, then load pixels."""
    path = Path(
        "/Users/cudmore/Dropbox/data/olson/IKA_A_102 Thy1_Spines_5MeO/"
        "Isak_Spines_8_30_23/Animal 148/Animal_148_Slice_2_Right.nd2"
    )
    if not path.exists():
        raise FileNotFoundError(f"ND2 file does not exist: {path}")

    acq_image = AcqImage(
        str(path),
        load_images=False,
        load_analysis_csv=False,
    )
    header = acq_image.images.header

    print(f"path: {path}")
    print(f"name: {acq_image.name}")
    print(f"shape: {header.shape}")
    print(f"dims: {header.dims}")
    print(f"sizes: {header.sizes}")
    print(f"dtype: {header.dtype}")
    print(f"num_channels: {header.num_channels}")
    print(f"num_scenes: {header.num_scenes}")
    print(f"physical_units: {header.physical_units}")
    print(f"physical_units_labels: {header.physical_units_labels}")
    print(f"file_size: {header.file_size}")
    print(f"images_loaded: {acq_image.images_loaded}")
    print(f"analysis_csv_loaded: {acq_image.analysis_csv_loaded}")

    print("")
    print("Loading ND2 pixels...")
    acq_image.load_images()
    pixels = acq_image.pixels
    print(f"images_loaded: {acq_image.images_loaded}")
    print(f"loaded shape: {pixels.data.shape}")
    print(f"loaded dtype: {pixels.data.dtype}")

    for channel in range(header.num_channels):
        plane = acq_image.images.get_slice_data_loaded(channel=channel, z=0, t=0)
        print(
            f"channel {channel} z0 plane: "
            f"shape={plane.shape}, dtype={plane.dtype}, min={plane.min()}, max={plane.max()}"
        )


if __name__ == "__main__":
    main()
