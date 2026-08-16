from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from pynwb import NWBFile, NWBHDF5IO
from pynwb.image import GrayscaleImage, Images


def build_test_nwb(path: Path) -> None:
    """Create an NWB file with two independent static image containers.

    The file contains:
    - One CYX dataset represented as two same-sized GrayscaleImage objects.
    - One independent YX dataset with a completely different shape.

    Args:
        path: Output NWB file path.

    Returns:
        None.
    """
    rng = np.random.default_rng(12345)

    # Synthetic CYX data: two independent channels sharing the same YX shape.
    cyx = rng.integers(
        low=0,
        high=65535,
        size=(2, 1024, 1024),
        dtype=np.uint16,
    )

    # Synthetic independent YX image with a very different geometry.
    yx = rng.integers(
        low=0,
        high=65535,
        size=(30000, 100),
        dtype=np.uint16,
    )

    nwbfile = NWBFile(
        session_description="Synthetic multiple static image container test",
        identifier="synthetic-multi-image-test",
        session_start_time=datetime.now(ZoneInfo("America/New_York")),
    )

    # PyNWB GrayscaleImage uses image data stored as X,Y.
    # AcqStore-style arrays are Y,X, so transpose at the serialization boundary.
    cyx_images = Images(
        name="acqimage_000",
        description="Synthetic CYX acquisition with two channels.",
    )

    for channel_index in range(cyx.shape[0]):
        cyx_images.add_image(
            GrayscaleImage(
                name=f"channel_{channel_index:04d}",
                data=cyx[channel_index].T,
                description=f"Synthetic channel {channel_index}.",
            )
        )

    yx_images = Images(
        name="acqimage_001",
        description="Independent synthetic YX acquisition.",
    )

    yx_images.add_image(
        GrayscaleImage(
            name="channel_0000",
            data=yx.T,
            description="Synthetic single-channel YX image.",
        )
    )

    # Each Images object is an independent NWB acquisition object.
    nwbfile.add_acquisition(cyx_images)
    nwbfile.add_acquisition(yx_images)

    with NWBHDF5IO(path, mode="w") as io:
        io.write(nwbfile)

    print(f"Wrote: {path}")
    print(f"CYX original shape: {cyx.shape}")
    print(f"YX original shape:  {yx.shape}")


def read_and_verify(path: Path) -> None:
    """Reopen the NWB file and verify the independent image shapes.

    Args:
        path: NWB file to inspect.

    Returns:
        None.
    """
    with NWBHDF5IO(path, mode="r", load_namespaces=True) as io:
        nwbfile = io.read()

        print("\nAcquisition objects:")
        for name, acquisition in nwbfile.acquisition.items():
            print(f"  {name}: {type(acquisition).__name__}")

        cyx_images = nwbfile.acquisition["acqimage_000"]
        yx_images = nwbfile.acquisition["acqimage_001"]

        cyx_channels = []

        for channel_index in range(2):
            image = cyx_images.images[f"channel_{channel_index:04d}"]

            # NWB XY -> AcqStore-style YX.
            restored_yx = np.asarray(image.data).T
            cyx_channels.append(restored_yx)

            print(
                f"acqimage_000 channel {channel_index}: "
                f"NWB shape={image.data.shape}, "
                f"restored YX={restored_yx.shape}"
            )

        restored_cyx = np.stack(cyx_channels, axis=0)

        yx_image = yx_images.images["channel_0000"]
        restored_yx = np.asarray(yx_image.data).T

        print(f"\nRestored CYX shape: {restored_cyx.shape}")
        print(f"Restored YX shape:  {restored_yx.shape}")

        assert restored_cyx.shape == (2, 1024, 1024)
        assert restored_yx.shape == (30000, 100)

    print("\nRound-trip shape checks passed.")


def main() -> None:
    """Run the synthetic multiple-image NWB experiment."""
    path = Path("synthetic_multiple_images.nwb")

    build_test_nwb(path)
    read_and_verify(path)

    print(
        "\nNext validation step:\n"
        "  python -m pynwb.validate synthetic_multiple_images.nwb\n"
    )


def main_2():
    from acqstore.acq_image import AcqImage
    from acqstore.nwb_io import NwbMetadata, NwbSubjectMetadata

    # load one image
    file_path = '/Users/cudmore/Sites/cs_project/cloudscope-data/data-samples/velocity-sample-data/7d Control/20251014/20251014_A98_0002.oir'
    acq_image = AcqImage(file_path)

    # fill in dandi info
    header = acq_image.images.header
    session_start_time = datetime.strptime(
        f'{header.date} {header.time}',
        '%Y%m%d %H:%M:%S',
    ).replace(tzinfo=ZoneInfo('America/New_York'))

    save_path = '/Users/cudmore/Desktop/tmp/my-nwb.nwb'
    metadata = NwbMetadata(
        subject=NwbSubjectMetadata(
            subject_id='A98',
            species='Mus musculus',
            sex='M',
            age='P14D',
            description='Subject A98.',
        ),
        session_start_time=session_start_time,
        experimenter=('Manning, Declan',),
        keywords=(
            'microscopy',
            'kymograph',
            'vascular imaging',
            'blood flow',
            'velocity analysis',
        ),
    )
    acq_image.save_as_nwb(
        save_path,
        metadata=metadata,
        overwrite=True,
    )

    acq_image_2 = AcqImage(save_path)

    acq_image_2 = AcqImage.from_nwb(save_path)
    pixels = acq_image_2.pixels
    print(f'pixels.shape: {pixels.shape}')

if __name__ == "__main__":
    # main()
    main_2()
