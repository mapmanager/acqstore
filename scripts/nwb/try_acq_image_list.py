"""Export a local AcqImageList to NWB, reload it, and verify the round trip."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pynwb.validation import validate

from acqstore.acq_image.acq_image import AcqImage
from acqstore.acq_image.acq_image_list import AcqImageList
from acqstore.nwb_io import NwbMetadata, NwbSubjectMetadata


# SOURCE_FOLDER = Path(
#     "/Users/cudmore/Sites/cs_project/cloudscope-data/data-samples/"
#     "velocity-sample-data/7d Control/20251014"
# )
SOURCE_FOLDER= '/Users/cudmore/Sites/cs_project/cloudscope-data/data-samples/diameter-sample-data'

NWB_PATH = Path("/Users/cudmore/Desktop/tmp/diameter-sample-data.nwb")


def _session_start_time(first_image: AcqImage) -> datetime:
    """Build the NWB session time from the first acquisition header.

    Args:
        first_image: First source acquisition in the list.

    Returns:
        Timezone-aware acquisition start time.

    Raises:
        ValueError: If the source header has no usable date or time.
    """
    header = first_image.images.header
    return datetime.strptime(
        f"{header.date} {header.time}",
        "%Y%m%d %H:%M:%S",
    ).replace(tzinfo=ZoneInfo("America/New_York"))


def _metadata(first_image: AcqImage) -> NwbMetadata:
    """Return explicit metadata for the local AcqImageList export."""
    return NwbMetadata(
        session_description="AcqStore AcqImageList round-trip demonstration",
        subject=NwbSubjectMetadata(
            subject_id="A98",
            species="Mus musculus",
            sex="M",
            age="P14D",
            description="Subject A98.",
        ),
        session_start_time=_session_start_time(first_image),
        experimenter=("Manning, Declan",),
        keywords=(
            "microscopy",
            "kymograph",
            "vascular imaging",
            "blood flow",
            "velocity analysis",
        ),
    )


def _compare_member(index: int, original: AcqImage, restored: AcqImage) -> None:
    """Materialize and compare one original/restored image pair.

    Args:
        index: Zero-based collection member index.
        original: Source acquisition.
        restored: NWB-backed acquisition.

    Returns:
        None.

    Raises:
        AssertionError: If meaningful AcqImage state differs.
    """
    original_header = original.images.header
    restored_header = restored.images.header

    assert restored.name == original.name
    assert restored_header.shape == original_header.shape
    assert restored_header.dims == original_header.dims
    assert restored_header.dtype == original_header.dtype
    assert restored_header.physical_units == original_header.physical_units
    assert restored_header.physical_units_labels == original_header.physical_units_labels
    assert restored_header.date == original_header.date
    assert restored_header.time == original_header.time
    assert restored.rois.to_list() == original.rois.to_list()
    assert (
        restored.analysis_set.serialize_json_analysis()
        == original.analysis_set.serialize_json_analysis()
    )

    original.load_images()
    restored.load_images()
    np.testing.assert_array_equal(
        restored.pixels.get_array(),
        original.pixels.get_array(),
    )

    original.load_analysis_csv()
    restored.load_analysis_csv()
    original_tables = original.analysis_set.results_tables_by_name()
    restored_tables = restored.analysis_set.results_tables_by_name()
    assert restored_tables.keys() == original_tables.keys()
    for analysis_name in original_tables:
        pd.testing.assert_frame_equal(
            restored_tables[analysis_name].reset_index(drop=True),
            original_tables[analysis_name].reset_index(drop=True),
            check_dtype=True,
        )

    print(
        f"  [{index + 1}] {restored.name}: "
        f"shape={restored_header.shape}, analyses={sorted(restored_tables)}"
    )

    original.unload_images()
    restored.unload_images()
    original.unload_analysis_csv()
    restored.unload_analysis_csv()


def main() -> None:
    """Run the local AcqImageList → NWB → AcqImageList round-trip."""
    print(f"Loading source AcqImageList: {SOURCE_FOLDER}")
    original = AcqImageList(str(SOURCE_FOLDER))
    if len(original) == 0:
        raise RuntimeError(f"No acquisition images found in {SOURCE_FOLDER}")
    print(f"Loaded {len(original)} source images")

    # NWB_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving NWB collection: {NWB_PATH}")
    original.save_as_nwb(
        NWB_PATH,
        metadata=_metadata(original.get_files()[0]),
        overwrite=True,
    )

    validation_errors = validate(path=NWB_PATH)
    if validation_errors:
        messages = "\n".join(str(error) for error in validation_errors)
        raise RuntimeError(f"PyNWB validation failed:\n{messages}")
    print("PyNWB validation passed")

    print("Reloading the NWB collection lazily")
    restored = AcqImageList.from_nwb(NWB_PATH)
    assert len(restored) == len(original)
    assert all(not image.images_loaded for image in restored)

    print("Comparing collection members")
    for index, (source_image, restored_image) in enumerate(
        zip(original, restored, strict=True)
    ):
        _compare_member(index, source_image, restored_image)

    print(f"Round trip passed for all {len(restored)} images")
    print(f"CloudScope test file: {NWB_PATH}")


if __name__ == "__main__":
    main()
