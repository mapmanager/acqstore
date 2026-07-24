"""Batch velocity analysis on all TIFs in a folder.

For each file: reuse the first existing ROI when present, otherwise add a
full-image ROI, run Radon velocity analysis, save.

Use ``INCLUDE_CONDITION_FOLDERS`` to limit processing to files whose path
contains one of those directory names as a path component. Comment out names
to process a smaller subset at a time. An empty list processes no files.

Run:

    uv run python scripts/dev/batch_velocity_analysis.py
"""

from __future__ import annotations

from pathlib import Path

from acqstore.acq_image import AcqImage, AcqImageList
from acqstore.acq_image.analysis import RadonVelocityAnalysis
from acqstore.acq_image.analysis.model import AnalysisKey

# --- CONFIGURATION ---
FOLDER_PATH = "/Users/cudmore/Desktop/declan_copied_results"
CHANNEL = 0
WINDOW_WIDTH = 64
FOLDER_DEPTH = 4

# Only process files whose path includes one of these folder names as a
# directory component. Comment out entries to shrink the batch. An empty list
# processes no files.
INCLUDE_CONDITION_FOLDERS: list[str] = [
    # "14d Saline",  # run 20260625 at 2am
    # "28d AngII",  # run 20260625 at 2:04 am
    # "28d AngII + Recovery",  # run 20260625 at 2:15 am
    # "28d Saline",  # run 20260625 at 2:30 am
    # "28d Saline + Recovery",  # run 20260625 at 2:38 am
]
# ---------------------


def is_included_condition_folder(acq_image: AcqImage, include_folders: list[str]) -> bool:
    """Return whether ``acq_image`` lies under one of ``include_folders``.

    A file matches when any :attr:`pathlib.Path.parts` entry equals one of the
    strings in ``include_folders``.

    Args:
        acq_image: Acquisition image to test.
        include_folders: Condition folder names to include. When empty, no
            files are processed.

    Returns:
        True when the file should be processed.
    """
    if not include_folders:
        return False
    path_parts = Path(acq_image.path).parts
    return any(part in include_folders for part in path_parts)


def main() -> None:
    """Run batch velocity analysis on every file in ``FOLDER_PATH``."""
    acq_image_list = AcqImageList(FOLDER_PATH, folder_depth=FOLDER_DEPTH)
    total = len(acq_image_list)
    print(f"Found {total} file(s) in {FOLDER_PATH}")
    if INCLUDE_CONDITION_FOLDERS:
        print(f"Including condition folders: {INCLUDE_CONDITION_FOLDERS}")

    failures: list[tuple[str, str]] = []
    successes = 0
    skipped = 0
    processed = 0

    for idx, acq_image in enumerate(acq_image_list):
        label = acq_image.name
        print(f"[{idx + 1}/{total}] {label}")

        if not is_included_condition_folder(acq_image, INCLUDE_CONDITION_FOLDERS):
            print("  skipped (condition folder not in INCLUDE_CONDITION_FOLDERS)")
            skipped += 1
            continue

        processed += 1

        try:
            # Reuse first ROI in creation order, or create a full-image rect ROI.
            roi_ids = acq_image.rois.get_roi_ids()
            if roi_ids:
                roi_id = roi_ids[0]
                print(f"  reusing existing roi_id={roi_id}")
            else:
                roi_id = acq_image.rois.create_rect_roi().roi_id
                print(f"  added roi_id={roi_id}")

            analysis_key = AnalysisKey(
                RadonVelocityAnalysis.analysis_name, CHANNEL, roi_id
            )
            if acq_image.analysis_set.get(analysis_key) is not None:
                print("  existing radon_velocity analysis; replacing")

            analysis = acq_image.analysis_set.create_and_run(
                RadonVelocityAnalysis,
                channel=CHANNEL,
                roi_id=roi_id,
                detection_params={"window_width": WINDOW_WIDTH},
                replace_existing=True,
            )
            print(f"  analysis complete: {analysis.key}")

            acq_image.save()
            print("  saved")
            successes += 1

        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            print(f"  FAILED: {msg}")
            failures.append((label, msg))

    print()
    print(
        f"Done. {successes}/{processed} processed succeeded, "
        f"{len(failures)} failed, {skipped} skipped."
    )
    if failures:
        print("Failures:")
        for name, msg in failures:
            print(f"  - {name}: {msg}")


if __name__ == "__main__":
    main()
