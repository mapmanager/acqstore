"""Batch heart-rate analysis on files that already have velocity results.

For each file: use the first existing ROI; when that ROI has a completed
``radon_velocity`` analysis, run ``heart_rate`` and save. Never creates ROIs.

Set ``DRY_RUN = True`` to print planned actions without running heart-rate
analysis or saving.

Use ``INCLUDE_CONDITION_FOLDERS`` to limit processing to files whose path
contains one of those directory names as a path component. Comment out names
to process a smaller subset at a time. An empty list processes no files.

Run:

    uv run python scripts/dev/batch_hr_analysis.py
"""

from __future__ import annotations

from pathlib import Path

from acqstore.acq_image import AcqImageList
from acqstore.acq_image.analysis import HeartRateAnalysis, RadonVelocityAnalysis
from acqstore.acq_image.analysis.model import AnalysisKey

# --- CONFIGURATION ---
FOLDER_PATH = "/Users/cudmore/Desktop/declan_copied_results"
CHANNEL = 0
FOLDER_DEPTH = 4
DRY_RUN = False  # Set False to run heart-rate analysis and save.

# Only process files whose path includes one of these folder names as a
# directory component. Comment out entries to shrink the batch. An empty list
# processes no files.
INCLUDE_CONDITION_FOLDERS: list[str] = [
    "14d Saline",
    "28d AngII",
    "28d AngII + Recovery",
    "28d Saline",
    "28d Saline + Recovery",
]
# ---------------------


def main() -> None:
    """Run batch heart-rate analysis on every eligible file in ``FOLDER_PATH``."""
    acq_image_list = AcqImageList(FOLDER_PATH, folder_depth=FOLDER_DEPTH)
    total = len(acq_image_list)
    print(f"Found {total} file(s) in {FOLDER_PATH}")
    print(f"DRY_RUN={DRY_RUN}")
    if INCLUDE_CONDITION_FOLDERS:
        print(f"Including condition folders: {INCLUDE_CONDITION_FOLDERS}")

    failures: list[tuple[str, str]] = []
    successes = 0
    skipped = 0
    would_run = 0
    attempted = 0

    for idx, acq_image in enumerate(acq_image_list):
        label = acq_image.name
        print(f"[{idx + 1}/{total}] {label}")

        # Condition-folder filter: any path component must match an include name.
        if not INCLUDE_CONDITION_FOLDERS:
            print("  skipped (INCLUDE_CONDITION_FOLDERS is empty)")
            skipped += 1
            continue
        if not any(part in INCLUDE_CONDITION_FOLDERS for part in Path(acq_image.path).parts):
            print("  skipped (condition folder not in INCLUDE_CONDITION_FOLDERS)")
            skipped += 1
            continue

        # Heart-rate batch never creates ROIs; use first ROI in creation order.
        roi_ids = acq_image.rois.get_roi_ids()
        if not roi_ids:
            print("  skipped (no ROIs)")
            skipped += 1
            continue

        roi_id = roi_ids[0]
        print(f"  first roi_id={roi_id}")

        velocity_key = AnalysisKey(
            RadonVelocityAnalysis.analysis_name, CHANNEL, roi_id
        )
        velocity = acq_image.analysis_set.get(velocity_key)
        if velocity is None:
            print("  skipped (no radon_velocity on first ROI)")
            skipped += 1
            continue

        if velocity.get_plot_data() is None:
            print("  skipped (radon_velocity has no plot data)")
            skipped += 1
            continue

        hr_key = AnalysisKey(HeartRateAnalysis.analysis_name, CHANNEL, roi_id)
        existing_hr = acq_image.analysis_set.get(hr_key)

        if DRY_RUN:
            if existing_hr is not None:
                print(
                    f"  would replace existing heart_rate on roi_id={roi_id} and save"
                )
            else:
                print(f"  would run heart_rate on roi_id={roi_id} and save")
            would_run += 1
            continue

        try:
            attempted += 1
            if existing_hr is not None:
                print("  existing heart_rate; replacing")

            analysis = acq_image.analysis_set.create_and_run(
                HeartRateAnalysis,
                channel=CHANNEL,
                roi_id=roi_id,
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
    if DRY_RUN:
        print(f"Done (dry run). {would_run} would run, {skipped} skipped.")
    else:
        print(
            f"Done. {successes}/{attempted} succeeded, "
            f"{len(failures)} failed, {skipped} skipped."
        )
    if failures:
        print("Failures:")
        for name, msg in failures:
            print(f"  - {name}: {msg}")


if __name__ == "__main__":
    main()
