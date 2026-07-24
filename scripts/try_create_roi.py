"""Phase 1 ROI sidecar harness for manual local verification.

Workflow:
1) Load AcqImage from one file path.
2) Create ROIs + set experiment metadata.
3) Save sidecar JSON.
4) Reload AcqImage from same file.
5) Verify ROIs + experiment metadata round-trip.

Also supports folder mode:
- Load AcqImageList from folder path.
- For each file in list, run create/save then reload/verify using separate functions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from acqstore.acq_image import AcqImage
from acqstore.acq_image import AcqImageList
from acqstore.acq_image.roi import LineEndpoints, RectRoiBounds


# Fill these in for quick local runs, or pass --file-path / --folder-path.
DEFAULT_FILE_PATH = "/Users/cudmore/Downloads/kymflow_app/acqimage-data/data/14d Saline/20251014/20251014_A98_0002.tif"
DEFAULT_FOLDER_PATH = ""


def create_and_save_for_file(file_path: str) -> dict[str, object]:
    """Create ROIs/metadata for one file and save sidecar JSON.

    Args:
        file_path: Acquisition file path.

    Returns:
        Summary dictionary used by reload verification.
    """
    acq = AcqImage(file_path)
    before_ids = acq.rois.get_roi_ids()

    rect = acq.rois.create_rect_roi(
        bounds=None,
        name="phase1_rect",
        note="created_by_try_create_roi",
    )

    # rect = acq.rois.create_rect_roi(
    #     RectRoiBounds(dim0_start=1, dim0_stop=6, dim1_start=2, dim1_stop=9),
    #     name="phase1_rect",
    #     note="created_by_try_create_roi",
    # )

    # line = acq.rois.create_line_roi(
    #     LineEndpoints(row0=1, col0=2, row1=5, col1=8),
    #     name="phase1_line",
    #     note="created_by_try_create_roi",
    # )

    acq.apply_metadata_patch(
        "experiment_metadata",
        {
            "species": "mouse",
            "genotype": "phase1-script",
        },
    )
    acq.save()

    sidecar_path = acq.get_sidecar_json_path()
    print(f"[SAVE] {file_path}")
    print(f"       sidecar: {sidecar_path}")
    print(f"       roi ids before: {before_ids}")
    print(f"       roi ids after : {acq.rois.get_roi_ids()}")
    return {
        "expected_rect_id": rect.roi_id,
        # "expected_line_id": line.roi_id,
        "expected_species": "mouse",
        "expected_genotype": "phase1-script",
    }


def reload_and_verify_for_file(file_path: str, expected: dict[str, object]) -> None:
    """Reload one file and verify ROI + experiment metadata persisted.

    Args:
        file_path: Acquisition file path.
        expected: Expected values from create/save step.

    Raises:
        AssertionError: If verification fails.
    """
    loaded = AcqImage(file_path)
    roi_ids = loaded.rois.get_roi_ids()
    exp = loaded.get_metadata_section("experiment_metadata")

    assert expected["expected_rect_id"] in roi_ids, "Expected rect ROI id not found after reload"
    # assert expected["expected_line_id"] in roi_ids, "Expected line ROI id not found after reload"
    assert exp.species == expected["expected_species"], "Species did not round-trip"
    assert exp.genotype == expected["expected_genotype"], "Genotype did not round-trip"

    print(f"[VERIFY] {file_path}")
    print(f"         roi ids: {roi_ids}")
    print(f"         species={exp.species!r} genotype={exp.genotype!r}")


def run_single_file_roundtrip(file_path: str) -> None:
    """Run Phase 1 round-trip on one file."""
    expected = create_and_save_for_file(file_path)
    reload_and_verify_for_file(file_path, expected)


def run_folder_roundtrip(folder_path: str) -> None:
    """Run Phase 1 round-trip on all files in an acquisition folder."""
    acq_list = AcqImageList(folder_path)
    files = list(acq_list.get_files())
    if not files:
        print(f"[LIST] No acquisition files found in folder: {folder_path}")
        return

    print(f"[LIST] Loaded {len(files)} files from: {folder_path}")
    for acq_file in files:
        path = acq_file.file_id
        expected = create_and_save_for_file(path)
        reload_and_verify_for_file(path, expected)


def _resolve_cli_or_default(value: str | None, default_value: str) -> str:
    """Pick CLI value when provided, otherwise fallback default."""
    chosen = (value or "").strip() or default_value.strip()
    return chosen


def main() -> None:
    """Run script entrypoint."""
    parser = argparse.ArgumentParser(description="Phase 1 ROI create/save/reload/verify harness.")
    parser.add_argument("--file-path", default=None, help="One acquisition file path.")
    parser.add_argument("--folder-path", default=None, help="Folder path for AcqImageList workflow.")
    args = parser.parse_args()

    file_path = _resolve_cli_or_default(args.file_path, DEFAULT_FILE_PATH)
    folder_path = _resolve_cli_or_default(args.folder_path, DEFAULT_FOLDER_PATH)

    if not file_path and not folder_path:
        raise ValueError(
            "Provide --file-path and/or --folder-path, or edit DEFAULT_FILE_PATH/DEFAULT_FOLDER_PATH."
        )

    if file_path:
        resolved_file = str(Path(file_path).expanduser().resolve())
        run_single_file_roundtrip(resolved_file)

    if folder_path:
        resolved_folder = str(Path(folder_path).expanduser().resolve())
        run_folder_roundtrip(resolved_folder)


if __name__ == "__main__":
    main()
