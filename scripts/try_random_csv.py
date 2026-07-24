"""Exercise AcqImageList manifest CSV randomization.

Edit the constants at the top of this script to point at a local dataset. The
script intentionally has no CLI arguments so it remains a simple development
probe for the backend API.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from acqstore.acq_image.acq_image_list import AcqImageList, PathKind

# Configure the dataset and sampling rule for this development probe.
LOAD_PATH = Path('/Users/cudmore/Sites/cloudscope-data/declan-data/manuscript_velocity_202606')
GROUPBY_COLUMN = 'grandparent'
N_PER_GROUP = 5


def main() -> None:
    """Load a folder, write randomized manifests, and reload the sample."""
    # Build per-run output names in the same folder as the source dataset.
    timestamp = datetime.now().strftime('%Y%m%d_%H_%M_%S')
    master_csv_path = LOAD_PATH / f'{timestamp}_randomized_master.csv'
    sampled_csv_path = LOAD_PATH / f'{timestamp}_randomized_sampled_n{N_PER_GROUP}.csv'

    # Load file metadata lazily before writing manifests.
    print(f'Loading: {LOAD_PATH}')
    result = AcqImageList.load_safe(
        str(LOAD_PATH),
        kind=PathKind.FOLDER,
        load_images=False,
        load_analysis_csv=False,
    )
    image_list = result.acq_image_list
    print(f'Loaded {len(image_list)} file(s); warnings={len(result.warnings)}')

    # Surface non-fatal load warnings so the sampled output can be interpreted.
    for warning in result.warnings:
        print(f'WARNING: {warning.message}: {warning.path}')

    # Write the complete randomized order once; the sampled CSV reads this file.
    image_list.to_randomized_manifest_master_csv(
        master_csv_path,
        groupby_column=GROUPBY_COLUMN,
        root_path=LOAD_PATH,
    )

    # Sample the first N rows per group from the saved randomized master CSV.
    image_list.to_randomized_manifest_csv(
        sampled_csv_path,
        master_csv_path=master_csv_path,
        n_per_group=N_PER_GROUP,
        allow_unbalanced=False,
    )
    print(f'Wrote randomized master: {master_csv_path}')
    print(f'Wrote randomized sample: {sampled_csv_path}')

    # Reload the sampled manifest as a smoke test that relative paths resolve.
    reload_result = AcqImageList.load_safe(
        str(sampled_csv_path),
        kind=PathKind.CSV,
        root_path=LOAD_PATH,
        load_images=False,
        load_analysis_csv=False,
    )
    print(
        f'Reloaded sampled manifest: {len(reload_result.acq_image_list)} file(s); '
        f'warnings={len(reload_result.warnings)}'
    )


if __name__ == '__main__':
    main()
