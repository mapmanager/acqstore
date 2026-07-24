"""Try the AcqImageList velocity analysis pool from a hardcoded folder path.

Edit ``LOAD_PATH`` before running.

Run:

    uv run python scripts/try_analysis_pool.py
"""

from __future__ import annotations

from pathlib import Path
from pprint import pprint

from acqstore.acq_image import AcqImageList
from acqstore.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)
setup_logging()

# Edit this path for your local machine.
LOAD_PATH = "/Users/cudmore/Sites/cloudscope-data/demo-velocity/20251030"


def print_pool_summary(images: AcqImageList) -> None:
    """Print a compact summary of the velocity analysis pool.

    Args:
        images: Loaded acquisition image list.

    Returns:
        None.
    """
    pool = images.velocity_analysis_pool
    df = pool.get_dataframe()

    print("\n=== VelocityAnalysisPool ===")
    print(f"loaded files: {len(images)}")
    print(f"pool rows: {len(df)}")
    print(f"pool columns: {len(df.columns)}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  {column}")

    print("\nHead:")
    if df.empty:
        print("  <empty>")
    else:
        print(df.head().to_string(index=False))


def try_refresh_first_row(images: AcqImageList) -> None:
    """Refresh the first pool row using the public row-refresh API.

    Args:
        images: Loaded acquisition image list.

    Returns:
        None.
    """
    pool = images.velocity_analysis_pool
    df = pool.get_dataframe()

    if df.empty:
        print("\nNo pool rows available to refresh.")
        return

    first = df.iloc[0]
    file_id = str(first["path"])
    channel = int(first["channel"])
    roi_id = int(first["roi_id"])

    print("\n=== Refresh first row ===")
    print(f"file_id: {file_id}")
    print(f"channel: {channel}")
    print(f"roi_id: {roi_id}")

    pool.refresh_row(file_id, channel=channel, roi_id=roi_id)

    refreshed = pool.get_dataframe()
    matching = refreshed.loc[refreshed["pool_row_id"] == first["pool_row_id"]]

    print("\nRefreshed row:")
    if matching.empty:
        print("  <missing>")
    else:
        pprint(matching.iloc[0].to_dict(), indent=4, sort_dicts=False)


def try_remove_and_rebuild(images: AcqImageList) -> None:
    """Remove one row from the pool, then rebuild from AcqImage state.

    Args:
        images: Loaded acquisition image list.

    Returns:
        None.
    """
    pool = images.velocity_analysis_pool
    df = pool.get_dataframe()

    if df.empty:
        print("\nNo pool rows available to remove/rebuild.")
        return

    first = df.iloc[0]
    file_id = str(first["path"])
    channel = int(first["channel"])
    roi_id = int(first["roi_id"])

    print("\n=== Remove first row then rebuild ===")
    print(f"starting rows: {len(df)}")

    pool.remove_row(file_id, channel=channel, roi_id=roi_id)
    print(f"after remove_row(): {len(pool.get_dataframe())}")

    pool.rebuild()
    print(f"after rebuild(): {len(pool.get_dataframe())}")


def try_export_csv(images: AcqImageList) -> None:
    """Export the pool DataFrame to a CSV file next to this script.

    Args:
        images: Loaded acquisition image list.

    Returns:
        None.
    """
    out_path = Path(__file__).with_name("try_analysis_pool_output.csv")

    images.velocity_analysis_pool.to_csv(out_path)

    print("\n=== Export CSV ===")
    print(f"wrote: {out_path}")


def main() -> None:
    """Run the manual velocity analysis pool workflow.

    Returns:
        None.
    """
    logger.info("Loading AcqImageList from %s", LOAD_PATH)
    images = AcqImageList(LOAD_PATH)

    print_pool_summary(images)
    try_refresh_first_row(images)
    try_remove_and_rebuild(images)
    try_export_csv(images)


if __name__ == "__main__":
    main()