"""Local development driver for AcqStore Web Dataset v1 export.

Edit SOURCE_PATH and DESTINATION_PATH below, then run this file directly.  The
real reusable API is ``load_and_export_web_dataset`` in ``web_export.py``.
"""

from pathlib import Path

from acqstore.acq_image.web_export import load_and_export_web_dataset


# Edit these two paths for your local data and desired export location.
SOURCE_PATH = Path("/Users/cudmore/Sites/cs_project/cloudscope-web/data/input")
DESTINATION_PATH = Path("/Users/cudmore/Sites/cs_project/cloudscope-web/data/output")

# Optional display name.  None uses the source-root folder name.
DATASET_NAME: str | None = None

# Development behavior: replace the complete prior export at DESTINATION_PATH.
OVERWRITE = True


def main() -> None:
    report = load_and_export_web_dataset(
        SOURCE_PATH,
        DESTINATION_PATH,
        name=DATASET_NAME,
        overwrite=OVERWRITE,
    )

    print(f"Exported {report.exported_images}/{report.discovered_images} images")
    print(f"Destination: {report.destination}")
    if report.warnings:
        print(f"Load warnings: {len(report.warnings)}")
        for warning in report.warnings:
            print(f"- {warning.message}: {warning.path or ''}")


if __name__ == "__main__":
    main()
