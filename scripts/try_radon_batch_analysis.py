"""Run Radon velocity batch analysis from hardcoded file paths.

Edit the constants near the top of the file before running.

Run:

    uv run python scripts/try_radon_batch_analysis.py
"""

from __future__ import annotations

from pathlib import Path

from acqstore.acq_image import AcqImage
from acqstore.acq_image.analysis.batch.acq_analysis_batch import AcqAnalysisBatch
from acqstore.acq_image.analysis.batch.radon_velocity_batch_strategy import (
    RadonVelocityBatchStrategy,
)
from acqstore.acq_image.analysis.batch.roi_mode import RoiBatchMode
from acqstore.acq_image.analysis import RadonVelocityAnalysis

SOURCE_PATHS = [
    "/path/to/file1.oir",
    "/path/to/file2.oir",
]
CHANNEL = 0
ROI_MODE = RoiBatchMode.ANALYZE_EXISTING_ROI
ROI_ID: int | None = 1
WINDOW_WIDTH = 64
MAX_PARALLEL_FILES = 2
USE_MULTIPROCESSING = True
PROCESSES: int | None = None


def load_acq_images(paths: list[str]) -> list[AcqImage]:
    """Load acquisition images.

    Args:
        paths: Acquisition file paths.

    Returns:
        Loaded acquisition images.
    """
    return [AcqImage(str(Path(path).expanduser())) for path in paths]


def build_detection_params(window_width: int) -> dict[str, object]:
    """Build validated Radon velocity detection parameters.

    Args:
        window_width: Radon analysis window width.

    Returns:
        Validated detection parameter dictionary.
    """
    params = RadonVelocityAnalysis.get_default_detection_params()
    params["window_width"] = int(window_width)
    RadonVelocityAnalysis.validate_detection_params(params)
    return params


def run_batch(acq_images: list[AcqImage]) -> None:
    """Run Radon velocity batch analysis and optionally save OK files.

    Args:
        acq_images: Acquisition images to analyze.

    Returns:
        None.
    """
    strategy = RadonVelocityBatchStrategy(
        channel=CHANNEL,
        roi_mode=ROI_MODE,
        roi_id=ROI_ID,
        detection_params=build_detection_params(WINDOW_WIDTH),
        use_multiprocessing=USE_MULTIPROCESSING,
        processes=PROCESSES,
    )
    batch = AcqAnalysisBatch(
        acq_images,
        strategy,
        max_parallel_files=MAX_PARALLEL_FILES,
    )
    batch.run(on_file_result=lambda result: print(result))

    # Save explicitly here when desired, for example:
    # for acq_image in acq_images:
    #     acq_image.save()


def main() -> None:
    """Run the manual Radon velocity batch workflow.

    Returns:
        None.
    """
    acq_images = load_acq_images(SOURCE_PATHS)
    run_batch(acq_images)


if __name__ == "__main__":
    main()
