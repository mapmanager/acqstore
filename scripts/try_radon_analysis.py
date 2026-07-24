"""Run one Radon velocity analysis from a hardcoded file path.

Edit the constants near the top of the file before running.

Run:

    uv run python scripts/try_radon_analysis.py
"""

from __future__ import annotations

# from pathlib import Path

from acqstore.acq_image import AcqImage
from acqstore.acq_image.analysis.model import AnalysisKey, AnalysisRunContext, BaseAnalysis
from acqstore.acq_image.analysis import RadonVelocityAnalysis

from acqstore.utils.logging import get_logger, setup_logging
logger = get_logger(__name__)

setup_logging()


def run_radon_analysis(acq_image: AcqImage) -> RadonVelocityAnalysis:
    """Create and run Radon velocity analysis on one file.

    Args:
        acq_image: Acquisition image.

    Returns:
        Completed Radon velocity analysis.
    """

    channel = 0
    # roi_id = 0
    window_width = 64

    # add the roi
    new_roi = acq_image.rois.create_rect_roi(name='test', note='test')
    roi_id = new_roi.roi_id

    # set detection params
    detection_params = RadonVelocityAnalysis.get_default_detection_params()
    detection_params["window_width"] = window_width
    RadonVelocityAnalysis.validate_detection_params(detection_params)

    key = AnalysisKey(RadonVelocityAnalysis.analysis_name, channel, roi_id)
    logger.info(f'key:{key}')


    # we start mutating here
    # dangerous as we can fail later for trivial reasons(like in acq_image.analysis_set.run_analysis)
    
    # make sure we do not already have analysis
    acq_image.analysis_set.remove(key)

    analysis = acq_image.analysis_set.create(
        RadonVelocityAnalysis.analysis_name,
        channel=channel,
        roi_id=roi_id,
        detection_params=detection_params,
    )
    logger.info(f'created analysis:{analysis}')

    # our callers should not have to do this level of logic
    # are we expecting acq_image.analysis_set.create to sometimes fail? if so, we need to define this in public api and document failure cases.
    # if not isinstance(analysis, RadonVelocityAnalysis):
    #     raise TypeError(f"Expected RadonVelocityAnalysis, got {type(analysis).__name__}")

    # should have good defaults
    # analysis.set_execution_options(
    #     use_multiprocessing=True,
    #     processes=None,
    # )

    context = AnalysisRunContext(
        progress_callback=lambda fraction, message: print(f"  === progress={fraction}: {message}")
    )

    logger.info(f'running analysis')
    acq_image.analysis_set.run_analysis(analysis.key, context=context)

    return analysis


def plot_radon_velocity(analysis: BaseAnalysis) -> None:
    """Plot Radon velocity results with matplotlib.

    Args:
        analysis: Radon velocity analysis object with ``time_s`` and
            ``velocity`` result columns.

    Returns:
        None.
    """
    import matplotlib.pyplot as plt

    time_s = analysis.get_column("time_s")
    velocity = analysis.get_column("velocity")
    plt.figure()
    plt.plot(time_s, velocity)
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity")
    plt.title("Radon velocity")
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Run the manual single-file Radon velocity workflow.

    Returns:
        None.
    """
    # load a folder of oir and grab first oir path
    path = '/Users/cudmore/Sites/cloudscope/example-data'
    from acqstore.acq_image import AcqImageList
    acq_image_list = AcqImageList(path)
    path = acq_image_list.file_list[1]
    
    print("Radon velocity detection schema (available):")
    for field in RadonVelocityAnalysis.get_detection_schema():
        print(
            f"  {field.name}: default={field.default!r}, "
            f"type={field.value_type.value}, choices={field.choices}, "
            f"description={field.description}"
        )

    acq_image = AcqImage(path)

    analysis = run_radon_analysis(acq_image)
    
    print("Analysis key:", analysis.key)
    print("Summary:", analysis.result.summary)
    print("Columns:", analysis.get_table_columns())
    
    logger.info('saving')
    acq_image.save()

    plot_radon_velocity(analysis)

    if 1:

        logger.info('reloading ...')
        reloaded = AcqImage(path)
        loaded = reloaded.analysis_set.get_required(analysis.key)
        print("  Reloaded:")
        print("  Analysis key:", loaded.key)
        print("  Summary:", loaded.result.summary)
        print("  Columns:", loaded.get_table_columns())


    # add a new roi with no analysis
    new_roi = acq_image.rois.create_rect_roi(name='test_no_analysis', note='note_test_no_analysis')

if __name__ == "__main__":
    main()
