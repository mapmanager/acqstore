"""
load from xxx

find a small dataset of in vivo line scans for example flow dataset

"""

from pprint import pprint

# todo: put acqimage and acimagelist into acqstore __init__.py
from acqstore.acq_image import AcqImage
from acqstore.acq_image import AcqImageList

from acqstore.acq_image.analysis import RadonVelocityAnalysis
from acqstore.acq_image.analysis import HeartRateAnalysis
from acqstore.acq_image.analysis.model import AnalysisRunContext
from acqstore.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)
setup_logging()

def run_radon_analysis(acq_image: AcqImage, channel: int, roi_id: int, window_width: int) -> RadonVelocityAnalysis:
    """Create and run Radon velocity analysis on one file.

    Args:
        acq_image: Acquisition image.
        channel: Channel index.
        roi_id: ROI identifier.
        window_width: Window width.

    Returns:
        Completed Radon velocity analysis.
    """

    # context = AnalysisRunContext(
    #     progress_callback=lambda fraction, message: print(f"  === progress={fraction}: {message}")
    # )

    logger.info('running analysis')
    analysis = acq_image.analysis_set.create_and_run(
        RadonVelocityAnalysis,
        channel=channel,
        roi_id=roi_id,
        detection_params={"window_width": window_width},
    )

    return analysis

def run():
    path = '/Users/cudmore/Sites/cloudscope-data/demo-velocity/20251030'

    window_width = 64
    channel = 0
    acq_image_list = AcqImageList(path)
    for _idx, acq_image in enumerate(acq_image_list):
        _schema_row= acq_image.get_schema_row()
        print(f'=== {_idx}:{len(acq_image_list)} {acq_image.name}')
        pprint(_schema_row, indent=4, width=120, sort_dicts=False)

        # add an roi
        new_roi = acq_image.rois.create_rect_roi(name='test', note='test')
        roi_id = new_roi.roi_id
        print(f'  added roi_id: {roi_id}')

        # run radon velocity analysis
        analysis = run_radon_analysis(acq_image, channel=channel, roi_id=roi_id, window_width=window_width)
        print(f'  analysis: {analysis}')

        # save acqimage
        acq_image.save()

        # run heart rate analysis
        # analysis = run_heart_rate(acq_image, channel=0, roi_id=roi_id)
        # print(f'  analysis: {analysis}')

        if _idx == 1:
            break

if __name__ == '__main__':
    run()