from acqstore.acq_image.acq_image_list import _build_file_list

from acqstore.acq_image import AcqImage
from acqstore.acq_image.acq_image import get_allowed_import_extensions

from acqstore.utils.logging import get_logger, setup_logging
logger = get_logger(__name__)

setup_logging()

if __name__ == "__main__":
    from pprint import pprint

    allowed = get_allowed_import_extensions()
    logger.info(f'allowed_import_extensions={allowed}')

    # tif
    if 1:
        path = '/Users/cudmore/Sites/cloudscope/tests/acqstore/data/tif-samples'
        path = '/Users/cudmore/Sites/cloudscope/tests/acqstore/data/oir-samples'
        # path = '/Users/cudmore/Sites/cloudscope/tests/acqstore/data/czi-samples'
        # files = sorted(_build_file_list(path, ALLOWED_IMPORT_EXTENSIONS))
        files = sorted(_build_file_list(path, allowed))
        for _idx, file in enumerate(files):
            print(f"=== {_idx}:{len(files)} {file}")
            
            acq_image = AcqImage(file)
            pprint(acq_image.images.header.as_dict(), indent=4, width=120, sort_dicts=False)
            
            acq_image.images.load_image_data()
            channel = 0
            img_data = acq_image.images.get_slice_data(channel)
            print(f'  channel "{channel}" img_data is: {img_data.shape} {img_data.dtype} min:{img_data.min()} max:{img_data.max()}')