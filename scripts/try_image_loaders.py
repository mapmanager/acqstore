from acqstore.acq_image.supported_import_extensions import get_allowed_import_extensions

from acqstore.acq_image.acq_image_list import _build_file_list

from acqstore.acq_image.file_loaders.oir_file_loader import OirFileLoader
from acqstore.acq_image.file_loaders.tiff_file_loader import TiffFileLoader
from acqstore.acq_image.file_loaders.czi_file_loader import CziFileLoader

from acqstore.utils.logging import setup_logging

if __name__ == "__main__":
    from pprint import pprint

    setup_logging()

    allowed = get_allowed_import_extensions()

    # tif
    if 1:
        path = '/Users/cudmore/Sites/cloudscope/tests/acqstore/data/tif-samples'
        files = sorted(_build_file_list(path, ["tif"]))
        for _idx, file in enumerate(files):
            print(f"=== {_idx}:{len(files)} {file}")
            tif_image = TiffFileLoader(file)
            pprint(tif_image.header.as_dict(), indent=4, width=120, sort_dicts=False)
            tif_image.load_image_data()
            channel = 0
            img_data = tif_image.get_slice_data(channel)
            print(f'  channel "{channel}" img_data is: {img_data.shape} {img_data.dtype} min:{img_data.min()} max:{img_data.max()}')
    # oir
    if 1:
        # path = "/Users/cudmore/Documents/KymFlow/declan_oir_20260409/20251030 WT Male 28d Saline"
        path = '/Users/cudmore/Sites/cloudscope/tests/acqstore/data/oir-samples'
        files = sorted(_build_file_list(path, ["oir"]))
        for _idx, file in enumerate(files):
            print(f"=== {_idx}:{len(files)} {file}")

            oir_image = OirFileLoader(file)
            # oir_image = AcqImage(file)

            pprint(oir_image.header.as_dict(), indent=4, width=120, sort_dicts=False)
            oir_image.load_image_data()
            channel = 0
            img_data = oir_image.get_slice_data(channel)
            print(
                f'  channel "{channel}" img_data is: {img_data.shape} {img_data.dtype} min:{img_data.min()} max:{img_data.max()}'
            )
            if _idx == 3:
                break

    # czi
    if 1:
        # path = '/Users/cudmore/Sites/kymflow_outer/kymflow/src/kymflow/core/image_loaders/image_loader_plugins/tests/fixtures/czi-samples/disjointedlinescansandframescans'
        path = '/Users/cudmore/Sites/cloudscope/tests/acqstore/data/czi-samples'
        files = _build_file_list(path, ["czi"])
        for file in files:
            czi_image = CziFileLoader(file)
            print("czi_image.header:")
            pprint(czi_image.header.as_dict(), indent=4, width=120, sort_dicts=False)
            czi_image.load_image_data()
            channel = 1
            img_data = czi_image.get_channel_data(channel)
            print(
                f'  channel "{channel}" img_data is: {img_data.shape} {img_data.dtype} min:{img_data.min()} max:{img_data.max()}'
            )
