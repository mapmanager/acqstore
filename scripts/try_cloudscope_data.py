"""Exercise the public AcqStore sample-data API.

This script lists the datasets published by ``cloudscope-data``, selects one
catalog entry, downloads and verifies its archive, and opens the installed
folder with ``AcqImageList``.

Change ``SAMPLE_ID`` to exercise a different catalog dataset.
"""

from pprint import pprint

from acqstore.acq_image import AcqImageList
from acqstore.sample_data import (
    ensure_sample,
    get_sample,
    get_sample_data_dir,
    list_samples,
)


# Stable dataset ID from cloudscope-data/catalog.json.
SAMPLE_ID = 'diameter-sample-data'


def run() -> None:
    """Exercise the complete public AcqStore sample-data workflow."""
    print('=== AcqStore sample-data cache')
    print(get_sample_data_dir())

    samples = list_samples()
    print(f'\n=== available sample datasets: {len(samples)}')
    for index, sample in enumerate(samples):
        print(f'{index}: {sample.name}')
        print(f'  label: {sample.label}')
        print(f'  description: {sample.description}')

    sample = get_sample(SAMPLE_ID)
    print(f'\n=== selected sample: {sample.name}')
    pprint(
        {
            'label': sample.label,
            'description': sample.description,
            'url': sample.url,
            'sha256': sample.sha256,
        },
        indent=4,
        width=120,
        sort_dicts=False,
    )

    print(f'\n=== ensuring sample: {sample.name}')
    sample_path = ensure_sample(sample.name)
    print(f'installed path: {sample_path}')

    print('\n=== loading sample with AcqImageList')
    acq_image_list = AcqImageList(str(sample_path), path_kind='folder')
    print(f'loaded acquisitions: {len(acq_image_list)}')

    for index, acq_image in enumerate(acq_image_list):
        print(f'\n{index}: {acq_image.name}')
        pprint(
            acq_image.get_schema_row(),
            indent=4,
            width=120,
            sort_dicts=False,
        )


if __name__ == '__main__':
    run()
