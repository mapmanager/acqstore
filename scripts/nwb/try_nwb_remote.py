"""Stream the public AcqStore NWB demo from DANDI without downloading it."""

from __future__ import annotations

import numpy as np

from acqstore.nwb_io import load_nwb


DANDI_URI = "dandi://DANDI/001947@draft/sub-A98/sub-A98.nwb"
DIRECT_S3_URL = (
    "https://dandiarchive.s3.amazonaws.com/"
    "blobs/caa/012/caa012c4-6218-46ef-b985-75e8b5c7c003"
)


def _exercise(source: str) -> tuple[tuple[int, ...], str]:
    """Load metadata lazily, then stream pixels and analysis tables."""
    image = load_nwb(source)
    print(f"  source: {source}")
    print(f"  identity: {image.path}")
    print(f"  display name: {image.name}")
    print(f"  shape: {image.images.header.shape}")
    print(f"  dims: {image.images.header.dims}")
    print(f"  pixels initially loaded: {image.images_loaded}")

    pixels = image.pixels.get_array()
    checksum = str(np.asarray(pixels, dtype=np.uint64).sum(dtype=np.uint64))
    print(f"  pixels loaded: {image.images_loaded}")
    print(f"  pixel checksum: {checksum}")

    image.load_analysis_csv()
    tables = image.analysis_set.results_tables_by_name()
    print(f"  analysis tables: {sorted(tables)}")
    print()
    return tuple(int(value) for value in pixels.shape), checksum


def main() -> None:
    """Exercise both stable DANDI identifiers and direct public S3 URLs."""
    print('=== DANDI_URI:')
    dandi_result = _exercise(DANDI_URI)
    
    print('=== DIRECT_S3_URL:')
    direct_result = _exercise(DIRECT_S3_URL)

    if dandi_result != direct_result:
        raise RuntimeError(
            "DANDI URI and direct S3 URL did not produce identical image results"
        )

    print("Remote NWB streaming demo passed.")


if __name__ == "__main__":
    main()
