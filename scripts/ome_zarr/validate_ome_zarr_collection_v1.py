"""Validate AcqStore OME-Zarr Collection v1 additive resources."""

from __future__ import annotations

import argparse
from pathlib import Path

from acqstore.acq_image.io.ome_zarr_collection_v1.validator import (
    ConformanceError,
    get_schema_path,
    validate_collection,
)


def main() -> int:
    """Validate a collection using the canonical packaged schema.

    Returns:
        Process exit status: zero for valid and one for invalid.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('collection_root', type=Path)
    args = parser.parse_args()
    try:
        validate_collection(
            args.collection_root.expanduser().resolve(strict=False),
            get_schema_path(),
        )
    except ConformanceError as exc:
        print(f'INVALID: {exc}')
        return 1
    print('VALID: AcqStore OME-Zarr Collection v1 additive resources')
    print('NOTE: validate each primary/reference image independently as OME-NGFF.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
