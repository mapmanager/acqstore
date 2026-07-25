"""Factory for concrete :class:`BaseFileLoader` instances by file path."""

from __future__ import annotations

from acqstore.acq_image.supported_import_extensions import (
    get_allowed_import_extensions,
    normalize_import_extension_for_path,
)

from .base_file_loader import BaseFileLoader
from .loader_registry import create_registered_file_loader


def create_file_loader(path: str) -> BaseFileLoader:
    """Return a file loader appropriate for ``path``.

    AcqStore selects a loader from the file extension. Supported formats include
    proprietary microscopy files (``.oir``, ``.czi``, ``.nd2``) and open formats
    (``.tif``, ``.ome.zarr``). The native ``.cs.ome.zarr`` variant is registered
    for stores that carry AcqStore metadata, ROIs, and analysis; that format is
    still under development.

    Only extensions listed in :func:`get_allowed_import_extensions` are supported.
    Comparison is case-insensitive. Directory-backed OME-Zarr stores are detected
    by compound suffixes such as ``.ome.zarr`` and ``.cs.ome.zarr``.

    Args:
        path: Filesystem path to an acquisition file or directory-backed store.

    Returns:
        A concrete loader instance.

    Raises:
        ValueError: If the path suffix is not a supported acquisition extension.
    """
    suffix = normalize_import_extension_for_path(path)
    allowed = set(get_allowed_import_extensions())
    if suffix not in allowed:
        allowed_text = ', '.join(sorted(allowed))
        raise ValueError(
            f'Unsupported acquisition file extension {suffix!r}; expected one of: {allowed_text}'
        )
    return create_registered_file_loader(path, suffix)
