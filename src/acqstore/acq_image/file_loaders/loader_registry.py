"""Registry for acquisition file loader implementations."""

from __future__ import annotations

from collections.abc import Callable

from .base_file_loader import BaseFileLoader
from .czi_file_loader import CziFileLoader
from .oir_file_loader import OirFileLoader
from .nd2_file_loader import Nd2FileLoader
from .ome_zarr_file_loader import OmeZarrFileLoader
from .tiff_file_loader import TiffFileLoader

FileLoaderFactory = Callable[[str], BaseFileLoader]

_FILE_LOADER_FACTORIES: dict[str, FileLoaderFactory] = {
    'tif': lambda path: TiffFileLoader(path, load_olympus_header=True),
    'oir': OirFileLoader,
    'czi': CziFileLoader,
    'nd2': Nd2FileLoader,
    'ome.zarr': OmeZarrFileLoader,
    'cs.ome.zarr': OmeZarrFileLoader,
    'ome.zarr.zip': OmeZarrFileLoader,
    'cs.ome.zarr.zip': OmeZarrFileLoader,
}


def get_registered_import_extensions() -> tuple[str, ...]:
    """Return import extensions with registered acquisition file loaders.

    Returns:
        Sorted canonical extensions without leading dots.
    """
    return tuple(sorted(_FILE_LOADER_FACTORIES))


def create_registered_file_loader(path: str, extension: str) -> BaseFileLoader:
    """Create the registered file loader for ``extension``.

    Args:
        path: Filesystem path to an acquisition file or directory-backed store.
        extension: Canonical or dotted extension to load.

    Returns:
        A concrete loader instance for ``path``.

    Raises:
        ValueError: If ``extension`` has no registered file loader.
    """
    normalized = extension.strip().lower().lstrip('.')
    try:
        factory = _FILE_LOADER_FACTORIES[normalized]
    except KeyError as exc:
        raise ValueError(f'No loader registered for extension {normalized!r}') from exc
    return factory(path)
