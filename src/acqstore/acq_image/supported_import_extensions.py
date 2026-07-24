"""Canonical acquisition import extensions (no leading dot).

These values are used for directory discovery and for :func:`create_file_loader`
validation. Extensions are compared case-insensitively on disk. Directory-backed
Zarr stores use compound suffixes such as ``ome.zarr`` and ``cs.ome.zarr``.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from acqstore.acq_image.file_loaders.loader_registry import get_registered_import_extensions

DEFAULT_IMPORT_EXTENSIONS: tuple[str, ...] = get_registered_import_extensions()
_runtime_import_extensions: set[str] = set(DEFAULT_IMPORT_EXTENSIONS)
_COMPOUND_ZARR_EXTENSIONS = tuple(
    sorted(
        (extension for extension in DEFAULT_IMPORT_EXTENSIONS if extension.endswith('.zarr') or extension.endswith('.zarr.zip')),
        key=len,
        reverse=True,
    )
)


def _normalize_extension(extension: str) -> str:
    """Normalize one extension to canonical storage form."""
    return extension.strip().lower().lstrip('.')


def _validate_supported_extensions(extensions: set[str]) -> None:
    unsupported = extensions.difference(DEFAULT_IMPORT_EXTENSIONS)
    if not unsupported:
        return
    supported_text = ', '.join(DEFAULT_IMPORT_EXTENSIONS)
    unsupported_text = ', '.join(sorted(unsupported))
    raise ValueError(
        f'Unsupported acquisition import extension(s): {unsupported_text}; '
        f'expected one of: {supported_text}'
    )


def get_supported_import_extensions() -> tuple[str, ...]:
    """Return acquisition import extensions with registered file loaders.

    Returns:
        Sorted canonical extensions without leading dots.
    """
    return DEFAULT_IMPORT_EXTENSIONS


def normalize_import_extension_for_path(path: str | Path) -> str:
    """Return the supported import extension represented by ``path``.

    ``Path.suffix`` only reports ``.zarr`` for OME-Zarr directory stores, so
    acqstore uses this helper wherever extension checks need to recognize
    compound store names.

    Args:
        path: File or directory-backed store path.

    Returns:
        Canonical extension without a leading dot, or an empty string when no
        suffix exists.
    """
    name = Path(path).name.lower()
    for compound in _COMPOUND_ZARR_EXTENSIONS:
        if name.endswith(f'.{compound}'):
            return compound
    return Path(name).suffix.lower().lstrip('.')


def path_has_allowed_import_extension(path: str | Path) -> bool:
    """Return whether ``path`` uses a currently allowed import extension.

    Args:
        path: File or directory-backed store path.

    Returns:
        True when ``path`` matches the runtime allowed import extension set.
    """
    return normalize_import_extension_for_path(path) in _runtime_import_extensions


def get_allowed_import_extensions() -> tuple[str, ...]:
    """Return runtime allowed import extensions in sorted order.

    Returns:
        Sorted canonical extensions without leading dots. The returned set is
        always a subset of :func:`get_supported_import_extensions`.
    """
    return tuple(sorted(_runtime_import_extensions))


def set_allowed_import_extensions(extensions: Iterable[str]) -> None:
    """Replace runtime allowed import extensions.

    Args:
        extensions: Iterable of extensions with or without leading dots.

    Raises:
        ValueError: If the normalized set is empty.
        ValueError: If any extension has no registered file loader.
    """
    normalized = {
        _normalize_extension(extension)
        for extension in extensions
        if extension.strip()
    }
    if not normalized:
        raise ValueError('Allowed import extensions must not be empty')
    _validate_supported_extensions(normalized)
    _runtime_import_extensions.clear()
    _runtime_import_extensions.update(normalized)


def add_allowed_import_extension(extension: str) -> None:
    """Add one runtime allowed import extension.

    Args:
        extension: Extension with or without a leading dot.

    Raises:
        ValueError: If ``extension`` is empty or has no registered file loader.
    """
    normalized = _normalize_extension(extension)
    if not normalized:
        raise ValueError('Extension must not be empty')
    _validate_supported_extensions({normalized})
    _runtime_import_extensions.add(normalized)


def remove_allowed_import_extension(extension: str) -> None:
    """Remove one runtime allowed import extension.

    Raises:
        KeyError: If extension is not currently allowed.
    """
    _runtime_import_extensions.remove(_normalize_extension(extension))


def reset_allowed_import_extensions() -> None:
    """Reset runtime allowed import extensions to loader-backed defaults.

    Returns:
        None.
    """
    _runtime_import_extensions.clear()
    _runtime_import_extensions.update(DEFAULT_IMPORT_EXTENSIONS)
