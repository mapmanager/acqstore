"""Persist uploaded acquisition files for AcqStore-backed applications."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from uuid import uuid4

from platformdirs import user_data_dir

from acqstore.acq_image.supported_import_extensions import (
    get_allowed_import_extensions,
    normalize_import_extension_for_path,
)
from acqstore.sample_data import DEFAULT_APP_NAME

UPLOAD_DIR_ENV = 'CLOUDSCOPE_UPLOAD_DIR'


class UploadError(RuntimeError):
    """Base error for uploaded acquisition file storage."""


class UploadCollisionError(UploadError):
    """Raised when an upload target filename already exists."""


class UnsupportedExtensionError(UploadError, ValueError):
    """Raised when an uploaded filename has an unsupported acquisition suffix."""


def get_upload_dir() -> Path:
    """Return the server-side directory used for uploaded acquisition files.

    Resolution order:

    1. ``CLOUDSCOPE_UPLOAD_DIR`` when set.
    2. ``platformdirs.user_data_dir("cloudscope") / "uploads"``.

    Returns:
        Existing upload directory path.

    Raises:
        OSError: If the directory cannot be created.
    """
    env_path = os.getenv(UPLOAD_DIR_ENV)
    if env_path:
        root = Path(env_path).expanduser().resolve(strict=False)
    else:
        root = Path(user_data_dir(DEFAULT_APP_NAME)) / 'uploads'
    root.mkdir(parents=True, exist_ok=True)
    return root


def validate_acq_filename(filename: str) -> str:
    """Validate one uploaded acquisition filename and return its basename.

    Args:
        filename: Original filename supplied by the upload event.

    Returns:
        Sanitized basename safe to place directly in the upload directory.

    Raises:
        TypeError: If ``filename`` is not a string.
        ValueError: If the filename is empty or includes path separators.
        UnsupportedExtensionError: If the suffix is not allowed by AcqStore.
    """
    if not isinstance(filename, str):
        raise TypeError(f'filename must be str, got {type(filename).__name__}')

    name = filename.strip()
    if not name:
        raise ValueError('Uploaded filename must not be empty')
    if '\x00' in name:
        raise ValueError('Uploaded filename must not contain null bytes')
    if '/' in name or '\\' in name or Path(name).name != name:
        raise ValueError(f'Uploaded filename must be a basename, got {filename!r}')
    if name in {'.', '..'}:
        raise ValueError(f'Uploaded filename is not valid: {filename!r}')

    suffix = normalize_import_extension_for_path(name)
    allowed = set(get_allowed_import_extensions())
    if suffix not in allowed:
        allowed_text = ', '.join(sorted(allowed))
        raise UnsupportedExtensionError(
            f'Unsupported acquisition file extension {suffix!r}; expected one of: {allowed_text}'
        )
    return name


def store_uploaded_file(
    src_path: Path,
    *,
    original_filename: str,
    upload_dir: Path | None = None,
) -> Path:
    """Copy one uploaded acquisition file into the server upload directory.

    Args:
        src_path: Readable temporary upload path produced by the upload widget.
        original_filename: Original client filename used for validation and the
            final stored basename.
        upload_dir: Optional target upload directory override for tests or
            deployment-specific callers.

    Returns:
        Final persisted file path.

    Raises:
        FileNotFoundError: If ``src_path`` does not exist or is not a file.
        UploadCollisionError: If the final filename already exists.
        UnsupportedExtensionError: If ``original_filename`` is unsupported.
        OSError: If copying or atomic replacement fails.
    """
    src = Path(src_path).expanduser()
    if not src.exists() or not src.is_file():
        raise FileNotFoundError(f'Uploaded source file does not exist: {src}')

    target_dir = Path(upload_dir).expanduser().resolve(strict=False) if upload_dir is not None else get_upload_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    filename = validate_acq_filename(original_filename)
    target = target_dir / filename
    if target.exists():
        raise UploadCollisionError(f'Uploaded file already exists: {target}')

    tmp_path = target_dir / f'.{target.name}.{uuid4().hex}.tmp'
    try:
        with src.open('rb') as src_handle, tmp_path.open('xb') as tmp_handle:
            shutil.copyfileobj(src_handle, tmp_handle)
        if target.exists():
            raise UploadCollisionError(f'Uploaded file already exists: {target}')
        tmp_path.replace(target)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    return target
