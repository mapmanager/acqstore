"""Storage helpers for local, ZIP, and S3-backed Zarr stores."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def is_s3_path(path: str | Path) -> bool:
    """Return whether ``path`` uses the ``s3://`` URI scheme.

    Args:
        path: Path-like value to inspect.

    Returns:
        True when ``path`` starts with ``s3://`` after string coercion.
    """
    return str(path).lower().startswith('s3://')


def is_zip_store_path(path: str | Path) -> bool:
    """Return whether ``path`` names a local ZIP-backed Zarr store.

    Args:
        path: Path-like value to inspect.

    Returns:
        True when the path ends with ``.zip``.
    """
    return str(path).lower().endswith('.zip')


def join_store_path(base: str | Path, *parts: str) -> str:
    """Join path parts for local filesystem or S3 URI destinations.

    Args:
        base: Base directory/store path.
        *parts: Relative path segments to append.

    Returns:
        Joined path string. S3 URIs retain the ``s3://`` prefix.
    """
    base_text = str(base).rstrip('/')
    cleaned = [part.strip('/') for part in parts if part]
    if not cleaned:
        return base_text
    if is_s3_path(base_text):
        return '/'.join([base_text, *cleaned])
    return str(Path(base_text).joinpath(*cleaned))


def ensure_store_absent(path: str | Path, *, overwrite: bool) -> None:
    """Remove an existing local destination when overwrite is requested.

    Args:
        path: Local destination path.
        overwrite: Whether to remove an existing destination.

    Raises:
        FileExistsError: If ``path`` exists and ``overwrite`` is false.
        ValueError: If called for an S3 path. S3 cleanup is delegated to writer
            libraries or AWS CLI because recursive deletion semantics are
            backend-specific.
    """
    if is_s3_path(path):
        if not overwrite:
            return
        raise ValueError('S3 overwrite cleanup is not implemented by acqstore; remove the S3 prefix first')
    dest = Path(path)
    if not dest.exists():
        return
    if not overwrite:
        raise FileExistsError(f'Destination already exists: {dest}')
    if dest.is_dir():
        shutil.rmtree(dest)
    else:
        dest.unlink()


def write_json_file(path: str | Path, payload: dict[str, Any]) -> None:
    """Write an indented JSON object to a local or S3 path.

    Args:
        path: Destination JSON path. ``s3://`` paths require ``s3fs`` and
            configured AWS credentials.
        payload: JSON-serializable object to write.

    Raises:
        ValueError: If ``path`` targets a member inside an existing ZIP store.
    """
    text = json.dumps(payload, indent=2, sort_keys=True)
    path_text = str(path)
    if is_s3_path(path_text):
        fs = _s3_filesystem()
        parent = path_text.rsplit('/', 1)[0]
        fs.makedirs(parent, exist_ok=True)
        with fs.open(path_text, 'w') as f:
            f.write(text)
        return
    if '.zip/' in path_text.lower():
        raise ValueError('Writing JSON directly inside an existing ZIP store is not supported')
    out = Path(path_text)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding='utf-8')


def read_json_file(path: str | Path) -> dict[str, Any]:
    """Read a JSON object from a local, S3, or ZIP-member path.

    Args:
        path: Source JSON path.

    Returns:
        Parsed JSON object.

    Raises:
        ValueError: If the JSON root is not an object.
    """
    path_text = str(path)
    if is_s3_path(path_text):
        fs = _s3_filesystem()
        with fs.open(path_text, 'r') as f:
            raw = json.load(f)
    elif '.zip/' in path_text.lower():
        zip_path, member = split_zip_member_path(path_text)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            raw = json.loads(zf.read(member).decode('utf-8'))
    else:
        raw = json.loads(Path(path_text).read_text(encoding='utf-8'))
    if not isinstance(raw, dict):
        raise ValueError(f'Expected JSON object in {path}')
    return raw


def path_exists(path: str | Path) -> bool:
    """Return whether a local, S3, or ZIP-member path exists.

    Args:
        path: Path to test.

    Returns:
        True when the path exists.
    """
    path_text = str(path)
    if is_s3_path(path_text):
        return bool(_s3_filesystem().exists(path_text))
    if '.zip/' in path_text.lower():
        zip_path, member = split_zip_member_path(path_text)
        if not Path(zip_path).is_file():
            return False
        with zipfile.ZipFile(zip_path, 'r') as zf:
            return member in set(zf.namelist())
    return Path(path_text).exists()


def split_zip_member_path(path: str | Path) -> tuple[Path, str]:
    """Split ``/path/store.zip/member`` into zip path and member name.

    Args:
        path: Combined ZIP-member path.

    Returns:
        ``(zip_path, member_name)``.

    Raises:
        ValueError: If ``path`` does not contain ``.zip/``.
    """
    text = str(path)
    lower = text.lower()
    marker = '.zip/'
    idx = lower.find(marker)
    if idx < 0:
        raise ValueError(f'Path does not identify a ZIP member: {path}')
    zip_text = text[: idx + len('.zip')]
    member = text[idx + len(marker) :].lstrip('/')
    if not member:
        raise ValueError(f'ZIP member path is empty: {path}')
    return Path(zip_text), member


def zip_directory_store(source_dir: str | Path, zip_path: str | Path, *, overwrite: bool = False) -> None:
    """Create a ZIP-backed store from a directory store.

    The archive contains the contents of ``source_dir`` at the ZIP root, not an
    extra top-level folder.

    Args:
        source_dir: Directory store to archive.
        zip_path: Destination ``.zip`` file.
        overwrite: Whether to replace an existing ZIP file.

    Raises:
        FileExistsError: If ``zip_path`` exists and ``overwrite`` is false.
        NotADirectoryError: If ``source_dir`` is not a directory.
    """
    src = Path(source_dir)
    if not src.is_dir():
        raise NotADirectoryError(f'Expected directory store: {src}')
    dest = Path(zip_path)
    if dest.exists():
        if not overwrite:
            raise FileExistsError(f'Destination already exists: {dest}')
        dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for item in sorted(src.rglob('*')):
            if item.is_file():
                zf.write(item, item.relative_to(src).as_posix())


def _s3_filesystem() -> Any:
    """Return an authenticated S3 filesystem.

    Returns:
        ``s3fs.S3FileSystem`` instance.

    Raises:
        ImportError: If ``s3fs`` is not installed.
    """
    try:
        import s3fs
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError('S3 paths require optional dependency s3fs. Install with: uv add s3fs boto3') from exc
    return s3fs.S3FileSystem()
