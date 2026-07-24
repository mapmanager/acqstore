"""Tests for server-side acquisition upload storage."""

from __future__ import annotations

from pathlib import Path

import pytest

import acqstore.upload_store as upload_store
from acqstore.acq_image.supported_import_extensions import reset_allowed_import_extensions
from acqstore.upload_store import (
    UnsupportedExtensionError,
    UploadCollisionError,
    get_upload_dir,
    store_uploaded_file,
    validate_acq_filename,
)


@pytest.fixture(autouse=True)
def reset_extensions() -> None:
    """Restore the default import-extension registry around each test."""
    reset_allowed_import_extensions()


def test_get_upload_dir_honors_environment_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    upload_dir = tmp_path / 'uploads'
    monkeypatch.setenv(upload_store.UPLOAD_DIR_ENV, str(upload_dir))

    assert get_upload_dir() == upload_dir.resolve(strict=False)
    assert upload_dir.is_dir()


def test_get_upload_dir_uses_platformdirs_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(upload_store.UPLOAD_DIR_ENV, raising=False)
    monkeypatch.setattr(upload_store, 'user_data_dir', lambda app_name: str(tmp_path / app_name))

    assert get_upload_dir() == tmp_path / upload_store.DEFAULT_APP_NAME / 'uploads'
    assert (tmp_path / upload_store.DEFAULT_APP_NAME / 'uploads').is_dir()


@pytest.mark.parametrize(
    'filename',
    ['file.tif', 'file.TIF', 'file.oir', 'file.czi', 'file.ome.zarr', 'file.cs.ome.zarr'],
)
def test_validate_acq_filename_accepts_allowed_extensions(filename: str) -> None:
    assert validate_acq_filename(filename) == filename


@pytest.mark.parametrize('filename', ['', '   ', 'folder/file.tif', r'folder\file.tif', '.', '..'])
def test_validate_acq_filename_rejects_invalid_names(filename: str) -> None:
    with pytest.raises(ValueError):
        validate_acq_filename(filename)


def test_validate_acq_filename_rejects_unsupported_extension() -> None:
    with pytest.raises(UnsupportedExtensionError, match='Unsupported acquisition file extension'):
        validate_acq_filename('file.png')


def test_store_uploaded_file_copies_content_to_upload_dir(tmp_path: Path) -> None:
    src = tmp_path / 'source.tmp'
    src.write_bytes(b'acq bytes')
    upload_dir = tmp_path / 'uploads'

    stored = store_uploaded_file(src, original_filename='sample.oir', upload_dir=upload_dir)

    assert stored == upload_dir / 'sample.oir'
    assert stored.read_bytes() == b'acq bytes'
    assert src.read_bytes() == b'acq bytes'


def test_store_uploaded_file_rejects_existing_target(tmp_path: Path) -> None:
    src = tmp_path / 'source.tmp'
    src.write_bytes(b'new')
    upload_dir = tmp_path / 'uploads'
    upload_dir.mkdir()
    existing = upload_dir / 'sample.oir'
    existing.write_bytes(b'existing')

    with pytest.raises(UploadCollisionError, match='already exists'):
        store_uploaded_file(src, original_filename='sample.oir', upload_dir=upload_dir)

    assert existing.read_bytes() == b'existing'


def test_store_uploaded_file_revalidates_extension(tmp_path: Path) -> None:
    src = tmp_path / 'source.tmp'
    src.write_bytes(b'data')

    with pytest.raises(UnsupportedExtensionError):
        store_uploaded_file(src, original_filename='sample.txt', upload_dir=tmp_path / 'uploads')


def test_store_uploaded_file_removes_temp_file_after_success(tmp_path: Path) -> None:
    src = tmp_path / 'source.tmp'
    src.write_bytes(b'data')
    upload_dir = tmp_path / 'uploads'

    stored = store_uploaded_file(src, original_filename='sample.czi', upload_dir=upload_dir)

    assert stored.exists()
    assert list(upload_dir.glob('*.tmp')) == []
    assert list(upload_dir.glob('.*.tmp')) == []
