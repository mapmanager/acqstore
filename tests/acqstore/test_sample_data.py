"""Tests for AcqStore sample-data catalog/download/extract helpers."""

from __future__ import annotations

import json
from pathlib import Path
import zipfile

import pytest

import acqstore.sample_data as sample_data_module
from acqstore.sample_data import (
    SampleDataError,
    SampleDataset,
    UnknownSampleError,
    ensure_sample,
    ensure_sample_file,
    get_sample,
    get_sample_data_dir,
    list_samples,
)


_SHA256 = 'a' * 64


def _catalog_item(*, sample_id: str = 'unit-sample', label: str = 'Unit Sample', **extra: object) -> dict[str, object]:
    return {
        'id': sample_id,
        'label': label,
        'description': 'Unit-test sample data.',
        'url': f'https://example.invalid/{sample_id}.zip',
        'sha256': _SHA256,
        **extra,
    }


def _catalog_text(*, sample_id: str = 'unit-sample', label: str = 'Unit Sample') -> str:
    return json.dumps([_catalog_item(sample_id=sample_id, label=label)])


def _sample(*, primary_file: str | None = None) -> SampleDataset:
    return SampleDataset(
        name='unit-sample',
        label='Unit Sample',
        description='Unit-test sample data.',
        url='https://example.invalid/unit-sample.zip',
        sha256=_SHA256,
        primary_file=primary_file,
    )


def _make_zip(path: Path, *, root_name: str = 'unit-sample') -> None:
    with zipfile.ZipFile(path, 'w') as zf:
        zf.writestr(f'{root_name}/cond1/a.oir', 'raw')
        zf.writestr(f'{root_name}/cond1/a.oir.json', '{}')


@pytest.fixture(autouse=True)
def _reset_catalog_cache(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(sample_data_module, '_CATALOG', None)
    monkeypatch.setenv('CLOUDSCOPE_SAMPLE_DATA_DIR', str(tmp_path / 'sample-cache'))


def test_list_samples_fetches_parses_and_caches_catalog(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sample_data_module, '_fetch_catalog', lambda: _catalog_text())

    samples = list_samples()

    assert samples == (_sample(),)
    cache_path = tmp_path / 'sample-cache' / '_catalog' / 'catalog.json'
    assert json.loads(cache_path.read_text(encoding='utf-8'))[0]['id'] == 'unit-sample'


def test_list_samples_preserves_catalog_display_order(monkeypatch) -> None:
    catalog = [
        {
            'id': 'velocity-sample-data',
            'label': 'Velocity Sample Data',
            'description': 'Velocity sample.',
            'url': 'https://example.invalid/velocity-sample-data.zip',
            'sha256': _SHA256,
        },
        {
            'id': 'diameter-sample-data',
            'label': 'Diameter Sample Data',
            'description': 'Diameter sample.',
            'url': 'https://example.invalid/diameter-sample-data.zip',
            'sha256': 'b' * 64,
        },
    ]
    monkeypatch.setattr(sample_data_module, '_fetch_catalog', lambda: json.dumps(catalog))

    assert [sample.name for sample in list_samples()] == [
        'velocity-sample-data',
        'diameter-sample-data',
    ]


def test_list_samples_reuses_in_memory_catalog(monkeypatch) -> None:
    calls = 0

    def _fetch() -> str:
        nonlocal calls
        calls += 1
        return _catalog_text()

    monkeypatch.setattr(sample_data_module, '_fetch_catalog', _fetch)

    assert list_samples() == list_samples()
    assert calls == 1


def test_list_samples_uses_cached_catalog_when_fetch_fails(tmp_path, monkeypatch) -> None:
    cache_path = tmp_path / 'sample-cache' / '_catalog' / 'catalog.json'
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text(_catalog_text(), encoding='utf-8')

    def _fail_fetch() -> str:
        raise SampleDataError('offline')

    monkeypatch.setattr(sample_data_module, '_fetch_catalog', _fail_fetch)

    assert list_samples() == (_sample(),)


def test_list_samples_rejects_duplicate_ids(monkeypatch) -> None:
    item = json.loads(_catalog_text())[0]
    monkeypatch.setattr(sample_data_module, '_fetch_catalog', lambda: json.dumps([item, item]))

    with pytest.raises(SampleDataError, match='duplicate id'):
        list_samples()


def test_get_sample_returns_catalog_sample(monkeypatch) -> None:
    monkeypatch.setattr(sample_data_module, '_fetch_catalog', lambda: _catalog_text())

    sample = get_sample('unit-sample')

    assert sample.name == 'unit-sample'
    assert sample.label == 'Unit Sample'


def test_get_sample_raises_for_unknown_sample(monkeypatch) -> None:
    monkeypatch.setattr(sample_data_module, '_fetch_catalog', lambda: _catalog_text())

    with pytest.raises(UnknownSampleError, match='Unknown sample dataset'):
        get_sample('missing')


def test_get_sample_data_dir_uses_env_override(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv('CLOUDSCOPE_SAMPLE_DATA_DIR', str(tmp_path / 'sample-cache'))
    assert get_sample_data_dir() == (tmp_path / 'sample-cache').resolve(strict=False)


def test_get_sample_data_dir_uses_acqstore_app_name(monkeypatch) -> None:
    monkeypatch.delenv('CLOUDSCOPE_SAMPLE_DATA_DIR', raising=False)
    monkeypatch.setattr(sample_data_module, 'user_data_dir', lambda app_name: f'/tmp/{app_name}')

    assert get_sample_data_dir() == Path('/tmp/acqstore/sample-data')


def test_ensure_sample_extracts_archive_and_returns_extracted_dir(tmp_path, monkeypatch) -> None:
    archive = tmp_path / 'archive.zip'
    _make_zip(archive)
    monkeypatch.setattr(sample_data_module, '_CATALOG', (_sample(),))
    monkeypatch.setattr('acqstore.sample_data._retrieve_archive', lambda _sample, _archive_dir: archive)

    load_path = ensure_sample('unit-sample', sample_data_dir=tmp_path / 'cache')

    assert load_path == tmp_path / 'cache' / f'unit-sample-{_SHA256[:12]}' / 'unit-sample'
    assert (load_path / 'cond1' / 'a.oir').is_file()
    assert (load_path.parent / '.acqstore_sample_extracted').is_file()


def test_ensure_sample_reuses_existing_extracted_sample(tmp_path, monkeypatch) -> None:
    sample = _sample()
    load_path = tmp_path / 'cache' / sample.cache_key / sample.name
    load_path.mkdir(parents=True)
    marker = load_path.parent / '.acqstore_sample_extracted'
    marker.write_text('done', encoding='utf-8')
    monkeypatch.setattr(sample_data_module, '_CATALOG', (sample,))

    def _fail_retrieve(_sample, _archive_dir):
        raise AssertionError('should not retrieve when marker and folder exist')

    monkeypatch.setattr('acqstore.sample_data._retrieve_archive', _fail_retrieve)

    assert ensure_sample(sample.name, sample_data_dir=tmp_path / 'cache') == load_path


def test_ensure_sample_raises_when_expected_directory_missing(tmp_path, monkeypatch) -> None:
    archive = tmp_path / 'archive.zip'
    _make_zip(archive, root_name='unexpected-root')
    sample = _sample()
    monkeypatch.setattr(sample_data_module, '_CATALOG', (sample,))
    monkeypatch.setattr('acqstore.sample_data._retrieve_archive', lambda _sample, _archive_dir: archive)

    with pytest.raises(SampleDataError, match='did not extract expected directory'):
        ensure_sample(sample.name, sample_data_dir=tmp_path / 'cache')


def test_ensure_sample_rejects_unsafe_zip_member(tmp_path, monkeypatch) -> None:
    archive = tmp_path / 'archive.zip'
    with zipfile.ZipFile(archive, 'w') as zf:
        zf.writestr('../evil.txt', 'bad')
    sample = _sample()
    monkeypatch.setattr(sample_data_module, '_CATALOG', (sample,))
    monkeypatch.setattr('acqstore.sample_data._retrieve_archive', lambda _sample, _archive_dir: archive)

    with pytest.raises(SampleDataError, match='Unsafe path'):
        ensure_sample(sample.name, sample_data_dir=tmp_path / 'cache')


def test_catalog_item_without_primary_file_is_a_folder_sample(monkeypatch) -> None:
    monkeypatch.setattr(sample_data_module, '_fetch_catalog', lambda: _catalog_text())

    sample = get_sample('unit-sample')

    assert sample.primary_file is None
    assert sample.is_single_file is False


def test_catalog_item_with_primary_file_is_a_single_file_sample(monkeypatch) -> None:
    catalog = [_catalog_item(primary_file='cond1/a.oir')]
    monkeypatch.setattr(sample_data_module, '_fetch_catalog', lambda: json.dumps(catalog))

    sample = get_sample('unit-sample')

    assert sample.primary_file == 'cond1/a.oir'
    assert sample.is_single_file is True


@pytest.mark.parametrize('primary_file', ['', '   ', '/abs/a.oir', '../a.oir', 'cond1/../a.oir', 'cond1\\a.oir', 'cond1/', 12])
def test_catalog_rejects_unsafe_primary_file(monkeypatch, primary_file) -> None:
    catalog = [_catalog_item(primary_file=primary_file)]
    monkeypatch.setattr(sample_data_module, '_fetch_catalog', lambda: json.dumps(catalog))

    with pytest.raises(SampleDataError, match="invalid 'primary_file'"):
        list_samples()


def test_ensure_sample_file_returns_primary_file_path(tmp_path, monkeypatch) -> None:
    archive = tmp_path / 'archive.zip'
    _make_zip(archive)
    sample = _sample(primary_file='cond1/a.oir')
    monkeypatch.setattr(sample_data_module, '_CATALOG', (sample,))
    monkeypatch.setattr('acqstore.sample_data._retrieve_archive', lambda _sample, _archive_dir: archive)

    file_path = ensure_sample_file('unit-sample', sample_data_dir=tmp_path / 'cache')

    assert file_path == tmp_path / 'cache' / f'unit-sample-{_SHA256[:12]}' / 'unit-sample' / 'cond1' / 'a.oir'
    assert file_path.is_file()


def test_ensure_sample_file_supports_directory_backed_primary_file(tmp_path, monkeypatch) -> None:
    archive = tmp_path / 'archive.zip'
    with zipfile.ZipFile(archive, 'w') as zf:
        zf.writestr('unit-sample/a.ome.zarr/.zgroup', '{}')
        zf.writestr('unit-sample/a.ome.zarr/0/0', 'raw')
    sample = _sample(primary_file='a.ome.zarr')
    monkeypatch.setattr(sample_data_module, '_CATALOG', (sample,))
    monkeypatch.setattr('acqstore.sample_data._retrieve_archive', lambda _sample, _archive_dir: archive)

    file_path = ensure_sample_file('unit-sample', sample_data_dir=tmp_path / 'cache')

    assert file_path.is_dir()
    assert (file_path / '.zgroup').is_file()


def test_ensure_sample_file_rejects_folder_sample(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sample_data_module, '_CATALOG', (_sample(),))

    def _fail_retrieve(_sample, _archive_dir):
        raise AssertionError('should not download a folder sample')

    monkeypatch.setattr('acqstore.sample_data._retrieve_archive', _fail_retrieve)

    with pytest.raises(SampleDataError, match='folder sample'):
        ensure_sample_file('unit-sample', sample_data_dir=tmp_path / 'cache')


def test_ensure_sample_file_raises_when_primary_file_missing(tmp_path, monkeypatch) -> None:
    archive = tmp_path / 'archive.zip'
    _make_zip(archive)
    sample = _sample(primary_file='cond1/missing.oir')
    monkeypatch.setattr(sample_data_module, '_CATALOG', (sample,))
    monkeypatch.setattr('acqstore.sample_data._retrieve_archive', lambda _sample, _archive_dir: archive)

    with pytest.raises(SampleDataError, match='missing its primary file'):
        ensure_sample_file('unit-sample', sample_data_dir=tmp_path / 'cache')


def test_ensure_sample_file_raises_for_unknown_sample(monkeypatch) -> None:
    monkeypatch.setattr(sample_data_module, '_fetch_catalog', lambda: _catalog_text())

    with pytest.raises(UnknownSampleError, match='Unknown sample dataset'):
        ensure_sample_file('missing')
