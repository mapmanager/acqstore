"""Download and prepare reusable AcqStore sample datasets.

This module is intentionally GUI-independent. Applications such as CloudScope
can call :func:`ensure_sample` and then pass the returned folder path to
``AcqImageList`` using the same code path as a user-selected folder.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen
import zipfile

from platformdirs import user_data_dir


SAMPLE_DATA_DIR_ENV = 'CLOUDSCOPE_SAMPLE_DATA_DIR'
DEFAULT_APP_NAME = 'acqstore'
CATALOG_URL = 'https://raw.githubusercontent.com/mapmanager/cloudscope-data/main/catalog.json'

_CATALOG_CACHE_DIR = '_catalog'
_CATALOG_CACHE_FILENAME = 'catalog.json'
_CATALOG: tuple['SampleDataset', ...] | None = None


@dataclass(frozen=True, slots=True)
class SampleDataset:
    """Catalog entry for one downloadable sample dataset.

    Attributes:
        name: Stable sample identifier from the catalog ``id`` field.
        label: User-facing sample label.
        description: Short description suitable for UI help text.
        url: Remote zip archive URL.
        sha256: Expected archive SHA-256 digest, without the ``sha256:`` prefix.
    """

    name: str
    label: str
    description: str
    url: str
    sha256: str

    @property
    def cache_key(self) -> str:
        """Return content-addressed cache directory name for this sample."""
        return f'{self.name}-{self.sha256[:12]}'

    @property
    def archive_filename(self) -> str:
        """Return deterministic local archive filename."""
        return f'{self.name}.zip'

    @property
    def known_hash(self) -> str:
        """Return Pooch-compatible hash string."""
        return f'sha256:{self.sha256}'


class SampleDataError(RuntimeError):
    """Base error for sample-data operations."""


class UnknownSampleError(SampleDataError):
    """Raised when a requested sample name is not present in the catalog."""


def list_samples() -> tuple[SampleDataset, ...]:
    """Return catalog sample datasets in catalog display order."""
    global _CATALOG
    if _CATALOG is None:
        _CATALOG = _load_catalog()
    return _CATALOG


def get_sample(name: str) -> SampleDataset:
    """Return one sample dataset from the catalog.

    Args:
        name: Catalog sample identifier.

    Returns:
        Sample dataset definition.

    Raises:
        UnknownSampleError: If ``name`` is not present in the catalog.
    """
    samples = {sample.name: sample for sample in list_samples()}
    try:
        return samples[name]
    except KeyError as exc:
        known = ', '.join(sorted(samples)) or '<none>'
        raise UnknownSampleError(f'Unknown sample dataset {name!r}; known samples: {known}') from exc


def get_sample_data_dir() -> Path:
    """Return root directory used for downloaded sample data.

    Resolution order:

    1. ``CLOUDSCOPE_SAMPLE_DATA_DIR`` when set.
    2. ``platformdirs.user_data_dir("acqstore") / "sample-data"``.

    On macOS the default is usually
    ``~/Library/Application Support/acqstore/sample-data``.
    """
    env_path = os.getenv(SAMPLE_DATA_DIR_ENV)
    if env_path:
        return Path(env_path).expanduser().resolve(strict=False)
    return Path(user_data_dir(DEFAULT_APP_NAME)) / 'sample-data'


def ensure_sample(name: str, *, sample_data_dir: str | Path | None = None) -> Path:
    """Ensure a sample dataset is downloaded/extracted and return its load folder.

    Args:
        name: Catalog sample identifier.
        sample_data_dir: Optional cache root override. Primarily useful for tests
            or scripts; deployment should normally use ``CLOUDSCOPE_SAMPLE_DATA_DIR``.

    Returns:
        Local folder path that can be passed to ``AcqImageList`` as a folder.

    Raises:
        UnknownSampleError: If ``name`` is not present in the catalog.
        SampleDataError: If the archive cannot be downloaded, validated, or
            extracted into the expected directory.
    """
    sample = get_sample(name)
    root = Path(sample_data_dir).expanduser().resolve(strict=False) if sample_data_dir is not None else get_sample_data_dir()
    sample_root = root / sample.cache_key
    load_path = sample_root / sample.name
    marker_path = sample_root / '.acqstore_sample_extracted'

    if load_path.is_dir() and marker_path.is_file():
        return load_path

    sample_root.mkdir(parents=True, exist_ok=True)
    archive_path = _retrieve_archive(sample, sample_root / '_archives')
    _extract_zip(archive_path, sample_root)

    if not load_path.is_dir():
        raise SampleDataError(f'Sample {sample.name!r} did not extract expected directory {sample.name!r} from {archive_path}')

    marker_path.write_text(f'{sample.name}\n{sample.sha256}\n', encoding='utf-8')
    return load_path


def _load_catalog() -> tuple[SampleDataset, ...]:
    """Fetch, cache, parse, and validate the sample catalog."""
    cache_path = get_sample_data_dir() / _CATALOG_CACHE_DIR / _CATALOG_CACHE_FILENAME
    try:
        catalog_text = _fetch_catalog()
    except SampleDataError:
        if not cache_path.is_file():
            raise
        try:
            catalog_text = cache_path.read_text(encoding='utf-8')
        except OSError as exc:
            raise SampleDataError(f'Could not read cached sample catalog {cache_path}: {exc}') from exc
    else:
        _write_catalog_cache(cache_path, catalog_text)

    return _parse_catalog(catalog_text)


def _fetch_catalog() -> str:
    """Download the catalog JSON text from ``cloudscope-data``."""
    try:
        with urlopen(CATALOG_URL, timeout=30) as response:  # noqa: S310 - fixed trusted HTTPS URL
            return response.read().decode('utf-8')
    except (OSError, UnicodeError, URLError) as exc:
        raise SampleDataError(f'Could not fetch sample catalog from {CATALOG_URL}: {exc}') from exc


def _write_catalog_cache(cache_path: Path, catalog_text: str) -> None:
    """Write catalog text atomically to the local cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix('.tmp')
    try:
        tmp_path.write_text(catalog_text, encoding='utf-8')
        tmp_path.replace(cache_path)
    except OSError as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        raise SampleDataError(f'Could not cache sample catalog at {cache_path}: {exc}') from exc


def _parse_catalog(catalog_text: str) -> tuple[SampleDataset, ...]:
    """Parse catalog JSON into validated sample definitions."""
    try:
        raw_catalog = json.loads(catalog_text)
    except json.JSONDecodeError as exc:
        raise SampleDataError(f'Sample catalog is not valid JSON: {exc}') from exc

    if not isinstance(raw_catalog, list):
        raise SampleDataError('Sample catalog must be a JSON array')

    samples: list[SampleDataset] = []
    seen_names: set[str] = set()
    for index, item in enumerate(raw_catalog):
        if not isinstance(item, dict):
            raise SampleDataError(f'Sample catalog item {index} must be a JSON object')
        sample = _parse_catalog_item(item, index=index)
        if sample.name in seen_names:
            raise SampleDataError(f'Sample catalog contains duplicate id {sample.name!r}')
        seen_names.add(sample.name)
        samples.append(sample)

    return tuple(samples)


def _parse_catalog_item(item: dict[str, Any], *, index: int) -> SampleDataset:
    """Parse and validate one catalog object."""
    required = ('id', 'label', 'description', 'url', 'sha256')
    values: dict[str, str] = {}
    for key in required:
        value = item.get(key)
        if not isinstance(value, str) or not value.strip():
            raise SampleDataError(f'Sample catalog item {index} has invalid {key!r}')
        values[key] = value.strip()

    sha256 = values['sha256'].lower()
    if len(sha256) != 64 or any(char not in '0123456789abcdef' for char in sha256):
        raise SampleDataError(f'Sample catalog item {index} has invalid SHA-256 digest')
    if not values['url'].startswith('https://'):
        raise SampleDataError(f'Sample catalog item {index} URL must use HTTPS')

    return SampleDataset(
        name=values['id'],
        label=values['label'],
        description=values['description'],
        url=values['url'],
        sha256=sha256,
    )


def _retrieve_archive(sample: SampleDataset, archive_dir: Path) -> Path:
    """Download/validate archive with Pooch and return local archive path."""
    try:
        import pooch
    except ImportError as exc:  # pragma: no cover - exercised only without optional dependency
        raise SampleDataError('Sample data support requires the pooch package. Install with: uv add pooch') from exc

    archive_dir.mkdir(parents=True, exist_ok=True)
    try:
        path = pooch.retrieve(
            url=sample.url,
            known_hash=sample.known_hash,
            fname=sample.archive_filename,
            path=archive_dir,
        )
    except Exception as exc:  # pragma: no cover - exact exception types vary by transport/hash failure
        raise SampleDataError(f'Could not retrieve sample dataset {sample.name!r}: {exc}') from exc
    return Path(path)


def _extract_zip(archive_path: Path, destination: Path) -> None:
    """Safely extract ``archive_path`` into ``destination``.

    Existing sample contents are replaced atomically enough for local app usage:
    extraction happens into a temporary sibling directory, then extracted entries
    are moved into ``destination``.
    """
    destination.mkdir(parents=True, exist_ok=True)
    tmp_dir = destination / '._extracting'
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    try:
        with zipfile.ZipFile(archive_path) as zf:
            for member in zf.infolist():
                _validate_zip_member(member.filename)
            zf.extractall(tmp_dir)

        for child in tmp_dir.iterdir():
            target = destination / child.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            child.replace(target)
    except zipfile.BadZipFile as exc:
        raise SampleDataError(f'Sample archive is not a valid zip file: {archive_path}') from exc
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)


def _validate_zip_member(name: str) -> None:
    """Reject zip entries that would escape the extraction directory."""
    path = Path(name)
    if path.is_absolute() or '..' in path.parts:
        raise SampleDataError(f'Unsafe path in sample archive: {name!r}')
