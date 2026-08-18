"""Validate the configured AcqStore OME-Zarr collection and its child images.

Run from the AcqStore repository with its UV environment::

    uv run python scripts/ome_zarr/validate_ome_zarr_collection.py PATH.ome.zarr
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import zarr

from acqstore.acq_image.io.ome_zarr import read_acq_pixels_ome_zarr


EXPECTED_FORMAT = 'acqstore-acq-image-collection'
EXPECTED_VERSION = 1


def main() -> None:
    """Validate the collection contract, child metadata, and pyramid data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('collection', type=Path, help='AcqImageCollection .ome.zarr directory')
    args = parser.parse_args()
    root = args.collection.expanduser().resolve(strict=True)
    manifest = _load_manifest(root)
    image_entries = manifest['acq_images']
    external_validator = Path(sys.executable).parent / 'ome-zarr-models'
    failures: list[str] = []
    decoded_levels = 0
    valid_primary_images = 0
    declared_references = 0
    valid_references = 0

    root_validation = _run_external_validator(external_validator, root)
    root_status = 'IMAGE MODEL PASS' if root_validation.returncode == 0 else 'COLLECTION GROUP (no root image multiscale)'

    for entry in image_entries:
        image_id = str(entry['id'])
        child = _contained_path(root, entry['ome_zarr_path'])
        result = _run_external_validator(external_validator, child)
        if result.returncode != 0:
            failures.append(f'{image_id}: external OME-Zarr validation failed')
            continue
        valid_primary_images += 1
        pixels = read_acq_pixels_ome_zarr(child, lazy=True)
        group = zarr.open_group(str(child), mode='r')
        datasets = group.attrs['ome']['multiscales'][0]['datasets']
        for dataset in datasets:
            np.asarray(group[dataset['path']])
            decoded_levels += 1
        print(
            f'PASS {image_id}: shape={pixels.shape} axes={pixels.axes} '
            f'dtype={pixels.dtype} levels={len(datasets)} chunks={group["0"].chunks}'
        )
        reference_raw = entry.get('reference_image_path')
        if reference_raw is None:
            continue
        declared_references += 1
        reference_path = _contained_path(root, reference_raw)
        reference_result = _run_external_validator(external_validator, reference_path)
        if reference_result.returncode != 0:
            failures.append(f'{image_id}/reference: external OME-Zarr validation failed')
            continue
        valid_references += 1
        reference_pixels = read_acq_pixels_ome_zarr(reference_path, lazy=True)
        reference_group = zarr.open_group(str(reference_path), mode='r')
        reference_datasets = reference_group.attrs['ome']['multiscales'][0]['datasets']
        for dataset in reference_datasets:
            np.asarray(reference_group[dataset['path']])
            decoded_levels += 1
        print(
            f'PASS {image_id}/reference: shape={reference_pixels.shape} '
            f'axes={reference_pixels.axes} dtype={reference_pixels.dtype} '
            f'levels={len(reference_datasets)} chunks={reference_group["0"].chunks}'
        )

    unexpected = sorted(path.relative_to(root).as_posix() for path in root.rglob('*') if path.name in {'.DS_Store', 'Thumbs.db'})
    if unexpected:
        failures.append(f'unexpected platform files: {", ".join(unexpected)}')

    print()
    print(f'Collection: {root}')
    print(f'Images declared: {len(image_entries)}')
    print(f'Primary OME-Zarr images valid: {valid_primary_images}/{len(image_entries)}')
    print(f'Reference OME-Zarr images valid: {valid_references}/{declared_references}')
    print(f'Pyramid levels decoded: {decoded_levels}')
    print('Manifest: PASS')
    print(f'Root OME image model: {root_status}')
    print(f'Unexpected files: {len(unexpected)}')
    if failures:
        for failure in failures:
            print(f'FAIL: {failure}')
        raise SystemExit(1)
    print('Overall child-image result: PASS')


def _load_manifest(root: Path) -> dict[str, object]:
    """Load and validate the root collection manifest.

    Args:
        root: Collection root directory.

    Returns:
        Parsed collection manifest.

    Raises:
        ValueError: If the manifest contract is malformed.
    """
    manifest_path = root / 'acqstore' / 'acq_image_collection.json'
    raw = json.loads(manifest_path.read_text(encoding='utf-8'))
    if not isinstance(raw, dict) or raw.get('format') != EXPECTED_FORMAT:
        raise ValueError('Invalid AcqStore collection manifest format')
    if raw.get('version') != EXPECTED_VERSION:
        raise ValueError(f'Unsupported AcqImageCollection version: {raw.get("version")!r}')
    images = raw.get('acq_images')
    if not isinstance(images, list) or not images:
        raise ValueError('Collection manifest must declare at least one image')
    ids: set[str] = set()
    paths: set[str] = set()
    for entry in images:
        if not isinstance(entry, dict):
            raise ValueError('Collection image entries must be objects')
        image_id = str(entry.get('id', ''))
        image_path = str(entry.get('ome_zarr_path', ''))
        if not image_id or image_id in ids:
            raise ValueError(f'Duplicate or empty image id: {image_id!r}')
        if not image_path or image_path in paths:
            raise ValueError(f'Duplicate or empty image path: {image_path!r}')
        ids.add(image_id)
        paths.add(image_path)
        _contained_path(root, image_path)
        _contained_path(root, entry.get('sidecar_path')).resolve(strict=True)
        _contained_path(root, entry.get('manifest_path')).resolve(strict=True)
        reference_path = entry.get('reference_image_path')
        if reference_path is not None:
            _contained_path(root, reference_path).resolve(strict=True)
    tables = raw.get('analysis_tables')
    if not isinstance(tables, dict):
        raise ValueError('Collection manifest tables must be an object')
    for table_path in tables.values():
        _contained_path(root, table_path).resolve(strict=True)
    return raw


def _contained_path(root: Path, raw: object) -> Path:
    """Resolve a manifest path while preventing collection-root escape.

    Args:
        root: Collection root directory.
        raw: Manifest path value.

    Returns:
        Resolved path beneath ``root``.

    Raises:
        ValueError: If the path is empty or escapes the collection root.
    """
    if not isinstance(raw, str) or not raw:
        raise ValueError(f'Invalid manifest path: {raw!r}')
    resolved = (root / raw).resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise ValueError(f'Manifest path escapes collection root: {raw!r}')
    return resolved


def _run_external_validator(executable: Path, target: Path) -> subprocess.CompletedProcess[str]:
    """Run the environment-local OME-Zarr validator without shell expansion.

    Args:
        executable: Validator executable inside AcqStore's virtual environment.
        target: Zarr group to validate.

    Returns:
        Completed validator process with captured output.
    """
    if not executable.is_file():
        raise FileNotFoundError(f"Validator is not installed in AcqStore's environment: {executable}")
    return subprocess.run(
        [str(executable), 'validate', str(target)],
        check=False,
        capture_output=True,
        text=True,
    )


if __name__ == '__main__':
    main()
