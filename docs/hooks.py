"""MkDocs hooks for AcqStore documentation."""

from __future__ import annotations

import shutil
import tomllib
from importlib.resources import files
from pathlib import Path
from typing import Any


def on_config(config: Any) -> Any:
    """Expose the AcqStore package version to MkDocs templates.

    The documentation footer displays the version from ``pyproject.toml`` so
    rendered docs clearly indicate which AcqStore source version was used to
    build the site.
    """
    pyproject_path = Path("pyproject.toml")
    version = "unknown"

    if pyproject_path.exists():
        data = tomllib.loads(pyproject_path.read_text())
        version = data.get("project", {}).get("version", "unknown")

    config.setdefault("extra", {})
    config["extra"]["acqstore_version"] = version
    return config


def on_post_build(config: Any) -> None:
    """Publish the canonical packaged v1 schema at its documented URL.

    Args:
        config: MkDocs configuration containing the built ``site_dir``.

    Returns:
        None.
    """
    schema = files('acqstore.acq_image.io.ome_zarr_collection_v1.schema').joinpath(
        'acqstore-ome-zarr-collection-v1.schema.json'
    )
    destination = Path(config['site_dir']) / 'schemas' / schema.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    with schema.open('rb') as source, destination.open('wb') as target:
        shutil.copyfileobj(source, target)
