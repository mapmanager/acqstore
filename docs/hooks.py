"""MkDocs hooks for AcqStore documentation."""

from __future__ import annotations

import tomllib
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
