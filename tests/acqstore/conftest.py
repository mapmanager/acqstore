"""Shared pytest fixtures for acqstore tests."""

from __future__ import annotations

import pytest

from acqstore.acq_image.supported_import_extensions import reset_allowed_import_extensions


@pytest.fixture(autouse=True)
def _reset_runtime_import_extensions() -> None:
    """Keep runtime extension configuration isolated across tests."""
    reset_allowed_import_extensions()
    yield
    reset_allowed_import_extensions()
