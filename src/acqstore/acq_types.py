"""Shared acquisition data types for acqstore."""

from __future__ import annotations

from enum import StrEnum


class AcqModality(StrEnum):
    """Supported acquisition data modalities."""

    IMAGE = 'image'
    TRACE = 'trace'
