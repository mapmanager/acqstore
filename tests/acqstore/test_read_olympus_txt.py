"""Tests for Olympus kymograph sidecar parsing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from acqstore.acq_image.file_loaders.read_olympus_txt import (
    _dtype_from_olympus_bits,
    _find_olympus_txt_file,
    _image_header_from_olympus_dict,
    _olympus_combined_datetime_to_yyyymmdd_hhmmss,
    _olympus_legacy_date_time_parts,
    read_olympus_txt_dict,
)


def _write_minimal_olympus_txt(path: Path) -> None:
    """Write a minimal sidecar matching the parser's line-split expectations."""
    path.write_text(
        "\n".join(
            [
                '"Channel Dimension" 1 1',
                '"X Dimension" 0 0 0 0 0 0.125',
                '"Image Size" "512" x "1000" y',
                '"T Dimension" 0 0 0 100.0',
                '"Date"\t"10/30/2025 02:54:36.454 PM"',
                '"Bits/Pixel" "16"',
            ]
        ),
        encoding="utf-8",
    )


def test_find_olympus_txt_file_uses_same_stem_as_tif(tmp_path: Path) -> None:
    """Companion txt should be resolved from the tif basename."""
    tif = tmp_path / "kymo.tif"
    txt = tmp_path / "kymo.txt"
    tif.write_bytes(b"not-a-real-tif")
    txt.write_text('"T Dimension" 0 0 0 0 0 1.0', encoding="utf-8")

    assert _find_olympus_txt_file(tif) == str(txt)


def test_read_olympus_txt_dict_returns_none_when_sidecar_missing(tmp_path: Path) -> None:
    """Missing sidecar should return None rather than raising."""
    tif = tmp_path / "solo.tif"
    tif.write_bytes(b"x")

    assert read_olympus_txt_dict(tif) is None


def test_read_olympus_txt_dict_parses_required_fields(tmp_path: Path) -> None:
    """Sidecar parser should populate geometry and timing fields."""
    tif = tmp_path / "sample.tif"
    tif.write_bytes(b"x")
    _write_minimal_olympus_txt(tmp_path / "sample.txt")

    parsed = read_olympus_txt_dict(tif)

    assert parsed is not None
    assert parsed["pixelsPerLine"] == 512
    assert parsed["numLines"] == 1000
    assert parsed["umPerPixel"] == pytest.approx(0.125)
    assert parsed["durImage_sec"] == pytest.approx(100.0)
    assert parsed["secondsPerLine"] == pytest.approx(0.1)
    assert parsed["bitsPerPixel"] == 16
    assert parsed["olympusDateTimeCombined"] == "10/30/2025 02:54:36.454 PM"
    assert parsed["tifChannelPaths"] == {1: tif}


def test_image_header_from_olympus_dict_maps_y_time_x_space(tmp_path: Path) -> None:
    """ImageHeader should follow kymograph axis policy: Y=lines, X=pixels."""
    tif = tmp_path / "sample.tif"
    parsed = {
        "numLines": 1000,
        "pixelsPerLine": 512,
        "secondsPerLine": 0.1,
        "umPerPixel": 0.125,
        "bitsPerPixel": 16,
        "numChannels": 1,
        "olympusDateTimeCombined": "10/30/2025 02:54:36.454 PM",
    }

    header = _image_header_from_olympus_dict(str(tif), parsed)

    assert header.shape == (1000, 512)
    assert header.dims == ("Y", "X")
    assert header.sizes == {"Y": 1000, "X": 512}
    assert header.physical_units == (0.1, 0.125)
    assert header.physical_units_labels == ("seconds", "um")
    assert header.dtype == np.dtype(np.uint16)
    assert header.date == "20251030"
    assert header.time == "14:54:36"


def test_image_header_from_olympus_dict_raises_when_required_fields_missing() -> None:
    """Missing geometry fields should fail fast with a clear error."""
    with pytest.raises(ValueError, match="missing required fields"):
        _image_header_from_olympus_dict("/tmp/x.tif", {"numLines": 10})


@pytest.mark.parametrize(
    ("bits", "expected"),
    [
        (None, np.uint16),
        (8, np.uint8),
        (16, np.uint16),
        (32, np.uint32),
    ],
)
def test_dtype_from_olympus_bits(bits: int | None, expected: np.dtype) -> None:
    """Bit depth mapping should follow the documented thresholds."""
    assert _dtype_from_olympus_bits(bits) == np.dtype(expected)


def test_olympus_combined_datetime_parser() -> None:
    """Combined US-style datetime strings should normalize to acqstore date/time."""
    date_s, time_s = _olympus_combined_datetime_to_yyyymmdd_hhmmss("10/30/2025 02:54:36.454 PM")
    assert date_s == "20251030"
    assert time_s == "14:54:36"


def test_olympus_legacy_date_time_parts_fallback() -> None:
    """Legacy split date/time fields should still parse when combined value is absent."""
    date_s, time_s = _olympus_legacy_date_time_parts("10/30/2025", "02:54:36 PM")
    assert date_s == "20251030"
    assert time_s == "14:54:36"
