"""Olympus kymograph sidecar ``.txt`` parsing helper.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from datetime import datetime

import numpy as np

from acqstore.utils.logging import get_logger
from .base_file_loader import ImageHeader

logger = get_logger(__name__)


def _dtype_from_olympus_bits(bits: int | None) -> np.dtype:
    if bits is None:
        return np.dtype(np.uint16)
    if bits <= 8:
        return np.dtype(np.uint8)
    if bits <= 16:
        return np.dtype(np.uint16)
    return np.dtype(np.uint32)

def _image_header_from_olympus_dict(path: str, d: dict[str, Any]) -> ImageHeader:
    """Build :class:`ImageHeader` from :func:`~olympus_txt_kym.read_olympus_txt_dict` output.

    Kymograph axes: ``Y`` = lines (time), ``X`` = pixels (space), matching 2D TIFF policy.
    """
    nl = d.get("numLines")
    pl = d.get("pixelsPerLine")
    spl = d.get("secondsPerLine")
    um = d.get("umPerPixel")
    if nl is None or pl is None or spl is None or um is None:
        raise ValueError(
            f"Olympus dict missing required fields for ImageHeader: "
            f"numLines={nl!r} pixelsPerLine={pl!r} secondsPerLine={spl!r} umPerPixel={um!r}"
        )
    shape = (int(nl), int(pl))
    dims = ("Y", "X")
    sizes = {"Y": shape[0], "X": shape[1]}
    bp = d.get("bitsPerPixel")
    bits: int | None
    if bp is None:
        bits = None
    else:
        try:
            bits = int(bp)
        except (TypeError, ValueError):
            bits = None
    dtype = _dtype_from_olympus_bits(bits)
    num_channels = int(d.get("numChannels", 1))
    physical_units = (float(spl), float(um))
    physical_units_labels = ("seconds", "um")
    combined = d.get("olympusDateTimeCombined")
    if isinstance(combined, str) and combined.strip():
        date_s, time_s = _olympus_combined_datetime_to_yyyymmdd_hhmmss(combined)
    else:
        date_s, time_s = _olympus_legacy_date_time_parts(
            d.get("dateStr"),
            d.get("timeStr"),
        )
    return ImageHeader(
        path=path,
        shape=shape,
        dims=dims,
        sizes=sizes,
        dtype=dtype,
        num_channels=num_channels,
        num_scenes=1,
        physical_units=physical_units,
        physical_units_labels=physical_units_labels,
        date=date_s,
        time=time_s,
    )

def _olympus_legacy_date_time_parts(
    date_str: object | None,
    time_str: object | None,
) -> tuple[str, str]:
    """Fallback when ``olympusDateTimeCombined`` is absent (best-effort)."""
    date_out = ""
    if isinstance(date_str, str) and date_str.strip():
        for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
            try:
                date_out = datetime.strptime(date_str.strip(), fmt).strftime("%Y%m%d")
                break
            except ValueError:
                continue
    time_out = ""
    if isinstance(time_str, str) and time_str.strip():
        for fmt in ("%H:%M:%S", "%I:%M:%S %p", "%I:%M:%S.%f %p"):
            try:
                time_out = datetime.strptime(time_str.strip(), fmt).strftime("%H:%M:%S")
                break
            except ValueError:
                continue
    return (date_out, time_out)

def _iso8601_datetime_str_to_yyyymmdd_hhmmss(s: str) -> tuple[str, str]:
    """Parse oirfile-style ISO 8601 string; return ``(YYYYMMDD, HH:MM:SS)`` or empty strings."""
    t = s.strip()
    if not t:
        return ("", "")
    if t.endswith("Z"):
        t = t[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(t)
    except ValueError:
        return ("", "")
    return (dt.strftime("%Y%m%d"), dt.strftime("%H:%M:%S"))


def _olympus_combined_datetime_to_yyyymmdd_hhmmss(combined: str | None) -> tuple[str, str]:
    """Parse Olympus ``.txt`` combined US-style date/time line."""
    if combined is None:
        return ("", "")
    c = str(combined).strip()
    if not c:
        return ("", "")
    fmts = (
        "%m/%d/%Y %I:%M:%S.%f %p",
        "%m/%d/%Y %I:%M:%S %p",
        "%m/%d/%Y %H:%M:%S.%f",
        "%m/%d/%Y %H:%M:%S",
    )
    for fmt in fmts:
        try:
            dt = datetime.strptime(c, fmt)
            return (dt.strftime("%Y%m%d"), dt.strftime("%H:%M:%S"))
        except ValueError:
            continue
    return ("", "")


def _get_channel_from_tif_filename(tif_path: str | Path) -> int | None:
    tif_file_name = os.path.basename(str(tif_path))
    if "_C001T" in tif_file_name:
        return 1
    if "_C002T" in tif_file_name:
        return 2
    if "_C003T" in tif_file_name:
        return 3
    return None


def _find_olympus_txt_file(tif_path: str | Path) -> str | None:
    """Return path to companion Olympus ``.txt`` if it exists, else ``None``."""
    tif_filename = os.path.basename(str(tif_path))
    channel = _get_channel_from_tif_filename(tif_path)

    if channel is None:
        olympus_txt_path = os.path.splitext(str(tif_path))[0] + ".txt"
    else:
        ch_stub = f"_C{channel:03d}"
        ch_stub_index = tif_filename.find(ch_stub)
        olympus_txt_file = tif_filename[0:ch_stub_index] + ".txt"
        olympus_txt_path = os.path.join(os.path.split(str(tif_path))[0], olympus_txt_file)

    if not os.path.isfile(olympus_txt_path):
        return None
    return olympus_txt_path


def read_olympus_txt_dict(tif_path: str | Path) -> dict[str, Any] | None:
    """Parse Olympus header text next to ``tif_path``; return dict or ``None`` if no file."""
    olympus_txt_path = _find_olympus_txt_file(tif_path)
    if olympus_txt_path is None:
        return None

    ret_dict: dict[str, Any] = {
        "dateStr": None,
        "timeStr": None,
        "umPerPixel": None,
        "secondsPerLine": None,
        "durImage_sec": None,
        "pixelsPerLine": None,
        "numLines": None,
        "bitsPerPixel": None,
        "olympusTxtPath": olympus_txt_path,
    }

    pixels_per_line: int | None = None

    with open(olympus_txt_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()

            if line.startswith('"Channel Dimension"'):
                one_line = line.replace('"', "")
                parts = one_line.split()
                if len(parts) >= 3:
                    ret_dict["numChannels"] = int(parts[2])

            if line.startswith('"X Dimension"'):
                parts = line.split()
                if len(parts) > 7:
                    ret_dict["umPerPixel"] = float(parts[7])

            if line.startswith('"T Dimension"'):
                parts = line.split()
                if len(parts) > 5:
                    ret_dict["durImage_sec"] = float(parts[5])

            if line.startswith('"Image Size"'):
                if pixels_per_line is None:
                    parts = line.split()
                    if len(parts) > 4:
                        pixels_per_line = int(parts[2].replace('"', ""))
                        num_lines = int(parts[4].replace('"', ""))
                        ret_dict["pixelsPerLine"] = pixels_per_line
                        ret_dict["numLines"] = num_lines

            if line.startswith('"Date"'):
                # Prefer tab-separated quoted blob: "Date"\t"10/30/2025 02:54:36.454 PM"
                if "\t" in line:
                    after_tab = line.split("\t", 1)[1].strip()
                    if after_tab.startswith('"') and after_tab.endswith('"'):
                        combined = after_tab[1:-1]
                        ret_dict["olympusDateTimeCombined"] = combined
                parts = line.split()
                if len(parts) > 2:
                    date_str = parts[1].replace('"', "")
                    time_str = parts[2]
                    dot_index = time_str.find(".")
                    if dot_index != -1:
                        time_str = time_str[0:dot_index]
                    ret_dict["dateStr"] = date_str
                    ret_dict["timeStr"] = time_str

            if line.startswith('"Bits/Pixel"'):
                parts = line.split()
                if len(parts) > 1:
                    bits = parts[1].replace('"', "")
                    ret_dict["bitsPerPixel"] = int(bits)

    if ret_dict["durImage_sec"] is None:
        logger.error("Olympus txt: did not get durImage_sec from %s", olympus_txt_path)
    else:
        nl = ret_dict["numLines"]
        if nl is not None and nl > 0:
            ret_dict["secondsPerLine"] = ret_dict["durImage_sec"] / nl

    if ret_dict["umPerPixel"] is None:
        logger.error("Olympus txt: did not get umPerPixel from %s", olympus_txt_path)

    given_channel_number = _get_channel_from_tif_filename(tif_path)
    channel_dict: dict[int, str | Path | None] = {}
    if given_channel_number is None:
        channel_dict = {1: tif_path}
    else:
        channel_dict = {given_channel_number: tif_path}
        ch_stub = f"C{given_channel_number:03d}"
        tif_file_name = os.path.basename(str(tif_path))
        n_ch = int(ret_dict.get("numChannels", 1))
        for channel_idx in range(n_ch):
            channel_number = channel_idx + 1
            if channel_number == given_channel_number:
                continue
            other = os.path.join(
                os.path.split(str(tif_path))[0],
                tif_file_name.replace(ch_stub, f"C{channel_number:03d}"),
            )
            if not os.path.isfile(other):
                logger.warning("Olympus: missing other channel tif: %s", other)
                channel_dict[channel_number] = None
            else:
                channel_dict[channel_number] = other

    ret_dict["tifChannelPaths"] = channel_dict
    return ret_dict
