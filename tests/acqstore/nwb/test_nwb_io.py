"""Tests for lazy single-AcqImage NWB import/export."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pynwb")
from pynwb import NWBHDF5IO, NWBFile
from pynwb.validation import validate

from acqstore.acq_image.acq_image import AcqImage
from acqstore.acq_image.analysis.velocity_analysis.radon_velocity_analysis import (
    RadonVelocityAnalysis,
)
from acqstore.acq_image.roi import RectRoiBounds
from acqstore.nwb_io import NwbMetadata, NwbSubjectMetadata, load_nwb, save_nwb


def _make_image(
    *,
    axes: tuple[str, ...],
    shape: tuple[int, ...] | None = None,
    source_id: str = "nwb-test",
    units: dict[str, str] | None = None,
    load_images: bool = True,
) -> AcqImage:
    """Create one deterministic YX/CYX in-memory acquisition.

    Args:
        axes: AcqStore axes, either YX or CYX.
        shape: Optional explicit shape matching ``axes``.
        source_id: Logical source identity.
        units: Optional Y/X unit labels.
        load_images: Whether the AcqPixels wrapper starts loaded.

    Returns:
        Deterministic uint16 AcqImage.
    """
    resolved_shape = shape or ((8, 12) if axes == ("Y", "X") else (2, 8, 12))
    data = np.arange(np.prod(resolved_shape), dtype=np.uint16).reshape(resolved_shape)
    return AcqImage.from_array(
        data,
        axes=axes,
        source_id=source_id,
        axis_spacing={"Y": 0.002, "X": 0.25},
        axis_units=units or {"Y": "um", "X": "um"},
        load_images=load_images,
    )


def _add_analysis(image: AcqImage) -> tuple[RadonVelocityAnalysis, pd.DataFrame]:
    """Add one analysis with a deterministic result table.

    Args:
        image: Image that will own the analysis.

    Returns:
        Tuple containing the created analysis and expected result DataFrame.
    """
    image.rois.create_rect_roi(bounds=RectRoiBounds(0, 8, 0, 12))
    analysis = RadonVelocityAnalysis(channel=0, roi_id=1)
    expected = pd.DataFrame({"time_s": [0.0, 0.1], "radon_velocity": [1.5, 2.5]})
    analysis.result.table = expected.copy()
    image.analysis_set.add(analysis)
    return analysis, expected


def test_yx_round_trip_is_lazy_and_preserves_json(tmp_path: Path) -> None:
    """YX NWB import should preserve JSON state without loading large data."""
    original = _make_image(axes=("Y", "X"))
    original.rois.create_rect_roi(name="roi", bounds=RectRoiBounds(1, 6, 2, 10))
    path = tmp_path / "yx.nwb"

    save_nwb(original, path)
    loaded = load_nwb(path)

    assert loaded.images_loaded is False
    assert loaded.analysis_csv_loaded is True  # no analyses means fully loaded by definition
    assert loaded.rois.to_list() == original.rois.to_list()
    assert loaded.pixels.axes == ("Y", "X")
    np.testing.assert_array_equal(loaded.pixels.get_array(), original.pixels.get_array())


def test_cyx_round_trip_preserves_channel_order(tmp_path: Path) -> None:
    """CYX channels should remain independent and ordered across NWB."""
    original = _make_image(axes=("C", "Y", "X"))
    path = tmp_path / "cyx.nwb"

    original.save_as_nwb(path)
    loaded = AcqImage.from_nwb(path)

    assert loaded.images_loaded is False
    np.testing.assert_array_equal(loaded.pixels.get_plane(c=0), original.pixels.get_plane(c=0))
    np.testing.assert_array_equal(loaded.pixels.get_plane(c=1), original.pixels.get_plane(c=1))


def test_kymograph_units_round_trip(tmp_path: Path) -> None:
    """A YX kymograph should remain YX and preserve seconds/microns labels."""
    original = _make_image(axes=("Y", "X"), units={"Y": "s", "X": "um"})
    path = tmp_path / "kymograph.nwb"

    save_nwb(original, path)
    loaded = load_nwb(path)
    header = loaded.get_metadata_section("acq_image_header").get_values()

    assert header["physical_unit_y"] == 0.002
    assert header["physical_unit_x"] == 0.25
    assert header["physical_label_y"] == "s"
    assert header["physical_label_x"] == "um"


def test_analysis_dynamic_table_is_lazy(tmp_path: Path) -> None:
    """NWB DynamicTables should not become DataFrames until explicitly loaded."""
    original = _make_image(axes=("Y", "X"))
    source_analysis, expected = _add_analysis(original)
    path = tmp_path / "analysis.nwb"

    save_nwb(original, path)
    loaded = load_nwb(path)

    assert loaded.analysis_csv_loaded is False
    analysis = loaded.analysis_set.get_required(source_analysis.key)
    assert analysis.result.table is None

    loaded.load_analysis_csv()

    assert loaded.analysis_csv_loaded is True
    restored = loaded.analysis_set.get_required(source_analysis.key).result.table
    assert restored is not None
    pd.testing.assert_frame_equal(restored.reset_index(drop=True), expected)

    loaded.unload_analysis_csv()
    assert loaded.analysis_csv_loaded is False
    assert loaded.analysis_set.get_required(source_analysis.key).result.table is None


def test_export_restores_lazy_source_state(tmp_path: Path) -> None:
    """Explicit export should materialize required NWB data then restore laziness."""
    original = _make_image(axes=("Y", "X"))
    _source_analysis, _expected = _add_analysis(original)
    first = tmp_path / "first.nwb"
    second = tmp_path / "second.nwb"
    save_nwb(original, first)

    lazy = load_nwb(first)
    assert lazy.images_loaded is False
    assert lazy.analysis_csv_loaded is False

    save_nwb(lazy, second)

    assert lazy.images_loaded is False
    assert lazy.analysis_csv_loaded is False
    copied = load_nwb(second, load_images=True, load_analysis_csv=True)
    assert copied.images_loaded is True
    assert copied.analysis_csv_loaded is True


def test_nwb_backed_save_rejects_in_place_persistence(tmp_path: Path) -> None:
    """AcqImage.save should reject NWB source mutation and require explicit export."""
    original = _make_image(axes=("Y", "X"))
    path = tmp_path / "source.nwb"
    save_nwb(original, path)
    loaded = load_nwb(path)

    with pytest.raises(RuntimeError, match="In-place NWB mutation is not supported"):
        loaded.save()


def test_metadata_defaults_and_subject(tmp_path: Path) -> None:
    """NWB metadata should use Eastern save time and structured subject fields."""
    original = _make_image(axes=("Y", "X"))
    path = tmp_path / "metadata.nwb"
    before = datetime.now(ZoneInfo("America/New_York"))
    metadata = NwbMetadata(
        subject=NwbSubjectMetadata(
            subject_id="subject-001",
            species="Mus musculus",
            age="P90D",
        )
    )

    save_nwb(original, path, metadata=metadata)
    after = datetime.now(ZoneInfo("America/New_York"))

    with NWBHDF5IO(path=path, mode="r") as io:
        nwbfile = io.read()
        assert nwbfile.identifier
        assert before <= nwbfile.session_start_time <= after
        assert nwbfile.subject.subject_id == "subject-001"
        assert nwbfile.subject.species == "Mus musculus"
        assert nwbfile.subject.sex == "U"


def test_unsupported_axes_rejected(tmp_path: Path) -> None:
    """ZYX should fail rather than being reinterpreted by the static-image v1."""
    original = AcqImage.from_array(
        np.zeros((3, 8, 12), dtype=np.uint16),
        axes=("Z", "Y", "X"),
        source_id="zyx",
    )

    with pytest.raises(ValueError, match="YX and CYX"):
        save_nwb(original, tmp_path / "zyx.nwb")


def test_non_acqstore_nwb_rejected(tmp_path: Path) -> None:
    """A valid NWB without AcqStore scratch should fail clearly."""
    path = tmp_path / "plain.nwb"
    nwbfile = NWBFile(
        session_description="plain NWB",
        identifier="plain",
        session_start_time=datetime.now(ZoneInfo("America/New_York")),
    )
    with NWBHDF5IO(path=path, mode="w") as io:
        io.write(nwbfile)

    with pytest.raises(ValueError, match="AcqStore scratch"):
        load_nwb(path)


def test_saved_file_passes_pynwb_validation(tmp_path: Path) -> None:
    """A single AcqStore NWB should pass PyNWB schema validation."""
    original = _make_image(axes=("C", "Y", "X"))
    path = tmp_path / "validated.nwb"

    save_nwb(original, path)

    assert validate(path=path) == []
