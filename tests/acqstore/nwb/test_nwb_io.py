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
from pynwb.image import GrayscaleImage, Images, RGBImage
from pynwb.validation import validate

from acqstore.acq_image.acq_image import AcqImage
from acqstore.acq_image.file_loaders.file_loader_factory import create_file_loader
from acqstore.acq_image.file_loaders.loader_registry import (
    get_registered_import_extensions,
)
from acqstore.acq_image.file_loaders.nwb_file_loader import NwbFileLoader
from acqstore.acq_image.supported_import_extensions import (
    get_allowed_import_extensions,
    get_supported_import_extensions,
)
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


def _write_stock_nwb(path: Path, images: dict[str, np.ndarray]) -> None:
    """Write standard static NWB images with no AcqStore metadata.

    Args:
        path: Destination NWB path.
        images: Mapping from image name to arrays already in NWB XY order.
    """
    nwbfile = NWBFile(
        session_description="stock static images",
        identifier="stock-static-images",
        session_start_time=datetime.now(ZoneInfo("America/New_York")),
    )
    container = Images(name="static_images", description="Independent static images")
    for name, data_xy in images.items():
        container.add_image(GrayscaleImage(name=name, data=data_xy))
    nwbfile.add_acquisition(container)
    with NWBHDF5IO(path=path, mode="w") as io:
        io.write(nwbfile)


def test_nwb_is_registered_as_an_ordinary_loader(tmp_path: Path) -> None:
    """NWB should participate in the normal acquisition loader registry."""
    expected_yx = np.arange(15, dtype=np.uint16).reshape(3, 5)
    path = tmp_path / "stock.nwb"
    _write_stock_nwb(path, {"image": expected_yx.T})

    assert "nwb" in get_registered_import_extensions()
    assert "nwb" in get_supported_import_extensions()
    assert "nwb" in get_allowed_import_extensions()
    assert isinstance(create_file_loader(str(path)), NwbFileLoader)


def test_stock_single_grayscale_image_loads_without_acqstore_metadata(
    tmp_path: Path,
) -> None:
    """A stock NWB GrayscaleImage should load with explicit XY-to-YX conversion."""
    expected_yx = (np.arange(15, dtype=np.uint16).reshape(3, 5) * 7) + 2
    path = tmp_path / "stock-single.nwb"
    _write_stock_nwb(path, {"vessel": expected_yx.T})

    lazy = load_nwb(path)
    assert lazy.images_loaded is False
    assert lazy.images.header.dims == ("Y", "X")
    assert lazy.images.header.shape == expected_yx.shape
    assert lazy.file_id.endswith("#static_images/vessel")
    np.testing.assert_array_equal(lazy.pixels.get_array(), expected_yx)

    eager = AcqImage(str(path))
    assert eager.images_loaded is True
    np.testing.assert_array_equal(eager.pixels.get_array(), expected_yx)


def test_stock_equal_shape_images_remain_independent_members(tmp_path: Path) -> None:
    """Equal shape must not be guessed to mean channels in a stock NWB file."""
    first = np.arange(12, dtype=np.uint16).reshape(3, 4)
    second = first + 100
    path = tmp_path / "stock-multiple.nwb"
    _write_stock_nwb(path, {"green": first.T, "red": second.T})

    with pytest.raises(ValueError, match="multiple supported logical images"):
        AcqImage(str(path))

    from acqstore.nwb_io import load_nwb_collection

    collection = load_nwb_collection(path)
    members = list(collection)
    assert [member.name for member in members] == [
        "static_images/green",
        "static_images/red",
    ]
    assert len({member.file_id for member in members}) == 2
    np.testing.assert_array_equal(members[0].pixels.get_array(), first)
    np.testing.assert_array_equal(members[1].pixels.get_array(), second)


def test_stock_collection_preserves_independent_shapes_and_laziness(
    tmp_path: Path,
) -> None:
    """Stock members may have unrelated shapes and load independently."""
    first = np.arange(12, dtype=np.uint16).reshape(3, 4)
    second = np.arange(35, dtype=np.uint16).reshape(5, 7)
    path = tmp_path / "stock-different-shapes.nwb"
    _write_stock_nwb(path, {"first": first.T, "second": second.T})

    from acqstore.nwb_io import load_nwb_collection

    members = list(load_nwb_collection(path))
    assert all(not member.images_loaded for member in members)
    members[0].load_images()
    assert members[0].pixels.shape == first.shape
    assert members[1].images_loaded is False
    assert members[1].images.header.shape == second.shape


def test_unsupported_stock_image_type_fails_clearly(tmp_path: Path) -> None:
    """Valid but unsupported image types must not be silently reinterpreted."""
    path = tmp_path / "rgb-only.nwb"
    nwbfile = NWBFile(
        session_description="RGB only",
        identifier="rgb-only",
        session_start_time=datetime.now(ZoneInfo("America/New_York")),
    )
    container = Images(name="rgb_images", description="Unsupported RGB image")
    container.add_image(
        RGBImage(name="rgb", data=np.zeros((4, 3, 3), dtype=np.uint8))
    )
    nwbfile.add_acquisition(container)
    with NWBHDF5IO(path=path, mode="w") as io:
        io.write(nwbfile)

    with pytest.raises(ValueError, match="Images/GrayscaleImage"):
        load_nwb(path)


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

    ordinary = AcqImage(str(path))
    assert ordinary.images_loaded is True
    assert ordinary.rois.to_list() == original.rois.to_list()
    np.testing.assert_array_equal(ordinary.pixels.get_array(), original.pixels.get_array())


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
        ),
        experimenter=("Researcher One",),
        experiment_description="Microscopy acquisition for NWB export testing.",
        institution="Example Research Institution",
        keywords=("microscopy", "vascular imaging"),
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
        assert list(nwbfile.experimenter) == ["Researcher One"]
        assert nwbfile.experiment_description == (
            "Microscopy acquisition for NWB export testing."
        )
        assert nwbfile.institution == "Example Research Institution"
        assert list(nwbfile.keywords) == ["microscopy", "vascular imaging"]


def test_unsupported_axes_rejected(tmp_path: Path) -> None:
    """ZYX should fail rather than being reinterpreted by the static-image v1."""
    original = AcqImage.from_array(
        np.zeros((3, 8, 12), dtype=np.uint16),
        axes=("Z", "Y", "X"),
        source_id="zyx",
    )

    with pytest.raises(ValueError, match="YX and CYX"):
        save_nwb(original, tmp_path / "zyx.nwb")


def test_nwb_without_supported_static_images_rejected(tmp_path: Path) -> None:
    """A valid NWB without supported image content should fail clearly."""
    path = tmp_path / "plain.nwb"
    nwbfile = NWBFile(
        session_description="plain NWB",
        identifier="plain",
        session_start_time=datetime.now(ZoneInfo("America/New_York")),
    )
    with NWBHDF5IO(path=path, mode="w") as io:
        io.write(nwbfile)

    with pytest.raises(ValueError, match="no AcqStore-supported static images"):
        load_nwb(path)


def test_saved_file_passes_pynwb_validation(tmp_path: Path) -> None:
    """A single AcqStore NWB should pass PyNWB schema validation."""
    original = _make_image(axes=("C", "Y", "X"))
    path = tmp_path / "validated.nwb"

    save_nwb(original, path)

    assert validate(path=path) == []


def test_export_exposes_standard_images_and_acqstore_round_trip_metadata(
    tmp_path: Path,
) -> None:
    """Third parties should see standard images while AcqStore sees full state."""
    original = _make_image(axes=("C", "Y", "X"))
    source_analysis, expected = _add_analysis(original)
    path = tmp_path / "standard-and-native.nwb"

    save_nwb(original, path)

    with NWBHDF5IO(path=path, mode="r") as io:
        nwbfile = io.read()
        container = nwbfile.acquisition["acqimage_0000_images"]
        assert isinstance(container, Images)
        assert sorted(container.images) == ["channel_0000", "channel_0001"]
        assert tuple(container.images["channel_0000"].data.shape) == (12, 8)
        assert "acqstore_manifest_json" in nwbfile.scratch
        assert any(name.startswith("acqstore_analysis__") for name in nwbfile.analysis)
        table = next(iter(nwbfile.analysis.values()))
        assert all(column.description != "no description" for column in table.columns)

    restored = AcqImage(str(path))
    assert restored.images.header.dims == ("C", "Y", "X")
    assert restored.images.header.shape == (2, 8, 12)
    assert restored.rois.to_list() == original.rois.to_list()
    assert restored._build_sidecar_payload() == original._build_sidecar_payload()
    table = restored.analysis_set.get_required(source_analysis.key).result.table
    assert table is not None
    pd.testing.assert_frame_equal(table.reset_index(drop=True), expected)
