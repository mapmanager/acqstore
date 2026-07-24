"""Tests for OIR file loader helpers and header axis labels."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator

import numpy as np
import pytest
from oirfile import METADATA

from acqstore.acq_image.acq_image import AcqImage
from acqstore.acq_image.file_loaders.oir_file_loader import (
    OirFileLoader,
    _enabled_axes_from_lsmimage_xml,
    _image_header_from_oir_scene,
    _is_y_timelapse_line_scan_axis,
    _oir_reference_spatial_coord_scales,
    _physical_units_for_oir_header,
    _reference_snapshot_from_oir_reference,
    _step_from_coord,
)

_OIR_SAMPLES = Path(__file__).resolve().parent / "data" / "oir-samples"
_KYMOGRAPH = _OIR_SAMPLES / "20251030_A106_0002.oir"
_ZSTACK = _OIR_SAMPLES / "20251030_A106.oir"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_OIR_DEBUG_0010 = _REPO_ROOT / "tmp/oir-debug/two-channel-oir/20260709_A131_0010.oir"

_TIMELAPSE_AXIS_XML = """
<root xmlns:commonparam="urn:test">
  <commonparam:axis enable="true">
    <commonparam:axis>TIMELAPSE</commonparam:axis>
    <commonparam:startPosition>0.0</commonparam:startPosition>
    <commonparam:endPosition>0.0</commonparam:endPosition>
    <commonparam:step>0.0</commonparam:step>
    <commonparam:maxSize>30000</commonparam:maxSize>
  </commonparam:axis>
  <commonparam:seriesInterval>1.142</commonparam:seriesInterval>
</root>
"""


class _FakeOirReference:
    """Minimal OIR reference object used by ``_reference_snapshot_from_oir_reference``."""

    dims = ("C", "Y", "X")
    sizes = {"C": 1, "Y": 8, "X": 9}
    line_roi = (1.0, 2.0, 7.0, 6.0)
    coord_units = {"X": "um", "Y": "um"}
    coord_scales = {"X": 0.25, "Y": 0.5}
    coords = {}

    def asarray(self) -> np.ndarray:
        """Return a fake channel-first OIR reference image."""
        return np.zeros((1, 8, 9), dtype=np.uint8)


class _FakeOirScene:
    """Minimal ``oirfile.OirFile``-like object for header unit tests."""

    def __init__(
        self,
        *,
        dims: tuple[str, ...],
        sizes: dict[str, int],
        coord_units: dict[str, str],
        coord_scales: dict[str, float],
        coords: dict[str, np.ndarray] | None = None,
        lsmimage_xml: str | None = None,
        coords_raises: bool = False,
        pixel_length_x: float | None = None,
    ) -> None:
        self.dims = dims
        self.sizes = sizes
        self.shape = tuple(sizes[d] for d in dims)
        self.dtype = np.dtype(np.uint16)
        self.coord_units = coord_units
        self.coord_scales = coord_scales
        self._coords = coords or {}
        self._coords_raises = coords_raises
        self._pixel_length_x = pixel_length_x
        self.datetime = None
        if lsmimage_xml is None:
            self.xml_metadata = MappingProxyType({})
        else:
            self.xml_metadata = MappingProxyType({METADATA.LSMIMAGE: [lsmimage_xml]})

    @property
    def coords(self) -> dict[str, np.ndarray]:
        if self._coords_raises:
            raise AssertionError("coords should not be accessed when coord_scales are complete")
        return self._coords


def test_oir_reference_image_populates_scan_path_from_line_roi() -> None:
    """OIR explicit line ROI endpoints are exposed through ReferenceImage scan path."""
    reference = _reference_snapshot_from_oir_reference(_FakeOirReference())

    assert reference.line_roi == (1.0, 2.0, 7.0, 6.0)
    assert reference.has_scan_path() is True
    scan_path = reference.get_scan_path()
    assert scan_path is not None
    np.testing.assert_array_equal(scan_path, np.asarray([[1.0, 7.0], [2.0, 6.0]]))
    x_pixels, y_pixels = reference.get_scan_path_plot()
    np.testing.assert_array_equal(x_pixels, np.asarray([1.0, 7.0]))
    np.testing.assert_array_equal(y_pixels, np.asarray([2.0, 6.0]))


def test_enabled_axes_from_lsmimage_xml_parses_timelapse() -> None:
    """LSMIMAGE XML exposes enabled TIMELAPSE axis metadata."""
    axes = _enabled_axes_from_lsmimage_xml(_TIMELAPSE_AXIS_XML)
    assert axes["TIMELAPSE"]["maxSize"] == 30000


def test_physical_units_for_oir_header_uses_coord_units_by_default() -> None:
    """Spatial OIR axes keep ``coord_units`` labels."""
    scene = _FakeOirScene(
        dims=("Z", "Y", "X"),
        sizes={"Z": 10, "Y": 512, "X": 512},
        coord_units={"Z": "µm", "Y": "µm", "X": "µm"},
        coord_scales={"Z": 1.7, "Y": 0.002, "X": 0.002},
    )

    units, labels = _physical_units_for_oir_header(scene)

    assert labels == ("µm", "µm", "µm")
    assert units == (1.7, 0.002, 0.002)


def test_physical_units_for_oir_header_relabels_y_for_line_scan_kymograph() -> None:
    """TIMELAPSE-on-Y line scans use seriesInterval (time) and pixel length (space)."""
    scene = _FakeOirScene(
        dims=("Y", "X"),
        sizes={"Y": 30000, "X": 24},
        coord_units={"Y": "µm", "X": "µm"},
        coord_scales={"Y": 0.000535, "X": 0.0114},
        lsmimage_xml=_TIMELAPSE_AXIS_XML,
        pixel_length_x=0.274,
    )

    assert _is_y_timelapse_line_scan_axis(scene) is True
    units, labels = _physical_units_for_oir_header(scene)

    assert labels == ("seconds", "µm")
    assert units == (pytest.approx(0.001142), pytest.approx(0.274))


def test_physical_units_for_oir_header_skips_coords_when_scales_complete() -> None:
    """Line-scan calibration does not need ``coords`` when scales are present."""
    scene = _FakeOirScene(
        dims=("Y", "X"),
        sizes={"Y": 30000, "X": 24},
        coord_units={"Y": "µm", "X": "µm"},
        coord_scales={"Y": 0.000535, "X": 0.0114},
        lsmimage_xml=_TIMELAPSE_AXIS_XML,
        pixel_length_x=0.274,
        coords_raises=True,
    )

    units, labels = _physical_units_for_oir_header(scene)

    assert labels == ("seconds", "µm")
    assert units == (pytest.approx(0.001142), pytest.approx(0.274))


def test_step_from_coord_returns_none_for_string_channel_names() -> None:
    """Channel-name coordinate arrays are categorical, not numeric spacing."""
    assert _step_from_coord(np.array(["CH1", "CH2"])) is None
    assert _step_from_coord(np.array([0.0, 0.5])) == pytest.approx(0.5)


def test_physical_units_for_oir_header_skips_channel_dim_c() -> None:
    """Multi-channel OIR dims include ``C`` with string coords; skip spatial step."""
    # TIMELAPSE maxSize must match Y for line-scan seconds labeling.
    timelapse_xml = """
<root xmlns:commonparam="urn:test">
  <commonparam:axis enable="true">
    <commonparam:axis>TIMELAPSE</commonparam:axis>
    <commonparam:startPosition>0.0</commonparam:startPosition>
    <commonparam:endPosition>0.0</commonparam:endPosition>
    <commonparam:step>0.0</commonparam:step>
    <commonparam:maxSize>10000</commonparam:maxSize>
  </commonparam:axis>
  <commonparam:seriesInterval>2.118</commonparam:seriesInterval>
</root>
"""
    scene = _FakeOirScene(
        dims=("C", "Y", "X"),
        sizes={"C": 2, "Y": 10000, "X": 512},
        coord_units={"Y": "micrometer", "X": "micrometer"},
        coord_scales={"Y": 0.000966, "X": 0.000966},
        coords={"C": np.array(["CH1", "CH2"])},
        lsmimage_xml=timelapse_xml,
        pixel_length_x=0.331,
    )

    units, labels = _physical_units_for_oir_header(scene)

    assert labels == ("", "seconds", "micrometer")
    assert units == (None, pytest.approx(0.002118), pytest.approx(0.331))


def test_physical_units_for_oir_header_skips_sample_dim_s() -> None:
    """RGB sample axis ``S`` is categorical like ``C``."""
    scene = _FakeOirScene(
        dims=("S", "Y", "X"),
        sizes={"S": 3, "Y": 64, "X": 64},
        coord_units={"Y": "micrometer", "X": "micrometer"},
        coord_scales={"Y": 0.1, "X": 0.1},
        coords={"S": np.array(["R", "G", "B"])},
    )

    units, labels = _physical_units_for_oir_header(scene)

    assert labels == ("", "micrometer", "micrometer")
    assert units == (None, pytest.approx(0.1), pytest.approx(0.1))


@pytest.mark.skipif(not _KYMOGRAPH.is_file(), reason="kymograph OIR fixture missing")
def test_oir_has_reference_image_cached_without_decoding_reference() -> None:
    """Reference existence is probed during header read without loading pixels."""
    loader = OirFileLoader(str(_KYMOGRAPH))

    assert loader.has_reference_image is True
    assert loader._referenceImage is None


@pytest.mark.skipif(not _KYMOGRAPH.is_file(), reason="kymograph OIR fixture missing")
def test_get_schema_row_uses_cached_reference_probe_without_reopening_oir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """File-list schema rows do not reopen OIR files just to populate reference emoji."""
    open_count = 0
    real_open = OirFileLoader._open_oir

    @contextmanager
    def _counting_open(self: OirFileLoader) -> Iterator[Any]:
        nonlocal open_count
        open_count += 1
        with real_open(self) as oir:
            yield oir

    monkeypatch.setattr(OirFileLoader, "_open_oir", _counting_open)

    acq = AcqImage(str(_KYMOGRAPH), load_images=False, load_analysis_csv=False)
    assert open_count == 1

    open_count = 0
    row = acq.get_schema_row()

    assert row["reference_image"] == "✅"
    assert open_count == 0


@pytest.mark.skipif(not _KYMOGRAPH.is_file(), reason="kymograph OIR fixture missing")
def test_oir_reference_image_still_decodes_lazily() -> None:
    """Explicit reference access still builds a decoded snapshot on demand."""
    loader = OirFileLoader(str(_KYMOGRAPH))

    reference = loader.reference_image

    assert reference is not None
    assert reference.array.ndim >= 2


@pytest.mark.skipif(not _KYMOGRAPH.is_file(), reason="kymograph OIR fixture missing")
def test_oir_kymograph_fixture_labels_y_seconds_x_um() -> None:
    """Real line-scan OIR matches Olympus TXT seconds/um calibration."""
    header = OirFileLoader(str(_KYMOGRAPH)).header

    assert header.dims == ("Y", "X")
    assert header.physical_units_labels == ("seconds", "micrometer")
    assert header.physical_units[0] == pytest.approx(0.001142, rel=1e-4)
    assert header.physical_units[1] == pytest.approx(0.274, rel=1e-3)
    acq = AcqImage(str(_KYMOGRAPH), load_images=False, load_analysis_csv=False)
    y_step, x_step = acq.get_image_physical_units()
    assert y_step == pytest.approx(0.001142, rel=1e-4)
    assert x_step == pytest.approx(0.274, rel=1e-3)


@pytest.mark.skipif(not _ZSTACK.is_file(), reason="Z-stack OIR fixture missing")
def test_oir_zstack_fixture_keeps_spatial_um_labels() -> None:
    """Z-stack OIR files keep spatial micrometer labels on all axes."""
    header = OirFileLoader(str(_ZSTACK)).header

    assert header.dims == ("Z", "Y", "X")
    assert header.physical_units_labels == ("micrometer", "micrometer", "micrometer")


@pytest.mark.skipif(not _KYMOGRAPH.is_file(), reason="kymograph OIR fixture missing")
def test_image_header_from_oir_scene_aligns_labels_with_dims() -> None:
    """OIR header builder aligns label tuple length with ``dims``."""
    import oirfile

    with oirfile.OirFile(_KYMOGRAPH) as oir:
        header = _image_header_from_oir_scene(str(_KYMOGRAPH), oir, num_scenes=1)

    assert len(header.physical_units_labels) == len(header.dims)
    assert header._physical_label_for_dim("Y") == "seconds"
    assert header._physical_label_for_dim("X") == "micrometer"


def test_oir_reference_spatial_coord_scales_override() -> None:
    """Reference planes use parent pixel length as µm/px, not coord deltas."""
    ref = _FakeOirReference()
    scales = _oir_reference_spatial_coord_scales(
        ref,
        pixel_length_x=0.331,
        pixel_length_y=0.331,
    )
    assert scales["X"] == pytest.approx(0.331)
    assert scales["Y"] == pytest.approx(0.331)

    snapshot = _reference_snapshot_from_oir_reference(
        ref,
        pixel_length_x=0.331,
        pixel_length_y=0.331,
    )
    plane = snapshot.get_plane(channel=0)
    assert plane.dx == pytest.approx(0.331)
    assert plane.dy == pytest.approx(0.331)


@pytest.mark.skipif(not _KYMOGRAPH.is_file(), reason="kymograph OIR fixture missing")
def test_oir_kymograph_reference_plane_matches_primary_spatial_x() -> None:
    """Reference overview uses spatial µm/px (primary X), not time axis step."""
    acq = AcqImage(str(_KYMOGRAPH), load_images=False, load_analysis_csv=False)
    _, primary_x_um = acq.get_image_physical_units()
    loader = OirFileLoader(str(_KYMOGRAPH))
    ref = loader.reference_image
    assert ref is not None
    plane = ref.get_plane(channel=0)
    assert plane.dx == pytest.approx(primary_x_um, rel=1e-3)
    assert plane.dy == pytest.approx(primary_x_um, rel=1e-3)
    assert plane.dx * plane.array.shape[0] == pytest.approx(primary_x_um * 512, rel=1e-3)


@pytest.mark.skipif(not _OIR_DEBUG_0010.is_file(), reason="oir-debug 0010 missing")
def test_oir_debug_0010_reference_matches_primary_x_and_txt_um_per_pixel() -> None:
    """Reference µm/px matches primary spatial X and Olympus TXT reference size."""
    acq = AcqImage(str(_OIR_DEBUG_0010), load_images=False, load_analysis_csv=False)
    _, primary_x_um = acq.get_image_physical_units()
    loader = OirFileLoader(str(_OIR_DEBUG_0010))
    ref = loader.reference_image
    assert ref is not None
    plane = ref.get_plane(channel=0)
    assert plane.dx == pytest.approx(primary_x_um, rel=1e-3)
    assert plane.dy == pytest.approx(primary_x_um, rel=1e-3)
    # Olympus TXT: 512 px reference field is 169.706 µm per side → ~0.331 µm/px.
    assert plane.dx == pytest.approx(169.706 / 512.0, rel=1e-3)


@pytest.mark.skipif(not _KYMOGRAPH.is_file(), reason="kymograph OIR fixture missing")
def test_oir_kymograph_reference_image_metadata_matches_plane_scales() -> None:
    """AcqImage reference metadata matches corrected ReferenceImage scales."""
    acq = AcqImage(str(_KYMOGRAPH), load_images=False, load_analysis_csv=False)
    ref_meta = acq.get_metadata_section('reference_image_metadata').get_values()
    plane = acq.images.reference_image.get_plane(channel=0)
    assert ref_meta['physical_unit_x'] == pytest.approx(plane.dy, rel=1e-6)
    assert ref_meta['physical_unit_y'] == pytest.approx(plane.dx, rel=1e-6)
    assert ref_meta['shape'] == str(plane.array.shape)
    assert ref_meta['physical_label_x'] == 'um'
    assert ref_meta['physical_label_y'] == 'um'


@pytest.mark.skipif(not _KYMOGRAPH.is_file(), reason="kymograph OIR fixture missing")
def test_oir_kymograph_reference_metadata_scan_path_matches_snapshot() -> None:
    """Reference metadata scan-path fields mirror ReferenceImage scan-path API."""
    acq = AcqImage(str(_KYMOGRAPH), load_images=False, load_analysis_csv=False)
    reference = acq.images.reference_image
    assert reference is not None
    ref_meta = acq.get_metadata_section('reference_image_metadata').get_values()

    assert ref_meta['has_scan_path'] is reference.has_scan_path()
    if not reference.has_scan_path():
        assert ref_meta['scan_path_num_points'] == 0
        assert ref_meta['scan_path_x_pixels'] == []
        assert ref_meta['scan_path_y_pixels'] == []
        return

    scan_path_plot = reference.get_scan_path_plot()
    assert scan_path_plot is not None
    x_pixels, y_pixels = scan_path_plot
    assert ref_meta['scan_path_num_points'] == len(x_pixels)
    assert ref_meta['scan_path_x_pixels'] == [float(value) for value in x_pixels]
    assert ref_meta['scan_path_y_pixels'] == [float(value) for value in y_pixels]
    line_roi = reference.get_line_roi()
    if line_roi is None:
        assert ref_meta['line_roi'] == ''
    else:
        assert ref_meta['line_roi'] == str(tuple(float(v) for v in line_roi))


@pytest.mark.skipif(not _OIR_DEBUG_0010.is_file(), reason="oir-debug 0010 missing")
def test_oir_debug_0010_reference_image_metadata_matches_primary_x() -> None:
    """Reference metadata spatial units match primary image X calibration."""
    acq = AcqImage(str(_OIR_DEBUG_0010), load_images=False, load_analysis_csv=False)
    _, primary_x_um = acq.get_image_physical_units()
    ref_meta = acq.get_metadata_section('reference_image_metadata').get_values()
    assert ref_meta['physical_unit_x'] == pytest.approx(primary_x_um, rel=1e-3)
    assert ref_meta['physical_unit_y'] == pytest.approx(primary_x_um, rel=1e-3)
    assert ref_meta['physical_unit_x'] == pytest.approx(169.706 / 512.0, rel=1e-3)
