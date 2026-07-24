"""Tests for :class:`CziFileLoader` header normalization."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from acqstore.acq_image.acq_image import AcqImage
from acqstore.acq_image.file_loaders import czi_file_loader as czi_file_loader_module
from acqstore.acq_image.file_loaders.base_file_loader import ImageHeader
from acqstore.acq_image.file_loaders.czi_file_loader import CziFileLoader


def _czi_header(
    *,
    dims: tuple[str, ...],
    shape: tuple[int, ...] | None = None,
    physical_units_labels: tuple[str, ...] | None = None,
) -> ImageHeader:
    """Build a minimal CZI-like header for unit tests."""
    if shape is None:
        shape = tuple(2 if dim == 'C' else 10 for dim in dims)
    sizes = {dim: shape[i] for i, dim in enumerate(dims)}
    num_channels = int(sizes['C']) if 'C' in sizes else 1
    n = len(dims)
    if physical_units_labels is None:
        physical_units_labels = dims
    return ImageHeader(
        path='/tmp/linescan.czi',
        shape=shape,
        dims=dims,
        sizes=sizes,
        dtype=np.dtype(np.uint16),
        num_channels=num_channels,
        num_scenes=1,
        physical_units=tuple(1.0 for _ in range(n)),
        physical_units_labels=physical_units_labels,
    )


@contextmanager
def _fake_open_czi(_self: CziFileLoader):
    """Yield a minimal czifile-like object with one scene."""
    czi_file = MagicMock()
    czi_file.scenes = [MagicMock()]
    yield czi_file


def _read_header_with_patch(raw_header: ImageHeader) -> ImageHeader:
    """Call :meth:`CziFileLoader._read_czi_header` without touching disk."""
    loader = CziFileLoader('/tmp/linescan.czi', header=raw_header)
    with patch.object(CziFileLoader, '_open_czi', _fake_open_czi):
        with patch(
            'acqstore.acq_image.file_loaders.czi_file_loader._image_header_from_scene',
            return_value=raw_header,
        ):
            return loader._read_czi_header()


def test_read_czi_header_remaps_linescan_t_to_y(monkeypatch: pytest.MonkeyPatch) -> None:
    """('C', 'T', 'X') line-scan headers gain a Y axis label."""
    mock_warning = MagicMock()
    monkeypatch.setattr(czi_file_loader_module.logger, 'warning', mock_warning)
    raw = _czi_header(dims=('C', 'T', 'X'), shape=(2, 30000, 24))
    header = _read_header_with_patch(raw)

    assert header.dims == ('C', 'Y', 'X')
    assert header.sizes == {'C': 2, 'Y': 30000, 'X': 24}
    assert header.physical_units_labels == ('C', 'Y', 'X')
    mock_warning.assert_called_once()
    assert "remapping 'T' axis to 'Y'" in mock_warning.call_args.args[0]


@pytest.mark.parametrize(
    'dims',
    [
        ('C', 'T', 'Y', 'X'),
        ('C', 'Y', 'X'),
    ],
)
def test_read_czi_header_leaves_existing_y_dims_unchanged(
    dims: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Frame stacks and 2D CZI layouts must not remap T when Y exists."""
    mock_warning = MagicMock()
    monkeypatch.setattr(czi_file_loader_module.logger, 'warning', mock_warning)
    raw = _czi_header(dims=dims)
    header = _read_header_with_patch(raw)

    assert header.dims == dims
    assert header.sizes == raw.sizes
    mock_warning.assert_not_called()


class _FakeContentType:
    """Minimal enum-like content type for CZI attachment tests."""

    def __init__(self, value: str) -> None:
        self.value = value


class _FakeAttachmentEntry:
    """Minimal CZI attachment entry with the fields used by CziFileLoader."""

    def __init__(self, name: str, content_file_type: str, filename: str = '') -> None:
        self.name = name
        self.content_file_type = _FakeContentType(content_file_type)
        self.filename = filename


class _FakeAttachment:
    """Minimal CZI attachment object exposing attachment_entry and data()."""

    def __init__(self, entry: _FakeAttachmentEntry, data: object) -> None:
        self.attachment_entry = entry
        self._data = data

    def data(self) -> object:
        """Return decoded fake attachment data."""
        return self._data


class _FakeCziWithReference:
    """Minimal CZI object exposing attachments(), xml_element, and metadata()."""

    def __init__(self, attachments: list[_FakeAttachment], xml: str | None = None) -> None:
        self._attachments = attachments
        self.xml_element = None
        self._xml = xml or '<ImageDocument />'
        scene = MagicMock()
        scene.shape = (2, 10, 10)
        scene.dims = ('C', 'Y', 'X')
        scene.sizes = {'C': 2, 'Y': 10, 'X': 10}
        scene.dtype = np.dtype(np.uint16)
        scene.coords = {}
        self.scenes = [scene]

    def attachments(self) -> list[_FakeAttachment]:
        """Return fake CZI attachments."""
        return self._attachments

    def metadata(self) -> str:
        """Return fake CZI XML metadata."""
        return self._xml


@contextmanager
def _fake_open_czi_reference(_self: CziFileLoader):
    """Yield a fake CZI containing one reference attachment and XML scaling."""
    reference = np.arange(2 * 64 * 96, dtype=np.uint8).reshape(2, 64, 96)
    raw_scan_path = np.vstack(
        [
            np.linspace(-1.5e-8, 1.5e-8, 128, dtype=np.float32),
            np.linspace(-2.5e-8, 2.5e-8, 128, dtype=np.float32),
        ]
    )
    attachments = [
        _FakeAttachment(
            _FakeAttachmentEntry('Thumbnail', 'JPG', 'thumbnail.jpg'),
            np.zeros((16, 16, 3), dtype=np.uint8),
        ),
        _FakeAttachment(
            _FakeAttachmentEntry('Image', 'ZISRAW', 'Image@123.zisraw'),
            reference,
        ),
        _FakeAttachment(
            _FakeAttachmentEntry('Image', 'ZISRAW', 'Image@scan.zisraw'),
            raw_scan_path,
        ),
    ]
    xml = '<ImageDocument><ScalingX>1.5e-8</ScalingX><ScalingY>2.5e-8</ScalingY></ImageDocument>'
    yield _FakeCziWithReference(attachments, xml=xml)


@contextmanager
def _fake_open_czi_without_reference(_self: CziFileLoader):
    """Yield a fake CZI without a channel-first Image/ZISRAW attachment."""
    attachments = [
        _FakeAttachment(
            _FakeAttachmentEntry('Image', 'ZISRAW', 'Image@thin.zisraw'),
            np.zeros((2, 1024), dtype=np.float32),
        )
    ]
    yield _FakeCziWithReference(attachments)


def test_czi_reference_image_from_zisraw_attachment() -> None:
    """CZI Image/ZISRAW ``(C, Y, X)`` attachment becomes ReferenceImage."""
    raw = _czi_header(dims=('C', 'Y', 'X'), shape=(2, 10, 10))
    loader = CziFileLoader('/tmp/reference.czi', header=raw)

    with patch.object(CziFileLoader, '_open_czi', _fake_open_czi_reference):
        reference = loader.reference_image

    assert reference is not None
    assert reference.array.shape == (2, 64, 96)
    assert reference.array.flags.writeable is False
    assert reference.dims == ('C', 'Y', 'X')
    assert reference.num_channels == 2
    assert reference.line_roi is None
    assert reference.coord_units == (('X', 'um'), ('Y', 'um'))
    scales = dict(reference.coord_scales)
    assert scales['X'] == pytest.approx(0.015)
    assert scales['Y'] == pytest.approx(0.025)
    assert reference.has_scan_path() is True
    scan_path = reference.get_scan_path()
    assert scan_path is not None
    assert scan_path.shape == (2, 128)
    assert scan_path.flags.writeable is False
    x_pixels, y_pixels = reference.get_scan_path_plot()
    assert x_pixels[0] == pytest.approx(47.0)
    assert x_pixels[-1] == pytest.approx(49.0)
    assert y_pixels[0] == pytest.approx(31.0)
    assert y_pixels[-1] == pytest.approx(33.0)

    plane = reference.get_plane(channel=1)
    assert plane.array.shape == (64, 96)
    assert plane.dx == pytest.approx(0.025)
    assert plane.dy == pytest.approx(0.015)
    assert plane.x_unit == 'um'
    assert plane.y_unit == 'um'


def test_czi_reference_image_ignores_non_reference_attachments() -> None:
    """Thin float ZISRAW arrays are not treated as reference images."""
    raw = _czi_header(dims=('C', 'Y', 'X'), shape=(2, 10, 10))
    loader = CziFileLoader('/tmp/no-reference.czi', header=raw)

    with patch.object(CziFileLoader, '_open_czi', _fake_open_czi_without_reference):
        reference = loader.reference_image

    assert reference is None


def test_czi_has_reference_image_cached_during_header_read() -> None:
    """Header read probes and caches reference existence without building snapshot."""
    with patch.object(CziFileLoader, '_open_czi', _fake_open_czi_reference):
        loader = CziFileLoader('/tmp/reference.czi')

    assert loader.has_reference_image is True
    assert loader._cached_reference_array is not None
    assert loader._cached_reference_array.shape == (2, 64, 96)
    assert loader._referenceImage is None


def test_czi_has_reference_image_false_for_scan_path_only_fixture() -> None:
    """Scan-path-only Image/ZISRAW attachments do not set the reference flag."""
    with patch.object(CziFileLoader, '_open_czi', _fake_open_czi_without_reference):
        loader = CziFileLoader('/tmp/no-reference.czi')

    assert loader.has_reference_image is False
    assert loader._cached_reference_array is None


def test_get_schema_row_does_not_reopen_czi_for_reference_column() -> None:
    """File-list schema rows use cached reference probe without a second CZI open."""
    open_count = 0

    @contextmanager
    def _counting_fake_reference_open(_self: CziFileLoader) -> Iterator[Any]:
        nonlocal open_count
        open_count += 1
        with _fake_open_czi_reference(_self) as czi:
            yield czi

    with patch.object(CziFileLoader, '_open_czi', _counting_fake_reference_open):
        acq = AcqImage('/tmp/reference.czi', load_images=False, load_analysis_csv=False)

    assert acq.images.has_reference_image is True
    assert open_count == 1

    open_count = 0
    row = acq.get_schema_row()

    assert row['reference_image'] == '✅'
    assert open_count == 0


def test_czi_reference_image_reuses_cached_array_without_rediscovering() -> None:
    """Lazy reference snapshot reuses the array decoded during header read."""
    with patch.object(CziFileLoader, '_open_czi', _fake_open_czi_reference):
        loader = CziFileLoader('/tmp/reference.czi')

    cached = loader._cached_reference_array
    assert cached is not None

    with patch.object(
        CziFileLoader,
        '_open_czi',
        _fake_open_czi_reference,
    ), patch(
        'acqstore.acq_image.file_loaders.czi_file_loader._find_czi_reference_array',
    ) as mock_find:
        reference = loader.reference_image

    assert reference is not None
    mock_find.assert_not_called()
    np.testing.assert_array_equal(reference.array, cached)
