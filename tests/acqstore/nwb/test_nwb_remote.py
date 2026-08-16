"""Tests for read-only NWB loading over HTTP and public DANDI identifiers."""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pynwb")
pytest.importorskip("remfile")
from pynwb import NWBHDF5IO, NWBFile
from pynwb.image import GrayscaleImage, Images

from acqstore.acq_image.acq_image import AcqImage
from acqstore.acq_image.acq_image_list import AcqImageList
from acqstore.nwb_io import (
    load_nwb,
    load_nwb_collection,
    save_nwb,
    save_nwb_collection,
)
from acqstore.nwb_source import NwbSource


class _RangeHandler(BaseHTTPRequestHandler):
    """Serve one in-memory file with the byte-range behavior remfile requires."""

    payload = b""

    def do_HEAD(self) -> None:  # noqa: N802 - stdlib handler API
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.payload)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        range_header = self.headers.get("Range")
        if range_header is None:
            start, end = 0, len(self.payload) - 1
            status = 200
        else:
            match = re.fullmatch(r"bytes=(\d+)-(\d*)", range_header)
            if match is None:
                self.send_error(416)
                return
            start = int(match.group(1))
            end = int(match.group(2)) if match.group(2) else len(self.payload) - 1
            end = min(end, len(self.payload) - 1)
            status = 206
        data = self.payload[start : end + 1]
        self.send_response(status)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Accept-Ranges", "bytes")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(self.payload)}")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        """Suppress per-range test-server logs."""


@contextmanager
def _serve_file(path: Path) -> Iterator[str]:
    """Serve ``path`` from a temporary loopback HTTP range server."""
    handler = type("RangeHandler", (_RangeHandler,), {"payload": path.read_bytes()})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/{path.name}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_remote_native_nwb_loads_pixels_lazily(tmp_path: Path) -> None:
    """A public HTTP NWB source should reconstruct native state and lazy pixels."""
    expected = np.arange(48, dtype=np.uint16).reshape(6, 8)
    original = AcqImage.from_array(expected, axes=("Y", "X"), source_id="remote-test")
    original.rois.create_rect_roi(name="remote-roi")
    path = tmp_path / "remote-native.nwb"
    save_nwb(original, path)

    with _serve_file(path) as url:
        loaded = load_nwb(url)
        assert loaded.path == url
        assert loaded.images_loaded is False
        assert loaded.rois.to_list() == original.rois.to_list()
        np.testing.assert_array_equal(loaded.pixels.get_array(), expected)


def test_remote_stock_nwb_uses_standard_grayscale_discovery(tmp_path: Path) -> None:
    """A stock remote GrayscaleImage should use the same conservative importer."""
    expected = np.arange(24, dtype=np.uint16).reshape(4, 6)
    nwbfile = NWBFile(
        session_description="remote stock image",
        identifier="remote-stock",
        session_start_time=datetime.now(UTC),
    )
    images = Images(name="stock_images", description="One independent image")
    images.add_image(GrayscaleImage(name="vessel", data=expected.T))
    nwbfile.add_acquisition(images)
    path = tmp_path / "remote-stock.nwb"
    with NWBHDF5IO(path=path, mode="w") as io:
        io.write(nwbfile)

    with _serve_file(path) as url:
        loaded = load_nwb(url)
        assert loaded.file_id == f"{url}#stock_images/vessel"
        np.testing.assert_array_equal(loaded.pixels.get_array(), expected)


def test_remote_native_collection_keeps_members_independently_lazy(tmp_path: Path) -> None:
    """Remote collection discovery must not eagerly load every member's pixels."""
    first = AcqImage.from_array(
        np.arange(24, dtype=np.uint16).reshape(4, 6),
        axes=("Y", "X"),
        source_id="first",
    )
    second = AcqImage.from_array(
        np.arange(60, dtype=np.uint16).reshape(2, 5, 6),
        axes=("C", "Y", "X"),
        source_id="second",
    )
    collection = AcqImageList.__new__(AcqImageList)
    collection.path = "memory://remote-collection"
    collection.source_root_path = None
    collection.file_list = [first.file_id, second.file_id]
    collection._files = [first, second]
    collection._files_by_id = {first.file_id: first, second.file_id: second}
    collection._attach_analysis_pools()
    path = tmp_path / "remote-collection.nwb"
    save_nwb_collection(collection, path)

    with _serve_file(path) as url:
        loaded = list(load_nwb_collection(url))
        assert all(not image.images_loaded for image in loaded)
        loaded[1].load_images()
        assert loaded[0].images_loaded is False
        assert loaded[1].pixels.shape == (2, 5, 6)


def test_remote_cache_is_explicit_and_reused_for_lazy_reads(tmp_path: Path) -> None:
    """A caller-selected cache should support discovery and later pixel reads."""
    expected = np.arange(30, dtype=np.uint16).reshape(5, 6)
    original = AcqImage.from_array(expected, axes=("Y", "X"), source_id="cache-test")
    path = tmp_path / "remote-cache.nwb"
    save_nwb(original, path)
    cache_dir = tmp_path / "byte-cache"

    with _serve_file(path) as url:
        loaded = load_nwb(url, remote_cache_dir=cache_dir)
        np.testing.assert_array_equal(loaded.pixels.get_array(), expected)

    assert cache_dir.is_dir()
    assert any(cache_dir.iterdir())


def test_remote_cache_is_rejected_for_local_source(tmp_path: Path) -> None:
    """A cache option on a local path should fail instead of being ignored."""
    local = tmp_path / "missing.nwb"
    local.touch()
    with pytest.raises(ValueError, match="only for remote"):
        NwbSource.from_value(local, cache_dir=tmp_path / "cache")


def test_remote_url_identity_drops_query_string() -> None:
    """Signed URL parameters must not leak into display or logical identity."""
    source = NwbSource.from_value("https://example.org/image.nwb?token=secret#fragment")

    assert source.location.endswith("?token=secret#fragment")
    assert source.identity == "https://example.org/image.nwb"
    assert "secret" not in repr(source)


class _JsonResponse(BytesIO):
    """Minimal context-managed response used for anonymous resolver tests."""

    def __enter__(self) -> _JsonResponse:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def test_public_dandi_uri_resolves_without_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """The production DANDI resolver must make anonymous public API requests."""
    responses = iter(
        [
            {"results": [{"asset_id": "asset-123"}]},
            {
                "contentUrl": [
                    "https://api.dandiarchive.org/api/assets/asset-123/download/",
                    "https://dandiarchive.s3.amazonaws.com/blobs/abc/123/blob-id",
                ]
            },
        ]
    )
    requests = []

    def fake_urlopen(request: object, timeout: int) -> _JsonResponse:
        requests.append(request)
        assert timeout == 30
        return _JsonResponse(json.dumps(next(responses)).encode())

    monkeypatch.setattr("acqstore.nwb_source.urlopen", fake_urlopen)
    uri = "dandi://DANDI/001947@draft/sub-A98/sub-A98.nwb"
    source = NwbSource.from_value(uri)

    assert source.identity == uri
    assert source.location == (
        "https://dandiarchive.s3.amazonaws.com/blobs/abc/123/blob-id"
    )
    assert len(requests) == 2
    assert all("Authorization" not in request.headers for request in requests)


@pytest.mark.parametrize(
    "value, message",
    [
        ("ftp://example.org/image.nwb", "Unsupported NWB source scheme"),
        ("dandi://sandbox/001947@draft/sub-A98/sub-A98.nwb", "production DANDI"),
        ("dandi://DANDI/not-an-id@draft/file.nwb", "six-digit-id"),
    ],
)
def test_invalid_remote_sources_fail_clearly(value: str, message: str) -> None:
    """Unsupported remote identities should fail rather than being guessed."""
    with pytest.raises(ValueError, match=message):
        NwbSource.from_value(value)
