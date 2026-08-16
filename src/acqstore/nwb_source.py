"""Read-only local and remote source handling for NWB/HDF5 files."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen


_DANDI_URI = re.compile(
    r"^dandi://(?P<instance>[^/]+)/(?P<dandiset>\d{6})@(?P<version>[^/]+)/(?P<path>.+)$",
    re.IGNORECASE,
)
_DANDI_API = "https://api.dandiarchive.org/api"


@dataclass(frozen=True, slots=True)
class NwbSource:
    """Normalized read-only NWB source.

    ``location`` is deliberately excluded from ``repr`` because callers may
    supply a signed URL. ``identity`` never contains a URL query string and is
    safe for display, logging, and AcqStore logical identifiers.
    """

    location: str = field(repr=False)
    identity: str
    is_remote: bool
    cache_dir: Path | None = None

    @classmethod
    def from_value(
        cls,
        value: str | Path | NwbSource,
        *,
        cache_dir: str | Path | None = None,
    ) -> NwbSource:
        """Normalize a local path, public URL, DANDI URI, or existing source.

        Args:
            value: Source to normalize.
            cache_dir: Optional persistent byte-range cache for remote reads.

        Returns:
            A validated source with a stable, query-free identity.

        Raises:
            FileNotFoundError: If a local source does not exist.
            ValueError: If a remote scheme or DANDI identifier is unsupported.
            RuntimeError: If a public DANDI asset cannot be resolved.
        """
        if isinstance(value, cls):
            if cache_dir is not None:
                raise ValueError("cache_dir cannot replace an existing NwbSource cache")
            return value
        resolved_cache = (
            Path(cache_dir).expanduser().resolve(strict=False)
            if cache_dir is not None
            else None
        )
        raw = str(value)
        if raw.lower().startswith("dandi://"):
            content_url = _resolve_public_dandi_uri(raw)
            return cls(
                location=content_url,
                identity=raw,
                is_remote=True,
                cache_dir=resolved_cache,
            )
        parsed = urlsplit(raw)
        if parsed.scheme.lower() in {"http", "https"}:
            if not parsed.netloc:
                raise ValueError(f"Remote NWB URL has no host: {_safe_url(raw)}")
            return cls(
                location=raw,
                identity=_safe_url(raw),
                is_remote=True,
                cache_dir=resolved_cache,
            )
        if parsed.scheme:
            raise ValueError(f"Unsupported NWB source scheme {parsed.scheme!r}")
        path = Path(raw).expanduser().resolve(strict=True)
        if resolved_cache is not None:
            raise ValueError("cache_dir is supported only for remote NWB sources")
        return cls(location=str(path), identity=str(path), is_remote=False)

    @property
    def local_path(self) -> Path | None:
        """Return the resolved local path, or ``None`` for a remote source."""
        return None if self.is_remote else Path(self.location)

    @contextmanager
    def open_nwb(self, nwb_hdf5_io: Any) -> Iterator[Any]:
        """Open this source as a PyNWB ``NWBHDF5IO`` reader.

        Remote resources are opened afresh for each operation and are closed in
        reverse ownership order. No network or HDF5 handle escapes the context.

        Args:
            nwb_hdf5_io: Imported PyNWB ``NWBHDF5IO`` class.

        Yields:
            An open read-only ``NWBHDF5IO`` instance.

        Raises:
            ImportError: If remote dependencies are not installed.
        """
        if not self.is_remote:
            with nwb_hdf5_io(
                path=self.location,
                mode="r",
                load_namespaces=True,
            ) as io:
                yield io
            return

        try:
            import h5py
            import remfile
        except ImportError as exc:
            raise ImportError(
                "Remote NWB support requires AcqStore's optional 'nwb-remote' "
                "dependencies. Install them with: uv sync --extra nwb-remote"
            ) from exc

        disk_cache = None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            disk_cache = remfile.DiskCache(str(self.cache_dir))
        remote_file = remfile.File(url=self.location, disk_cache=disk_cache)
        try:
            with h5py.File(remote_file, mode="r") as h5_file:
                with nwb_hdf5_io(
                    file=h5_file,
                    mode="r",
                    load_namespaces=True,
                ) as io:
                    yield io
        finally:
            remote_file.close()


def _safe_url(url: str) -> str:
    """Return a URL without query parameters or fragments."""
    parsed = urlsplit(url)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def _resolve_public_dandi_uri(uri: str) -> str:
    """Resolve one public production DANDI asset to its direct content URL."""
    match = _DANDI_URI.fullmatch(uri)
    if match is None:
        raise ValueError(
            "DANDI NWB source must use "
            "dandi://DANDI/<six-digit-id>@<version>/<asset-path>"
        )
    if match.group("instance").casefold() != "dandi":
        raise ValueError("Only the public production DANDI instance is supported")

    dandiset = match.group("dandiset")
    version = match.group("version")
    asset_path = match.group("path")
    query = urlencode({"path": asset_path})
    listing_url = (
        f"{_DANDI_API}/dandisets/{quote(dandiset)}/versions/"
        f"{quote(version, safe='')}/assets/?{query}"
    )
    listing = _read_public_json(listing_url, uri)
    results = listing.get("results") if isinstance(listing, dict) else None
    if not isinstance(results, list) or len(results) != 1:
        count = len(results) if isinstance(results, list) else 0
        raise RuntimeError(
            f"Public DANDI source {uri!r} resolved to {count} assets; expected exactly one"
        )
    asset_id = results[0].get("asset_id") if isinstance(results[0], dict) else None
    if not isinstance(asset_id, str) or not asset_id:
        raise RuntimeError(f"Public DANDI source {uri!r} returned no asset identifier")

    metadata = _read_public_json(f"{_DANDI_API}/assets/{quote(asset_id)}/", uri)
    content_urls = metadata.get("contentUrl") if isinstance(metadata, dict) else None
    if not isinstance(content_urls, list):
        raise RuntimeError(f"Public DANDI source {uri!r} returned no content URLs")
    direct = next(
        (
            value
            for value in content_urls
            if isinstance(value, str)
            and value.startswith("https://dandiarchive.s3.amazonaws.com/")
        ),
        None,
    )
    if direct is None:
        direct = next(
            (value for value in content_urls if isinstance(value, str) and value.startswith("https://")),
            None,
        )
    if direct is None:
        raise RuntimeError(f"Public DANDI source {uri!r} has no HTTPS content URL")
    return direct


def _read_public_json(url: str, source_identity: str) -> object:
    """Read anonymous JSON from the production DANDI API."""
    request = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=30) as response:
            return json.load(response)
    except HTTPError as exc:
        if exc.code in {401, 403}:
            raise RuntimeError(
                f"DANDI source {source_identity!r} is not publicly accessible; "
                "authenticated remote loading is not implemented"
            ) from exc
        raise RuntimeError(
            f"DANDI API request failed for {source_identity!r}: HTTP {exc.code}"
        ) from exc
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not resolve public DANDI source {source_identity!r}") from exc
