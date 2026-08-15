"""Lazy image loader for one logical AcqImage stored inside a local NWB file."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .base_file_loader import BaseFileLoader, ImageHeader


@dataclass(frozen=True, slots=True)
class _NwbLoaderApi:
    """Container for lazily imported PyNWB runtime classes.

    Args:
        NWBHDF5IO: PyNWB HDF5 I/O class.
    """

    NWBHDF5IO: Any


def _require_nwb_loader_api() -> _NwbLoaderApi:
    """Import PyNWB only when NWB pixels are actually requested.

    Returns:
        Runtime API required to read the local NWB file.

    Raises:
        ImportError: If AcqStore's optional ``nwb`` extra is not installed.
    """
    try:
        from pynwb import NWBHDF5IO
    except ImportError as exc:
        raise ImportError(
            "NWB support requires AcqStore's optional 'nwb' dependencies. "
            "Install them with: uv sync --extra nwb"
        ) from exc
    return _NwbLoaderApi(NWBHDF5IO=NWBHDF5IO)


class NwbFileLoader(BaseFileLoader):
    """Lazy pixel loader for one member of an AcqStore NWB file.

    Args:
        path: Physical local NWB file path.
        member_id: Logical AcqStore member identifier in the NWB manifest.
        images_container: NWB ``Images`` acquisition-container name.
        channel_images: Ordered NWB ``GrayscaleImage`` names for the member.
        axes: AcqStore axes, currently exactly ``YX`` or ``CYX``.
        header: Metadata-only ImageHeader built from the NWB manifest.
    """

    def __init__(
        self,
        path: str,
        *,
        member_id: str,
        images_container: str,
        channel_images: tuple[str, ...],
        axes: tuple[str, ...],
        header: ImageHeader,
    ) -> None:
        """Create a lazy NWB member loader without reading pixel datasets.

        Args:
            path: Physical local NWB file path.
            member_id: Logical AcqStore member identifier.
            images_container: NWB acquisition-container name.
            channel_images: Ordered image names inside the container.
            axes: AcqStore pixel axes.
            header: Metadata-only image header.

        Returns:
            None.
        """
        self.member_id = member_id
        self.images_container = images_container
        self.channel_images = channel_images
        self.axes = axes
        super().__init__(path, header=header)

    def read_header(self) -> ImageHeader:
        """Reject header reads because NWB headers are supplied from the manifest.

        Returns:
            Never returns.

        Raises:
            RuntimeError: Always, because construction must provide ``header``.
        """
        raise RuntimeError("NwbFileLoader requires a manifest-derived ImageHeader")

    def _load_full_image_array(self) -> np.ndarray:
        """Load only this logical member's YX/CYX pixels from the NWB file.

        Returns:
            NumPy array in AcqStore YX or CYX axis order.

        Raises:
            ValueError: If the expected Images container or channel image is
                missing, or if the stored channel count is inconsistent.
        """
        api = _require_nwb_loader_api()
        channels: list[np.ndarray] = []

        # PyNWB/HDF5 objects are valid only while the IO handle remains open.
        # Materialize this one AcqImage, then close the file immediately.
        with api.NWBHDF5IO(path=self.path, mode="r", load_namespaces=True) as io:
            nwbfile = io.read()
            images = nwbfile.acquisition.get(self.images_container)
            if images is None:
                raise ValueError(
                    f"NWB Images container {self.images_container!r} is missing "
                    f"for member {self.member_id!r}"
                )
            for image_name in self.channel_images:
                image = images.images.get(image_name)
                if image is None:
                    raise ValueError(
                        f"NWB image {image_name!r} is missing for member {self.member_id!r}"
                    )
                # GrayscaleImage uses XY storage in this AcqStore NWB contract;
                # transpose back to AcqStore's public YX convention.
                channels.append(np.asarray(image.data[:]).T)

        if self.axes == ("Y", "X"):
            if len(channels) != 1:
                raise ValueError(
                    f"YX member {self.member_id!r} must contain exactly one channel"
                )
            return channels[0]
        if self.axes == ("C", "Y", "X"):
            if not channels:
                raise ValueError(f"CYX member {self.member_id!r} has no channels")
            return np.stack(channels, axis=0)
        raise ValueError(f"Unsupported NWB member axes: {self.axes!r}")
