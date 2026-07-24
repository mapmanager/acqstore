"""Header metadata models for sweep-based trace acquisitions."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True)
class TraceHeader:
    """Header metadata for a sweep-based trace recording.

    Args:
        path: Filesystem path to the source recording.
        file_size: Finder-style decimal file size string, or empty when unknown.
        num_sweeps: Number of sweeps in the recording.
        num_channels: Number of recorded input channels.
        samples_per_second: Sampling rate in Hz.
        samples_per_sweep: Number of samples in each sweep.
        channel_names: Input channel names, one per channel.
        channel_units: Input channel units, one per channel.
        command_names: Command/DAC channel names reported by the source file.
        command_units: Command/DAC channel units reported by the source file.
        protocol: Source protocol name, if available.
        acquisition_datetime: Acquisition datetime string, if available.
        x_units: Time-axis units. Defaults to seconds.
    """

    path: str
    file_size: str
    num_sweeps: int
    num_channels: int
    samples_per_second: float
    samples_per_sweep: int
    channel_names: tuple[str, ...]
    channel_units: tuple[str, ...]
    command_names: tuple[str, ...]
    command_units: tuple[str, ...]
    protocol: str = ''
    acquisition_datetime: str = ''
    x_units: str = 's'

    def __post_init__(self) -> None:
        """Validate header consistency after construction.

        Raises:
            ValueError: If counts or metadata lengths are inconsistent.
        """
        if self.num_sweeps < 0:
            raise ValueError(f'num_sweeps must be >= 0, got {self.num_sweeps}')
        if self.num_channels <= 0:
            raise ValueError(f'num_channels must be > 0, got {self.num_channels}')
        if self.samples_per_second <= 0:
            raise ValueError(
                'samples_per_second must be > 0, '
                f'got {self.samples_per_second}'
            )
        if self.samples_per_sweep < 0:
            raise ValueError(
                'samples_per_sweep must be >= 0, '
                f'got {self.samples_per_sweep}'
            )
        if len(self.channel_names) != self.num_channels:
            raise ValueError(
                'channel_names length must match num_channels, '
                f'got {len(self.channel_names)} and {self.num_channels}'
            )
        if len(self.channel_units) != self.num_channels:
            raise ValueError(
                'channel_units length must match num_channels, '
                f'got {len(self.channel_units)} and {self.num_channels}'
            )

    def as_dict(self) -> dict[str, Any]:
        """Return the header as a plain dictionary.

        Returns:
            Dictionary containing all dataclass fields.
        """
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def format_dims_display(self) -> str:
        """Return compact dimensions for display.

        Returns:
            String such as ``"sweeps:17 channels:2 samples:1600"``.
        """
        return (
            f'sweeps:{self.num_sweeps} '
            f'channels:{self.num_channels} '
            f'samples:{self.samples_per_sweep}'
        )
