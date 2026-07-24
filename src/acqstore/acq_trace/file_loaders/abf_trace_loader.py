"""pyABF-backed loader for Axon Binary Format trace recordings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from acqstore.acq_image.file_loaders.base_file_loader import format_file_size
from acqstore.acq_trace.epoch_data import EpochInterval, EpochTable
from acqstore.acq_trace.sweep_data import SweepData
from acqstore.acq_trace.trace_header import TraceHeader


class AbfTraceLoader:
    """Load ABF trace metadata and sweeps using pyABF.

    The loader reads header-like metadata during construction and loads numeric
    sweep arrays only when :meth:`get_sweep` is called. It does not expose image
    pixels, image ROIs, image contrast, or image header metadata.

    Args:
        path: Path to an ``.abf`` file.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If ``path`` is not a file.
        RuntimeError: If pyABF is not installed.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = str(Path(path).expanduser())
        path_obj = Path(self.path)
        if not path_obj.exists():
            raise FileNotFoundError(f'ABF file does not exist: {self.path}')
        if not path_obj.is_file():
            raise ValueError(f'ABF path is not a file: {self.path}')
        try:
            import pyabf  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                'pyABF is required to load ABF files. Install the project '
                'dependencies or add pyabf to the environment.'
            ) from exc
        self._abf: Any = pyabf.ABF(self.path)
        self._header = self._build_header()

    @property
    def header(self) -> TraceHeader:
        """Return trace header metadata.

        Returns:
            Immutable :class:`TraceHeader` describing the ABF file.
        """
        return self._header

    def get_sweep(self, *, channel_index: int, sweep_index: int) -> SweepData:
        """Return one channel/sweep recording.

        Args:
            channel_index: Zero-based input channel index.
            sweep_index: Zero-based sweep index.

        Returns:
            :class:`SweepData` containing time, recorded values, command values,
            and epoch labels for the requested channel/sweep.

        Raises:
            ValueError: If ``channel_index`` or ``sweep_index`` is out of range.
        """
        self._validate_channel_index(channel_index)
        self._validate_sweep_index(sweep_index)
        self._abf.setSweep(sweepNumber=sweep_index, channel=channel_index)
        time_sec = np.asarray(self._abf.sweepX, dtype=float).copy()
        values = np.asarray(self._abf.sweepY, dtype=float).copy()
        command_raw = getattr(self._abf, 'sweepC', None)
        command_values = None
        if command_raw is not None:
            command_values = np.asarray(command_raw, dtype=float).copy()
        epoch_table = self._build_epoch_table()
        epoch_index_values = epoch_table.to_sample_labels(len(time_sec))
        return SweepData(
            sweep_index=sweep_index,
            channel_index=channel_index,
            time_sec=time_sec,
            values=values,
            value_units=self._channel_unit(channel_index),
            command_values=command_values,
            command_units=self._command_unit(channel_index),
            epoch_index_values=epoch_index_values,
            epoch_table=epoch_table,
        )

    def info(self) -> str:
        """Return a human-readable ABF file summary.

        Returns:
            Multiline string describing the source file, dimensions, channels,
            command outputs, protocol, and acquisition datetime.
        """
        header = self.header
        lines = [
            'ABF Trace File',
            f'  path: {header.path}',
            f'  file_size: {header.file_size}',
            f'  sweeps: {header.num_sweeps}',
            f'  channels: {header.num_channels}',
            f'  samples_per_sweep: {header.samples_per_sweep}',
            f'  samples_per_second: {header.samples_per_second:g}',
            f'  protocol: {header.protocol}',
            f'  acquisition_datetime: {header.acquisition_datetime}',
            '  input_channels:',
        ]
        for index, (name, unit) in enumerate(zip(header.channel_names, header.channel_units, strict=True)):
            lines.append(f'    [{index}] {name} ({unit})')
        lines.append('  command_channels:')
        for index, name in enumerate(header.command_names):
            unit = header.command_units[index] if index < len(header.command_units) else ''
            lines.append(f'    [{index}] {name} ({unit})')
        return '\n'.join(lines)

    def _build_header(self) -> TraceHeader:
        """Build header metadata from the pyABF object.

        Returns:
            Trace header populated from pyABF attributes.
        """
        return TraceHeader(
            path=self.path,
            file_size=format_file_size(self.path),
            num_sweeps=int(self._abf.sweepCount),
            num_channels=int(self._abf.channelCount),
            samples_per_second=float(self._abf.dataRate),
            samples_per_sweep=int(self._abf.sweepPointCount),
            channel_names=tuple(str(x) for x in self._abf.adcNames),
            channel_units=tuple(str(x) for x in self._abf.adcUnits),
            command_names=tuple(str(x) for x in getattr(self._abf, 'dacNames', ())),
            command_units=tuple(str(x) for x in getattr(self._abf, 'dacUnits', ())),
            protocol=str(getattr(self._abf, 'protocol', '') or ''),
            acquisition_datetime=str(getattr(self._abf, 'abfDateTime', '') or ''),
            x_units='s',
        )

    def _build_epoch_table(self) -> EpochTable:
        """Build an epoch table for the current pyABF sweep.

        Returns:
            Epoch table extracted from ``abf.sweepEpochs``. Returns an empty
            table when epoch metadata is unavailable.
        """
        sweep_epochs = getattr(self._abf, 'sweepEpochs', None)
        if sweep_epochs is None:
            return EpochTable(())
        starts = list(getattr(sweep_epochs, 'p1s', []) or [])
        ends = list(getattr(sweep_epochs, 'p2s', []) or [])
        levels = list(getattr(sweep_epochs, 'levels', []) or [])
        types = list(getattr(sweep_epochs, 'types', []) or [])
        digital_states = list(getattr(sweep_epochs, 'digitalStates', []) or [])
        intervals: list[EpochInterval] = []
        for index, (start, end) in enumerate(zip(starts, ends, strict=False)):
            level = float(levels[index]) if index < len(levels) else float('nan')
            epoch_type = str(types[index]) if index < len(types) else ''
            states_raw = digital_states[index] if index < len(digital_states) else []
            states = tuple(int(x) for x in states_raw)
            intervals.append(
                EpochInterval(
                    epoch_index=index,
                    start_sample=int(start),
                    end_sample=int(end),
                    level=level,
                    epoch_type=epoch_type,
                    digital_states=states,
                )
            )
        return EpochTable(tuple(intervals))

    def _validate_channel_index(self, channel_index: int) -> None:
        """Validate a channel index.

        Args:
            channel_index: Zero-based input channel index.

        Raises:
            ValueError: If the index is out of range.
        """
        if channel_index < 0 or channel_index >= self.header.num_channels:
            raise ValueError(
                'channel_index out of range: '
                f'{channel_index} not in [0, {self.header.num_channels})'
            )

    def _validate_sweep_index(self, sweep_index: int) -> None:
        """Validate a sweep index.

        Args:
            sweep_index: Zero-based sweep index.

        Raises:
            ValueError: If the index is out of range.
        """
        if sweep_index < 0 or sweep_index >= self.header.num_sweeps:
            raise ValueError(
                'sweep_index out of range: '
                f'{sweep_index} not in [0, {self.header.num_sweeps})'
            )

    def _channel_unit(self, channel_index: int) -> str:
        """Return the input unit for a channel.

        Args:
            channel_index: Zero-based input channel index.

        Returns:
            Unit string, or empty string when unavailable.
        """
        if channel_index < len(self.header.channel_units):
            return self.header.channel_units[channel_index]
        return ''

    def _command_unit(self, channel_index: int) -> str:
        """Return the command unit associated with a channel.

        Args:
            channel_index: Zero-based input channel index.

        Returns:
            Command unit string, falling back to the first command unit when the
            ABF file has fewer DAC unit labels than input channels.
        """
        if channel_index < len(self.header.command_units):
            return self.header.command_units[channel_index]
        if self.header.command_units:
            return self.header.command_units[0]
        return ''
