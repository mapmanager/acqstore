"""Acquisition object for non-image trace recording files."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from acqstore.acq_image.metadata import ExperimentMetadata
from acqstore.acq_trace.file_loaders.abf_trace_loader import AbfTraceLoader
from acqstore.acq_trace.sweep_data import SweepData
from acqstore.acq_trace.trace_header import TraceHeader
from acqstore.acq_types import AcqModality
from acqstore.utils.logging import get_logger

if TYPE_CHECKING:
    from acqstore.acq_trace.analysis.trace_peak_params import TracePeakDetectionParams
    from acqstore.acq_trace.analysis.trace_peak_result import TracePeakDetectionResult

logger = get_logger(__name__)

_ACQTRACE_SIDECAR_VERSION = 1
_ACQTRACE_SIDECAR_REQUIRED_KEYS = {
    'version',
    'modality',
    'accepted',
    'experiment_metadata',
    'trace_header_metadata',
}
_ACQTRACE_SIDECAR_OPTIONAL_KEYS: set[str] = set()


class AcqTrace:
    """Root object for one non-image trace acquisition file.

    ``AcqTrace`` is the trace/electrophysiology sibling of ``AcqImage``. It owns
    source-file metadata, experimental metadata, and lazy access to sweep data.
    It intentionally does not expose image pixels, image ROIs, image contrast,
    reference images, or image-specific analysis state.

    The constructor opens the source file through the trace loader and then
    attempts to hydrate persisted sidecar state from ``<source-file>.json`` when
    that file exists. Calling :meth:`save` writes trace-specific sidecar JSON.

    Args:
        path: Path to a supported trace recording file. ABF is currently
            supported.
        accepted: Whether the file is accepted for downstream processing.
        experiment_metadata: Optional experimental metadata. Defaults to an
            empty :class:`ExperimentMetadata` instance.

    Raises:
        ValueError: If the file extension is unsupported.
        FileNotFoundError: If the path does not exist.
    """

    modality = AcqModality.TRACE

    def __init__(
        self,
        path: str | Path,
        *,
        accepted: bool = True,
        experiment_metadata: ExperimentMetadata | None = None,
    ) -> None:
        self.path = str(Path(path).expanduser())
        self.accepted = bool(accepted)
        self.experiment_metadata = experiment_metadata or ExperimentMetadata()
        suffix = Path(self.path).suffix.lower()
        if suffix != '.abf':
            raise ValueError(f'Unsupported trace file extension: {suffix or "<none>"}')
        self._loader = AbfTraceLoader(self.path)
        self.load_sidecar_json()

    @property
    def name(self) -> str:
        """Return the source file name.

        Returns:
            Basename of :attr:`path`.
        """
        return Path(self.path).name

    @property
    def file_id(self) -> str:
        """Return the stable file identifier for this trace.

        Returns:
            Current source path. This mirrors the simple file identity currently
            used by scripting code before mixed acquisition-list support lands.
        """
        return self.path

    @property
    def trace_header(self) -> TraceHeader:
        """Return trace header metadata.

        Returns:
            Immutable :class:`TraceHeader` for the source recording.
        """
        return self._loader.header

    @property
    def is_dirty(self) -> bool:
        """Return whether this trace has unsaved sidecar changes.

        Returns:
            True when a metadata section has unsaved edits.
        """
        return self.experiment_metadata.is_dirty()

    def get_trace_header(self) -> TraceHeader:
        """Return trace header metadata.

        Returns:
            Immutable :class:`TraceHeader` for the source recording.
        """
        return self.trace_header

    def get_sweep(self, *, channel_index: int, sweep_index: int) -> SweepData:
        """Return one channel/sweep recording.

        Args:
            channel_index: Zero-based input channel index.
            sweep_index: Zero-based sweep index.

        Returns:
            :class:`SweepData` for the requested channel/sweep.

        Raises:
            ValueError: If either index is out of range.
        """
        return self._loader.get_sweep(channel_index=channel_index, sweep_index=sweep_index)

    def iter_sweeps(self, *, channel_index: int) -> Iterator[SweepData]:
        """Iterate over all sweeps for one channel.

        Args:
            channel_index: Zero-based input channel index.

        Yields:
            :class:`SweepData` for each sweep in ascending sweep-index order.

        Raises:
            ValueError: If ``channel_index`` is out of range.
        """
        for sweep_index in range(self.trace_header.num_sweeps):
            yield self.get_sweep(channel_index=channel_index, sweep_index=sweep_index)

    def get_sweep_trace_table(self, *, channel_index: int, sweep_index: int) -> pd.DataFrame:
        """Return a per-sample table for one channel/sweep pair.

        Args:
            channel_index: Zero-based input channel index.
            sweep_index: Zero-based sweep index.

        Returns:
            DataFrame with ``time_sec``, ``value``, ``command``, and ``epoch``
            columns.

        Raises:
            ValueError: If either index is out of range.
        """
        return self.get_sweep(
            channel_index=channel_index,
            sweep_index=sweep_index,
        ).as_dataframe()

    def get_channel_trace_table(self, *, channel_index: int) -> pd.DataFrame:
        """Return a wide per-sample table for all sweeps of one channel.

        Args:
            channel_index: Zero-based input channel index.

        Returns:
            DataFrame with one ``time_sec`` column plus ``sweep_N``,
            ``sweep_N_command``, and ``sweep_N_epoch`` columns for every sweep.

        Raises:
            ValueError: If ``channel_index`` is out of range or sweep time axes
                are inconsistent.
        """
        table: pd.DataFrame | None = None
        for sweep in self.iter_sweeps(channel_index=channel_index):
            sweep_table = sweep.as_dataframe()
            if table is None:
                table = pd.DataFrame({'time_sec': sweep_table['time_sec']})
            elif not table['time_sec'].equals(sweep_table['time_sec']):
                raise ValueError(
                    'Cannot build a wide channel trace table because sweep '
                    f'{sweep.sweep_index} has a different time axis'
                )
            prefix = f'sweep_{sweep.sweep_index}'
            table[prefix] = sweep_table['value']
            table[f'{prefix}_command'] = sweep_table['command']
            table[f'{prefix}_epoch'] = sweep_table['epoch']
        if table is None:
            return pd.DataFrame({'time_sec': []})
        return table

    def get_epoch_table(
        self,
        *,
        channel_index: int,
        sweep_index: int | None = None,
    ) -> pd.DataFrame:
        """Return epoch interval metadata for one or more sweeps.

        Args:
            channel_index: Zero-based input channel index.
            sweep_index: Optional zero-based sweep index. When omitted, epochs
                from all sweeps for the channel are returned.

        Returns:
            DataFrame with one row per epoch interval.

        Raises:
            ValueError: If ``channel_index`` or ``sweep_index`` is out of range.
        """
        if sweep_index is not None:
            sweep = self.get_sweep(channel_index=channel_index, sweep_index=sweep_index)
            return sweep.get_epoch_table(samples_per_second=self.trace_header.samples_per_second)
        frames = [
            sweep.get_epoch_table(samples_per_second=self.trace_header.samples_per_second)
            for sweep in self.iter_sweeps(channel_index=channel_index)
        ]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)


    def run_peak_detection(
        self,
        *,
        channel_index: int,
        sweep_index: int | None = None,
        params: TracePeakDetectionParams | None = None,
    ) -> TracePeakDetectionResult:
        """Run peak detection on one channel of this trace acquisition.

        Args:
            channel_index: Zero-based input channel index.
            sweep_index: Optional zero-based sweep index. When None, all sweeps
                for ``channel_index`` are analyzed.
            params: Optional trace peak-detection parameters. Defaults to
                positive peak detection with no height, prominence, or distance
                filter.

        Returns:
            Trace peak-detection result with per-sample and per-peak tables.

        Raises:
            ValueError: If indices or detection parameters are invalid.
        """
        from acqstore.acq_trace.analysis.trace_peak_detection import (
            run_trace_peak_detection,
        )

        return run_trace_peak_detection(
            self,
            channel_index=channel_index,
            sweep_index=sweep_index,
            params=params,
        )

    def info(self) -> str:
        """Return a human-readable trace file summary.

        Returns:
            Multiline string describing the trace recording.
        """
        return self._loader.info()

    def to_summary_dict(self) -> dict[str, object]:
        """Return a structured summary of this trace acquisition.

        Returns:
            Dictionary of file identity, modality, acceptance state, and trace
            header values for scripting and future GUI adapters.
        """
        return {
            'file_id': self.file_id,
            'name': self.name,
            'path': self.path,
            'modality': self.modality.value,
            'accepted': self.accepted,
            'is_dirty': self.is_dirty,
            'trace_header': self.trace_header.as_dict(),
        }

    def load_lazy_data(self) -> None:
        """Load lazy trace data for interactive use.

        Returns:
            None. ABF sweep data remains loaded on demand in this backend-only
            phase, so this method is currently a no-op.
        """
        return None

    def unload_lazy_data(self) -> None:
        """Unload lazy trace data that can be reloaded later.

        Returns:
            None. ABF sweep data is not cached by :class:`AcqTrace` in this
            backend-only phase, so this method is currently a no-op.
        """
        return None

    def get_sidecar_json_path(self) -> str:
        """Return sidecar JSON path for this trace file.

        Returns:
            Sidecar path using full source filename with extension plus
            ``.json`` suffix, for example ``recording.abf.json``.
        """
        return str(Path(f'{self.path}.json'))

    def save(self) -> None:
        """Persist trace sidecar JSON.

        Returns:
            None. Numeric source data are not modified.
        """
        self.save_sidecar_json()
        self.experiment_metadata.set_clean()

    def save_sidecar_json(self) -> None:
        """Persist sidecar JSON for this trace file.

        Returns:
            None.
        """
        sidecar_path = Path(self.get_sidecar_json_path())
        sidecar_path.write_text(
            json.dumps(self._build_sidecar_payload(), indent=2, sort_keys=True),
            encoding='utf-8',
        )

    def load_sidecar_json(self) -> None:
        """Load sidecar JSON into runtime state when present.

        Invalid sidecar content is ignored with a warning.
        """
        sidecar_path = Path(self.get_sidecar_json_path())
        if not sidecar_path.is_file():
            return
        try:
            self._load_sidecar_payload(
                json.loads(sidecar_path.read_text(encoding='utf-8')),
                source=str(sidecar_path),
            )
        except Exception as exc:  # pragma: no cover - defensive load path
            logger.warning('Failed to load trace sidecar JSON for %s: %s', self.path, exc)

    def _build_sidecar_payload(self) -> dict[str, object]:
        """Build sidecar JSON payload for this trace file.

        Returns:
            JSON-serializable sidecar payload.
        """
        return {
            'version': _ACQTRACE_SIDECAR_VERSION,
            'modality': self.modality.value,
            'accepted': bool(self.accepted),
            'experiment_metadata': self.experiment_metadata.to_dict(),
            'trace_header_metadata': self.trace_header.as_dict(),
        }

    def _load_sidecar_payload(self, raw: object, *, source: str) -> None:
        """Validate and apply one trace sidecar payload.

        Args:
            raw: Parsed JSON value.
            source: Human-readable sidecar source path for warning messages.

        Raises:
            ValueError: If required fields are missing or unsupported.
        """
        if not isinstance(raw, dict):
            raise ValueError('Trace sidecar JSON payload must be an object')
        missing = sorted(_ACQTRACE_SIDECAR_REQUIRED_KEYS - set(raw.keys()))
        if missing:
            raise ValueError(f'Trace sidecar JSON missing required keys: {missing}')
        extra = sorted(
            set(raw.keys())
            - _ACQTRACE_SIDECAR_REQUIRED_KEYS
            - _ACQTRACE_SIDECAR_OPTIONAL_KEYS
        )
        if extra:
            logger.warning('Ignoring unknown AcqTrace sidecar keys for %s: %s', source, extra)
        version = raw['version']
        if version != _ACQTRACE_SIDECAR_VERSION:
            raise ValueError(
                f'Unsupported AcqTrace sidecar version {version!r}; '
                f'expected {_ACQTRACE_SIDECAR_VERSION!r}'
            )
        modality = raw['modality']
        if modality != self.modality.value:
            raise ValueError(
                f'Unsupported AcqTrace sidecar modality {modality!r}; '
                f'expected {self.modality.value!r}'
            )
        exp_obj = raw['experiment_metadata']
        if exp_obj is not None and not isinstance(exp_obj, dict):
            raise ValueError("Trace sidecar field 'experiment_metadata' must be an object")
        header_obj = raw['trace_header_metadata']
        if header_obj is not None and not isinstance(header_obj, dict):
            raise ValueError("Trace sidecar field 'trace_header_metadata' must be an object")
        self.accepted = bool(raw['accepted'])
        self.experiment_metadata = ExperimentMetadata.from_dict(exp_obj)
