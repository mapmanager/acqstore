# AcqTrace backend development notes

`acqstore.acq_trace` is the backend-only model for non-image electrophysiology recordings such as ABF files loaded through `pyABF`. It is intentionally developed as a sibling of `AcqImage`, not as an image-shaped object with fake pixels or fake ROIs.

## Current scope

The current implementation supports:

- `AcqTrace` for one trace recording file.
- `AbfTraceLoader` for `.abf` files.
- `TraceHeader` for file-level sweep/channel metadata.
- `SweepData` for one `(channel_index, sweep_index)` pair.
- `EpochInterval` and `EpochTable` for ABF command epoch intervals.
- Sidecar JSON for trace-specific acceptance and experimental metadata.
- Trace table APIs:
  - `get_sweep_trace_table(channel_index, sweep_index)`
  - `get_channel_trace_table(channel_index)`
  - `get_epoch_table(channel_index, sweep_index=None)`
- Peak detection APIs:
  - `detect_peaks_1d(...)` in `acqstore.common_analysis.peak_detection_core`
  - `run_trace_peak_detection(...)` in `acqstore.acq_trace.analysis.trace_peak_detection`
  - `AcqTrace.run_peak_detection(...)` as a convenience wrapper

The package remains pure backend Python. It does not import from `cloudscope` or `nicewidgets`.

## What AcqTrace deliberately does not include

`AcqTrace` does not expose or emulate:

- image pixels
- image header metadata
- rectangular ROIs
- image contrast
- reference images
- velocity analysis
- diameter analysis
- CloudScope primary image view behavior
- CloudScope ROI/contrast toolbar behavior

Future GUI integration should branch by modality and use trace-specific views/toolbars for ABF data.

## Modality

Trace files use `AcqModality.TRACE` from `src/acqstore/acq_types.py`.

Future image code can expose `AcqModality.IMAGE` when the mixed acquisition-list work begins. A future mixed list should use a small shared item protocol for file identity and lifecycle, while image-only and trace-only APIs remain separate.

## Trace table versus epoch table

Terminology used in this package:

- **Sweep data**: in-memory arrays for one `(channel_index, sweep_index)` pair.
- **Trace table**: one row per sample.
- **Epoch table**: one row per command epoch interval.
- **Peak table**: one row per detected peak.

For one sweep, `SweepData.as_dataframe()` returns:

```text
time_sec
value
command
epoch
```

For all sweeps in one channel, `AcqTrace.get_channel_trace_table(...)` returns a wide table:

```text
time_sec
sweep_0
sweep_0_command
sweep_0_epoch
sweep_1
sweep_1_command
sweep_1_epoch
...
```

`AcqTrace.get_epoch_table(channel_index, sweep_index=None)` returns compact interval metadata:

```text
channel_index
sweep_index
epoch_index
start_sample
end_sample
duration_samples
start_sec
end_sec
duration_sec
level
epoch_type
digital_states
```

## Peak detection design

Peak detection is split into two layers.

### Shared core

`src/acqstore/common_analysis/peak_detection_core.py` contains modality-neutral one-dimensional peak detection.

The shared core accepts only:

- `time_sec`
- `values`
- `PeakDetectionCoreParams`

It returns:

- a per-sample trace table with `is_peak` and `peak_id`
- a peak table with peak index/time/value/prominence/width fields

The shared core does not know about ABF files, sweeps, channels, epochs, image ROIs, CSV files, or GUI state.

### Trace wrapper

`src/acqstore/acq_trace/analysis/trace_peak_detection.py` wraps the shared core for `AcqTrace` and adds trace-specific identifiers:

- `channel_index`
- `channel_name`
- `sweep_index`
- `epoch_index`
- epoch interval metadata
- value units
- global peak IDs across all analyzed sweeps

Default behavior analyzes all sweeps for a channel:

```python
result = trace.run_peak_detection(channel_index=0, sweep_index=None)
```

A caller can analyze one sweep explicitly:

```python
result = trace.run_peak_detection(channel_index=0, sweep_index=3)
```

This avoids silently analyzing only sweep 0.

## Why AcqAnalysisSet is deferred

Current image analysis uses `AcqAnalysisSet` and an image-centric `AnalysisKey` shaped around:

```text
analysis_name + channel + roi_id
```

ABF peak analysis is naturally organized around:

```text
analysis_name + channel_index + sweep_index
```

Each detected peak also belongs to an epoch interval. Epoch membership is stored in each peak row rather than forced into an ROI identifier.

For this reason, `AcqTrace` does not currently attach the existing image `AcqAnalysisSet`. A future backend pass should design a trace-specific analysis set, probably with:

- `TraceAnalysisKey`
- `TraceAnalysisSet`
- trace analysis result persistence

This should not reuse `roi_id` for sweep or epoch identifiers.

## Sidecar JSON

`AcqTrace.save()` writes trace-specific sidecar JSON at:

```text
<source.abf>.json
```

The sidecar currently stores:

- `version`
- `modality`
- `accepted`
- `experiment_metadata`
- `trace_header_metadata`

It intentionally does not store image-only fields such as ROIs, image contrast, or image header metadata.

## Development script

`scripts/try_abfile.py` exercises the public API. It hard-codes an ABF path, loads the file, prints summaries, runs peak detection on all sweeps for one channel, and plots three linked Plotly rows:

1. recorded values with detected peak markers
2. DAC command values
3. epoch labels

The script has no command-line arguments by design. Edit the hard-coded path before running.

## Future CloudScope integration notes

Future GUI integration should remain modality-aware:

- Image files use image views, ROI tools, contrast controls, velocity, diameter, and image sum-intensity analysis.
- Trace files use trace-specific channel/sweep controls and peak-analysis views.

The existing `SumIntensityPlotView` may eventually be refactored into a reusable peak-trace plot component, but ABF should not pretend to be image-derived sum intensity.

Future mixed-list work should likely introduce a small common acquisition item protocol containing only safe shared fields such as file identity, modality, accepted state, dirty state, save/load lifecycle, and schema/tree rows. It should not include image-only methods.
