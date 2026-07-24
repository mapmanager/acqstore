"""Configuration model for synthetic sum-intensity image generation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SyntheticSumIntensityConfig:
    """Configuration for synthetic line-scan sum-intensity data.

    The generator creates an actual two-dimensional image with shape
    ``(num_timepoints, num_spacepoints)``. This lets tests and scripts exercise
    the same row-sum, filtering, detrending, df/f0, detection, and width
    measurement pipeline used for real ROI images.

    Event timing can be supplied explicitly with ``event_times_sec`` for
    deterministic tests, or generated from a Poisson process with
    ``event_rate_hz`` for exploratory scripts. When both are supplied,
    ``event_times_sec`` takes precedence.

    Args:
        num_timepoints: Number of line-scan time samples.
        num_spacepoints: Number of spatial samples per line.
        seconds_per_line: Sampling interval in seconds.
        um_per_pixel: Spatial sampling interval in microns.
        event_times_sec: Optional explicit event onset times in seconds.
        event_rate_hz: Optional Poisson event rate used when explicit event
            times are not supplied.
        event_jitter_sec: Standard deviation of Gaussian timing jitter applied
            to generated or explicit events.
        event_amplitudes: Optional explicit event amplitudes in fluorescence
            units. When omitted, amplitudes are generated from
            ``event_amplitude`` and ``event_amplitude_jitter_fraction``.
        event_amplitude: Default peak fluorescence increment for synthetic
            events before bleaching.
        event_amplitude_jitter_fraction: Relative amplitude jitter standard
            deviation. For example, 0.1 gives approximately 10 percent jitter.
        tau_rise_sec: Event rise time constant in seconds.
        tau_decay_sec: Event decay time constant in seconds.
        f0: Baseline fluorescence before bleaching.
        bleach_tau_sec: Photobleaching decay time constant in seconds. Use
            ``None`` to disable bleaching.
        bleach_floor_fraction: Long-time bleaching floor as a fraction of the
            initial fluorescence.
        noise_sigma: Additive Gaussian noise standard deviation applied to the
            one-dimensional fluorescence trace.
        spatial_noise_sigma: Additive Gaussian noise standard deviation applied
            independently to each image pixel.
        pop_probability: Per-line probability of a one-line acquisition pop.
        pop_amplitude: Absolute fluorescence increment for line-pop artifacts.
        spatial_gain_sigma: Standard deviation of per-column multiplicative
            gain noise. Use 0 for a flat spatial profile.
        seed: Random seed for deterministic output.
    """

    num_timepoints: int = 2500
    num_spacepoints: int = 281
    seconds_per_line: float = 0.00393
    um_per_pixel: float = 0.414
    event_times_sec: tuple[float, ...] | None = (1.0, 2.5, 4.0, 6.2, 8.1)
    event_rate_hz: float | None = None
    event_jitter_sec: float = 0.0
    event_amplitudes: tuple[float, ...] | None = None
    event_amplitude: float = 300.0
    event_amplitude_jitter_fraction: float = 0.0
    tau_rise_sec: float = 0.030
    tau_decay_sec: float = 0.250
    f0: float = 1000.0
    bleach_tau_sec: float | None = 20.0
    bleach_floor_fraction: float = 0.35
    noise_sigma: float = 5.0
    spatial_noise_sigma: float = 2.0
    pop_probability: float = 0.001
    pop_amplitude: float = 150.0
    spatial_gain_sigma: float = 0.0
    seed: int | None = 1
