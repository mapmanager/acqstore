"""Synthetic sum-intensity image generator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from acqstore.acq_image.analysis.sum_intensity_analysis.synthetic.synthetic_config import (
    SyntheticSumIntensityConfig,
)
from acqstore.acq_image.analysis.sum_intensity_analysis.synthetic.synthetic_data import (
    SyntheticSumIntensityData,
)
from acqstore.acq_image.analysis.sum_intensity_analysis.synthetic.synthetic_events import (
    generate_poisson_event_times,
)


def make_synthetic_sum_intensity_image(
    config: SyntheticSumIntensityConfig | None = None,
) -> SyntheticSumIntensityData:
    """Create a synthetic line-scan image for sum-intensity analysis.

    The model is intentionally simple and inspectable:

    ``F(t) = bleach(t) * (F0 + sum(events(t))) + noise + pop_artifacts``

    The returned image then expands the one-dimensional fluorescence trace over
    the spatial axis:

    ``image[t, x] = F(t) * spatial_profile[x] + spatial_noise[t, x]``

    Args:
        config: Optional synthetic-data configuration. Defaults are chosen to
            produce a small, deterministic trace with several positive events.

    Returns:
        Synthetic image, ground-truth traces, and ground-truth event table.

    Raises:
        ValueError: If configuration values are invalid.
    """
    cfg = config or SyntheticSumIntensityConfig()
    _validate_config(cfg)
    rng = np.random.default_rng(cfg.seed)

    time_sec = np.arange(cfg.num_timepoints, dtype=float) * cfg.seconds_per_line
    duration_sec = float(time_sec[-1] + cfg.seconds_per_line)
    event_times = _event_times(cfg, duration_sec=duration_sec, rng=rng)
    amplitudes = _event_amplitudes(cfg, n_events=event_times.size, rng=rng)

    event_trace = np.zeros(cfg.num_timepoints, dtype=float)
    ideal_df_f = np.zeros(cfg.num_timepoints, dtype=float)
    event_records: list[dict[str, float | int]] = []
    for event_id, (event_time, amplitude) in enumerate(zip(event_times, amplitudes), start=1):
        kernel = _difference_of_exponentials(
            time_sec,
            onset_time_sec=float(event_time),
            tau_rise_sec=cfg.tau_rise_sec,
            tau_decay_sec=cfg.tau_decay_sec,
        )
        event_trace += float(amplitude) * kernel
        ideal_df_f += (float(amplitude) / cfg.f0) * kernel
        peak_index = int(np.argmax(kernel)) if kernel.size else 0
        event_records.append(
            {
                "event_id": event_id,
                "onset_time_sec": float(event_time),
                "peak_time_sec": float(time_sec[peak_index]),
                "amplitude": float(amplitude),
            }
        )

    bleach = _bleach_trace(time_sec, cfg)
    fluorescence = bleach * (float(cfg.f0) + event_trace)
    if cfg.noise_sigma > 0:
        fluorescence = fluorescence + rng.normal(0.0, cfg.noise_sigma, size=fluorescence.shape)
    fluorescence = _add_line_pops(fluorescence, cfg=cfg, rng=rng)

    spatial_profile = _spatial_profile(cfg, rng=rng)
    image = fluorescence[:, np.newaxis] * spatial_profile[np.newaxis, :]
    if cfg.spatial_noise_sigma > 0:
        image = image + rng.normal(0.0, cfg.spatial_noise_sigma, size=image.shape)

    return SyntheticSumIntensityData(
        image=np.asarray(image, dtype=float),
        time_sec=time_sec,
        fluorescence_trace=np.asarray(fluorescence, dtype=float),
        ideal_df_f_trace=np.asarray(ideal_df_f, dtype=float),
        ground_truth_events=pd.DataFrame(event_records),
        seconds_per_line=float(cfg.seconds_per_line),
        um_per_pixel=float(cfg.um_per_pixel),
        f0=float(cfg.f0),
    )


def _validate_config(config: SyntheticSumIntensityConfig) -> None:
    """Validate synthetic generator configuration.

    Args:
        config: Configuration to validate.

    Raises:
        ValueError: If a required value is invalid.
    """
    if config.num_timepoints < 2:
        raise ValueError("num_timepoints must be >= 2")
    if config.num_spacepoints < 1:
        raise ValueError("num_spacepoints must be >= 1")
    if config.seconds_per_line <= 0:
        raise ValueError("seconds_per_line must be > 0")
    if config.um_per_pixel <= 0:
        raise ValueError("um_per_pixel must be > 0")
    if config.tau_rise_sec <= 0 or config.tau_decay_sec <= 0:
        raise ValueError("tau_rise_sec and tau_decay_sec must be > 0")
    if config.f0 <= 0:
        raise ValueError("f0 must be > 0")
    if config.bleach_tau_sec is not None and config.bleach_tau_sec <= 0:
        raise ValueError("bleach_tau_sec must be > 0 when supplied")
    if config.bleach_floor_fraction < 0 or config.bleach_floor_fraction > 1:
        raise ValueError("bleach_floor_fraction must be between 0 and 1")
    if config.noise_sigma < 0 or config.spatial_noise_sigma < 0:
        raise ValueError("noise sigma values must be >= 0")
    if config.pop_probability < 0 or config.pop_probability > 1:
        raise ValueError("pop_probability must be between 0 and 1")
    if config.pop_amplitude < 0:
        raise ValueError("pop_amplitude must be >= 0")
    if config.spatial_gain_sigma < 0:
        raise ValueError("spatial_gain_sigma must be >= 0")


def _event_times(
    config: SyntheticSumIntensityConfig,
    *,
    duration_sec: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return configured event times with optional jitter.

    Args:
        config: Synthetic configuration.
        duration_sec: Trace duration in seconds.
        rng: Random generator.

    Returns:
        Sorted event times in seconds.
    """
    if config.event_times_sec is not None:
        times = np.asarray(config.event_times_sec, dtype=float)
    elif config.event_rate_hz is not None:
        times = generate_poisson_event_times(
            duration_sec=duration_sec,
            rate_hz=float(config.event_rate_hz),
            rng=rng,
        )
    else:
        times = np.asarray((), dtype=float)

    if config.event_jitter_sec > 0 and times.size:
        times = times + rng.normal(0.0, config.event_jitter_sec, size=times.shape)
    times = times[(times >= 0.0) & (times < duration_sec)]
    return np.asarray(np.sort(times), dtype=float)


def _event_amplitudes(
    config: SyntheticSumIntensityConfig,
    *,
    n_events: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return one amplitude per event.

    Args:
        config: Synthetic configuration.
        n_events: Number of events.
        rng: Random generator.

    Returns:
        Event amplitudes in fluorescence units.
    """
    if n_events == 0:
        return np.asarray((), dtype=float)
    if config.event_amplitudes is not None:
        values = np.asarray(config.event_amplitudes, dtype=float)
        if values.size != n_events:
            raise ValueError("event_amplitudes length must match event count")
        return values
    amplitudes = np.full(n_events, float(config.event_amplitude), dtype=float)
    if config.event_amplitude_jitter_fraction > 0:
        sigma = abs(float(config.event_amplitude) * config.event_amplitude_jitter_fraction)
        amplitudes = amplitudes + rng.normal(0.0, sigma, size=n_events)
    return np.maximum(amplitudes, 0.0)


def _difference_of_exponentials(
    time_sec: np.ndarray,
    *,
    onset_time_sec: float,
    tau_rise_sec: float,
    tau_decay_sec: float,
) -> np.ndarray:
    """Return a unit-peak difference-of-exponentials event kernel.

    Args:
        time_sec: Time axis in seconds.
        onset_time_sec: Event onset time in seconds.
        tau_rise_sec: Rise time constant in seconds.
        tau_decay_sec: Decay time constant in seconds.

    Returns:
        Event kernel with maximum value 1 when possible.
    """
    dt = np.asarray(time_sec, dtype=float) - float(onset_time_sec)
    kernel = np.zeros_like(dt, dtype=float)
    mask = dt >= 0.0
    if not np.any(mask):
        return kernel
    active = dt[mask]
    kernel[mask] = (1.0 - np.exp(-active / tau_rise_sec)) * np.exp(-active / tau_decay_sec)
    peak = float(np.max(kernel))
    if peak > 0:
        kernel = kernel / peak
    return kernel


def _bleach_trace(time_sec: np.ndarray, config: SyntheticSumIntensityConfig) -> np.ndarray:
    """Return multiplicative photobleaching trend.

    Args:
        time_sec: Time axis in seconds.
        config: Synthetic configuration.

    Returns:
        Bleaching multiplier, one value per time point.
    """
    if config.bleach_tau_sec is None:
        return np.ones_like(time_sec, dtype=float)
    floor = float(config.bleach_floor_fraction)
    return floor + (1.0 - floor) * np.exp(-time_sec / float(config.bleach_tau_sec))


def _add_line_pops(
    fluorescence: np.ndarray,
    *,
    cfg: SyntheticSumIntensityConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add one-line acquisition pop artifacts.

    Args:
        fluorescence: One-dimensional fluorescence trace.
        cfg: Synthetic configuration.
        rng: Random generator.

    Returns:
        Fluorescence trace with sparse line artifacts added.
    """
    if cfg.pop_probability <= 0 or cfg.pop_amplitude <= 0:
        return np.asarray(fluorescence, dtype=float)
    result = np.asarray(fluorescence, dtype=float).copy()
    pop_mask = rng.random(result.size) < cfg.pop_probability
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=int(np.sum(pop_mask)))
    result[pop_mask] += signs * float(cfg.pop_amplitude)
    return result


def _spatial_profile(
    config: SyntheticSumIntensityConfig,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return per-column multiplicative spatial gain profile.

    Args:
        config: Synthetic configuration.
        rng: Random generator.

    Returns:
        One-dimensional profile with length ``num_spacepoints``.
    """
    profile = np.ones(config.num_spacepoints, dtype=float)
    if config.spatial_gain_sigma > 0:
        profile = profile + rng.normal(0.0, config.spatial_gain_sigma, size=profile.shape)
    return np.maximum(profile, 0.0)
