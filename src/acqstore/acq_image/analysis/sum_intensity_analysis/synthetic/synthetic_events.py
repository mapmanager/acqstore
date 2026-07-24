"""Event-time helpers for synthetic sum-intensity data."""

from __future__ import annotations

import numpy as np


def generate_poisson_event_times(
    *,
    duration_sec: float,
    rate_hz: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate event times from a homogeneous Poisson process.

    Args:
        duration_sec: Duration of the synthetic trace in seconds.
        rate_hz: Average event rate in events per second.
        rng: NumPy random generator.

    Returns:
        Sorted event times in seconds within ``[0, duration_sec)``.

    Raises:
        ValueError: If duration or rate are not positive.
    """
    if duration_sec <= 0:
        raise ValueError("duration_sec must be > 0")
    if rate_hz <= 0:
        raise ValueError("rate_hz must be > 0")

    times: list[float] = []
    current = 0.0
    while True:
        current += float(rng.exponential(1.0 / rate_hz))
        if current >= duration_sec:
            break
        times.append(current)
    return np.asarray(times, dtype=float)
