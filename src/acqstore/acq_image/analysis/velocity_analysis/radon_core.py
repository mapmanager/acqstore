"""Pure Radon-transform velocity analysis core.

This module intentionally has no imports from ``acqstore``. It operates on
plain NumPy arrays and returns plain Python/pandas-friendly outputs so it can be
used from scripts, tests, GUI workers, and future batch runners.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import queue
import time
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing.pool import Pool
from typing import Any

import numpy as np
import pandas as pd
from skimage.transform import radon

RADON_VELOCITY_COLUMNS: tuple[str, ...] = (
    "time_s",
    "time_index",
    "theta_deg",
    "velocity",
)


class RadonVelocityCancelled(RuntimeError):
    """Raised when Radon velocity analysis is cancelled."""


@dataclass(frozen=True, slots=True)
class RadonVelocityCoreResult:
    """Pure Radon velocity output.

    Args:
        summary: Small JSON-serializable result summary.
        table: Per-window result table.
        spread_matrix_fine: Fine-angle variance matrix with shape
            ``(n_windows, n_fine_angles)``.
    """

    summary: dict[str, Any]
    table: pd.DataFrame
    spread_matrix_fine: np.ndarray


def radon_worker(
    data_window: np.ndarray,
    angles: np.ndarray,
    angles_fine: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Calculate flow angle for one time window using Radon transforms.

    This is the multiprocessing-safe worker port of the historical Radon
    implementation. The algorithm subtracts the window mean, performs a coarse
    Radon search over ``angles``, then refines around the best coarse angle with
    ``angles_fine`` offsets.

    Args:
        data_window: Two-dimensional ``(time, space)`` image window.
        angles: Coarse angle grid in degrees.
        angles_fine: Fine angle offsets in degrees around the best coarse angle.

    Returns:
        Tuple ``(best_theta_deg, spread_fine)`` where ``spread_fine`` contains
        the variance of the fine Radon projections.
    """
    data_window = data_window.astype(np.float32, copy=False)
    data_window = data_window - float(np.mean(data_window))

    radon_coarse = radon(data_window, theta=angles, circle=False)
    spread_coarse = np.var(radon_coarse, axis=0)
    coarse_theta = float(angles[int(np.argmax(spread_coarse))])

    fine_angles = coarse_theta + angles_fine
    radon_fine = radon(data_window, theta=fine_angles, circle=False)
    spread_fine = np.var(radon_fine, axis=0)
    best_theta = coarse_theta + float(angles_fine[int(np.argmax(spread_fine))])

    return best_theta, spread_fine.astype(np.float32, copy=False)


def analyze_radon_windows(
    data: np.ndarray,
    *,
    window_width: int,
    dim0_start: int | None = None,
    dim0_stop: int | None = None,
    dim1_start: int | None = None,
    dim1_stop: int | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    progress_every: int = 1,
    cancel_callback: Callable[[], bool] | None = None,
    use_multiprocessing: bool = True,
    processes: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the historical windowed Radon analysis on a 2D array.

    This function preserves the old core behavior: 25 percent window overlap,
    coarse angle search over 0..179 degrees, and fine search from -2 to +2
    degrees in 0.25 degree steps.

    Args:
        data: Two-dimensional ``(time, space)`` kymograph/ROI image.
        window_width: Number of time samples per analysis window.
        dim0_start: Optional start index along time axis.
        dim0_stop: Optional stop index along time axis.
        dim1_start: Optional start index along space axis.
        dim1_stop: Optional stop index along space axis.
        progress_callback: Optional callback receiving ``(completed, total)``.
        progress_every: Emit progress every N completed windows.
        cancel_callback: Optional callback returning True when cancellation is
            requested.
        use_multiprocessing: Whether to use ``multiprocessing.Pool``.
        processes: Optional number of worker processes for the inner Radon
            computation. If None, uses ``os.cpu_count() - 1`` clamped to at
            least one.

    Returns:
        Tuple ``(thetas, time_indices, spread_matrix_fine)``.

    Raises:
        ValueError: If inputs are invalid.
        RadonVelocityCancelled: If cancellation is requested.
    """
    if data.ndim != 2:
        raise ValueError(f"data must be 2D (time, space); got shape={data.shape}")

    dim0_start = 0 if dim0_start is None else int(dim0_start)
    dim0_stop = data.shape[0] if dim0_stop is None else int(dim0_stop)
    dim1_start = 0 if dim1_start is None else int(dim1_start)
    dim1_stop = data.shape[1] if dim1_stop is None else int(dim1_stop)

    if window_width <= 0:
        raise ValueError(f"window_width must be positive, got {window_width}")

    n_time = dim0_stop - dim0_start
    stepsize = int(0.25 * window_width)
    if stepsize <= 0:
        raise ValueError(f"window_width too small to compute stepsize: {window_width}")

    nsteps = math.floor(n_time / stepsize) - 3
    if nsteps <= 0:
        raise ValueError(
            f"Invalid nsteps={nsteps}. Check window_width={window_width} and data.shape={data.shape}"
        )

    angles = np.arange(180, dtype=np.float32)
    fine_step = 0.25
    angles_fine = np.arange(-2.0, 2.0 + fine_step, fine_step, dtype=np.float32)

    thetas = np.zeros(nsteps, dtype=np.float32)
    time_indices = np.ones(nsteps, dtype=np.float32) * np.nan
    spread_matrix_fine = np.zeros((nsteps, len(angles_fine)), dtype=np.float32)

    completed = 0
    last_emit = 0

    def cancelled() -> bool:
        return bool(cancel_callback is not None and cancel_callback())

    def maybe_progress(force: bool = False) -> None:
        nonlocal last_emit
        if progress_callback is None:
            return
        if force or (completed - last_emit) >= max(1, progress_every):
            progress_callback(completed, nsteps)
            last_emit = completed

    maybe_progress(force=True)

    if use_multiprocessing and nsteps > 1:
        proc_count = processes or (os.cpu_count() or 1) - 1
        proc_count = max(1, int(proc_count))
        with Pool(processes=proc_count) as pool:
            result_objs = []
            for k in range(nsteps):
                if cancelled():
                    pool.terminate()
                    pool.join()
                    raise RadonVelocityCancelled(
                        "Radon velocity analysis cancelled before submitting all windows."
                    )
                time_indices[k] = dim0_start + (k * stepsize) + (window_width / 2.0)
                t_start = dim0_start + (k * stepsize)
                t_stop = t_start + window_width
                data_window = data[t_start:t_stop, dim1_start:dim1_stop]
                result_objs.append(pool.apply_async(radon_worker, (data_window, angles, angles_fine)))

            for k, result in enumerate(result_objs):
                if cancelled():
                    pool.terminate()
                    pool.join()
                    raise RadonVelocityCancelled(
                        "Radon velocity analysis cancelled while processing windows."
                    )
                worker_theta, worker_spread_fine = result.get()
                thetas[k] = worker_theta
                spread_matrix_fine[k, :] = worker_spread_fine
                completed += 1
                maybe_progress()
    else:
        for k in range(nsteps):
            if cancelled():
                raise RadonVelocityCancelled("Radon velocity analysis cancelled.")
            time_indices[k] = dim0_start + (k * stepsize) + (window_width / 2.0)
            t_start = dim0_start + (k * stepsize)
            t_stop = t_start + window_width
            data_window = data[t_start:t_stop, dim1_start:dim1_stop]
            worker_theta, worker_spread_fine = radon_worker(data_window, angles, angles_fine)
            thetas[k] = worker_theta
            spread_matrix_fine[k, :] = worker_spread_fine
            completed += 1
            maybe_progress()

    maybe_progress(force=True)
    return thetas, time_indices, spread_matrix_fine


def theta_to_velocity(
    theta_deg: np.ndarray,
    *,
    step_time: float,
    step_space: float,
) -> np.ndarray:
    """Convert Radon angles to velocity estimates.

    Args:
        theta_deg: Angle estimates in degrees.
        step_time: Physical step along the time/line axis.
        step_space: Physical step along the spatial axis.

    Returns:
        Velocity estimates in ``step_space / step_time`` units.
    """
    theta_rad = np.deg2rad(theta_deg.astype(np.float64, copy=False))
    tangent = np.tan(theta_rad)
    with np.errstate(divide="ignore", invalid="ignore"):
        velocity = (step_space / step_time) / tangent
    velocity[~np.isfinite(velocity)] = np.nan
    return velocity.astype(np.float64, copy=False)


def run_radon_velocity(
    image: np.ndarray,
    *,
    window_width: int,
    physical_units: tuple[float, float],
    progress_callback: Callable[[float | None, str], None] | None = None,
    cancel_callback: Callable[[], bool] | None = None,
    use_multiprocessing: bool = True,
    processes: int | None = None,
) -> RadonVelocityCoreResult:
    """Run Radon velocity analysis and return summary/table outputs.

    Args:
        image: Two-dimensional ``(time, space)`` ROI image.
        window_width: Number of time samples per Radon window.
        physical_units: Tuple ``(step_time, step_space)`` aligned with image
            axes. The result table ``time_s`` is computed from the window-center
            time index and ``step_time``.
        progress_callback: Optional callback receiving ``(fraction, message)``.
        cancel_callback: Optional cancellation callback.
        use_multiprocessing: Whether to use multiprocessing for Radon windows.
        processes: Number of worker processes for one file. ``None`` chooses a
            CPU-count based default.

    Returns:
        ``RadonVelocityCoreResult`` with summary, table, and fine spread matrix.

    Raises:
        ValueError: If input validation fails.
        RadonVelocityCancelled: If cancellation is requested.
    """
    start_time = time.time()
    if len(physical_units) != 2:
        raise ValueError("physical_units must be a (step_time, step_space) tuple")
    step_time = float(physical_units[0])
    step_space = float(physical_units[1])
    if step_time <= 0 or step_space <= 0:
        raise ValueError(f"physical_units must be positive, got {physical_units!r}")

    def on_window_progress(completed: int, total: int) -> None:
        if progress_callback is None:
            return
        fraction = None if total <= 0 else completed / total
        progress_callback(fraction, f"Radon windows {completed}/{total}")

    thetas, time_indices, spread_matrix_fine = analyze_radon_windows(
        image,
        window_width=int(window_width),
        progress_callback=on_window_progress,
        cancel_callback=cancel_callback,
        use_multiprocessing=use_multiprocessing,
        processes=processes,
    )
    time_s = time_indices.astype(np.float64) * step_time
    velocity = theta_to_velocity(thetas, step_time=step_time, step_space=step_space)
    table = pd.DataFrame(
        {
            "time_s": time_s,
            "time_index": time_indices.astype(np.float64),
            "theta_deg": thetas.astype(np.float64),
            "velocity": velocity,
        },
        columns=list(RADON_VELOCITY_COLUMNS),
    )
    summary = {
        # "window_width": int(window_width),  # not needed, stored in detection_params
        "num_windows": int(len(table)),
        "velocity_mean": float(np.nanmean(velocity)) if len(velocity) else float("nan"),
        "velocity_median": float(np.nanmedian(velocity)) if len(velocity) else float("nan"),
        "velocity_cv": float(np.nanstd(velocity)) / float(np.nanmean(velocity)) if len(velocity) else float("nan"),
        # "elapsed_s": float(time.time() - start_time),  # time the algorithm took
        # "use_multiprocessing": bool(use_multiprocessing),
        # "processes": None if processes is None else int(processes),
    }
    return RadonVelocityCoreResult(
        summary=summary,
        table=table,
        spread_matrix_fine=spread_matrix_fine,
    )
