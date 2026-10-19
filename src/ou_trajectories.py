"""Simulation utilities for the Ornstein--Uhlenbeck (OU) process.

The vector process is parameterized as

    dX_t = theta @ (mu - X_t) dt + sigma @ dW_t,

where ``W_t`` is a vector Brownian motion.
"""

from __future__ import annotations

import numpy as np


def smooth_ou_trajectories(
    trajectories: np.ndarray,
    window_size: int = 5,
) -> np.ndarray:
    """Smooth OU trajectories by convolution with a discrete mollifier.

    Smoothing is applied independently to every trajectory and state
    component along the time axis.  Scalar trajectories have shape
    ``(n_trajectories, n_times)`` and vector trajectories have shape
    ``(n_trajectories, n_times, state_dim)``.  Each trajectory is first
    shifted by its initial value, so the smoothed trajectory starts at zero.
    The shifted signal uses an odd reflection at the initial boundary and an
    even reflection at the final boundary, avoiding an artificial jump at
    either edge.
    The kernel is obtained by sampling and normalizing the smooth,
    compactly supported mollifier
    ``phi(x) = exp(-1 / (1 - x**2))`` for ``abs(x) < 1`` and zero otherwise.

    Parameters
    ----------
    trajectories : numpy.ndarray
        Sampled OU trajectories.
    window_size : int, default=5
        Odd number of temporal samples in the mollifier support.  It must be
        a positive odd integer and no larger than the number of time points.

    Returns
    -------
    numpy.ndarray
        Smoothed trajectories with the same shape and dtype as the input.
    """
    values = np.asarray(trajectories)
    if values.ndim not in (2, 3):
        raise ValueError(
            "trajectories must have shape (n_trajectories, n_times) or "
            "(n_trajectories, n_times, state_dim)."
        )
    if values.shape[1] < 1:
        raise ValueError("trajectories must contain at least one time point.")
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer.")
    if window_size > values.shape[1]:
        raise ValueError(
            "window_size cannot exceed the number of time points."
        )

    radius = window_size // 2
    initial_values = values[:, 0, ...]
    initial_values_expanded = (
        initial_values[:, None]
        if values.ndim == 2
        else initial_values[:, None, :]
    )
    if radius == 0:
        result = values - initial_values_expanded
        return result.astype(values.dtype, copy=False)

    shifted_values = values - initial_values_expanded

    kernel_coordinates = np.arange(-radius, radius + 1, dtype=float) / radius
    kernel = np.zeros(window_size, dtype=float)
    interior = np.abs(kernel_coordinates) < 1.0
    kernel[interior] = np.exp(
        -1.0 / (1.0 - kernel_coordinates[interior] ** 2)
    )
    kernel /= kernel.sum()

    pad_width = ((0, 0), (radius, radius)) + ((0, 0),) * (values.ndim - 2)
    padded = np.pad(shifted_values, pad_width=pad_width, mode="reflect")
    # Odd reflection about the zero initial state prevents a slope jump.
    padded[:, :radius] *= -1.0
    values_3d = padded[..., np.newaxis] if values.ndim == 2 else padded
    smoothed = np.empty(
        (values_3d.shape[0], values.shape[1], values_3d.shape[2]),
        dtype=np.result_type(values, float),
    )
    for trajectory_index in range(values_3d.shape[0]):
        for component_index in range(values_3d.shape[2]):
            smoothed[trajectory_index, :, component_index] = np.convolve(
                values_3d[trajectory_index, :, component_index],
                kernel,
                mode="valid",
            )
    smoothed = smoothed[:, :, 0] if values.ndim == 2 else smoothed
    # Symmetry of the mollifier and odd reflection make this zero exactly up
    # to floating-point roundoff; assign it explicitly for a strict anchor.
    smoothed[:, 0, ...] = 0.0
    return smoothed.astype(values.dtype, copy=False)


def generate_ou_trajectories(
    theta: np.ndarray | float,
    mu: np.ndarray | float,
    sigma: np.ndarray | float,
    x0: np.ndarray | float,
    n_trajectories: int = 10,
    T: float = 1.0,
    dt: float = 0.01,
    *,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate OU trajectories using the Euler--Maruyama method.

    Parameters
    ----------
    theta : numpy.ndarray, shape (state_dim, state_dim)
        Mean-reversion matrix.
    mu : numpy.ndarray, shape (state_dim,)
        Long-run mean vector.
    sigma : numpy.ndarray, shape (state_dim, brownian_dim)
        Diffusion matrix. The Brownian motion has ``brownian_dim`` components.
    x0 : numpy.ndarray, shape (state_dim,)
        Initial state for every trajectory.
    n_trajectories : int, default=10
        Number of independent sample paths.
    T : float, default=1.0
        Final simulation time.
    dt : float, default=0.01
        Time step. The last step is shortened when needed to end exactly at ``T``.
    seed : int | None, optional
        Seed for reproducible random paths.

    Returns
    -------
    times : numpy.ndarray, shape (n_steps + 1,)
        Times at which the process is evaluated.
    paths : numpy.ndarray
        Simulated trajectories with shape
        ``(n_trajectories, n_steps + 1, state_dim)`` for vector inputs. For
        scalar inputs, the legacy shape ``(n_trajectories, n_steps + 1)`` is
        returned.
    """
    if n_trajectories < 1:
        raise ValueError("n_trajectories must be at least 1.")
    if T <= 0:
        raise ValueError("T must be positive.")
    if dt <= 0:
        raise ValueError("dt must be positive.")

    theta_array = np.asarray(theta, dtype=float)
    mu_array = np.asarray(mu, dtype=float)
    sigma_array = np.asarray(sigma, dtype=float)
    x0_array = np.asarray(x0, dtype=float)

    scalar_input = (
        theta_array.ndim == 0
        and mu_array.ndim == 0
        and sigma_array.ndim == 0
        and x0_array.ndim == 0
    )
    if theta_array.ndim == 0:
        theta_array = theta_array.reshape(1, 1)
    if mu_array.ndim == 0:
        mu_array = mu_array.reshape(1)
    if x0_array.ndim == 0:
        x0_array = x0_array.reshape(1)
    if mu_array.ndim != 1:
        raise ValueError("mu must be a vector.")
    if x0_array.ndim != 1:
        raise ValueError("x0 must be a vector.")
    if theta_array.ndim != 2 or theta_array.shape[0] != theta_array.shape[1]:
        raise ValueError("theta must be a square matrix.")
    state_dim = mu_array.size
    if theta_array.shape != (state_dim, state_dim):
        raise ValueError("theta must have shape (state_dim, state_dim).")
    if x0_array.shape != (state_dim,):
        raise ValueError("x0 must have shape (state_dim,).")
    if sigma_array.ndim == 0:
        sigma_array = sigma_array.reshape(1, 1)
    if sigma_array.ndim != 2 or sigma_array.shape[0] != state_dim:
        raise ValueError("sigma must have shape (state_dim, brownian_dim).")

    # Include a final, potentially shorter, step so the endpoint is exactly T.
    n_full_steps = int(np.floor(T / dt))
    times = np.arange(n_full_steps + 1, dtype=float) * dt
    if T - times[-1] > 1e-12:
        times = np.append(times, T)
    else:
        times[-1] = T  # Avoid small floating-point endpoint discrepancies.

    paths = np.empty((n_trajectories, len(times), state_dim), dtype=float)
    paths[:, 0, :] = x0_array
    rng = np.random.default_rng(seed)

    for step, step_size in enumerate(np.diff(times), start=1):
        previous = paths[:, step - 1, :]
        dW = rng.normal(
            0.0, np.sqrt(step_size), size=(n_trajectories, sigma_array.shape[1])
        )
        drift = (mu_array - previous) @ theta_array.T
        noise = dW @ sigma_array.T
        paths[:, step, :] = previous + drift * step_size + noise

    if scalar_input:
        return times, paths[..., 0]
    return times, paths


if __name__ == "__main__":
    import logging

    from src.logging_utils import setup_logging

    setup_logging()
    logger = logging.getLogger(__name__)
    times, trajectories = generate_ou_trajectories(
        theta=2.0, mu=1.0, sigma=0.3, x0=0.0, seed=42
    )
    logger.info("Generated %d trajectories over [0, %s].", len(trajectories), times[-1])
    logger.info("Final values: %s", trajectories[:, -1])
