"""Compute vector-valued xi and sigma from batches of strain trajectories e(t).

For each input path, this module solves the constitutive relation from the
``Multidimensional Learning`` section of ``ou_process_workflow.tex``:

    d xi / dt = ||e(t) - 2 xi(t)||^(1/p) sign(e(t) - 2 xi(t)),
    sigma(t) = e(t) - xi(t).
"""

from __future__ import annotations

import numpy as np


def compute_xi_and_sigma(
    e_trajectories: np.ndarray,
    dt: float = 0.01,
    *,
    p: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve for vector-valued ``xi`` and ``sigma`` from strain trajectories.

    ``e_trajectories`` is sampled on a uniform time grid. For vector-valued
    strain, it has shape ``(n_trajectories, n_times, m)``. Forward Euler is
    used for the internal-variable ODE:

        r[n] = e[n] - 2 * xi[n]
        xi[n + 1] = xi[n] + dt * ||r[n]||^(1/p) * r[n] / ||r[n]||.

    The stress-like output is evaluated at every supplied strain value:

        sigma[n] = e[n] - xi[n].

    Parameters
    ----------
    e_trajectories : numpy.ndarray, shape (n_trajectories, n_times, m)
        Input trajectories of e(t). A one-dimensional input is interpreted as
        one scalar trajectory and a two-dimensional input as scalar trajectories.
    dt : float, default=0.01
        Uniform time step between columns of ``e_trajectories``.
    p : float, default=1.0
        Power-law exponent parameter. Must satisfy ``0 < p <= 1``.

    Returns
    -------
    xi : numpy.ndarray
        Internal-variable trajectories with the same shape as the normalized
        strain input.
    sigma : numpy.ndarray
        Stress-like trajectories with the same shape as ``xi``.
    """
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if not 0 < p <= 1:
        raise ValueError("p must satisfy 0 < p <= 1.")

    e = np.asarray(e_trajectories, dtype=float)
    scalar_input = e.ndim <= 2
    if e.ndim == 1:
        e = e[np.newaxis, :, np.newaxis]
    elif e.ndim == 2:
        e = e[:, :, np.newaxis]
    if e.ndim != 3:
        raise ValueError(
            "e_trajectories must be a 1D, 2D, or 3D array."
        )
    if e.shape[1] == 0:
        raise ValueError("e_trajectories must contain at least one time point.")

    n_trajectories, n_times, _ = e.shape
    xi = np.zeros_like(e)

    for time_index in range(n_times - 1):
        residual = e[:, time_index+1] - 2.0 * xi[:, time_index]
        residual_norm = np.linalg.norm(residual, axis=-1, keepdims=True)
        direction = np.divide(
            residual,
            residual_norm,
            out=np.zeros_like(residual),
            where=residual_norm > 0,
        )
        xi[:, time_index + 1] = xi[:, time_index] + dt * (
            residual_norm ** (1.0 / p)
        ) * direction

    sigma = e - xi
    if scalar_input:
        return xi[..., 0], sigma[..., 0]
    return xi, sigma


if __name__ == "__main__":
    example_e = np.ones((2, 101))
    xi, sigma = compute_xi_and_sigma(example_e)
    print("xi final values:", xi[:, -1])
    print("sigma final values:", sigma[:, -1])
