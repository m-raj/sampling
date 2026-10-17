"""Shared helpers for evaluating learned constitutive models."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

# np.trapezoid replaced np.trapz in NumPy 2.0.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def _as_numpy(values) -> np.ndarray:
    """Convert a NumPy array or torch tensor to a float NumPy array."""
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=float)


def _as_trajectories(values: np.ndarray) -> np.ndarray:
    """Return values with shape ``(batch, time, dim)``.

    Accepted layouts follow the rest of the code base:
    ``(time,)`` for one scalar history, ``(batch, time)`` for scalar
    histories such as the 1D stress returned by ``stress_history`` or
    ``VariationalODESolver.solve``, and ``(batch, time, dim)`` for
    vector-valued histories.
    """
    if values.ndim == 1:
        return values[np.newaxis, :, np.newaxis]
    if values.ndim == 2:
        return values[:, :, np.newaxis]
    if values.ndim == 3:
        return values
    raise ValueError(
        "Expected shape (time,), (batch, time), or (batch, time, dim), "
        f"got {values.shape}."
    )


def relative_l2_error(
    true_values,
    predicted_values,
    times: Optional[np.ndarray] = None,
    reduction: str = "mean",
    eps: float = 1e-12,
):
    """Relative L2-in-time error between true and predicted histories.

    For each trajectory the error is

        ||f_true - f_pred||_{L2(0, T)} / ||f_true||_{L2(0, T)},

    with ``||f||_{L2(0, T)}^2 = int_0^T |f(t)|^2 dt`` and ``|.|`` the
    Euclidean norm over components. The time integral uses the
    trapezoidal rule. Works for stresses, strains, or internal variables.

    Parameters
    ----------
    true_values, predicted_values : numpy.ndarray or torch.Tensor
        Histories of shape ``(time,)``, ``(batch, time)``, or
        ``(batch, time, dim)``.
    times : numpy.ndarray, optional
        Time grid of shape ``(time,)``. If omitted a uniform grid is
        assumed; the relative error does not depend on the uniform step.
    reduction : {"mean", "none"}
        ``"mean"`` averages over trajectories; ``"none"`` returns the
        error of every trajectory with shape ``(batch,)``.
    eps : float
        Lower bound on the denominator to avoid division by zero.
    """
    true_array = _as_trajectories(_as_numpy(true_values))
    predicted_array = _as_trajectories(_as_numpy(predicted_values))
    if true_array.shape != predicted_array.shape:
        raise ValueError(
            "true_values and predicted_values must have the same shape, "
            f"got {true_array.shape} and {predicted_array.shape}."
        )

    if times is not None:
        times = _as_numpy(times)
        if times.shape != (true_array.shape[1],):
            raise ValueError(
                f"times must have shape ({true_array.shape[1]},), got {times.shape}."
            )

    def squared_l2_norm(values: np.ndarray) -> np.ndarray:
        pointwise = np.sum(values**2, axis=-1)
        if pointwise.shape[1] == 1:
            return pointwise[:, 0]
        if times is None:
            return _trapezoid(pointwise, axis=1)
        return _trapezoid(pointwise, x=times, axis=1)

    error_norm = np.sqrt(squared_l2_norm(true_array - predicted_array))
    true_norm = np.sqrt(squared_l2_norm(true_array))
    relative_errors = error_norm / np.maximum(true_norm, eps)

    if reduction == "mean":
        return float(relative_errors.mean())
    if reduction == "none":
        return relative_errors
    raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}.")
