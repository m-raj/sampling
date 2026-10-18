"""Reference linear constitutive law from the LaTeX formulation.

The internal variable and stress satisfy

    xi_dot = strain - 2 * xi,
    stress = strain - xi,
    xi(0) = 0.

Forward Euler is used for the internal-variable update so this model can be
benchmarked against the same time-discrete history used by the learned model.
"""

from __future__ import annotations

import numpy as np


class KinematicHardeningPlasticity:
    """Evaluate the analytical linear constitutive relation on strain paths."""

    def __init__(self, node_count: int, model_dt: float, m: float = 1.0, E: float = 1.0, H: float = 0.1, sigma_y: float = 0.01) -> None:
        if node_count < 1:
            raise ValueError("node_count must be positive.")
        if model_dt <= 0:
            raise ValueError("model_dt must be positive.")
        self.node_count = node_count
        self.model_dt = model_dt
        self.m = m
        self.E = E  # Young's modulus
        self.H = H  # Hardening modulus
        self.sigma_y = sigma_y  # Yield stress

    def stress_history(self, strain_history: np.ndarray) -> np.ndarray:
        """Return stress for every time and node in a strain history.

        Parameters
        ----------
        strain_history : numpy.ndarray, shape (time, node)
            Strain values sampled at uniform model time steps.
        """
        strain = np.asarray(strain_history, dtype=float)
        if strain.ndim != 2 or strain.shape[1] != self.node_count:
            raise ValueError("strain_history must have shape (time, node_count).")
        if strain.shape[0] == 0:
            raise ValueError("strain_history must contain at least one time point.")

        plastic_strain = np.zeros_like(strain)
        stress = np.zeros_like(strain)
        stress[0] = self.E * (strain[0] - plastic_strain[0])
        for time_index in range(strain.shape[0] - 1):
            elastic_drive = (self.E * (strain[time_index+1] - plastic_strain[time_index]) - self.H * plastic_strain[time_index])/self.sigma_y
            direction = np.sign(elastic_drive)
            if self.m>1e-2:
                plastic_strain_increment = np.power(np.abs(elastic_drive), 1.0/self.m)* direction
                plastic_strain[time_index+1] = plastic_strain[time_index] + self.model_dt * plastic_strain_increment
                stress[time_index+1] = self.E * (strain[time_index+1] - plastic_strain[time_index+1])
            else:
                plastic_update = np.where(elastic_drive>0, (self.E*strain[time_index+1] - self.sigma_y)/(self.E + self.H), (self.E*strain[time_index+1] + self.sigma_y)/(self.E + self.H))
                plastic_update = np.where(np.abs(elastic_drive) < 1, plastic_strain[time_index], plastic_update)
                plastic_strain[time_index+1] = plastic_update 
                stress[time_index+1] = self.E * (strain[time_index+1] - plastic_strain[time_index+1])
        return plastic_strain, stress

    def stress(self, strain_history: np.ndarray) -> np.ndarray:
        """Return the final stress state for the FEM residual."""
        return self.stress_history(strain_history)[-1]



# class HardeningPlasticity:
#     """One-dimensional plasticity with isotropic hardening.

#     The model is

#         stress = strain - xi,
#         xi_dot = |stress / yield_stress|**(1 / p) sign(stress),
#         yield_stress_dot = H * |xi_dot|.

#     For ``p > 0`` the equations are integrated with forward Euler. For
#     ``p == 0`` the rate-independent response is computed with a 1D return map.
#     """

#     def __init__(
#         self,
#         node_count: int,
#         model_dt: float,
#         sigma0: float = 0.1,
#         hardening_modulus: float = 0.0,
#         p: float = 1.0,
#     ) -> None:
#         if node_count < 1:
#             raise ValueError("node_count must be positive.")
#         if model_dt <= 0:
#             raise ValueError("model_dt must be positive.")
#         if sigma0 <= 0:
#             raise ValueError("sigma0 must be positive.")
#         if hardening_modulus < 0:
#             raise ValueError("hardening_modulus must be non-negative.")
#         if p < 0:
#             raise ValueError("p must be non-negative.")
#         self.node_count = node_count
#         self.model_dt = model_dt
#         self.sigma0 = sigma0
#         self.hardening_modulus = hardening_modulus
#         self.p = p

#     def stress_history(self, strain_history: np.ndarray) -> np.ndarray:
#         """Return stress for every time and node in a strain history."""
#         strain = np.asarray(strain_history, dtype=float)
#         if strain.ndim != 2 or strain.shape[1] != self.node_count:
#             raise ValueError("strain_history must have shape (time, node_count).")
#         if strain.shape[0] == 0:
#             raise ValueError("strain_history must contain at least one time point.")

#         internal_variable = np.zeros(self.node_count, dtype=float)
#         plastic_strain = np.zeros(self.node_count, dtype=float)
#         stress = np.empty_like(strain)
#         stress[0] = strain[0] - internal_variable

#         for time_index in range(strain.shape[0] - 1):
#             plastic_increment_pos = (-internal_variable - plastic_strain + strain[time_index + 1])/(2 + 1/self.model_dt)
#             plastic_increment_neg = (internal_variable - plastic_strain + strain[time_index + 1])/(2 - 1/self.model_dt)
#             if plastic_increment_pos > 0:
#                 plastic_increment = plastic_increment_pos
#             elif plastic_increment_neg <= 0:
#                 plastic_increment = plastic_increment_neg
#             else: 
#                 raise ValueError("Plastic increment is undefined.")
#             plastic_strain += plastic_increment
#             internal_variable += np.abs(plastic_increment)
#             stress[time_index + 1] = strain[time_index + 1] -plastic_strain
#         return stress

#     def stress(self, strain_history: np.ndarray) -> np.ndarray:
#         """Return the final stress state for the FEM residual."""
#         return self.stress_history(strain_history)[-1]




class IsotropicHardeningPlasticity:
    """Variational plasticity model from the final appendix derivation.

    The stored variables are plastic strain ``epsilon_p`` and accumulated
    hardening ``xi``. The energy and dissipation potential are

        W = E / 2 * (strain - epsilon_p)**2 + H / 2 * xi**2,
        D = 1 / (1 + m) * abs(xi_dot)**(1 + m),
        xi_dot = abs(epsilon_p_dot).

    The update is evaluated independently at every node for a strain history.
    For ``m == 1`` it uses the closed-form update derived in the appendix. For
    other positive ``m`` it solves the corresponding monotone scalar equation by
    bisection.
    """
    def __init__(
        self,
        node_count: int,
        model_dt: float,
        elastic_modulus: float = 1.0,
        hardening_modulus: float = 1.0,
        dissipation_exponent: float = 1.0,
        bisection_iterations: int = 80,
    ) -> None:
        if node_count < 1:
            raise ValueError("node_count must be positive.")
        if model_dt <= 0:
            raise ValueError("model_dt must be positive.")
        if elastic_modulus <= 0:
            raise ValueError("elastic_modulus must be positive.")
        if hardening_modulus < 0:
            raise ValueError("hardening_modulus must be non-negative.")
        if dissipation_exponent <= 0:
            raise ValueError("dissipation_exponent must be positive.")
        if bisection_iterations < 1:
            raise ValueError("bisection_iterations must be positive.")

        self.node_count = node_count
        self.model_dt = model_dt
        self.elastic_modulus = elastic_modulus
        self.hardening_modulus = hardening_modulus
        self.dissipation_exponent = dissipation_exponent
        self.bisection_iterations = bisection_iterations

    def _plastic_increment(self, driving_force: np.ndarray) -> np.ndarray:
        """Solve the non-negative increment equation for each node."""
        magnitude = np.maximum(driving_force, 0.0)
        if self.dissipation_exponent == 1.0:
            return magnitude / (
                self.hardening_modulus
                + self.elastic_modulus
                + 1.0 / self.model_dt
            )

        lower = np.zeros_like(magnitude)
        upper = np.maximum(
            magnitude / (self.hardening_modulus + self.elastic_modulus),
            self.model_dt * magnitude ** (1.0 / self.dissipation_exponent),
        )
        for _ in range(self.bisection_iterations):
            middle = 0.5 * (lower + upper)
            residual = (
                (self.hardening_modulus + self.elastic_modulus) * middle
                + (middle / self.model_dt) ** self.dissipation_exponent
                - magnitude
            )
            lower = np.where(residual < 0.0, middle, lower)
            upper = np.where(residual < 0.0, upper, middle)
        return 0.5 * (lower + upper)

    def stress_history(self, strain_history: np.ndarray) -> np.ndarray:
        """Return stress at every time and node for a strain history."""
        strain = np.asarray(strain_history, dtype=float)
        if strain.ndim != 2 or strain.shape[1] != self.node_count:
            raise ValueError("strain_history must have shape (time, node_count).")
        if strain.shape[0] == 0:
            raise ValueError("strain_history must contain at least one time point.")

        plastic_strain = np.zeros(self.node_count, dtype=float)
        hardening_variable = np.zeros(self.node_count, dtype=float)
        stress = np.empty_like(strain)
        stress[0] = self.elastic_modulus * strain[0]

        for time_index in range(strain.shape[0] - 1):
            elastic_drive = self.elastic_modulus * (
                strain[time_index + 1] - plastic_strain
            )
            direction = np.sign(elastic_drive)
            driving_force = (
                np.abs(elastic_drive)
                - self.hardening_modulus * hardening_variable
            )
            increment_magnitude = self._plastic_increment(driving_force)
            plastic_increment = direction * increment_magnitude

            plastic_strain += plastic_increment
            hardening_variable += np.abs(plastic_increment)
            stress[time_index + 1] = self.elastic_modulus * (
                strain[time_index + 1] - plastic_strain
            )

        return stress

    def stress(self, strain_history: np.ndarray) -> np.ndarray:
        """Return the final stress state for a FEM residual."""
        return self.stress_history(strain_history)[-1]
