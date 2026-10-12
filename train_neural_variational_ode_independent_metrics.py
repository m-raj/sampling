"""Train a vector-valued variational ODE with independent stress/strain scales.

The strain and stress normalization matrices are independent.  Their induced
operator G = E_*^T S_* / w_* converts the normalized energy gradient into
normalized stress and is also used for the normalized thermodynamic driving
force.  The dataset must contain ``e_trajectories``, ``xi_target``, and
``sigma_target`` with shape ``(batch, time, dimension)``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.plugins.environments import LightningEnvironment
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from neural_variational_ode import Dissipation, Energy


def load_matrix(path: str | Path) -> Tensor:
    """Load a square matrix from ``.pt`` or ``.npy``."""
    matrix_path = Path(path)
    if matrix_path.suffix == ".npy":
        value = np.load(matrix_path)
    else:
        value = torch.load(matrix_path, map_location="cpu", weights_only=True)
    matrix = torch.as_tensor(value, dtype=torch.float32)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("A normalization matrix must be square.")
    if torch.linalg.matrix_rank(matrix) != matrix.shape[0]:
        raise ValueError("A normalization matrix must be nonsingular.")
    return matrix


class IndependentNormalization(nn.Module):
    """Normalize strain and stress with separate matrices."""

    def __init__(
        self,
        strain_matrix: Tensor,
        stress_matrix: Tensor,
        energy_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if strain_matrix.shape != stress_matrix.shape:
            raise ValueError("Strain and stress matrices must have equal shapes.")
        if energy_scale <= 0:
            raise ValueError("energy_scale must be positive.")
        self.register_buffer("strain_matrix", strain_matrix)
        self.register_buffer("stress_matrix", stress_matrix)
        self.energy_scale = energy_scale

    def normalize_strain(self, strain: Tensor) -> Tensor:
        return torch.linalg.solve(
            self.strain_matrix, strain.transpose(-1, -2)
        ).transpose(-1, -2)

    def normalize_stress(self, stress: Tensor) -> Tensor:
        return torch.linalg.solve(
            self.stress_matrix, stress.transpose(-1, -2)
        ).transpose(-1, -2)

    def denormalize_stress(self, normalized_stress: Tensor) -> Tensor:
        return normalized_stress @ self.stress_matrix.T

    @property
    def induced_metric(self) -> Tensor:
        return self.strain_matrix.T @ self.stress_matrix / self.energy_scale


class MetricVariationalODESolver(nn.Module):
    """Solve the variational ODE using the normalization-induced metric."""

    def __init__(
        self,
        energy: Energy,
        dissipation: Dissipation,
        metric: Tensor,
    ) -> None:
        super().__init__()
        if energy.e_dim != energy.xi_dim or energy.xi_dim != dissipation.xi_dim:
            raise ValueError("Metric solver requires equal strain and xi dimensions.")
        if metric.shape != (energy.e_dim, energy.e_dim):
            raise ValueError("metric has incompatible dimensions.")
        self.energy = energy
        self.dissipation = dissipation
        self.register_buffer("metric", metric)

    @staticmethod
    def _differentiable_input(value: Tensor) -> Tensor:
        return value if value.requires_grad else value.clone().requires_grad_(True)

    def solve(
        self,
        normalized_strain: Tensor,
        dt: float,
        *,
        create_graph: bool,
    ) -> tuple[Tensor, Tensor]:
        if normalized_strain.ndim != 3:
            raise ValueError(
                "normalized_strain must have shape (batch, time, dimension)."
            )
        if dt <= 0:
            raise ValueError("dt must be positive.")
        batch_size, n_times, _ = normalized_strain.shape
        xi = normalized_strain.new_zeros(batch_size, self.energy.xi_dim)
        xi_paths = [xi]
        stress_paths: list[Tensor] = []

        for time_index in range(n_times - 1):
            strain_t = self._differentiable_input(
                normalized_strain[:, time_index]
            )
            xi_t = self._differentiable_input(xi)
            energy_gradient, xi_gradient = self.energy.derivatives(
                strain_t, xi_t, create_graph=create_graph
            )
            normalized_stress = torch.linalg.solve(
                self.metric, energy_gradient.T
            ).T
            driving_force = torch.linalg.solve(
                self.metric, xi_gradient.T
            ).T
            xi_dot = self.dissipation.derivative(
                self._differentiable_input(-driving_force),
                create_graph=create_graph,
            )
            stress_paths.append(normalized_stress)
            xi = xi_t + dt * xi_dot
            xi_paths.append(xi)

        strain_final = self._differentiable_input(normalized_strain[:, -1])
        xi_final = self._differentiable_input(xi)
        energy_gradient, _ = self.energy.derivatives(
            strain_final, xi_final, create_graph=create_graph
        )
        stress_paths.append(torch.linalg.solve(self.metric, energy_gradient.T).T)
        return torch.stack(xi_paths, dim=1), torch.stack(stress_paths, dim=1)


class IndependentMetricVariationalODE(pl.LightningModule):
    """Train and return physical stress with independent normalization matrices."""

    def __init__(
        self,
        strain_matrix: Tensor,
        stress_matrix: Tensor,
        *,
        energy_scale: float = 1.0,
        dt: float = 0.01,
        learning_rate: float = 1e-3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["strain_matrix", "stress_matrix"])
        self.normalization = IndependentNormalization(
            strain_matrix, stress_matrix, energy_scale=energy_scale
        )
        dimension = strain_matrix.shape[0]
        self.solver = MetricVariationalODESolver(
            Energy(e_dim=dimension, xi_dim=dimension),
            Dissipation(xi_dim=dimension),
            self.normalization.induced_metric,
        )
        self.loss = nn.MSELoss()

    def forward(self, strain: Tensor) -> tuple[Tensor, Tensor]:
        normalized_strain = self.normalization.normalize_strain(strain)
        return self.solver.solve(
            normalized_strain,
            dt=self.hparams.dt,
            create_graph=self.training,
        )

    def predicted_stress(self, strain: Tensor) -> Tensor:
        _, normalized_stress = self(strain)
        return self.normalization.denormalize_stress(normalized_stress)

    def training_step(
        self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        strain, _, stress = batch
        _, normalized_prediction = self(strain)
        normalized_target = self.normalization.normalize_stress(stress)
        loss = self.loss(normalized_prediction, normalized_target)
        self.log("loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", default="training_dataset.pt")
    parser.add_argument("--strain-matrix", required=True)
    parser.add_argument("--stress-matrix", required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dt", type=float, default=1 / 500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--energy-scale", type=float, default=1.0)
    parser.add_argument("--run-id", default="independent_metrics")
    args = parser.parse_args()

    strain_matrix = load_matrix(args.strain_matrix)
    stress_matrix = load_matrix(args.stress_matrix)
    dataset = torch.load(args.dataset_path, map_location="cpu", weights_only=True)
    strain = torch.as_tensor(dataset["e_trajectories"], dtype=torch.float32)
    xi = torch.as_tensor(dataset["xi_target"], dtype=torch.float32)
    stress = torch.as_tensor(dataset["sigma_target"], dtype=torch.float32)
    if strain.ndim != 3 or strain.shape != xi.shape or xi.shape != stress.shape:
        raise ValueError(
            "e_trajectories, xi_target, and sigma_target must have matching "
            "(batch, time, dimension) shapes."
        )
    if strain_matrix.shape[0] != strain.shape[-1]:
        raise ValueError("Normalization matrix dimensions must match the data.")

    loader = DataLoader(
        TensorDataset(strain, xi, stress),
        batch_size=args.batch_size,
        shuffle=True,
    )
    model = IndependentMetricVariationalODE(
        strain_matrix,
        stress_matrix,
        energy_scale=args.energy_scale,
        dt=args.dt,
        learning_rate=args.learning_rate,
    )
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="cpu",
        devices=1,
        plugins=[LightningEnvironment()],
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, train_dataloaders=loader)
    torch.save(model.state_dict(), f"neural_variational_ode_{args.run_id}.pt")


if __name__ == "__main__":
    main()
