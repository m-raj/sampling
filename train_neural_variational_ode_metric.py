"""Train a vector-valued variational ODE with a shared normalization matrix.

The supplied matrix is used to normalize both strain and stress.  The neural
variational model is trained in these normalized coordinates with ordinary
Euclidean derivatives; no normalization-induced metric operator is used.

The dataset must contain ``e_trajectories``, ``xi_target``, and
``sigma_target``.  Trajectories have shape ``(batch, time, dimension)``.
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

from neural_variational_ode import Dissipation, Energy, VariationalODESolver


def load_matrix(path: str | Path) -> Tensor:
    """Load a square normalization matrix from ``.pt`` or ``.npy``."""
    matrix_path = Path(path)
    if matrix_path.suffix == ".npy":
        value = np.load(matrix_path)
    else:
        value = torch.load(matrix_path, map_location="cpu", weights_only=True)
    matrix = torch.as_tensor(value, dtype=torch.float32)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("The normalization matrix must be square.")
    if torch.linalg.matrix_rank(matrix) != matrix.shape[0]:
        raise ValueError("The normalization matrix must be nonsingular.")
    return matrix


class SharedMetricNormalization(nn.Module):
    """Normalize stress and strain with one user-supplied matrix."""

    def __init__(
        self,
        matrix: Tensor,
        strain_reference: Tensor | None = None,
        stress_reference: Tensor | None = None,
    ) -> None:
        super().__init__()
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix must be square.")
        dimension = matrix.shape[0]
        self.register_buffer("matrix", matrix)
        self.register_buffer(
            "strain_reference",
            torch.zeros(dimension) if strain_reference is None else strain_reference,
        )
        self.register_buffer(
            "stress_reference",
            torch.zeros(dimension) if stress_reference is None else stress_reference,
        )

    def normalize_strain(self, strain: Tensor) -> Tensor:
        return torch.linalg.solve(
            self.matrix, (strain - self.strain_reference).transpose(-1, -2)
        ).transpose(-1, -2)

    def normalize_stress(self, stress: Tensor) -> Tensor:
        return torch.linalg.solve(
            self.matrix, (stress - self.stress_reference).transpose(-1, -2)
        ).transpose(-1, -2)

    def denormalize_stress(self, normalized_stress: Tensor) -> Tensor:
        return normalized_stress @ self.matrix.T + self.stress_reference


class NormalizedVariationalODE(pl.LightningModule):
    """Train and evaluate the variational ODE in shared normalized coordinates."""

    def __init__(
        self,
        normalization_matrix: Tensor,
        *,
        dt: float = 0.01,
        learning_rate: float = 1e-3,
        e_dim: int | None = None,
        xi_dim: int | None = None,
    ) -> None:
        super().__init__()
        dimension = normalization_matrix.shape[0]
        e_dim = dimension if e_dim is None else e_dim
        xi_dim = dimension if xi_dim is None else xi_dim
        if e_dim != dimension:
            raise ValueError("e_dim must match the normalization matrix dimension.")
        if xi_dim != dimension:
            raise ValueError(
                "This script uses the shared matrix for vector-valued xi, "
                "so xi_dim must match its dimension."
            )
        self.save_hyperparameters(ignore=["normalization_matrix"])
        self.normalization = SharedMetricNormalization(normalization_matrix)
        self.solver = VariationalODESolver(
            Energy(e_dim=e_dim, xi_dim=xi_dim),
            Dissipation(xi_dim=xi_dim),
        )
        self.loss = nn.MSELoss()

    def forward(self, strain: Tensor) -> tuple[Tensor, Tensor]:
        return self.solver.solve(
            self.normalization.normalize_strain(strain),
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
    parser.add_argument("--normalization-matrix", required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dt", type=float, default=1 / 500)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--run-id", default="shared_metric")
    args = parser.parse_args()

    dataset = torch.load(args.dataset_path, map_location="cpu", weights_only=True)
    strain = torch.as_tensor(dataset["e_trajectories"], dtype=torch.float32)
    xi = torch.as_tensor(dataset["xi_target"], dtype=torch.float32)
    stress = torch.as_tensor(dataset["sigma_target"], dtype=torch.float32)
    if strain.ndim != 3 or strain.shape[-1] != xi.shape[-1] or xi.shape != stress.shape:
        raise ValueError(
            "e_trajectories, xi_target, and sigma_target must have matching "
            "(batch, time, dimension) shapes."
        )

    normalization_matrix = load_matrix(args.normalization_matrix)
    if normalization_matrix.shape[0] != strain.shape[-1]:
        raise ValueError("The normalization matrix dimension must match the data.")
    loader = DataLoader(
        TensorDataset(strain, xi, stress),
        batch_size=args.batch_size,
        shuffle=True,
    )
    model = NormalizedVariationalODE(
        normalization_matrix,
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
