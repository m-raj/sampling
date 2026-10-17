"""Train the neural variational ODE on the kinematic-hardening dataset.

Loads the dataset written by ``generate_kinematic_hardening_dataset.py``,
applies the stress/strain/time normalization of
``kinematic_hardening_normalized.ipynb``, and trains Energy and
Dissipation by backpropagating through ``VariationalODESolver.solve``.
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from neural_variational_ode import Dissipation, Energy, VariationalODESolver
from utilities import relative_l2_error


def normalization_scales(strain: torch.Tensor, E: float, T: float) -> dict:
    """Dependent stress/strain scales, sigma_* = E epsilon_*, and time scale T."""
    strain_scale = strain.abs().amax().clamp_min(torch.finfo(strain.dtype).eps)
    return {
        "strain_scale": float(strain_scale),
        "stress_scale": float(E * strain_scale),
        "time_scale": float(T),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="kinematic_hardening_dataset.pt")
    parser.add_argument("--output", default="kinematic_hardening_model.pt")
    # The notebook used (64, 64), but with the current squared-ReLU
    # InputConvexNetwork the forward-Euler solve diverges within a few steps.
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[64])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    dataset = torch.load(args.dataset, weights_only=False)
    metadata = dataset["metadata"]
    dt = metadata["ou"]["dt"]
    times = dataset["times"].numpy()
    strain = dataset["strain"].to(torch.float32)
    stress = dataset["stress"].to(torch.float32)

    scales = normalization_scales(
        strain, E=metadata["constitutive_model"]["E"], T=metadata["ou"]["T"]
    )
    normalized_strain = strain / scales["strain_scale"]
    normalized_stress = stress / scales["stress_scale"]
    normalized_dt = dt / scales["time_scale"]

    training_dataset = TensorDataset(normalized_strain, normalized_stress)
    train_loader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True)

    energy_model = Energy(e_dim=1, xi_dim=1, hidden_sizes=tuple(args.hidden_sizes))
    dissipation_model = Dissipation(xi_dim=1, hidden_sizes=tuple(args.hidden_sizes))
    variational_solver = VariationalODESolver(energy_model, dissipation_model)
    optimizer = torch.optim.Adam(variational_solver.parameters(), lr=args.lr)
    loss_function = torch.nn.MSELoss()

    training_losses = []
    stress_relative_errors = []
    for epoch in tqdm(range(args.epochs), desc="Training", unit="epoch"):
        epoch_loss = 0.0
        epoch_stress_error = 0.0
        for strain_batch, stress_batch in train_loader:
            optimizer.zero_grad()
            _, predicted_stress = variational_solver.solve(
                strain_batch,
                dt=normalized_dt,
                create_graph=True,
            )
            loss = loss_function(predicted_stress, stress_batch)
            loss.backward()
            optimizer.step()
            batch_size = strain_batch.shape[0]
            epoch_loss += loss.detach().item() * batch_size
            epoch_stress_error += (
                relative_l2_error(stress_batch, predicted_stress, times=times) * batch_size
            )
        epoch_loss /= len(training_dataset)
        epoch_stress_error /= len(training_dataset)
        training_losses.append(epoch_loss)
        stress_relative_errors.append(epoch_stress_error)
        tqdm.write(
            f"Epoch {epoch + 1:3d}/{args.epochs}: loss = {epoch_loss:.6e}, "
            f"stress rel. error = {epoch_stress_error:.4e}"
        )

    torch.save(
        {
            "energy_state_dict": energy_model.state_dict(),
            "dissipation_state_dict": dissipation_model.state_dict(),
            "hidden_sizes": list(args.hidden_sizes),
            "scales": scales,
            "normalized_dt": normalized_dt,
            "training_losses": training_losses,
            "stress_relative_errors": stress_relative_errors,
            "train_args": vars(args),
            "dataset_metadata": metadata,
        },
        args.output,
    )
    print(f"Saved trained model to {args.output}")


if __name__ == "__main__":
    main()
