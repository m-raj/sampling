"""Train the neural variational ODE on the kinematic-hardening dataset.

Loads the train and validation splits written by
``generate_kinematic_hardening_dataset.py`` and, without any normalization,
trains Energy and Dissipation with the discrete adjoint gradients of
``VariationalODESolver.compute_adjoint_gradients``, as in
``kinematic_hardening_adjoint.ipynb``.

PyTorch Lightning handles the training loop, logging (CSV), periodic
checkpoints (last and best validation error), resuming (``--resume``),
and saving a checkpoint when training is interrupted (Ctrl-C, SIGTERM,
or an exception).

Every new training gets its own folder ``<runs-dir>/<timestamp>[_name]``:

    checkpoints/   last.ckpt, best-epoch=N.ckpt, interrupted.ckpt
    logs/          version_*/metrics.csv and hparams.yaml, one per (re)start
    plots/         training_curves.pdf, and test_kinematic_hardening.py figures

Resuming continues in the folder of the given checkpoint.
"""

from __future__ import annotations

import argparse
import glob
import os
import signal
from datetime import datetime

import lightning.pytorch as pl
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, OnExceptionCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from neural_variational_ode import Dissipation, Energy, VariationalODESolver
from utilities import apply_plot_style, relative_l2_error, save_pdf


class AdjointVariationalODE(pl.LightningModule):
    """Variational ODE trained with discrete adjoint gradients."""

    def __init__(
        self,
        dt: float,
        hidden_sizes: tuple[int, ...] = (64,),
        learning_rate: float = 1e-2,
        dataset_path: str | None = None,
        dataset_metadata: dict | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        # Adjoint gradients are assigned to ``.grad`` by hand.
        self.automatic_optimization = False
        self.solver = VariationalODESolver(
            Energy(e_dim=1, xi_dim=1, hidden_sizes=hidden_sizes),
            Dissipation(xi_dim=1, hidden_sizes=hidden_sizes),
        )

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        strain_batch, stress_batch = batch
        optimizer = self.optimizers()
        optimizer.zero_grad()
        loss, gradients, _, predicted_stress = self.solver.compute_adjoint_gradients(
            strain_batch, stress_batch, dt=self.hparams.dt
        )
        for parameter, gradient in zip(self.solver.parameters(), gradients):
            parameter.grad = gradient / stress_batch.numel()
        optimizer.step()

        batch_size = strain_batch.shape[0]
        self.log("train_loss", loss / stress_batch.numel(), batch_size=batch_size,
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_stress_rel_error",
                 relative_l2_error(stress_batch, predicted_stress),
                 batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        strain_batch, stress_batch = batch
        # The solver differentiates W and D, so it needs autograd even here.
        with torch.enable_grad():
            _, predicted_stress = self.solver.solve(
                strain_batch, dt=self.hparams.dt, create_graph=False
            )
        self.log("val_stress_rel_error",
                 relative_l2_error(stress_batch, predicted_stress),
                 batch_size=strain_batch.shape[0], prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.solver.parameters(), lr=self.hparams.learning_rate)


def split_loader(split: dict, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(split["strain"].to(torch.float32), split["stress"].to(torch.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_dir_from_checkpoint(checkpoint_path: str) -> str:
    """Run folder that holds ``checkpoint_path`` (``<run>/checkpoints/x.ckpt``)."""
    checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    if os.path.basename(checkpoint_dir) == "checkpoints":
        return os.path.dirname(checkpoint_dir)
    return checkpoint_dir


def new_run_dir(runs_dir: str, run_name: str | None) -> str:
    """Fresh, timestamped run folder such as ``runs/20261017-142501_name``."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(runs_dir, f"{stamp}_{run_name}" if run_name else stamp)
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


def plot_training_curves(run_dir: str) -> None:
    """Plot loss and relative errors against epoch from all logs of a run.

    Every (re)start of a run writes its own ``logs/version_*/metrics.csv``;
    they are concatenated, keeping the latest value logged for each epoch.
    """
    metric_files = sorted(glob.glob(os.path.join(run_dir, "logs", "version_*", "metrics.csv")))
    if not metric_files:
        return
    metrics = pd.concat([pd.read_csv(path) for path in metric_files], ignore_index=True)
    metrics = metrics.groupby("epoch").last()

    apply_plot_style()
    fig, (loss_axis, error_axis) = plt.subplots(1, 2, figsize=(11, 4))
    if "train_loss" in metrics:
        loss_axis.semilogy(metrics.index + 1, metrics["train_loss"], color="#2a78d6")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Training loss")
    loss_axis.grid(alpha=0.2)
    for column, color, linestyle, label in (
        ("train_stress_rel_error", "#2a78d6", "-", "train"),
        ("val_stress_rel_error", "#eb6834", "--", "validation"),
    ):
        if column in metrics:
            values = metrics[column].dropna()
            error_axis.semilogy(values.index + 1, values, color=color, linestyle=linestyle, label=label)
    error_axis.set_xlabel("Epoch")
    error_axis.set_ylabel(r"Relative $L^2$ stress error")
    error_axis.legend()
    error_axis.grid(alpha=0.2)
    for axis in (loss_axis, error_axis):
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))

    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    save_pdf(fig, os.path.join(plot_dir, "training_curves.pdf"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="datasets/kinematic_hardening_dataset.pt")
    parser.add_argument("--runs-dir", default="runs",
                        help="Every new training gets its own timestamped folder in here.")
    parser.add_argument("--run-name", default=None,
                        help="Optional suffix for the run folder name.")
    # The notebook used (64, 64), but with the current squared-ReLU
    # InputConvexNetwork the forward-Euler solve diverges within a few steps.
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[64])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=100,
                        help="Total number of epochs, including those of a resumed run.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=1,
                        help="Save last.ckpt every N epochs.")
    parser.add_argument("--val-every", type=int, default=1,
                        help="Evaluate the validation split every N epochs.")
    parser.add_argument("--accelerator", default="cpu")
    parser.add_argument("--resume", default=None,
                        help="Checkpoint to continue from, e.g. runs/<run>/checkpoints/last.ckpt "
                        "or interrupted.ckpt. Training continues in that run's folder.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed)

    # Lightning handles Ctrl-C and SIGTERM; treat a closed terminal the same way.
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.default_int_handler)

    run_dir = run_dir_from_checkpoint(args.resume) if args.resume else new_run_dir(args.runs_dir, args.run_name)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    print(f"Run folder: {os.path.abspath(run_dir)}")

    dataset = torch.load(args.dataset, weights_only=False)
    metadata = dataset["metadata"]
    train_loader = split_loader(dataset["train"], args.batch_size, shuffle=True)
    val_loader = split_loader(dataset["validation"], args.batch_size, shuffle=False)

    model = AdjointVariationalODE(
        dt=metadata["ou"]["dt"],
        hidden_sizes=tuple(args.hidden_sizes),
        learning_rate=args.lr,
        dataset_path=os.path.abspath(args.dataset),
        dataset_metadata=metadata,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best-{epoch}",
            monitor="val_stress_rel_error",
            save_last=True,
            every_n_epochs=args.checkpoint_every,
        ),
        OnExceptionCheckpoint(dirpath=checkpoint_dir, filename="interrupted"),
    ]
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        callbacks=callbacks,
        # One logs/version_* folder per (re)start, so resuming never overwrites logs.
        logger=CSVLogger(run_dir, name="logs"),
        check_val_every_n_epoch=args.val_every,
        # Validation needs autograd for the solver's derivatives.
        inference_mode=False,
        log_every_n_steps=1,
    )
    try:
        trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume)
    finally:
        # Also runs when training is interrupted.
        plot_training_curves(run_dir)
        print(f"Model, logs and plots saved in {os.path.abspath(run_dir)}")


if __name__ == "__main__":
    main()
