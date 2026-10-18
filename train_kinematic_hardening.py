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
import time
import warnings
from datetime import datetime

import lightning.pytorch as pl
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, NullFormatter
import pandas as pd
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, OnExceptionCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from src.neural_variational_invertible_ode import (
    InvertibleDissipation,
    InvertibleEnergy,
    InvertibleVariationalODESolver,
)
from src.neural_variational_ode import Dissipation, Energy, VariationalODESolver
from src.utilities import apply_plot_style, relative_l2_error, save_pdf

MODELS = ("potential", "invertible")


def build_solver(model: str, hidden_size: int):
    """Variational ODE solver of the requested type.

    Both models have a single hidden layer of width ``hidden_size``.

    ``potential``: input-convex potentials W and D, differentiated with
    autograd (``neural_variational_ode``).
    ``invertible``: direct invertible curl-free maps for the stress, driving
    force and xi_dot (``neural_variational_invertible_ode``).
    """

    if model == "potential":
        return VariationalODESolver(
            Energy(e_dim=1, xi_dim=1, hidden_sizes=(hidden_size,)),
            Dissipation(xi_dim=1, hidden_sizes=(hidden_size,)),
        )
    if model == "invertible":
        return InvertibleVariationalODESolver(
            InvertibleEnergy(e_dim=1, xi_dim=1, hidden_size=hidden_size),
            InvertibleDissipation(xi_dim=1, hidden_size=hidden_size),
        )
    raise ValueError(f"model must be one of {MODELS}, got {model!r}.")


class AdjointVariationalODE(pl.LightningModule):
    """Variational ODE trained with discrete adjoint gradients."""

    def __init__(
        self,
        dt: float,
        hidden_size: int = 64,
        learning_rate: float = 1e-2,
        dataset_path: str | None = None,
        dataset_metadata: dict | None = None,
        model: str = "potential",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        # Adjoint gradients are assigned to ``.grad`` by hand.
        self.automatic_optimization = False
        self.solver = build_solver(model, hidden_size)
        # Wall-clock training time accumulated by earlier runs of a resumed training.
        self.hours_before_start = 0.0
        self._clock_start: float | None = None

    def elapsed_hours(self) -> float:
        """Cumulative wall-clock training time in hours, across resumes."""
        if self._clock_start is None:
            return self.hours_before_start
        return self.hours_before_start + (time.perf_counter() - self._clock_start) / 3600.0

    def on_train_start(self) -> None:
        self._clock_start = time.perf_counter()

    def on_train_epoch_end(self) -> None:
        self.log("elapsed_hours", self.elapsed_hours(), on_step=False, on_epoch=True)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["elapsed_hours"] = self.elapsed_hours()

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        self.hours_before_start = checkpoint.get("elapsed_hours", 0.0)

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
    # The data are in-memory tensors: a full epoch loads in under a millisecond
    # without workers, while worker processes add ~0.7 s startup and make each
    # epoch slower, so loading in the main process (0 workers) is fastest.
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


class EpochProgressBar(Callback):
    """A progress bar over all epochs, with the latest metrics on the line below."""

    METRICS = {
        "train_loss": "loss",
        "train_stress_rel_error": "train err",
        "val_stress_rel_error": "val err",
    }

    def on_train_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        # A resumed run starts part-way through the total number of epochs.
        self.bar = tqdm(
            total=trainer.max_epochs,
            initial=trainer.current_epoch,
            desc="Training",
            unit="epoch",
            position=0,
            dynamic_ncols=True,
        )
        # A bar-less tqdm line that only displays text, kept below the bar.
        self.status = tqdm(total=0, bar_format="{desc}", position=1)

    def _show_metrics(self, trainer: pl.Trainer) -> None:
        metrics = trainer.callback_metrics
        self.status.set_description_str(
            ", ".join(
                f"{label} = {float(metrics[name]):.3e}"
                for name, label in self.METRICS.items()
                if name in metrics
            )
        )

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        # Validation runs only every --val-every epochs; show it when it does.
        if hasattr(self, "bar") and not trainer.sanity_checking:
            self._show_metrics(trainer)

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._show_metrics(trainer)
        self.bar.update(1)
        # Close before Lightning prints its "max_epochs reached" message, which
        # would otherwise land between (and duplicate) the two lines.
        if self.bar.n >= self.bar.total:
            self._close()

    def _close(self) -> None:
        for line in ("bar", "status"):
            if hasattr(self, line):
                getattr(self, line).close()
                delattr(self, line)

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._close()

    def on_exception(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        exception: BaseException,
    ) -> None:
        self._close()


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


def log_interp(x, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """Interpolate (and linearly extrapolate) in log-log space."""
    x = np.asarray(x, dtype=float)
    log_x = np.log(np.clip(x, 1e-300, None))
    log_xp, log_fp = np.log(xp), np.log(fp)
    log_y = np.interp(log_x, log_xp, log_fp)
    if len(xp) > 1:
        left, right = log_x < log_xp[0], log_x > log_xp[-1]
        left_slope = (log_fp[1] - log_fp[0]) / (log_xp[1] - log_xp[0])
        right_slope = (log_fp[-1] - log_fp[-2]) / (log_xp[-1] - log_xp[-2])
        log_y = np.where(left, log_fp[0] + (log_x - log_xp[0]) * left_slope, log_y)
        log_y = np.where(right, log_fp[-1] + (log_x - log_xp[-1]) * right_slope, log_y)
    return np.exp(log_y)


def scientific_y_axis(axis: plt.Axes) -> None:
    """Scientific y tick labels with one common exponent shown above the axis.

    Each tick is labelled by its value divided by 10**exponent, where the
    exponent is that of the largest visible value; call after the y limits
    are final.
    """
    exponent = int(np.floor(np.log10(axis.get_ylim()[1])))
    scale = 10.0**exponent
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / scale:g}"))
    axis.yaxis.set_minor_formatter(NullFormatter())
    # Left of the frame's top corner, clear of the top x-axis tick labels.
    axis.text(-0.035, 1.01, rf"$\times 10^{{{exponent}}}$", transform=axis.transAxes,
              ha="right", va="bottom")


def plot_training_curves(run_dir: str) -> None:
    """Plot loss and relative errors against training time, log-log.

    The bottom axis is cumulative wall-clock training time in hours and the
    top axis the corresponding epoch. Every (re)start of a run writes its own
    ``logs/version_*/metrics.csv``; they are concatenated, keeping the latest
    value logged for each epoch.
    """
    metric_files = sorted(glob.glob(os.path.join(run_dir, "logs", "version_*", "metrics.csv")))
    if not metric_files:
        return
    metrics = pd.concat([pd.read_csv(path) for path in metric_files], ignore_index=True)
    metrics = metrics.groupby("epoch").last()
    epochs = metrics.index.to_numpy() + 1.0
    # Runs logged before training time was recorded fall back to epochs.
    has_time = "elapsed_hours" in metrics and metrics["elapsed_hours"].notna().all()
    x_values = metrics["elapsed_hours"].to_numpy() if has_time else epochs

    apply_plot_style()
    fig, (loss_axis, error_axis) = plt.subplots(1, 2, figsize=(11, 5.5))
    if "train_loss" in metrics:
        loss_axis.loglog(x_values, metrics["train_loss"], color="#2a78d6")
    loss_axis.set_ylabel("Training loss")
    for column, color, linestyle, label in (
        ("train_stress_rel_error", "#2a78d6", "-", "train"),
        ("val_stress_rel_error", "#eb6834", "--", "validation"),
    ):
        if column in metrics:
            valid = metrics[column].notna().to_numpy()
            error_axis.loglog(
                x_values[valid], metrics[column].to_numpy()[valid],
                color=color, linestyle=linestyle, label=label,
                marker="o" if valid.sum() < 10 else None,
            )
    error_axis.set_ylabel(r"Relative $L^2$ stress error")
    error_axis.legend()

    for axis in (loss_axis, error_axis):
        axis.grid(alpha=0.2, which="both")
        axis.set_xlim(x_values.min(), x_values.max())
        scientific_y_axis(axis)
        if has_time:
            axis.set_xlabel("Training time (h)")
            top = axis.secondary_xaxis(
                "top",
                functions=(
                    lambda hours: log_interp(hours, x_values, epochs),
                    lambda epoch: log_interp(epoch, epochs, x_values),
                ),
            )
            top.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
            top.set_xlabel("Epoch")
        else:
            axis.set_xlabel("Epoch")

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
    parser.add_argument(
        "--model",
        choices=MODELS,
        default="invertible",
        help="potential: input-convex W and D (neural_variational_ode); "
        "invertible: invertible curl-free maps (neural_variational_invertible_ode).",
    )
    parser.add_argument(
        "--hidden-size", type=int, default=64, help="Width of the single hidden layer."
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=100,
                        help="Total number of epochs, including those of a resumed run.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Save last.ckpt every N epochs.",
    )
    parser.add_argument(
        "--val-every",
        type=int,
        default=100,
        help="Evaluate the validation split every N epochs.",
    )
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

    model_type, hidden_size = args.model, args.hidden_size
    if args.resume:
        # A resumed run keeps the architecture it was trained with.
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)["hyper_parameters"]
        model_type = saved.get("model", "potential")
        hidden_size = saved["hidden_size"]
    model = AdjointVariationalODE(
        dt=metadata["ou"]["dt"],
        hidden_size=hidden_size,
        learning_rate=args.lr,
        dataset_path=os.path.abspath(args.dataset),
        dataset_metadata=metadata,
        model=model_type,
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
        EpochProgressBar(),
    ]
    # Workers are deliberately 0 (see split_loader); silence Lightning's generic advice.
    warnings.filterwarnings("ignore", message=".*does not have many workers.*")
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
        # Replaced by EpochProgressBar: one bar for all epochs instead of one per epoch.
        enable_progress_bar=False,
    )
    try:
        trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume)
        # Periodic checkpoints only happen every --checkpoint-every epochs, so
        # always save the final state as well.
        trainer.save_checkpoint(os.path.join(checkpoint_dir, "last.ckpt"))
    finally:
        # Also runs when training is interrupted.
        plot_training_curves(run_dir)
        print(f"Model, logs and plots saved in {os.path.abspath(run_dir)}")


if __name__ == "__main__":
    main()
