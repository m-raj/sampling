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
"""

from __future__ import annotations

import argparse
import os
import signal

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, OnExceptionCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from neural_variational_ode import Dissipation, Energy, VariationalODESolver
from utilities import relative_l2_error


class AdjointVariationalODE(pl.LightningModule):
    """Variational ODE trained with discrete adjoint gradients."""

    def __init__(
        self,
        dt: float,
        hidden_sizes: tuple[int, ...] = (64,),
        learning_rate: float = 1e-2,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="kinematic_hardening_dataset.pt")
    parser.add_argument("--output-dir", default="runs/kinematic_hardening",
                        help="Directory for checkpoints and CSV logs.")
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
                        help="Checkpoint to continue from, e.g. runs/kinematic_hardening/last.ckpt "
                        "or interrupted.ckpt.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed)

    # Lightning handles Ctrl-C and SIGTERM; treat a closed terminal the same way.
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, signal.default_int_handler)

    dataset = torch.load(args.dataset, weights_only=False)
    metadata = dataset["metadata"]
    train_loader = split_loader(dataset["train"], args.batch_size, shuffle=True)
    val_loader = split_loader(dataset["validation"], args.batch_size, shuffle=False)

    model = AdjointVariationalODE(
        dt=metadata["ou"]["dt"],
        hidden_sizes=tuple(args.hidden_sizes),
        learning_rate=args.lr,
        dataset_metadata=metadata,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=args.output_dir,
            filename="best-{epoch}",
            monitor="val_stress_rel_error",
            save_last=True,
            every_n_epochs=args.checkpoint_every,
        ),
        OnExceptionCheckpoint(dirpath=args.output_dir, filename="interrupted"),
    ]
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        callbacks=callbacks,
        logger=CSVLogger(args.output_dir, name="logs"),
        check_val_every_n_epoch=args.val_every,
        # Validation needs autograd for the solver's derivatives.
        inference_mode=False,
        log_every_n_steps=1,
    )
    trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume)
    print(f"Checkpoints and logs saved in {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
