"""Pretrain potential and invertible networks on a quadratic reference.

``ICNNPretrainer`` fits an ``InputConvexNetwork`` to the quadratic potential

    psi(x) = 1/2 x^T x,

and ``CurlFreeINNPretrainer`` fits an ``InvertibleCurlFreeNetwork`` to its
gradient, the identity map x -> x. Both generate their own synthetic data,
sampled uniformly from the ball of radius ``radius`` (default 1) centred at
the origin, with the dimension of the network input.

Usage:

    python -m src.pretraining --model icnn --input-dim 1 --output icnn.pt
    python -m src.pretraining --model invertible --input-dim 3 --radius 2
"""

from __future__ import annotations

import argparse
import logging

import torch
from torch import Tensor, nn
from tqdm.contrib.logging import logging_redirect_tqdm

from src.logging_utils import progress, setup_logging
from src.neural_variational_invertible_ode import InvertibleCurlFreeNetwork
from src.neural_variational_ode import InputConvexNetwork

logger = logging.getLogger(__name__)


def sample_ball(
    n: int,
    dim: int,
    radius: float = 1.0,
    *,
    generator: torch.Generator | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return ``n`` points uniformly distributed in the ``dim``-ball of ``radius``."""
    if radius <= 0:
        raise ValueError("radius must be positive.")
    direction = torch.randn(n, dim, generator=generator, dtype=dtype, device=device)
    direction = direction / direction.norm(dim=-1, keepdim=True)
    u = torch.rand(n, 1, generator=generator, dtype=dtype, device=device)
    return radius * u ** (1.0 / dim) * direction


class _BallPretrainer:
    """Fit ``network`` to ``target`` on uniform samples from a ball."""

    def __init__(
        self,
        network: nn.Module,
        radius: float = 1.0,
        *,
        n_train: int = 4096,
        n_val: int = 1024,
        batch_size: int = 256,
        epochs: int = 500,
        lr: float = 1e-2,
        seed: int = 0,
        log_every: int = 50,
    ) -> None:
        if radius <= 0:
            raise ValueError("radius must be positive.")
        self.network = network
        self.radius = radius
        self.n_train = n_train
        self.n_val = n_val
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.log_every = log_every
        self.train_data: tuple[Tensor, Tensor] | None = None
        self.val_data: tuple[Tensor, Tensor] | None = None

    @property
    def input_dim(self) -> int:
        raise NotImplementedError

    def target(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def generate_data(self) -> None:
        """Sample the training and validation sets with their targets."""
        parameter = next(self.network.parameters())
        generator = torch.Generator().manual_seed(self.seed)
        x = sample_ball(
            self.n_train + self.n_val, self.input_dim, self.radius,
            generator=generator, dtype=parameter.dtype,
        ).to(parameter.device)
        y = self.target(x)
        self.train_data = (x[: self.n_train], y[: self.n_train])
        self.val_data = (x[self.n_train :], y[self.n_train :])

    @torch.no_grad()
    def evaluate(self) -> float:
        """Relative L2 error on the validation set."""
        if self.val_data is None:
            self.generate_data()
        x, y = self.val_data
        return ((self.network(x) - y).norm() / y.norm()).item()

    def fit(self) -> list[float]:
        """Train the network in place and return the per-epoch training MSE."""
        if self.train_data is None:
            self.generate_data()
        x, y = self.train_data
        optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        generator = torch.Generator().manual_seed(self.seed + 1)
        history: list[float] = []

        self.network.train()
        # Print log lines above the progress bar instead of breaking it.
        with logging_redirect_tqdm():
            for epoch in progress(range(1, self.epochs + 1), desc="Pretraining", unit="epoch"):
                permutation = torch.randperm(self.n_train, generator=generator).to(x.device)
                epoch_loss = 0.0
                for start in range(0, self.n_train, self.batch_size):
                    index = permutation[start : start + self.batch_size]
                    loss = nn.functional.mse_loss(self.network(x[index]), y[index])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item() * len(index)
                history.append(epoch_loss / self.n_train)
                if epoch % self.log_every == 0 or epoch == self.epochs:
                    logger.info(
                        "epoch %d: train MSE %.3e, val relative error %.3e",
                        epoch, history[-1], self.evaluate(),
                    )
        self.network.eval()
        return history


class ICNNPretrainer(_BallPretrainer):
    """Pretrain an ``InputConvexNetwork`` on the potential 1/2 x^T x."""

    def __init__(self, network: InputConvexNetwork, radius: float = 1.0, **kwargs) -> None:
        super().__init__(network, radius, **kwargs)

    @property
    def input_dim(self) -> int:
        return self.network.input_layers[0].in_features

    def target(self, x: Tensor) -> Tensor:
        return 0.5 * (x * x).sum(-1)


class CurlFreeINNPretrainer(_BallPretrainer):
    """Pretrain an ``InvertibleCurlFreeNetwork`` on the identity map x -> x."""

    def __init__(
        self, network: InvertibleCurlFreeNetwork, radius: float = 1.0, **kwargs
    ) -> None:
        super().__init__(network, radius, **kwargs)

    @property
    def input_dim(self) -> int:
        return self.network.W.shape[0]

    def target(self, x: Tensor) -> Tensor:
        return x.clone()


def pretrain_solver(solver: nn.Module, radius: float = 1.0, **kwargs) -> dict[str, float]:
    """Pretrain the elastic, inelastic and dissipation networks of a solver.

    Works for both ``VariationalODESolver`` (input-convex networks, fitted to
    1/2 x^T x) and ``InvertibleVariationalODESolver`` (curl-free maps, fitted
    to the identity). ``kwargs`` are passed to each pretrainer. Returns the
    validation relative error of each network.
    """
    networks = {
        "elastic": solver.energy.elastic_network,
        "inelastic": solver.energy.inelastic_network,
        "dissipation": solver.dissipation.network,
    }
    errors: dict[str, float] = {}
    for name, network in networks.items():
        if isinstance(network, InputConvexNetwork):
            pretrainer = ICNNPretrainer(network, radius, **kwargs)
        elif isinstance(network, InvertibleCurlFreeNetwork):
            pretrainer = CurlFreeINNPretrainer(network, radius, **kwargs)
        else:
            raise TypeError(f"Cannot pretrain {name} network of type {type(network).__name__}.")
        logger.info("Pretraining %s network (%s)", name, type(network).__name__)
        pretrainer.fit()
        errors[name] = pretrainer.evaluate()
        logger.info("Pretrained %s network: val relative error %.3e", name, errors[name])
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=("icnn", "invertible"), required=True)
    parser.add_argument("--input-dim", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-train", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None,
                        help="Path to save the pretrained state_dict (.pt).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    torch.manual_seed(args.seed)

    kwargs = dict(
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        n_train=args.n_train, seed=args.seed,
    )
    if args.model == "icnn":
        network = InputConvexNetwork(args.input_dim, hidden_sizes=(args.hidden_size,))
        pretrainer = ICNNPretrainer(network, args.radius, **kwargs)
    else:
        network = InvertibleCurlFreeNetwork(args.input_dim, args.hidden_size)
        pretrainer = CurlFreeINNPretrainer(network, args.radius, **kwargs)

    pretrainer.fit()
    logger.info("Final validation relative error: %.3e", pretrainer.evaluate())
    if args.output is not None:
        torch.save(network.state_dict(), args.output)
        logger.info("Saved pretrained weights to %s", args.output)


if __name__ == "__main__":
    main()
