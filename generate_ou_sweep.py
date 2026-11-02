"""Generate kinematic-hardening datasets over a grid of OU parameters.

Builds one dataset (see ``generate_kinematic_hardening_dataset.py``) for every
pair of ``theta`` and ``sigma`` values. Each grid is either log-spaced
(``--theta-range`` / ``--sigma-range``) or listed explicitly (``--thetas`` /
``--sigmas``). All other options keep the single-dataset defaults and can be
overridden on the command line, e.g.

    python generate_ou_sweep.py --thetas 1 2 3 4 5 --sigma-range 0.5 2 5 \\
        --n-trajectories 128 --T 2.0

Every dataset uses the same ``--seed`` and ``--val-seed``, so they share the
same underlying noise and differ only through theta and sigma.

The smoothing window scales with theta (``--window-fraction`` times 1/theta),
so every dataset is smoothed by the same relative amount.

Files are written to ``--output-dir`` as ``theta<θ>_sigma<σ>.pt``, together
with ``index.json`` (one entry per dataset) and ``sweep.log``.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import logging
import os

import numpy as np

from generate_kinematic_hardening_dataset import (
    build_dataset,
    build_parser,
    mollifier_window,
    save_dataset,
)
from src.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> tuple[argparse.Namespace, argparse.Namespace]:
    """Return (sweep options, base dataset options)."""
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        # Otherwise --theta would be read as an abbreviation of --theta-range.
        allow_abbrev=False,
        epilog="Any other option of generate_kinematic_hardening_dataset.py "
        "(except --theta, --sigma, --output) is passed through.",
    )
    for name, default in (("theta", [1.0, 10.0, 4]), ("sigma", [0.5, 2.0, 4])):
        group = parser.add_mutually_exclusive_group()
        group.add_argument(f"--{name}-range", type=float, nargs=3, default=default,
                           metavar=("MIN", "MAX", "N"),
                           help=f"N log-spaced {name} values from MIN to MAX "
                           f"(default: {default[0]:g} {default[1]:g} {default[2]:g}).")
        group.add_argument(f"--{name}s", type=float, nargs="+", default=None, metavar="V",
                           help=f"Explicit {name} values, instead of --{name}-range.")
    parser.add_argument("--output-dir", default="datasets/ou_sweep")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Do not regenerate datasets whose file already exists.")
    sweep_args, rest = parser.parse_known_args()

    for option in ("--theta", "--sigma", "--output"):
        if any(arg == option or arg.startswith(option + "=") for arg in rest):
            parser.error(f"{option} is set by the sweep; use --theta-range, "
                         "--sigma-range or --output-dir instead.")
    base_args = build_parser().parse_args(rest)
    return sweep_args, base_args


def log_grid(value_range: list[float], name: str) -> np.ndarray:
    low, high, count = value_range
    if low <= 0 or high <= 0:
        raise ValueError(f"--{name}-range bounds must be positive for log spacing.")
    if count < 1 or count != int(count):
        raise ValueError(f"--{name}-range N must be a positive integer.")
    return np.geomspace(low, high, int(count))


def value_grid(sweep_args: argparse.Namespace, name: str) -> np.ndarray:
    """Explicit values (``--thetas``) if given, otherwise the log-spaced range."""
    values = getattr(sweep_args, f"{name}s")
    if values is None:
        return log_grid(getattr(sweep_args, f"{name}_range"), name)
    if min(values) <= 0:
        raise ValueError(f"--{name}s values must be positive.")
    return np.unique(values)


def main() -> None:
    sweep_args, base_args = parse_args()
    setup_logging(os.path.join(sweep_args.output_dir, "sweep.log"))

    thetas = value_grid(sweep_args, "theta")
    sigmas = value_grid(sweep_args, "sigma")
    logger.info("theta values: %s", np.round(thetas, 4).tolist())
    logger.info("sigma values: %s", np.round(sigmas, 4).tolist())

    entries = []
    combinations = list(itertools.product(thetas, sigmas))
    for index, (theta, sigma) in enumerate(combinations, start=1):
        file_name = f"theta{theta:.3g}_sigma{sigma:.3g}.pt"
        path = os.path.join(sweep_args.output_dir, file_name)
        args = copy.copy(base_args)
        args.theta = float(theta)
        args.sigma = float(sigma)
        args.output = path
        window = mollifier_window(args)
        entry = {"theta": args.theta, "sigma": args.sigma, "file": file_name,
                 "mollifier_window": window}
        entries.append(entry)

        if sweep_args.skip_existing and os.path.exists(path):
            logger.info("[%d/%d] %s exists, skipping", index, len(combinations), file_name)
            continue

        logger.info("[%d/%d] theta=%.4g sigma=%.4g, mollifier window %d samples (%.4g s)",
                    index, len(combinations), theta, sigma, window, window * args.dt)
        if window < 5:
            logger.warning("Window below 5 samples: almost no smoothing.")
        dataset = build_dataset(args)
        entry["shapes"] = dataset["metadata"]["shapes"]
        save_dataset(dataset, path)

    base = {key: value for key, value in vars(base_args).items()
            if key not in ("theta", "sigma", "output")}
    index_path = os.path.join(sweep_args.output_dir, "index.json")
    with open(index_path, "w") as file:
        json.dump({"thetas": thetas.tolist(), "sigmas": sigmas.tolist(),
                   "base_args": base, "datasets": entries}, file, indent=2)
    logger.info("Wrote %d datasets, index at %s", len(entries), index_path)


if __name__ == "__main__":
    main()
