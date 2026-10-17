"""Generate the kinematic-hardening training dataset.

Reproduces the data pipeline of ``kinematic_hardening_normalized.ipynb``:

1. sample strain trajectories with an OU process,
2. smooth them with the mollifier convolution,
3. compute plastic strain and stress with ``KinematicHardeningPlasticity``.

The physical (unnormalized) histories are saved together with the
metadata needed to reproduce them or to normalize them for training.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np
import torch
from tqdm.auto import tqdm

from ou_trajectories import generate_ou_trajectories, smooth_ou_trajectories
from true_constitutive_law import KinematicHardeningPlasticity


def build_dataset(args: argparse.Namespace) -> dict:
    times, strain_trajectories = generate_ou_trajectories(
        theta=args.theta,
        mu=args.mu,
        sigma=args.sigma,
        x0=args.x0,
        n_trajectories=args.n_trajectories,
        T=args.T,
        dt=args.dt,
        seed=args.seed,
    )
    strain_trajectories_smooth = smooth_ou_trajectories(
        strain_trajectories,
        window_size=args.mollifier_window,
    )

    kinematic_model = KinematicHardeningPlasticity(
        node_count=1,
        model_dt=args.dt,
        m=args.rate_exponent,
        E=args.E,
        H=args.H,
        sigma_y=args.sigma_y,
    )
    stress_trajectories = np.empty_like(strain_trajectories_smooth)
    plastic_strain_trajectories = np.empty_like(strain_trajectories_smooth)
    for trajectory_index, strain_path in enumerate(
        tqdm(strain_trajectories_smooth, desc="Stress histories", unit="traj")
    ):
        plastic_strain, stress_path = kinematic_model.stress_history(
            strain_path[:, np.newaxis]
        )
        plastic_strain_trajectories[trajectory_index] = plastic_strain[:, 0]
        stress_trajectories[trajectory_index] = stress_path[:, 0]

    metadata = {
        "description": "OU strain paths, mollifier-smoothed, with kinematic-hardening stress.",
        "created": datetime.now(timezone.utc).isoformat(),
        "ou": {
            "n_trajectories": args.n_trajectories,
            "T": args.T,
            "dt": args.dt,
            "theta": args.theta,
            "mu": args.mu,
            "sigma": args.sigma,
            "x0": args.x0,
            "seed": args.seed,
        },
        "smoothing": {"mollifier_window": args.mollifier_window},
        "constitutive_model": {
            "name": "KinematicHardeningPlasticity",
            "E": args.E,
            "H": args.H,
            "sigma_y": args.sigma_y,
            "rate_exponent": args.rate_exponent,
        },
        "shapes": {
            "times": list(times.shape),
            "trajectories": list(strain_trajectories_smooth.shape),
        },
    }

    return {
        "times": torch.as_tensor(times, dtype=torch.float64),
        "strain_raw": torch.as_tensor(strain_trajectories, dtype=torch.float64),
        "strain": torch.as_tensor(strain_trajectories_smooth, dtype=torch.float64),
        "plastic_strain": torch.as_tensor(plastic_strain_trajectories, dtype=torch.float64),
        "stress": torch.as_tensor(stress_trajectories, dtype=torch.float64),
        "metadata": metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default="kinematic_hardening_dataset.pt")
    # OU process.
    parser.add_argument("--n-trajectories", type=int, default=64)
    parser.add_argument("--T", type=float, default=2.0)
    parser.add_argument("--dt", type=float, default=1.0 / 100.0)
    parser.add_argument("--theta", type=float, default=3.0)
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--sigma", type=float, default=0.08)
    parser.add_argument("--x0", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    # Smoothing.
    parser.add_argument("--mollifier-window", type=int, default=21)
    # Kinematic-hardening model.
    parser.add_argument("--E", type=float, default=1.0)
    parser.add_argument("--H", type=float, default=1.0)
    parser.add_argument("--sigma-y", type=float, default=1.0)
    parser.add_argument("--rate-exponent", type=float, default=0.3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = build_dataset(args)
    torch.save(dataset, args.output)
    print(f"Saved {dataset['metadata']['shapes']['trajectories']} trajectories to {args.output}")


if __name__ == "__main__":
    main()
