"""Generate the kinematic-hardening train, validation and test datasets.

Train and validation follow the data pipeline of
``kinematic_hardening_normalized.ipynb`` with the same OU parameters but
different seeds:

1. sample strain trajectories with an OU process,
2. smooth them with the mollifier convolution,
3. compute plastic strain and stress with ``KinematicHardeningPlasticity``.

The test set uses realistic piecewise-linear loading programs instead of
OU paths: tensile loading--unloading cycles, tension--compression cycles,
and tension--compression cycles of growing amplitude, for a family of
strain rates and maximum tensile/compressive strains.

All splits share one time grid and are saved, in physical units, in a
single ``.pt`` file together with the metadata needed to reproduce them.
"""

from __future__ import annotations

import argparse
import itertools
from datetime import datetime, timezone

import numpy as np
import torch
from tqdm.auto import tqdm

from ou_trajectories import generate_ou_trajectories, smooth_ou_trajectories
from true_constitutive_law import KinematicHardeningPlasticity

TEST_PATTERNS = ("tension_cycles", "tension_compression", "growing_tension_compression")


def stress_histories(
    kinematic_model: KinematicHardeningPlasticity, strain_trajectories: np.ndarray, desc: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return plastic strain and stress for every strain trajectory."""
    stress_trajectories = np.empty_like(strain_trajectories)
    plastic_strain_trajectories = np.empty_like(strain_trajectories)
    for trajectory_index, strain_path in enumerate(
        tqdm(strain_trajectories, desc=desc, unit="traj")
    ):
        plastic_strain, stress_path = kinematic_model.stress_history(
            strain_path[:, np.newaxis]
        )
        plastic_strain_trajectories[trajectory_index] = plastic_strain[:, 0]
        stress_trajectories[trajectory_index] = stress_path[:, 0]
    return plastic_strain_trajectories, stress_trajectories


def ou_split(args: argparse.Namespace, n_trajectories: int, seed: int) -> tuple[np.ndarray, dict]:
    """Sample and smooth OU strain paths with the shared OU parameters."""
    times, strain_trajectories = generate_ou_trajectories(
        theta=args.theta,
        mu=args.mu,
        sigma=args.sigma,
        x0=args.x0,
        n_trajectories=n_trajectories,
        T=args.T,
        dt=args.dt,
        seed=seed,
    )
    strain_trajectories_smooth = smooth_ou_trajectories(
        strain_trajectories,
        window_size=args.mollifier_window,
    )
    return times, {"strain_raw": strain_trajectories, "strain": strain_trajectories_smooth}


def strain_waypoints(
    pattern: str, tension: float, compression: float, n_ramps: int
) -> list[float]:
    """Strain turning points of a loading program, starting from zero.

    ``tension`` and ``compression`` are positive magnitudes; compressive
    peaks are negative strains.
    """
    waypoints = [0.0]
    cycle = 0
    while len(waypoints) <= n_ramps:
        cycle += 1
        if pattern == "tension_cycles":
            waypoints += [tension, 0.0]
        elif pattern == "tension_compression":
            waypoints += [tension, 0.0, -compression, 0.0]
        elif pattern == "growing_tension_compression":
            # Peaks grow by a third of the final amplitude per cycle.
            growth = min(1.0, cycle / 3.0)
            waypoints += [growth * tension, 0.0, -growth * compression, 0.0]
        else:
            raise ValueError(f"Unknown loading pattern {pattern!r}.")
    return waypoints


def loading_program(
    times: np.ndarray, pattern: str, tension: float, compression: float, rate: float
) -> np.ndarray:
    """Piecewise-linear strain history with constant |strain rate| ``rate``."""
    # Enough ramps to cover [0, T] even for the smallest peaks.
    smallest_peak = min(tension, compression) / 3.0
    n_ramps = int(np.ceil(times[-1] * rate / smallest_peak)) + 4
    waypoints = np.asarray(strain_waypoints(pattern, tension, compression, n_ramps))
    waypoint_times = np.concatenate(([0.0], np.cumsum(np.abs(np.diff(waypoints)) / rate)))
    return np.interp(times, waypoint_times, waypoints)


def test_split(args: argparse.Namespace, times: np.ndarray, train_strain: np.ndarray) -> tuple[dict, dict]:
    """Family of realistic loading programs scaled to the training data.

    Peak strains are fractions of the median (over trajectories) of the
    largest |strain| in each training path; strain rates are fractions of
    the median |strain rate| of the training paths.
    """
    reference_strain = float(np.median(np.abs(train_strain).max(axis=1)))
    reference_rate = float(np.median(np.abs(np.diff(train_strain, axis=1)) / args.dt))
    tensions = [fraction * reference_strain for fraction in args.test_tension_fractions]
    compressions = [fraction * reference_strain for fraction in args.test_compression_fractions]
    rates = [fraction * reference_rate for fraction in args.test_rate_fractions]

    strain_paths = []
    programs = []
    for pattern in TEST_PATTERNS:
        # Purely tensile cycles do not depend on the compressive peak.
        pattern_compressions = compressions[:1] if pattern == "tension_cycles" else compressions
        for tension, compression, rate in itertools.product(tensions, pattern_compressions, rates):
            strain_paths.append(loading_program(times, pattern, tension, compression, rate))
            programs.append(
                {
                    "pattern": pattern,
                    "max_tensile_strain": tension,
                    "max_compressive_strain": 0.0 if pattern == "tension_cycles" else -compression,
                    "strain_rate": rate,
                }
            )

    metadata = {
        "description": "Piecewise-linear loading programs (not OU).",
        "patterns": list(TEST_PATTERNS),
        "reference_strain": reference_strain,
        "reference_strain_rate": reference_rate,
        "tension_fractions": list(args.test_tension_fractions),
        "compression_fractions": list(args.test_compression_fractions),
        "rate_fractions": list(args.test_rate_fractions),
        "programs": programs,
    }
    return {"strain": np.stack(strain_paths)}, metadata


def build_dataset(args: argparse.Namespace) -> dict:
    if args.val_seed == args.seed:
        raise ValueError("--val-seed must differ from --seed.")

    kinematic_model = KinematicHardeningPlasticity(
        node_count=1,
        model_dt=args.dt,
        m=args.rate_exponent,
        E=args.E,
        H=args.H,
        sigma_y=args.sigma_y,
    )

    times, train = ou_split(args, args.n_trajectories, args.seed)
    _, validation = ou_split(args, args.n_val, args.val_seed)
    test, test_metadata = test_split(args, times, train["strain"])

    splits = {"train": train, "validation": validation, "test": test}
    for name, split in splits.items():
        split["plastic_strain"], split["stress"] = stress_histories(
            kinematic_model, split["strain"], desc=f"Stress histories ({name})"
        )

    metadata = {
        "description": "Kinematic-hardening stress for OU (train/validation) and "
        "loading-program (test) strain paths.",
        "created": datetime.now(timezone.utc).isoformat(),
        "ou": {
            "n_trajectories": args.n_trajectories,
            "n_val": args.n_val,
            "T": args.T,
            "dt": args.dt,
            "theta": args.theta,
            "mu": args.mu,
            "sigma": args.sigma,
            "x0": args.x0,
            "seed": args.seed,
            "val_seed": args.val_seed,
        },
        "smoothing": {"mollifier_window": args.mollifier_window},
        "test": test_metadata,
        "constitutive_model": {
            "name": "KinematicHardeningPlasticity",
            "E": args.E,
            "H": args.H,
            "sigma_y": args.sigma_y,
            "rate_exponent": args.rate_exponent,
        },
        "shapes": {name: list(split["strain"].shape) for name, split in splits.items()},
    }

    dataset = {
        name: {key: torch.as_tensor(value, dtype=torch.float64) for key, value in split.items()}
        for name, split in splits.items()
    }
    dataset["times"] = torch.as_tensor(times, dtype=torch.float64)
    dataset["metadata"] = metadata
    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default="kinematic_hardening_dataset.pt")
    # OU process (shared by train and validation).
    parser.add_argument("--n-trajectories", type=int, default=64)
    parser.add_argument("--n-val", type=int, default=16)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=1.0 / 12000.0)
    parser.add_argument("--theta", type=float, default=3.0)
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--x0", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-seed", type=int, default=43)
    # Smoothing.
    parser.add_argument("--mollifier-window", type=int, default=21)
    # Test loading programs, as fractions of the training strain and rate scales.
    parser.add_argument("--test-tension-fractions", type=float, nargs="+", default=[0.5, 1.0])
    parser.add_argument("--test-compression-fractions", type=float, nargs="+", default=[0.5, 1.0])
    parser.add_argument("--test-rate-fractions", type=float, nargs="+", default=[0.25, 0.5, 1.0])
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
    print(f"Saved {dataset['metadata']['shapes']} to {args.output}")


if __name__ == "__main__":
    main()
