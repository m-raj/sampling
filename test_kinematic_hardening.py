"""Evaluate a trained kinematic-hardening model on the saved datasets.

Loads a checkpoint written by ``train_kinematic_hardening.py`` and the
dataset written by ``generate_kinematic_hardening_dataset.py``, predicts
stress for the train, validation and test splits, and reports the relative
L2-in-time stress error of every split.

Figures (saved as PDF in the run's ``plots`` folder by default):

1. ``test_stress_time``: true and predicted stress against time for the
   test samples with the worst, median and best relative error.
2. ``test_stress_strain``: the same samples as stress--strain curves.
3. ``error_boxplots``: box plots of the train, validation and test errors.
"""

from __future__ import annotations

import argparse
import logging
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from train_kinematic_hardening import AdjointVariationalODE, run_dir_from_checkpoint
from src.logging_utils import setup_logging
from src.utilities import apply_plot_style, relative_l2_error, save_pdf

logger = logging.getLogger(__name__)

SPLITS = ("train", "validation", "test")
# First three slots of the dataviz skill's validated categorical palette.
BLUE, ORANGE, AQUA = "#1f75de", "#eb6834", "#1baf7a"
TRUE_STYLE = {"color": BLUE, "linestyle": "-", "linewidth": 2.0, "label": "true"}
PREDICTED_STYLE = {"color": ORANGE, "linestyle": "--", "linewidth": 2.0, "label": "predicted"}
SPLIT_COLORS = {"train": BLUE, "validation": ORANGE, "test": AQUA}


def predict_stress(model: AdjointVariationalODE, strain: torch.Tensor, batch_size: int) -> np.ndarray:
    """Predicted stress for every strain trajectory, solved in batches."""
    predictions = []
    for strain_batch in torch.split(strain.to(torch.float32), batch_size):
        # The solver differentiates W and D, so it needs autograd.
        with torch.enable_grad():
            _, stress_batch = model.solver.solve(
                strain_batch, dt=model.hparams.dt, create_graph=False
            )
        predictions.append(stress_batch.detach().cpu().numpy())
    return np.concatenate(predictions)


def describe_program(program: dict, strain: np.ndarray) -> str:
    # Report the strain range actually reached: slow programs may end before
    # reaching their nominal peaks.
    pattern = program["pattern"].replace("_", " ")
    return (
        f"{pattern}, rate {program['strain_rate']:.3g}, "
        f"strain [{strain.min():.3g}, {strain.max():.3g}]"
    )


def representative_samples(errors: np.ndarray) -> dict[str, int]:
    """Indices of the worst, median and best relative error."""
    order = np.argsort(errors)
    return {"worst": int(order[-1]), "median": int(order[len(order) // 2]), "best": int(order[0])}


def comparison_figure(
    x_values: dict[int, np.ndarray],
    x_label: str,
    strain: np.ndarray,
    true_stress: np.ndarray,
    predicted_stress: np.ndarray,
    samples: dict[str, int],
    errors: np.ndarray,
    programs: list[dict],
) -> plt.Figure:
    """One row per representative sample: true vs predicted stress."""

    fig, axes = plt.subplots(
        len(samples), 1, figsize=(8, 3.2 * len(samples)), squeeze=False
    )
    for axis, (name, index) in zip(axes[:, 0], samples.items()):
        axis.plot(x_values[index], true_stress[index], **TRUE_STYLE)
        axis.plot(x_values[index], predicted_stress[index], **PREDICTED_STYLE)
        axis.set_title(
            f"{name.capitalize()} (test no. {index}, rel. error {errors[index]:.3g})\n"
            f"{describe_program(programs[index], strain[index])}"
        )
        axis.set_xlabel(x_label)
        axis.set_ylabel("Stress")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(loc="best")
    return fig


def error_boxplot_figure(errors: dict[str, np.ndarray]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    boxes = ax.boxplot(
        [errors[split] for split in SPLITS],
        tick_labels=[f"{split}\n(n={len(errors[split])})" for split in SPLITS],
        patch_artist=True,
        widths=0.5,
        medianprops={"color": "black", "linewidth": 1.5},
        flierprops={"marker": "o", "markersize": 4, "markerfacecolor": "none"},
    )
    for patch, split in zip(boxes["boxes"], SPLITS):
        patch.set_facecolor(SPLIT_COLORS[split])
        patch.set_alpha(0.6)
    all_errors = np.concatenate(list(errors.values()))
    # Errors spanning orders of magnitude are easier to compare on a log axis.
    if all_errors.min() > 0 and all_errors.max() / all_errors.min() > 100:
        ax.set_yscale("log")
    ax.set_ylabel(r"Relative $L^2$ stress error")
    ax.grid(axis="y", alpha=0.2)
    return fig


def save_figure(fig: plt.Figure, output_dir: str, name: str) -> None:
    save_pdf(fig, os.path.join(output_dir, f"{name}.pdf"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", default=None,
                        help="Run folder written by train_kinematic_hardening.py; "
                        "its checkpoints/last.ckpt is evaluated.")
    parser.add_argument("--checkpoint", default=None,
                        help="Specific checkpoint to evaluate, e.g. checkpoints/best-epoch=N.ckpt.")
    parser.add_argument("--dataset", default=None,
                        help="Defaults to the dataset the model was trained on.")
    parser.add_argument("--output-dir", default=None,
                        help="Defaults to the run's plots folder.")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.checkpoint is None:
        if args.run_dir is None:
            parser.error("give --run-dir or --checkpoint")
        args.checkpoint = os.path.join(args.run_dir, "checkpoints", "last.ckpt")
    return args


def main() -> None:
    # Scripts only save figures; set the backend here rather than at import
    # time so importing this module (e.g. in a notebook) keeps its backend.
    matplotlib.use("Agg")
    args = parse_args()
    apply_plot_style()
    output_dir = args.output_dir or os.path.join(run_dir_from_checkpoint(args.checkpoint), "plots")
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(os.path.join(output_dir, "test.log"))

    # The model type (potential or invertible) is restored from the checkpoint.
    model = AdjointVariationalODE.load_from_checkpoint(args.checkpoint, map_location="cpu")
    model.eval()
    logger.info("Model: %s (%s)", model.hparams.model, type(model.solver).__name__)
    dataset_path = args.dataset or model.hparams.get("dataset_path")
    if dataset_path is None:
        raise ValueError("Checkpoint does not record its dataset; pass --dataset.")
    logger.info("Dataset: %s", dataset_path)
    dataset = torch.load(dataset_path, weights_only=False)
    dataset_dt = dataset["metadata"]["ou"]["dt"]
    if not np.isclose(model.hparams.dt, dataset_dt):
        raise ValueError(f"Model was trained with dt={model.hparams.dt}, dataset has dt={dataset_dt}.")

    predictions = {}
    errors = {}
    for split in SPLITS:
        predictions[split] = predict_stress(model, dataset[split]["strain"], args.batch_size)
        errors[split] = relative_l2_error(
            dataset[split]["stress"], predictions[split], reduction="none"
        )

    # Tables are logged as one multi-line message so their columns stay aligned.
    table = [f"Relative L2 stress error ({args.checkpoint})",
             f"{'split':<11}{'n':>4}{'mean':>11}{'median':>11}{'min':>11}{'max':>11}"]
    for split in SPLITS:
        e = errors[split]
        table.append(f"{split:<11}{len(e):>4}{e.mean():>11.4e}{np.median(e):>11.4e}{e.min():>11.4e}{e.max():>11.4e}")
    logger.info("\n".join(table))

    programs = dataset["metadata"]["test"]["programs"]
    samples = representative_samples(errors["test"])
    table = ["Representative test samples:"]
    for name, index in samples.items():
        table.append(f"  {name:<7}#{index:<3} error {errors['test'][index]:.4e}  {describe_program(programs[index], dataset['test']['strain'][index].numpy())}")
    logger.info("\n".join(table))

    times = dataset["times"].numpy()
    test_strain = dataset["test"]["strain"].numpy()
    test_stress = dataset["test"]["stress"].numpy()
    save_figure(
        comparison_figure({i: times for i in samples.values()}, "Time", test_strain, test_stress,
                          predictions["test"], samples, errors["test"], programs),
        output_dir, "test_stress_time",
    )
    save_figure(
        comparison_figure({i: test_strain[i] for i in samples.values()}, "Strain", test_strain, test_stress,
                          predictions["test"], samples, errors["test"], programs),
        output_dir, "test_stress_strain",
    )
    save_figure(error_boxplot_figure(errors), output_dir, "error_boxplots")

    np.savez(
        os.path.join(output_dir, "relative_errors.npz"),
        **{f"{split}_errors": errors[split] for split in SPLITS},
        test_predicted_stress=predictions["test"],
    )
    logger.info("Figures and errors saved in %s", os.path.abspath(output_dir))


if __name__ == "__main__":
    main()
