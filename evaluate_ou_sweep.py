"""Cross-evaluate OU sweep models: every model on every dataset.

``--model-name NAME`` (one job per model, see ``submit_ou_sweep.py evaluate``)
loads the newest run of the model trained on dataset ``NAME`` and computes
the relative L2-in-time stress error of every trajectory in the train,
validation and test splits of every sweep dataset. The errors are saved to
``<runs-dir>/cross_eval/NAME.npz`` under keys ``<dataset>__<split>``.

``--collect`` combines all saved results into

- ``summary.csv``: mean and median error for every (trained on, evaluated on,
  split), in long format,
- ``heatmap_<split>.pdf``: median error with rows the training dataset and
  columns the evaluation dataset, both ordered by (theta, sigma). The
  diagonal (evaluated on its own dataset) is outlined and missing models are
  left blank.
- ``own_error_vs_sigma.pdf`` and ``own_error_vs_theta.pdf``: mean error of
  every model on its own dataset, one panel per split. The first has sigma on
  the x-axis and one line per theta, the second the other way round.
"""

from __future__ import annotations

import argparse
import logging
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, LogLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.logging_utils import setup_logging
from src.utilities import apply_plot_style, relative_l2_error
from submit_ou_sweep import dataset_name, find_checkpoint, latest_run, load_index
from test_kinematic_hardening import predict_stress
from train_kinematic_hardening import AdjointVariationalODE

logger = logging.getLogger(__name__)

SPLITS = ("train", "validation", "test")


def sorted_entries(index: dict) -> list[dict]:
    return sorted(index["datasets"], key=lambda entry: (entry["theta"], entry["sigma"]))


def evaluate_model(args: argparse.Namespace) -> None:
    index = load_index(args.sweep_dir)
    entries = {dataset_name(entry): entry for entry in index["datasets"]}
    if args.model_name not in entries:
        raise ValueError(f"{args.model_name} is not in {args.sweep_dir}/index.json.")
    run_dir = latest_run(os.path.join(args.runs_dir, args.model_name))
    if run_dir is None:
        raise FileNotFoundError(f"No run for {args.model_name} in {args.runs_dir}.")
    checkpoint = find_checkpoint(run_dir, args.checkpoint_name)
    if checkpoint is None:
        raise FileNotFoundError(f"No {args.checkpoint_name} checkpoint in {run_dir}/checkpoints.")

    output_dir = os.path.join(args.runs_dir, "cross_eval")
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(os.path.join(output_dir, f"{args.model_name}.log"))
    logger.info("Model %s: %s", args.model_name, os.path.abspath(checkpoint))
    model = AdjointVariationalODE.load_from_checkpoint(checkpoint, map_location="cpu")
    model.eval()

    results = {}
    table = [f"Relative L2 stress error of {args.model_name}",
             f"{'evaluated on':<24}{'split':<12}{'mean':>11}{'median':>11}{'max':>11}"]
    for entry in sorted_entries(index):
        name = dataset_name(entry)
        dataset = torch.load(os.path.join(args.sweep_dir, entry["file"]), weights_only=False)
        dataset_dt = dataset["metadata"]["ou"]["dt"]
        if not np.isclose(model.hparams.dt, dataset_dt):
            raise ValueError(f"Model was trained with dt={model.hparams.dt}, {name} has dt={dataset_dt}.")
        for split in SPLITS:
            predicted = predict_stress(model, dataset[split]["strain"], args.batch_size)
            errors = relative_l2_error(dataset[split]["stress"], predicted, reduction="none")
            results[f"{name}__{split}"] = errors
            table.append(f"{name:<24}{split:<12}{errors.mean():>11.4e}"
                         f"{np.median(errors):>11.4e}{errors.max():>11.4e}")
        logger.info("Evaluated on %s", name)
    logger.info("\n".join(table))

    trained_on = entries[args.model_name]
    path = os.path.join(output_dir, f"{args.model_name}.npz")
    np.savez(path, **results, model_theta=trained_on["theta"], model_sigma=trained_on["sigma"],
             checkpoint=os.path.abspath(checkpoint))
    logger.info("Saved %s", path)


def grid_label(entry: dict) -> str:
    return rf"$\theta={entry['theta']:.3g}$, $\sigma={entry['sigma']:.3g}$"


def heatmap(matrix: np.ndarray, entries: list[dict], split: str) -> plt.Figure:
    count = len(entries)
    size = max(6.0, 0.32 * count + 2.5)
    fig, ax = plt.subplots(figsize=(size + 1.2, size))
    finite = matrix[np.isfinite(matrix) & (matrix > 0)]
    norm = LogNorm(vmin=finite.min(), vmax=finite.max()) if finite.size else None
    image = ax.imshow(np.ma.masked_invalid(matrix), cmap="viridis", norm=norm)
    # A colorbar axis of the same height as the heatmap.
    colorbar_axis = make_axes_locatable(ax).append_axes("right", size="3%", pad=0.15)
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label(rf"Median relative $L^2$ stress error ({split})")
    for i in range(count):
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="white", linewidth=1.5))
    labels = [grid_label(entry) for entry in entries]
    ax.set_xticks(range(count), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(count), labels, fontsize=7)
    ax.minorticks_off()
    ax.tick_params(top=False, right=False)
    ax.set_xlabel("Evaluated on")
    ax.set_ylabel("Trained on")
    return fig


def own_error_figure(
    own_errors: dict[str, dict[tuple[float, float], float]],
    thetas: list[float],
    sigmas: list[float],
    x_name: str,
) -> plt.Figure:
    """Error of every model on its own dataset, one panel per split.

    ``own_errors[split][(theta, sigma)]`` is the error of the model trained on
    (theta, sigma). With ``x_name="sigma"`` each line is one theta and the
    x-axis is sigma; with ``x_name="theta"`` the other way round.
    """
    x_values, line_values = (sigmas, thetas) if x_name == "sigma" else (thetas, sigmas)
    line_name = "theta" if x_name == "sigma" else "sigma"
    colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(line_values)))
    splits = list(own_errors)
    fig, axes = plt.subplots(1, len(splits), figsize=(5.2 * len(splits), 4.6),
                             sharey=True, squeeze=False)
    for axis, split in zip(axes[0], splits):
        for color, line_value in zip(colors, line_values):
            keys = [(line_value, x) if x_name == "sigma" else (x, line_value) for x in x_values]
            y = np.array([own_errors[split].get(key, np.nan) for key in keys])
            axis.plot(x_values, y, marker="o", markersize=5, color=color,
                      label=rf"$\{line_name}={line_value:.3g}$")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xticks(x_values, [f"{x:.3g}" for x in x_values])
        axis.minorticks_off()
        # Errors span about one decade, so label 1-2-5 steps, not just powers of ten.
        axis.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        axis.set_xlabel(rf"$\{x_name}$")
        axis.set_title(split)
        axis.grid(alpha=0.2, which="both")
    axes[0, 0].set_ylabel(r"Mean relative $L^2$ stress error")
    axes[0, -1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    return fig


def save_own_error_plots(
    own_errors: dict[str, dict[tuple[float, float], float]],
    thetas: list[float],
    sigmas: list[float],
    output_dir: str,
) -> None:
    for x_name in ("sigma", "theta"):
        fig = own_error_figure(own_errors, thetas, sigmas, x_name)
        fig.savefig(os.path.join(output_dir, f"own_error_vs_{x_name}.pdf"), bbox_inches="tight")
        plt.close(fig)


def collect(args: argparse.Namespace) -> None:
    output_dir = os.path.join(args.runs_dir, "cross_eval")
    setup_logging(os.path.join(output_dir, "collect.log"))
    entries = sorted_entries(load_index(args.sweep_dir))
    names = [dataset_name(entry) for entry in entries]

    rows, missing = [], []
    matrices = {split: np.full((len(names), len(names)), np.nan) for split in SPLITS}
    own_means = {split: {} for split in SPLITS}
    for i, (model_name, model_entry) in enumerate(zip(names, entries)):
        path = os.path.join(output_dir, f"{model_name}.npz")
        if not os.path.exists(path):
            missing.append(model_name)
            continue
        results = np.load(path)
        for j, (eval_name, eval_entry) in enumerate(zip(names, entries)):
            for split in SPLITS:
                key = f"{eval_name}__{split}"
                if key not in results:
                    continue
                errors = results[key]
                matrices[split][i, j] = np.median(errors)
                if i == j:
                    own_means[split][(model_entry["theta"], model_entry["sigma"])] = errors.mean()
                rows.append({
                    "train_dataset": model_name, "train_theta": model_entry["theta"],
                    "train_sigma": model_entry["sigma"],
                    "eval_dataset": eval_name, "eval_theta": eval_entry["theta"],
                    "eval_sigma": eval_entry["sigma"], "split": split,
                    "mean_error": errors.mean(), "median_error": np.median(errors),
                    "max_error": errors.max(), "n": len(errors),
                })
    if missing:
        logger.warning("No results for %d of %d models: %s", len(missing), len(names), ", ".join(missing))
    if not rows:
        raise FileNotFoundError(f"No results in {output_dir}; run --model-name jobs first.")

    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "summary.csv"), index=False)
    apply_plot_style()
    for split, matrix in matrices.items():
        if np.isnan(matrix).all():
            logger.warning("No %s errors in the results; skipping its plots.", split)
            continue
        # Not save_pdf: it makes every axis square, which would squash the colorbar.
        fig = heatmap(matrix, entries, split)
        fig.savefig(os.path.join(output_dir, f"heatmap_{split}.pdf"), bbox_inches="tight")
        plt.close(fig)
    thetas = sorted({entry["theta"] for entry in entries})
    sigmas = sorted({entry["sigma"] for entry in entries})
    save_own_error_plots({split: means for split, means in own_means.items() if means},
                         thetas, sigmas, output_dir)
    diagonal = np.diag(matrices["test"])
    logger.info("Median test error on own dataset: %s",
                ", ".join(f"{n} {e:.3e}" for n, e in zip(names, diagonal) if np.isfinite(e)))
    logger.info("Wrote summary.csv, heatmaps and own-error plots to %s", os.path.abspath(output_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--model-name", help="Evaluate the model trained on this dataset, e.g. theta1_sigma0.5.")
    mode.add_argument("--collect", action="store_true", help="Combine all saved results.")
    parser.add_argument("--sweep-dir", default="datasets/ou_sweep")
    parser.add_argument("--runs-dir", default="runs/ou_sweep")
    parser.add_argument("--checkpoint-name", default="last.ckpt",
                        help="Checkpoint file to evaluate, or 'best' for the run's "
                        "best-epoch=N.ckpt (lowest validation error).")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    # Scripts only save figures.
    matplotlib.use("Agg")
    args = parse_args()
    if args.collect:
        collect(args)
    else:
        evaluate_model(args)


if __name__ == "__main__":
    main()
