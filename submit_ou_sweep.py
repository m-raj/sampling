"""Submit one SLURM job per OU sweep dataset: training or cross-evaluation.

Reads the sweep written by ``generate_ou_sweep.py`` (``--sweep-dir``, with its
``index.json``) and gives every dataset its own folder ``<runs-dir>/<name>``,
where ``name`` is the dataset file name without ``.pt`` (e.g.
``theta2_sigma0.5``). ``train_kinematic_hardening.py`` creates its usual
timestamped run inside it; the newest run is the model of that dataset.

    python submit_ou_sweep.py train --partition P --setup "source venv/bin/activate" \\
        -- --model invertible --epochs 1000 --pretrain
    python submit_ou_sweep.py evaluate --partition P --setup "source venv/bin/activate"

``train`` forwards every option it does not know to
``train_kinematic_hardening.py``. It is safe to run again: for every dataset it

- skips it while a job submitted from here is still queued or running,
- skips it once its newest run has trained for ``--epochs`` epochs,
- resumes the newest run from its latest checkpoint if it stopped early
  (e.g. at the time limit, where SLURM's SIGTERM makes Lightning save
  ``interrupted.ckpt``),
- and otherwise starts a new run. ``--fresh`` starts new runs for all.

``evaluate`` submits ``evaluate_ou_sweep.py --model-name <name>`` for every
finished model that has no results yet (``--force`` redoes them). Combine the
results with ``python evaluate_ou_sweep.py --collect`` afterwards.

Job scripts and SLURM output go to ``<runs-dir>/<name>/`` and every
submission is recorded in ``<runs-dir>/jobs.json``. ``--dry-run`` writes the
job scripts and prints what would be submitted without submitting.
"""

from __future__ import annotations

import argparse
import getpass
import glob
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# Set by this script for every job, so they cannot be passed through.
RESERVED_TRAIN_OPTIONS = ("--dataset", "--runs-dir", "--run-name", "--resume")
# Must match the default of train_kinematic_hardening.py.
DEFAULT_EPOCHS = 100


# ---------------------------------------------------------------------------
# Sweep layout (also used by evaluate_ou_sweep.py)
# ---------------------------------------------------------------------------

def load_index(sweep_dir: str) -> dict:
    with open(os.path.join(sweep_dir, "index.json")) as file:
        return json.load(file)


def dataset_name(entry: dict) -> str:
    return os.path.splitext(entry["file"])[0]


def latest_run(model_dir: str) -> str | None:
    """Newest run folder of one dataset (run folders start with a timestamp)."""
    runs = sorted(path for path in glob.glob(os.path.join(model_dir, "*"))
                  if os.path.isdir(os.path.join(path, "checkpoints")))
    return runs[-1] if runs else None


def latest_checkpoint(run_dir: str) -> str | None:
    """The more recent of ``last.ckpt`` and ``interrupted.ckpt``."""
    candidates = [os.path.join(run_dir, "checkpoints", name)
                  for name in ("last.ckpt", "interrupted.ckpt")]
    candidates = [path for path in candidates if os.path.exists(path)]
    return max(candidates, key=os.path.getmtime) if candidates else None


def completed_epochs(checkpoint_path: str) -> int:
    """Number of epochs fully trained when the checkpoint was saved."""
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    loop = checkpoint.get("loops", {}).get("fit_loop", {})
    progress = loop.get("epoch_progress", {}).get("current", {})
    if "completed" in progress:
        return int(progress["completed"])
    return int(checkpoint.get("epoch", -1)) + 1


# ---------------------------------------------------------------------------
# SLURM
# ---------------------------------------------------------------------------

def active_job_ids() -> set[str]:
    """Job IDs of this user that SLURM still lists (queued or running)."""
    result = subprocess.run(["squeue", "-h", "-u", getpass.getuser(), "-o", "%i"],
                            capture_output=True, text=True, check=True)
    return set(result.stdout.split())


def load_jobs(runs_dir: str) -> list[dict]:
    path = os.path.join(runs_dir, "jobs.json")
    if not os.path.exists(path):
        return []
    with open(path) as file:
        return json.load(file)


def record_job(runs_dir: str, job: dict) -> None:
    jobs = load_jobs(runs_dir)
    jobs.append(job)
    with open(os.path.join(runs_dir, "jobs.json"), "w") as file:
        json.dump(jobs, file, indent=2)


def busy_datasets(runs_dir: str, task: str, dry_run: bool) -> set[str]:
    """Datasets with a ``task`` job from jobs.json that is still queued or running."""
    jobs = [job for job in load_jobs(runs_dir) if job["task"] == task]
    if not jobs:
        return set()
    try:
        active = active_job_ids()
    except (OSError, subprocess.CalledProcessError):
        if not dry_run:
            raise
        print("squeue not available: not checking for queued or running jobs.")
        return set()
    return {job["name"] for job in jobs if job["job_id"] in active}


def job_script(args: argparse.Namespace, job_name: str, log_dir: str, command: list[str]) -> str:
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --output={os.path.join(os.path.abspath(log_dir), job_name + '-%j.out')}",
        f"#SBATCH --chdir={PROJECT_ROOT}",
        f"#SBATCH --time={args.time}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={args.cpus}",
        f"#SBATCH --mem={args.mem}",
        # SIGTERM to the batch shell 5 minutes before the time limit. The shell
        # is replaced by Python (exec below), so Lightning receives it and saves
        # interrupted.ckpt; resubmitting `train` resumes from there.
        "#SBATCH --signal=B:TERM@300",
    ]
    if args.partition:
        lines.append(f"#SBATCH --partition={args.partition}")
    if args.account:
        lines.append(f"#SBATCH --account={args.account}")
    lines += ["", "set -e", f"cd {shlex.quote(PROJECT_ROOT)}", *args.setup,
              'export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"',
              'export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"',
              "", "exec " + shlex.join(command), ""]
    return "\n".join(lines)


def submit(args: argparse.Namespace, task: str, name: str, command: list[str]) -> None:
    model_dir = os.path.join(args.runs_dir, name)
    os.makedirs(model_dir, exist_ok=True)
    script_path = os.path.join(model_dir, f"{task}.sbatch")
    with open(script_path, "w") as file:
        file.write(job_script(args, f"{task}_{name}", model_dir, command))
    if args.dry_run:
        print(f"  [dry run] sbatch {script_path}\n            {shlex.join(command)}")
        return
    result = subprocess.run(["sbatch", "--parsable", script_path],
                            capture_output=True, text=True, check=True)
    job_id = result.stdout.strip().split(";")[0]
    record_job(args.runs_dir, {
        "name": name, "task": task, "job_id": job_id, "command": shlex.join(command),
        "script": script_path, "submitted": datetime.now().isoformat(timespec="seconds"),
    })
    print(f"  submitted job {job_id}")


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

def submit_training(args: argparse.Namespace, entries: list[dict], train_options: list[str]) -> None:
    for option in RESERVED_TRAIN_OPTIONS:
        if any(arg == option or arg.startswith(option + "=") for arg in train_options):
            sys.exit(f"{option} is set by this script for every job; do not pass it.")
    epochs_parser = argparse.ArgumentParser(add_help=False)
    epochs_parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    epochs = epochs_parser.parse_known_args(train_options)[0].epochs
    print(f"Target: {epochs} epochs per model. Training options: {shlex.join(train_options) or '(defaults)'}")

    busy = busy_datasets(args.runs_dir, "train", args.dry_run)
    for entry in entries:
        name = dataset_name(entry)
        model_dir = os.path.join(args.runs_dir, name)
        command = [args.python, "train_kinematic_hardening.py",
                   "--dataset", os.path.join(args.sweep_dir, entry["file"])]
        print(f"{name}:", end=" ")
        if name in busy:
            print("job still queued or running, skipping")
            continue

        run_dir = None if args.fresh else latest_run(model_dir)
        checkpoint = latest_checkpoint(run_dir) if run_dir else None
        if checkpoint is None:
            print("new run")
            command += ["--runs-dir", model_dir]
        else:
            done = completed_epochs(checkpoint)
            if done >= epochs:
                print(f"finished ({done} epochs in {run_dir}), skipping")
                continue
            print(f"resuming {checkpoint} after {done} epochs")
            command += ["--resume", checkpoint]
        submit(args, "train", name, command + train_options)


def submit_evaluation(args: argparse.Namespace, entries: list[dict]) -> None:
    busy = busy_datasets(args.runs_dir, "evaluate", args.dry_run)
    for entry in entries:
        name = dataset_name(entry)
        print(f"{name}:", end=" ")
        run_dir = latest_run(os.path.join(args.runs_dir, name))
        if run_dir is None or not os.path.exists(os.path.join(run_dir, "checkpoints", args.checkpoint_name)):
            print(f"no {args.checkpoint_name} yet, skipping")
            continue
        if name in busy:
            print("job still queued or running, skipping")
            continue
        results = os.path.join(args.runs_dir, "cross_eval", f"{name}.npz")
        if os.path.exists(results) and not args.force:
            print("already evaluated, skipping (--force to redo)")
            continue
        print(f"evaluating {run_dir}")
        command = [args.python, "evaluate_ou_sweep.py", "--model-name", name,
                   "--sweep-dir", args.sweep_dir, "--runs-dir", args.runs_dir,
                   "--checkpoint-name", args.checkpoint_name]
        submit(args, "evaluate", name, command)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    shared = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    shared.add_argument("--sweep-dir", default="datasets/ou_sweep")
    shared.add_argument("--runs-dir", default="runs/ou_sweep")
    shared.add_argument("--only", nargs="+", metavar="NAME",
                        help="Only these datasets, e.g. theta1_sigma0.5.")
    shared.add_argument("--dry-run", action="store_true",
                        help="Write the job scripts and print the commands; submit nothing.")
    slurm = shared.add_argument_group("SLURM")
    slurm.add_argument("--partition", default=None)
    slurm.add_argument("--account", default=None)
    slurm.add_argument("--time", default="12:00:00", help="Time limit per job.")
    slurm.add_argument("--cpus", type=int, default=4, help="CPUs per job.")
    slurm.add_argument("--mem", default="8G", help="Memory per job.")
    slurm.add_argument("--setup", action="append", default=[], metavar="LINE",
                       help="Shell line run before Python, e.g. 'module load python' "
                       "or 'source venv/bin/activate'. Repeatable.")
    slurm.add_argument("--python", default="python", help="Python executable in the job.")

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    tasks = parser.add_subparsers(dest="task", required=True)
    tasks.add_parser(
        "train", parents=[shared], allow_abbrev=False,
        help="Train one model per dataset.",
        epilog="Other options (after --) go to train_kinematic_hardening.py.",
    ).add_argument("--fresh", action="store_true",
                   help="Start new runs even where a run exists.")
    evaluate = tasks.add_parser(
        "evaluate", parents=[shared], allow_abbrev=False,
        help="Evaluate every trained model on all datasets.",
    )
    evaluate.add_argument("--checkpoint-name", default="last.ckpt")
    evaluate.add_argument("--force", action="store_true",
                          help="Re-evaluate models that already have results.")

    args, rest = parser.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]
    if args.task != "train" and rest:
        parser.error(f"unrecognized arguments: {' '.join(rest)}")
    return args, rest


def main() -> None:
    args, train_options = parse_args()
    entries = load_index(args.sweep_dir)["datasets"]
    if args.only:
        names = {dataset_name(entry) for entry in entries}
        unknown = set(args.only) - names
        if unknown:
            sys.exit(f"Unknown datasets: {sorted(unknown)}. Available: {sorted(names)}")
        entries = [entry for entry in entries if dataset_name(entry) in args.only]
    os.makedirs(args.runs_dir, exist_ok=True)

    if args.task == "train":
        submit_training(args, entries, train_options)
    else:
        submit_evaluation(args, entries)


if __name__ == "__main__":
    main()
