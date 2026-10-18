"""Solve a displacement-controlled 1D bar with a learned constitutive law.

The bar occupies ``(0, 1)`` and uses linear finite elements. The left end is
fixed and the right-end displacement increases from zero to one, so the
prescribed macroscopic strain is the load parameter. The constitutive model is
the trained ``VariationalODESolver`` from ``neural_variational_ode.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import root

try:
    import fenics as df
except ImportError as exc:  # pragma: no cover - depends on the local FEM stack
    raise ImportError(
        "This solver requires legacy FEniCS/DOLFIN: install the fenics package."
    ) from exc

from src.neural_variational_ode import Dissipation, Energy, VariationalODESolver
from src.true_constitutive_law import TrueConstitutiveLaw


class NeuralConstitutiveLaw:
    """Evaluate learned stress at all mesh nodes for a strain history."""

    def __init__(self, checkpoint: Path, node_count: int, model_dt: float) -> None:
        if model_dt <= 0:
            raise ValueError("model_dt must be positive.")
        self.solver = VariationalODESolver(Energy(1, 1), Dissipation(1))
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.solver.load_state_dict(state)
        self.solver.eval()
        self.node_count = node_count
        self.model_dt = model_dt

    def stress(self, strain_history: np.ndarray) -> np.ndarray:
        """Return the final learned stress at every node."""
        if strain_history.ndim != 2 or strain_history.shape[1] != self.node_count:
            raise ValueError("strain_history must have shape (time, node_count).")
        paths = torch.as_tensor(strain_history.T, dtype=torch.float32)
        with torch.enable_grad():
            _, stress = self.solver.solve(
                paths, dt=self.model_dt, create_graph=False
            )
        return stress[:, -1].detach().cpu().numpy().reshape(self.node_count)


class NeuralBarProblem:
    """FEniCS mesh and weak residual for the displacement-controlled bar."""

    def __init__(
        self,
        node_count: int,
        constitutive_law: NeuralConstitutiveLaw,
        body_force: float,
    ):
        self.mesh = df.IntervalMesh(node_count - 1, 0.0, 1.0)
        self.space = df.FunctionSpace(self.mesh, "CG", 1)
        self.test_function = df.TestFunction(self.space)
        self.stress_function = df.Function(self.space)
        self.constitutive_law = constitutive_law
        self.node_count = node_count
        self.body_force = df.Constant(body_force)
        self.previous_strains = np.zeros((1, node_count), dtype=np.float64)
        self.previous_displacement = np.zeros(node_count, dtype=np.float64)

    def nodal_strain(self, displacement: np.ndarray) -> np.ndarray:
        """Recover nodal strain by averaging adjacent P1 element gradients."""
        gradients = np.diff(displacement) * (self.node_count - 1)
        strain = np.empty(self.node_count, dtype=np.float64)
        strain[0] = gradients[0]
        strain[-1] = gradients[-1]
        strain[1:-1] = 0.5 * (gradients[:-1] + gradients[1:])
        return strain

    def residual(self, free_displacement: np.ndarray, load: float) -> np.ndarray:
        displacement = np.empty(self.node_count, dtype=np.float64)
        displacement[0] = 0.0
        displacement[-1] = load
        displacement[1:-1] = free_displacement

        current_strain = self.nodal_strain(displacement)
        history = np.vstack((self.previous_strains, current_strain))
        stress = self.constitutive_law.stress(history)
        self.stress_function.vector()[:] = stress
        weak_residual = df.assemble(
            (
                self.stress_function * self.test_function.dx(0)
                - self.body_force * self.test_function
            )
            * df.dx
        )
        return weak_residual.get_local()[1:-1]

    def solve_load_step(self, load: float) -> np.ndarray:
        """Solve internal equilibrium for one prescribed end displacement."""
        guess = self.previous_displacement[1:-1].copy()
        solution = root(
            lambda values: self.residual(values, load), guess, method="hybr"
        )
        residual_norm = np.linalg.norm(solution.fun, ord=np.inf)
        if not solution.success or residual_norm > 1e-7:
            raise RuntimeError(
                f"FEM solve failed at load {load:.6g}: {solution.message}; "
                f"residual={residual_norm:.3e}"
            )
        displacement = np.concatenate(([0.0], solution.x, [load]))
        self.previous_displacement = displacement
        self.previous_strains = np.vstack(
            (self.previous_strains, self.nodal_strain(displacement))
        )
        return displacement


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=51)
    parser.add_argument("--load-steps", type=int, default=10)
    parser.add_argument(
        "--constitutive-law",
        choices=("neural", "true"),
        default="true",
        help="Constitutive model used by FEM (default: neural).",
    )
    parser.add_argument(
        "--body-force",
        type=float,
        default=1.0,
        help="Constant body force b in -d(sigma)/dx = b (default: 1).",
    )
    parser.add_argument(
        "--model-dt",
        type=float,
        default=0.01,
        help="Time step used when training the neural variational ODE.",
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("one-d/neural_variational_ode_1.pt")
    )
    parser.add_argument("--output", type=Path, default=Path("neural_bar_results.npz"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.nodes < 3 or args.load_steps < 1:
        raise ValueError("--nodes must be at least 3 and --load-steps must be positive.")
    if args.constitutive_law == "neural" and not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")

    if args.constitutive_law == "true":
        law = TrueConstitutiveLaw(args.nodes, args.model_dt)
    else:
        law = NeuralConstitutiveLaw(args.checkpoint, args.nodes, args.model_dt)
    problem = NeuralBarProblem(args.nodes, law, args.body_force)
    loads = np.linspace(0.0, 1.0, args.load_steps + 1)
    displacements = [problem.previous_displacement.copy()]
    for load in loads[1:]:
        displacements.append(problem.solve_load_step(float(load)))

    displacement_history = np.asarray(displacements)
    strain_history = np.asarray(
        [problem.nodal_strain(displacement) for displacement in displacement_history]
    )
    stress_history = np.asarray(
        [law.stress(strain_history[: index + 1]) for index in range(len(loads))]
    )
    np.savez(
        args.output,
        x=np.linspace(0.0, 1.0, args.nodes),
        load=loads,
        displacement=displacement_history,
        strain=strain_history,
        stress=stress_history,
    )
    print(f"Saved {len(loads)} load steps to {args.output}")


if __name__ == "__main__":
    main()