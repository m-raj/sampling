"""Neural variational ODE model for trajectories of e(t).

The model represents the formulation

    sigma = W_e(e, xi),       d = W_xi(e, xi),
    xi_dot = D_{xi_dot}(-d),  xi(0) = 0,

where W and D are neural networks with scalar outputs and subscripts denote
derivatives with respect to their vector inputs.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

class ReLUS(nn.Module):
    """ReLU activation with a small slope for negative inputs."""

    def __init__(self, negative_slope: float = 0.01) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.where(
            x >= 0, x**2, 0
        )

def _mlp(input_size: int, hidden_sizes: Sequence[int], bias: bool = True) -> nn.Sequential:
    """Create a scalar-output multilayer perceptron."""
    layers: list[nn.Module] = []
    previous_size = input_size
    for hidden_size in hidden_sizes:
        layers.extend((nn.Linear(previous_size, hidden_size), ReLUS()))
        previous_size = hidden_size
    layers.append(nn.Linear(previous_size, 1, bias=bias))
    return nn.Sequential(*layers)


class _PositiveLinear(nn.Module):
    """Linear layer with non-negative weights."""

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.raw_weight = nn.Parameter(torch.empty(output_size, input_size))
        self.bias = nn.Parameter(torch.empty(output_size))
        nn.init.xavier_uniform_(self.raw_weight)
        nn.init.zeros_(self.bias)
        self.activation = ReLUS()

    def forward(self, x: Tensor) -> Tensor:
        return nn.functional.linear(
            x, nn.functional.softplus(self.raw_weight), self.bias
        )


class InputConvexNetwork(nn.Module):
    """Scalar input-convex neural network."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int] = (64,),
        *,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if input_size < 1:
            raise ValueError("input_size must be positive.")
        if any(hidden_size < 1 for hidden_size in hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integers.")

        hidden_sizes = tuple(hidden_sizes)
        self.input_layers = nn.ModuleList()
        self.state_layers = nn.ModuleList()
        previous_size = input_size
        for layer_index, hidden_size in enumerate(hidden_sizes):
            self.input_layers.append(
                nn.Linear(input_size, hidden_size, bias=bias)
            )
            if layer_index == 0:
                self.state_layers.append(nn.Identity())
            else:
                self.state_layers.append(
                    _PositiveLinear(previous_size, hidden_size)
                )
            previous_size = hidden_size

        self.output_input = nn.Linear(input_size, 1, bias=bias)
        self.output_state = (
            _PositiveLinear(previous_size, 1) if hidden_sizes else None
        )

    def forward(self, x: Tensor) -> Tensor:
        state: Tensor | None = None
        for input_layer, state_layer in zip(
            self.input_layers, self.state_layers
        ):
            input_part = input_layer(x)
            if state is None:
                state = input_part
            else:
                state = input_part + state_layer(state)
            state = torch.where(state >= 0, state**2, torch.zeros_like(state))

        output = self.output_input(x)
        if state is not None and self.output_state is not None:
            output = output + self.output_state(state)
        return output.squeeze(-1)


class InvertibleNeuralNetwork(nn.Module):
    """Single-layer curl-free map with convexity-preserving initialization.

    When ``invertible`` is true, the hidden weights are parameterized by a
    smooth positive function, as required for a convex potential gradient.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        bias: bool = True,
        invertible: bool = False,
    ) -> None:
        super().__init__()
        if input_size < 1 or hidden_size < 1:
            raise ValueError("input_size and hidden_size must be positive.")

        self.A = nn.Parameter(torch.empty(input_size, hidden_size))
        nn.init.xavier_uniform_(self.A)

        if bias:
            self.bias: nn.Parameter | None = nn.Parameter(
                torch.zeros(hidden_size)
            )
        else:
            self.register_parameter("bias", None)

        # Initialize the effective positive weights near one, rather than
        # initializing their unconstrained parameters near zero.  This avoids
        # a badly scaled map after the softplus transformation.
        positive_weight = torch.ones(hidden_size, dtype=torch.float32)
        self.P = nn.Parameter(self.inverse_softplus(positive_weight))
        self.activation = nn.Softplus()
        self.invertible = invertible

    @staticmethod
    def inverse_softplus(value: Tensor) -> Tensor:
        """Map strictly positive effective weights to raw parameters."""
        if torch.any(value <= 0):
            raise ValueError("value must be strictly positive.")
        return value + torch.log(-torch.expm1(-value))

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate the curl-free map for inputs ending in ``input_size``."""
        if x.shape[-1] != self.A.shape[0]:
            raise ValueError(
                f"Expected input dimension {self.A.shape[0]}, "
                f"got {x.shape[-1]}."
            )
        hidden = self.activation(
            torch.matmul(x, self.A)
            + (self.bias if self.bias is not None else 0.0)
        )
        weights = self.weight_activation(self.P) if self.invertible else self.P
        return torch.matmul(hidden * weights, self.A.mT)

    @staticmethod
    def weight_activation(x: Tensor) -> Tensor:
        """Constrain invertible-map weights to be strictly positive."""
        return nn.functional.softplus(x)


class Energy(nn.Module):
    """Neural scalar energy split into elastic and inelastic contributions.

    The energy is represented as

    ``W(e, xi) = W_elastic(e - xi) + W_inelastic(xi)``.
    """

    def __init__(
        self,
        e_dim: int = 1,
        xi_dim: int = 1,
        hidden_sizes: Sequence[int] = (64,),
    ) -> None:
        super().__init__()
        if e_dim < 1 or xi_dim < 1:
            raise ValueError("e_dim and xi_dim must be positive.")
        if e_dim != xi_dim:
            raise ValueError(
                "e_dim and xi_dim must match because the elastic network "
                "receives e - xi."
            )
        self.e_dim = e_dim
        self.xi_dim = xi_dim
        self.elastic_network = InputConvexNetwork(
            input_size=e_dim, hidden_sizes=hidden_sizes, bias=False
        )
        self.inelastic_network = InputConvexNetwork(
            input_size=xi_dim, hidden_sizes=hidden_sizes, bias=False
        )

    def forward(self, e: Tensor, xi: Tensor) -> Tensor:
        """Evaluate W(e, xi), preserving all leading batch dimensions."""
        if e.shape[:-1] != xi.shape[:-1]:
            raise ValueError("e and xi must have matching leading dimensions.")
        if e.shape[-1] != self.e_dim or xi.shape[-1] != self.xi_dim:
            raise ValueError(
                f"Expected e and xi dimensions ({self.e_dim}, {self.xi_dim}), "
                f"got ({e.shape[-1]}, {xi.shape[-1]})."
            )
        elastic_energy = self.elastic_network(e - xi).squeeze(-1)
        inelastic_energy = self.inelastic_network(xi).squeeze(-1)
        return elastic_energy + inelastic_energy

    def derivatives(
        self, e: Tensor, xi: Tensor, *, create_graph: bool
    ) -> tuple[Tensor, Tensor]:
        """Return (W_e, W_xi). Inputs must have ``requires_grad=True``."""
        energy = self(e, xi)
        return torch.autograd.grad(
            energy.sum(), (e, xi), create_graph=create_graph
        )


class Dissipation(nn.Module):
    """Neural scalar dissipation potential D(xi_dot)."""

    def __init__(
        self, xi_dim: int = 1, hidden_sizes: Sequence[int] = (64,)
    ) -> None:
        super().__init__()
        if xi_dim < 1:
            raise ValueError("xi_dim must be positive.")
        self.xi_dim = xi_dim
        self.network = InputConvexNetwork(
            input_size=xi_dim, hidden_sizes=hidden_sizes
        )

    def forward(self, xi_dot: Tensor) -> Tensor:
        """Evaluate D(xi_dot), preserving all leading batch dimensions."""
        if xi_dot.shape[-1] != self.xi_dim:
            raise ValueError(
                f"Expected xi_dot dimension {self.xi_dim}, got {xi_dot.shape[-1]}."
            )
        return self.network(xi_dot).squeeze(-1)

    def derivative(self, xi_dot: Tensor, *, create_graph: bool) -> Tensor:
        """Return D_xi_dot evaluated at ``xi_dot``."""
        dissipation = self(xi_dot)
        (derivative,) = torch.autograd.grad(
            dissipation.sum(), xi_dot, create_graph=create_graph
        )
        return derivative


class VariationalODESolver(nn.Module):
    """Integrate the neural variational ODE for a batch of e trajectories."""

    def __init__(self, energy: Energy, dissipation: Dissipation) -> None:
        super().__init__()
        if energy.e_dim < 1 or energy.xi_dim != dissipation.xi_dim:
            raise ValueError("Energy and dissipation dimensions are incompatible.")
        self.energy = energy
        self.dissipation = dissipation

    @staticmethod
    def _differentiable_input(value: Tensor) -> Tensor:
        """Ensure an input can be differentiated without mutating caller data."""
        if value.requires_grad:
            return value
        return value.clone().requires_grad_(True)

    def solve(
        self,
        e_trajectories: Tensor,
        dt: float = 0.01,
        *,
        create_graph: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Return xi and sigma for e trajectories of shape ``(batch, time, e_dim)``.

        The ODE is integrated with forward Euler. ``create_graph=True`` is
        required during training so gradients can flow through W's and D's
        partial derivatives to their network parameters.
        """
        scalar_input = e_trajectories.ndim == 2
        if scalar_input:
            e_trajectories = e_trajectories.unsqueeze(-1)
        if e_trajectories.ndim != 3:
            raise ValueError(
                "e_trajectories must have shape (batch, time, e_dim)."
            )
        if e_trajectories.shape[1] < 1:
            raise ValueError("e_trajectories must contain at least one time point.")
        if dt <= 0:
            raise ValueError("dt must be positive.")
        if e_trajectories.shape[-1] != self.energy.e_dim:
            raise ValueError(
                f"Expected e dimension {self.energy.e_dim}, "
                f"got {e_trajectories.shape[-1]}."
            )

        batch_size, n_times, _ = e_trajectories.shape
        xi = e_trajectories.new_zeros(batch_size, self.energy.xi_dim)
        xi_paths: list[Tensor] = [xi]
        sigma_paths: list[Tensor] = []

        for time_index in range(n_times - 1):
            e_t = self._differentiable_input(e_trajectories[:, time_index])
            xi_t = self._differentiable_input(xi)
            sigma_t, d_t = self.energy.derivatives(
                e_t, xi_t, create_graph=create_graph
            )
            negative_d = self._differentiable_input(-d_t)
            xi_dot = self.dissipation.derivative(
                negative_d, create_graph=create_graph
            )
            sigma_paths.append(sigma_t)
            xi = xi_t + dt * xi_dot
            xi_paths.append(xi)

        # Evaluate sigma at the final time after xi has been advanced there.
        e_final = self._differentiable_input(e_trajectories[:, -1])
        xi_final = self._differentiable_input(xi)
        sigma_final, _ = self.energy.derivatives(
            e_final, xi_final, create_graph=create_graph
        )
        sigma_paths.append(sigma_final)

        xi_result = torch.stack(xi_paths, dim=1)
        sigma_result = torch.stack(sigma_paths, dim=1)
        if scalar_input and self.energy.xi_dim == 1:
            xi_result = xi_result.squeeze(-1)
        if scalar_input and self.energy.e_dim == 1:
            sigma_result = sigma_result.squeeze(-1)
        return xi_result, sigma_result

    def compute_adjoint_gradients(
        self,
        e_trajectories: Tensor,
        sigma_target: Tensor,
        dt: float = 0.01,
    ) -> tuple[Tensor, tuple[Tensor, ...], Tensor, Tensor]:
        """Compute parameter gradients with a discrete adjoint sweep.

        The forward Euler state equation used by :meth:`solve` is
        differentiated with its discrete adjoint.  Forward states and model
        outputs are detached before the backward sweep; each individual time
        step is then recomputed with autograd and released immediately.
        Consequently, the complete forward computational graph is never
        retained.

        The objective is the forward-Euler quadrature

        ``0.5 * dt * sum((sigma - sigma_target)**2)``.

        Parameters
        ----------
        e_trajectories : Tensor
            Strain trajectories with shape ``(batch, time, e_dim)`` or
            ``(batch, time)`` when ``e_dim == 1``.
        sigma_target : Tensor
            Target stresses with the same shape as the returned stress
            trajectory.
        dt : float, default=0.01
            Forward Euler time step.

        Returns
        -------
        loss : Tensor
            Detached value of the quadrature objective.
        gradients : tuple[Tensor, ...]
            Gradients aligned with ``tuple(self.parameters())``.  They are
            detached tensors and are not assigned to ``parameter.grad``.
        xi : Tensor
            Detached inelastic-variable trajectory.
        sigma : Tensor
            Detached stress trajectory.
        """
        if dt <= 0:
            raise ValueError("dt must be positive.")

        scalar_input = e_trajectories.ndim == 2
        if scalar_input:
            e_trajectories = e_trajectories.unsqueeze(-1)
        if e_trajectories.ndim != 3:
            raise ValueError(
                "e_trajectories must have shape (batch, time, e_dim)."
            )
        if e_trajectories.shape[1] < 1:
            raise ValueError("e_trajectories must contain at least one time point.")
        if e_trajectories.shape[-1] != self.energy.e_dim:
            raise ValueError(
                f"Expected e dimension {self.energy.e_dim}, "
                f"got {e_trajectories.shape[-1]}."
            )

        if scalar_input:
            target = sigma_target.unsqueeze(-1)
        else:
            target = sigma_target
        if target.shape != (
            e_trajectories.shape[0],
            e_trajectories.shape[1],
            self.energy.e_dim,
        ):
            raise ValueError("sigma_target must match the stress trajectory shape.")

        parameters = tuple(self.parameters())
        xi = e_trajectories.new_zeros(
            e_trajectories.shape[0], self.energy.xi_dim
        )
        xi_paths: list[Tensor] = [xi.detach()]
        sigma_paths: list[Tensor] = []

        # Build and discard one graph per forward step.  Detached states are
        # sufficient because the adjoint sweep reconstructs state sensitivities.
        for time_index in range(e_trajectories.shape[1] - 1):
            e_t = e_trajectories[:, time_index].detach().requires_grad_(True)
            xi_t = xi.detach().requires_grad_(True)
            sigma_t, d_t = self.energy.derivatives(
                e_t, xi_t, create_graph=True
            )
            xi_dot = self.dissipation.derivative(
                (-d_t).requires_grad_(True), create_graph=True
            )
            xi = (xi_t + dt * xi_dot).detach()
            sigma_paths.append(sigma_t.detach())
            xi_paths.append(xi)

        e_final = e_trajectories[:, -1].detach().requires_grad_(True)
        xi_final = xi.detach().requires_grad_(True)
        sigma_final, _ = self.energy.derivatives(
            e_final, xi_final, create_graph=True
        )
        sigma_paths.append(sigma_final.detach())

        xi_path = torch.stack(xi_paths, dim=1)
        sigma_path = torch.stack(sigma_paths, dim=1)
        parameter_gradients = [
            torch.zeros_like(parameter) for parameter in parameters
        ]

        # Terminal adjoint: lambda(T) = d loss(T) / d xi(T).
        xi_state = xi_path[:, -1].detach().requires_grad_(True)
        e_state = e_trajectories[:, -1].detach().requires_grad_(True)
        sigma_state, _ = self.energy.derivatives(
            e_state, xi_state, create_graph=True
        )
        terminal_loss = 0.5 * dt * (sigma_state - target[:, -1]).square().sum()
        terminal_gradients = torch.autograd.grad(
            terminal_loss,
            (xi_state, *parameters),
            allow_unused=True,
            retain_graph=False,
        )
        adjoint = terminal_gradients[0].detach()
        for index, gradient in enumerate(terminal_gradients[1:]):
            if gradient is not None:
                parameter_gradients[index].add_(gradient.detach())

        # Discrete backward adjoint:
        # lambda_k = d(l_k + lambda_{k+1}^T F_k)/d xi_k,
        # dJ/dtheta += d(l_k + lambda_{k+1}^T F_k)/d theta.
        for time_index in range(e_trajectories.shape[1] - 2, -1, -1):
            e_state = e_trajectories[:, time_index].detach().requires_grad_(True)
            xi_state = xi_path[:, time_index].detach().requires_grad_(True)
            sigma_state, d_state = self.energy.derivatives(
                e_state, xi_state, create_graph=True
            )
            xi_dot = self.dissipation.derivative(
                (-d_state).requires_grad_(True), create_graph=True
            )
            xi_next = xi_state + dt * xi_dot
            local_loss = 0.5 * dt * (
                sigma_state - target[:, time_index]
            ).square().sum()
            augmented_local_loss = local_loss + torch.sum(adjoint.detach() * xi_next)
            local_gradients = torch.autograd.grad(
                augmented_local_loss,
                (xi_state, *parameters),
                allow_unused=True,
                retain_graph=False,
            )
            adjoint = local_gradients[0].detach()
            for index, gradient in enumerate(local_gradients[1:]):
                if gradient is not None:
                    parameter_gradients[index].add_(gradient.detach())

        loss = 0.5 * dt * (sigma_path - target.detach()).square().sum()
        if scalar_input and self.energy.xi_dim == 1:
            xi_path = xi_path.squeeze(-1)
        if scalar_input and self.energy.e_dim == 1:
            sigma_path = sigma_path.squeeze(-1)
        return (
            loss.detach(),
            tuple(parameter_gradients),
            xi_path,
            sigma_path,
        )
