"""Neural variational ODE model for trajectories of e(t).

The model represents the formulation

    sigma = W_e(e, xi),       d = W_xi(e, xi),
    xi_dot = D_{xi_dot}(-d),  xi(0) = 0,

where W and D are neural networks with scalar outputs and subscripts denote
derivatives with respect to their vector inputs.
"""

from __future__ import annotations

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


class InvertibleCurlFreeNetwork(nn.Module):
    """One-hidden-layer curl-free map from Appendix K.

    For ``x`` in R^n and hidden state ``v`` in R^h, the map is

    ``v = activation(W.T @ x + b)``
    ``y = (v * P) @ W.T + c``.

    Thus the input and output dimensions are both ``input_size`` while
    ``hidden_size`` controls the arbitrary hidden dimension.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if input_size < 1 or hidden_size < 1:
            raise ValueError("input_size and hidden_size must be positive.")

        self.W = nn.Parameter(torch.empty(input_size, hidden_size))
        nn.init.xavier_uniform_(self.W)

        self.b = nn.Parameter(torch.zeros(hidden_size))

        # Initialize the effective positive weights near one, rather than
        # initializing their unconstrained parameters near zero.  This avoids
        # a badly scaled map after the softplus transformation.
        positive_weight = torch.ones(
            hidden_size, dtype=self.W.dtype, device=self.W.device
        )
        self.raw_P = nn.Parameter(self.inverse_softplus(positive_weight))
        if bias:
            self.c: nn.Parameter | None = nn.Parameter(
                torch.zeros(input_size)
            )
        else:
            self.register_parameter("c", None)
        self.activation = nn.Softplus()

    @property
    def P(self) -> Tensor:
        """Return the strictly positive effective hidden-weight vector."""
        return self.weight_activation(self.raw_P)

    @staticmethod
    def inverse_softplus(value: Tensor) -> Tensor:
        """Map strictly positive effective weights to raw parameters."""
        if torch.any(value <= 0):
            raise ValueError("value must be strictly positive.")
        return value + torch.log(-torch.expm1(-value))

    def forward(self, x: Tensor) -> Tensor:
        """Evaluate the curl-free map for inputs ending in ``input_size``."""
        if x.shape[-1] != self.W.shape[0]:
            raise ValueError(
                f"Expected input dimension {self.W.shape[0]}, "
                f"got {x.shape[-1]}."
            )
        hidden = self.activation(
            torch.matmul(x, self.W) + self.b
        )
        output = torch.matmul(hidden * self.P, self.W.mT)
        if self.c is not None:
            output = output + self.c
        return output

    @staticmethod
    def weight_activation(x: Tensor) -> Tensor:
        """Constrain invertible-map weights to be strictly positive."""
        return torch.abs(x)


class Energy(nn.Module):
    """Direct elastic stress and inelastic driving-force maps.

    The associated potential split is
    ``W(e, xi) = W_elastic(e - xi) + W_inelastic(xi)``.
    The chain rule gives ``sigma = sigma_elastic(e - xi)`` and
    ``d = -sigma + d_inelastic(xi)``.
    """

    def __init__(
        self,
        e_dim: int = 1,
        xi_dim: int = 1,
        hidden_size: int = 64,
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
        self.elastic_network = InvertibleCurlFreeNetwork(
            e_dim, hidden_size
        )
        self.inelastic_network = InvertibleCurlFreeNetwork(
            xi_dim, hidden_size
        )

    def forward(self, e: Tensor, xi: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(sigma, d)`` for matching strain and internal state."""
        if e.shape[:-1] != xi.shape[:-1]:
            raise ValueError("e and xi must have matching leading dimensions.")
        if e.shape[-1] != self.e_dim or xi.shape[-1] != self.xi_dim:
            raise ValueError(
                f"Expected e and xi dimensions ({self.e_dim}, {self.xi_dim}), "
                f"got ({e.shape[-1]}, {xi.shape[-1]})."
            )
        sigma = self.elastic_network(e - xi)
        d = -sigma + self.inelastic_network(xi)
        return sigma, d


class Dissipation(nn.Module):
    """Direct dissipation-rate map evaluated at ``-d``."""

    def __init__(
        self,
        xi_dim: int = 1,
        hidden_size: int = 64,
    ) -> None:
        super().__init__()
        if xi_dim < 1:
            raise ValueError("xi_dim must be positive.")
        self.xi_dim = xi_dim
        self.network = InvertibleCurlFreeNetwork(
            xi_dim, hidden_size
        )

    def forward(self, negative_d: Tensor) -> Tensor:
        """Return ``xi_dot`` evaluated at the force ``-d``."""
        if negative_d.shape[-1] != self.xi_dim:
            raise ValueError(
                f"Expected dissipation dimension {self.xi_dim}, "
                f"got {negative_d.shape[-1]}."
            )
        return self.network(negative_d)


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

        The ODE is integrated with forward Euler. ``create_graph=True`` keeps
        the direct-map computation graph available for ordinary backprop.  The
        argument is retained for call compatibility; direct maps do not need
        nested input differentiation.
        """
        _ = create_graph
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
            sigma_t, d_t = self.energy(e_t, xi_t)
            xi_dot = self.dissipation(-d_t)
            sigma_paths.append(sigma_t)
            xi = xi_t + dt * xi_dot
            xi_paths.append(xi)

        # Evaluate sigma at the final time after xi has been advanced there.
        e_final = self._differentiable_input(e_trajectories[:, -1])
        xi_final = self._differentiable_input(xi)
        sigma_final, _ = self.energy(e_final, xi_final)
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
            sigma_t, d_t = self.energy(e_t, xi_t)
            xi_dot = self.dissipation(-d_t)
            xi = (xi_t + dt * xi_dot).detach()
            sigma_paths.append(sigma_t.detach())
            xi_paths.append(xi)

        e_final = e_trajectories[:, -1].detach().requires_grad_(True)
        xi_final = xi.detach().requires_grad_(True)
        sigma_final, _ = self.energy(e_final, xi_final)
        sigma_paths.append(sigma_final.detach())

        xi_path = torch.stack(xi_paths, dim=1)
        sigma_path = torch.stack(sigma_paths, dim=1)
        parameter_gradients = [
            torch.zeros_like(parameter) for parameter in parameters
        ]

        # Terminal adjoint: lambda(T) = d loss(T) / d xi(T).
        xi_state = xi_path[:, -1].detach().requires_grad_(True)
        e_state = e_trajectories[:, -1].detach().requires_grad_(True)
        sigma_state, _ = self.energy(e_state, xi_state)
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
            sigma_state, d_state = self.energy(e_state, xi_state)
            xi_dot = self.dissipation(-d_state)
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
