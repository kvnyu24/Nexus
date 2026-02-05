"""GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection.

Reference:
    Zhao, J., et al. "GaLore: Memory-Efficient LLM Training by Gradient
    Low-Rank Projection." ICML 2024. https://arxiv.org/abs/2403.03507

GaLore projects gradients into a low-rank subspace via SVD to reduce
optimizer memory. Unlike LoRA, which constrains the weight update to a
fixed low-rank form, GaLore projects the full-rank gradient at each step
and only maintains optimizer states (momentum, variance) for the projected
low-rank gradient. The projection matrices are updated periodically to
track the changing gradient subspace throughout training.

Key advantages over LoRA:
    - Full-rank weight updates (not constrained to rank-r subspace)
    - Reduced optimizer state memory (up to 65% reduction)
    - Compatible with any existing optimizer (Adam, SGD, etc.)
    - No architectural changes to the model
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Type
import math

from nexus.core.base import NexusModule


@dataclass
class GaLoreConfig:
    """Configuration for GaLore gradient projection.

    Attributes:
        rank: Rank of the gradient projection subspace. Controls the
            trade-off between memory savings and approximation quality.
        update_interval: Number of training steps between projection
            matrix updates via SVD. More frequent updates track the
            gradient subspace better but add computation overhead.
        scale: Scaling factor applied to the projected gradient.
        proj_type: Projection type. "std" for standard projection,
            "reverse_std" for projection from the right side, or
            "full" for both left and right projections.
    """
    rank: int = 128
    update_interval: int = 200
    scale: float = 1.0
    proj_type: str = "std"


class GaLoreProjector(NexusModule):
    """Projects gradients into a low-rank subspace using SVD.

    Given a gradient matrix G of shape (m, n), the projector computes
    the truncated SVD to obtain a rank-r projection matrix P, then
    projects the gradient as:
        - "std":         P.T @ G        (left projection, when m >= n)
        - "reverse_std": G @ P.T        (right projection, when m < n)
        - "full":        P_left.T @ G @ P_right  (both sides)

    The projection matrices are updated every `update_interval` steps
    to track the evolving gradient subspace.

    Args:
        rank: Target rank for the projection.
        update_interval: Steps between SVD re-computation.
        scale: Scaling factor for projected gradients.
        proj_type: Projection strategy ("std", "reverse_std", or "full").
    """

    def __init__(
        self,
        rank: int = 128,
        update_interval: int = 200,
        scale: float = 1.0,
        proj_type: str = "std",
    ):
        config = {
            "rank": rank,
            "update_interval": update_interval,
            "scale": scale,
            "proj_type": proj_type,
        }
        super().__init__(config)

        self.rank = rank
        self.update_interval = update_interval
        self.scale = scale
        self.proj_type = proj_type

        # Projection matrices (lazily initialized)
        self.ortho_matrix: Optional[torch.Tensor] = None
        self.ortho_matrix_right: Optional[torch.Tensor] = None
        self._step_count = 0

    def _compute_projection(self, gradient: torch.Tensor) -> None:
        """Compute the SVD-based projection matrix from the current gradient.

        For a gradient G of shape (m, n):
            - If proj_type is "std", compute left singular vectors U[:, :rank].
            - If proj_type is "reverse_std", compute right singular vectors V[:, :rank].
            - If proj_type is "full", compute both U and V truncations.

        Args:
            gradient: The full-rank gradient tensor of shape (m, n).
        """
        with torch.no_grad():
            if gradient.dim() != 2:
                gradient = gradient.reshape(gradient.shape[0], -1)

            m, n = gradient.shape
            device = gradient.device
            dtype = gradient.dtype

            # Use float32 for SVD numerical stability
            grad_fp32 = gradient.float()

            if self.proj_type == "std":
                # Left projection: use top-r left singular vectors
                U, _, _ = torch.linalg.svd(grad_fp32, full_matrices=False)
                self.ortho_matrix = U[:, :self.rank].to(dtype=dtype, device=device)

            elif self.proj_type == "reverse_std":
                # Right projection: use top-r right singular vectors
                _, _, Vh = torch.linalg.svd(grad_fp32, full_matrices=False)
                self.ortho_matrix = Vh[:self.rank, :].T.to(dtype=dtype, device=device)

            elif self.proj_type == "full":
                U, _, Vh = torch.linalg.svd(grad_fp32, full_matrices=False)
                self.ortho_matrix = U[:, :self.rank].to(dtype=dtype, device=device)
                self.ortho_matrix_right = Vh[:self.rank, :].T.to(
                    dtype=dtype, device=device
                )

    def project(self, gradient: torch.Tensor) -> torch.Tensor:
        """Project a gradient into the low-rank subspace.

        If the projection matrices are stale (or uninitialized), a new
        SVD is computed first.

        Args:
            gradient: Full-rank gradient of shape (m, n) or higher-dim.

        Returns:
            Projected gradient in the low-rank subspace.
        """
        original_shape = gradient.shape
        if gradient.dim() != 2:
            gradient = gradient.reshape(gradient.shape[0], -1)

        # Recompute projection if needed
        if self.ortho_matrix is None or self._step_count % self.update_interval == 0:
            self._compute_projection(gradient)

        self._step_count += 1

        if self.proj_type == "std":
            # P.T @ G -> (rank, n)
            projected = self.ortho_matrix.T @ gradient
        elif self.proj_type == "reverse_std":
            # G @ P -> (m, rank)
            projected = gradient @ self.ortho_matrix
        elif self.proj_type == "full":
            # P_left.T @ G @ P_right -> (rank, rank)
            projected = self.ortho_matrix.T @ gradient @ self.ortho_matrix_right
        else:
            raise ValueError(f"Unknown projection type: {self.proj_type}")

        return projected * self.scale

    def project_back(self, low_rank_gradient: torch.Tensor) -> torch.Tensor:
        """Reconstruct the full-rank gradient from the projected representation.

        This is the inverse projection that maps the optimizer-updated
        low-rank gradient back to the original parameter shape.

        Args:
            low_rank_gradient: Projected gradient from the subspace.

        Returns:
            Full-rank gradient approximation.
        """
        if self.proj_type == "std":
            # P @ projected -> (m, n)
            return (self.ortho_matrix @ low_rank_gradient) * self.scale
        elif self.proj_type == "reverse_std":
            # projected @ P.T -> (m, n)
            return (low_rank_gradient @ self.ortho_matrix.T) * self.scale
        elif self.proj_type == "full":
            return (
                self.ortho_matrix @ low_rank_gradient @ self.ortho_matrix_right.T
            ) * self.scale
        else:
            raise ValueError(f"Unknown projection type: {self.proj_type}")

    def forward(self, gradient: torch.Tensor) -> torch.Tensor:
        """Project gradient into low-rank subspace (alias for project).

        Args:
            gradient: The full-rank gradient tensor.

        Returns:
            Low-rank projected gradient.
        """
        return self.project(gradient)


class GaLoreOptimizer:
    """Optimizer wrapper that applies GaLore gradient projection.

    Wraps any PyTorch optimizer to maintain optimizer states only in
    the low-rank projected gradient space. This reduces the memory
    footprint of optimizer states (e.g., Adam's first and second
    moments) proportionally to the projection rank.

    Usage:
        base_optimizer = torch.optim.AdamW
        optimizer = GaLoreOptimizer(
            model.parameters(),
            base_optimizer,
            lr=1e-3,
            galore_config=GaLoreConfig(rank=128),
        )
        # Standard training loop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    Args:
        params: Iterable of parameter groups. Each group can optionally
            contain a "galore_config" key for per-group projection settings.
        optimizer_cls: The base optimizer class (e.g., torch.optim.AdamW).
        galore_config: Default GaLore configuration for all parameter groups
            that do not specify their own.
        **optimizer_kwargs: Arguments passed to the base optimizer (lr, etc.).
    """

    def __init__(
        self,
        params,
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.AdamW,
        galore_config: Optional[GaLoreConfig] = None,
        **optimizer_kwargs,
    ):
        self.galore_config = galore_config or GaLoreConfig()
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs

        # Process parameter groups
        self.param_groups: List[Dict[str, Any]] = []
        self.projectors: Dict[int, GaLoreProjector] = {}
        self._galore_params: Dict[int, bool] = {}

        self._process_params(params)

        # Create projected parameter placeholders for the base optimizer
        self._projected_params: Dict[int, torch.Tensor] = {}
        self._setup_optimizer()

    def _process_params(self, params) -> None:
        """Parse parameters into groups and assign projectors.

        Parameters with 2D weights (linear layers) are assigned GaLore
        projectors. 1D parameters (biases, norms) bypass projection.
        """
        if isinstance(params, dict):
            params = [params]

        param_list = []
        for group_or_param in params:
            if isinstance(group_or_param, dict):
                for p in group_or_param.get("params", []):
                    param_list.append(p)
            elif isinstance(group_or_param, torch.Tensor):
                param_list.append(group_or_param)
            elif isinstance(group_or_param, nn.Parameter):
                param_list.append(group_or_param)
            else:
                # It may be a generator
                for p in group_or_param:
                    param_list.append(p)

        galore_params = []
        regular_params = []

        for param in param_list:
            pid = id(param)
            if param.dim() >= 2 and param.requires_grad:
                self._galore_params[pid] = True
                self.projectors[pid] = GaLoreProjector(
                    rank=self.galore_config.rank,
                    update_interval=self.galore_config.update_interval,
                    scale=self.galore_config.scale,
                    proj_type=self.galore_config.proj_type,
                )
                galore_params.append(param)
            else:
                self._galore_params[pid] = False
                regular_params.append(param)

        self.param_groups = [
            {"params": galore_params, "galore": True},
            {"params": regular_params, "galore": False},
        ]

    def _setup_optimizer(self) -> None:
        """Create the base optimizer with appropriate parameter groups."""
        opt_groups = []
        for group in self.param_groups:
            if group["galore"]:
                # For GaLore params, create projected placeholders
                projected = []
                for param in group["params"]:
                    pid = id(param)
                    projector = self.projectors[pid]
                    if param.dim() >= 2:
                        m, n = param.shape[0], param.reshape(param.shape[0], -1).shape[1]
                        rank = min(projector.rank, m, n)

                        if projector.proj_type == "std":
                            shape = (rank, n)
                        elif projector.proj_type == "reverse_std":
                            shape = (m, rank)
                        elif projector.proj_type == "full":
                            shape = (rank, rank)
                        else:
                            shape = (rank, n)
                    else:
                        shape = param.shape

                    placeholder = torch.zeros(
                        shape, dtype=param.dtype, device=param.device, requires_grad=True
                    )
                    self._projected_params[pid] = placeholder
                    projected.append(placeholder)

                opt_groups.append({"params": projected, **self.optimizer_kwargs})
            else:
                if group["params"]:
                    opt_groups.append({"params": group["params"], **self.optimizer_kwargs})

        if opt_groups:
            self.base_optimizer = self.optimizer_cls(opt_groups)
        else:
            self.base_optimizer = self.optimizer_cls(
                [{"params": [torch.zeros(1, requires_grad=True)]}],
            )

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero out gradients for all parameter groups.

        Args:
            set_to_none: If True, set gradients to None instead of zero
                for memory efficiency.
        """
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    if set_to_none:
                        param.grad = None
                    else:
                        param.grad.zero_()

        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None) -> Optional[float]:
        """Perform a single optimization step with gradient projection.

        For each GaLore parameter:
            1. Project the full-rank gradient into the low-rank subspace.
            2. Pass the projected gradient to the base optimizer.
            3. Project the optimizer update back to full rank.
            4. Apply the full-rank update to the parameter.

        Non-GaLore parameters are updated normally.

        Args:
            closure: Optional closure that reevaluates the model loss.

        Returns:
            Optional loss from the closure.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Project gradients for GaLore parameters
        for group in self.param_groups:
            if not group["galore"]:
                continue

            for param in group["params"]:
                if param.grad is None:
                    continue

                pid = id(param)
                projector = self.projectors[pid]
                placeholder = self._projected_params[pid]

                # Project gradient to low-rank subspace
                projected_grad = projector.project(param.grad)

                # Assign to placeholder for the base optimizer
                if placeholder.grad is None:
                    placeholder.grad = projected_grad.clone()
                else:
                    placeholder.grad.copy_(projected_grad)

        # Step the base optimizer (updates placeholders)
        self.base_optimizer.step()

        # Project updates back and apply to original parameters
        for group in self.param_groups:
            if not group["galore"]:
                continue

            for param in group["params"]:
                if param.grad is None:
                    continue

                pid = id(param)
                projector = self.projectors[pid]
                placeholder = self._projected_params[pid]

                # The optimizer has modified placeholder.data; compute the
                # full-rank update direction from the projected space
                full_rank_update = projector.project_back(placeholder.data)

                # Apply update to the original parameter
                with torch.no_grad():
                    param.data.add_(full_rank_update.reshape(param.shape))

                # Reset placeholder
                placeholder.data.zero_()

        return loss

    def state_dict(self) -> Dict[str, Any]:
        """Return the optimizer state dict including projector states."""
        return {
            "base_optimizer": self.base_optimizer.state_dict(),
            "step_counts": {
                pid: proj._step_count
                for pid, proj in self.projectors.items()
            },
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load optimizer state dict.

        Args:
            state_dict: State dict from a previous state_dict() call.
        """
        self.base_optimizer.load_state_dict(state_dict["base_optimizer"])
        if "step_counts" in state_dict:
            for pid, count in state_dict["step_counts"].items():
                if pid in self.projectors:
                    self.projectors[pid]._step_count = count

    @property
    def param_groups_all(self) -> List[Dict[str, Any]]:
        """Return all base optimizer parameter groups."""
        return self.base_optimizer.param_groups


def apply_galore(
    model: nn.Module,
    optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.AdamW,
    config: Optional[GaLoreConfig] = None,
    **optimizer_kwargs,
) -> GaLoreOptimizer:
    """Create a GaLore-wrapped optimizer for the given model.

    This is a convenience function that collects model parameters,
    identifies which ones benefit from gradient projection (2D weight
    matrices), and returns a GaLoreOptimizer ready for training.

    Args:
        model: The model to optimize.
        optimizer_cls: Base optimizer class.
        config: GaLore configuration.
        **optimizer_kwargs: Arguments passed to the base optimizer
            (e.g., lr=1e-3, weight_decay=0.01).

    Returns:
        A GaLoreOptimizer instance wrapping the specified base optimizer.
    """
    config = config or GaLoreConfig()
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    return GaLoreOptimizer(
        params=trainable_params,
        optimizer_cls=optimizer_cls,
        galore_config=config,
        **optimizer_kwargs,
    )
