"""Muon Optimizer: Momentum + Orthogonalization.

Reference: "Muon: An optimizer for hidden layers in neural networks"
(Jordan et al., 2024)

Muon applies SGD with Nesterov momentum followed by Newton-Schulz
orthogonalization to compute the nearest orthogonal matrix to each
2D parameter update. This provides a principled way to normalize
gradient updates while preserving directional information. For non-2D
parameters (embeddings, biases, normalization layers), it falls back
to standard AdamW.

Key properties:
    - Applies orthogonalization to 2D parameter updates via Newton-Schulz
    - Newton-Schulz iteration approximates polar decomposition
    - Non-2D parameters (embeddings, biases, norms) use AdamW
    - Particularly effective for training transformers
    - 5 Newton-Schulz steps are typically sufficient
"""

import torch
import torch.nn as nn
import math
from torch.optim import Optimizer
from typing import Iterable, Optional, Callable, List, Dict, Any


def _newton_schulz_orthogonalize(
    matrix: torch.Tensor,
    num_steps: int = 5,
) -> torch.Tensor:
    """Compute the nearest orthogonal matrix using Newton-Schulz iteration.

    The Newton-Schulz iteration converges to the polar factor U of the
    polar decomposition A = U * S, where U is orthogonal and S is
    positive semi-definite.

    Iteration:
        X_{k+1} = 0.5 * X_k * (3I - X_k^T * X_k)

    This converges quadratically when the spectral norm of X_0 < sqrt(3).

    Args:
        matrix: Input matrix to orthogonalize. Shape: (m, n).
        num_steps: Number of Newton-Schulz iterations. Default: 5.

    Returns:
        Orthogonalized matrix of the same shape.
    """
    assert matrix.dim() == 2, "Newton-Schulz requires 2D input"

    # Transpose if needed so m >= n for numerical stability
    transposed = False
    if matrix.shape[0] < matrix.shape[1]:
        matrix = matrix.T
        transposed = True

    # Normalize to ensure convergence (spectral norm < sqrt(3))
    norm = torch.norm(matrix, "fro")
    if norm > 0:
        matrix = matrix / norm

    # Newton-Schulz iteration
    X = matrix
    for _ in range(num_steps):
        # X = 0.5 * X * (3I - X^T X)
        XtX = X.T @ X
        identity = torch.eye(XtX.shape[0], device=XtX.device, dtype=XtX.dtype)
        X = 0.5 * X @ (3.0 * identity - XtX)

    if transposed:
        X = X.T

    return X


class Muon(Optimizer):
    """Implements the Muon optimizer (Momentum + Orthogonalization).

    Muon applies Nesterov momentum SGD to 2D parameters and then
    orthogonalizes the update direction using Newton-Schulz iteration.
    Non-2D parameters are optimized using AdamW.

    This provides a principled way to constrain updates to lie on the
    Stiefel manifold (set of orthogonal matrices), which has been shown
    to improve training dynamics for transformer models.

    Args:
        params: Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr: Learning rate for 2D parameters (with orthogonalization).
            Default: 0.02.
        momentum: Momentum factor for Nesterov SGD on 2D params.
            Default: 0.95.
        ns_steps: Number of Newton-Schulz orthogonalization steps.
            Default: 5.
        weight_decay: Weight decay for 2D parameters. Default: 0.0.
        adamw_lr: Learning rate for AdamW on non-2D parameters.
            Default: 3e-4.
        adamw_betas: Beta coefficients for AdamW. Default: (0.95, 0.95).
        adamw_eps: Epsilon for AdamW. Default: 1e-8.
        adamw_wd: Weight decay for AdamW on non-2D params. Default: 0.0.

    Example:
        >>> optimizer = Muon(model.parameters(), lr=0.02, momentum=0.95)
        >>> for batch in dataloader:
        ...     loss = model(batch)
        ...     loss.backward()
        ...     optimizer.step()
        ...     optimizer.zero_grad()
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.02,
        momentum: float = 0.95,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        adamw_lr: float = 3e-4,
        adamw_betas: tuple = (0.95, 0.95),
        adamw_eps: float = 1e-8,
        adamw_wd: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if ns_steps < 1:
            raise ValueError(f"Invalid ns_steps: {ns_steps}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            adamw_lr=adamw_lr,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            adamw_wd=adamw_wd,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[torch.Tensor]:
        """Perform a single optimization step.

        2D parameters are updated using Nesterov momentum SGD followed
        by Newton-Schulz orthogonalization. Non-2D parameters are
        updated using AdamW.

        Args:
            closure: A closure that reevaluates the model and returns the loss.
                Optional.

        Returns:
            Loss value if closure is provided, None otherwise.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]
            adamw_lr = group["adamw_lr"]
            adamw_betas = group["adamw_betas"]
            adamw_eps = group["adamw_eps"]
            adamw_wd = group["adamw_wd"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")

                state = self.state[p]

                is_2d = p.dim() == 2

                if is_2d:
                    # Nesterov momentum SGD + orthogonalization for 2D params
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)

                    buf = state["momentum_buffer"]

                    # Nesterov momentum update
                    buf.mul_(momentum).add_(grad)
                    nesterov_grad = grad.add(buf, alpha=momentum)

                    # Orthogonalize the update direction
                    update = _newton_schulz_orthogonalize(
                        nesterov_grad, num_steps=ns_steps
                    )

                    # Scale update to match original gradient norm
                    # This preserves the effective step size
                    original_scale = grad.norm() / (update.norm() + 1e-8)
                    update = update * original_scale

                    # Weight decay (decoupled)
                    if weight_decay != 0.0:
                        p.mul_(1.0 - lr * weight_decay)

                    # Apply update
                    p.add_(update, alpha=-lr)

                else:
                    # AdamW for non-2D parameters (embeddings, biases, norms)
                    if len(state) == 0:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)

                    state["step"] += 1
                    step = state["step"]
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    beta1, beta2 = adamw_betas

                    # AdamW update
                    exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(
                        grad, grad, value=1.0 - beta2
                    )

                    # Bias correction
                    bias_correction1 = 1.0 - beta1 ** step
                    bias_correction2 = 1.0 - beta2 ** step

                    step_size = adamw_lr / bias_correction1
                    denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(
                        adamw_eps
                    )

                    # Weight decay (decoupled)
                    if adamw_wd != 0.0:
                        p.mul_(1.0 - adamw_lr * adamw_wd)

                    p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
