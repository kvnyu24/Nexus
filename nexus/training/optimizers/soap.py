"""SOAP Optimizer: Shampoo with Adam Optimizer Preconditioning.

SOAP combines the benefits of Adam's adaptive learning rates with Shampoo's
second-order preconditioning. It maintains separate preconditioners for each
parameter dimension, providing better conditioning than Adam while being more
memory-efficient than full Shampoo.

Reference:
    "SOAP: Improving and Stabilizing Shampoo using Adam"
    Nikhil Vyas et al., 2024
    https://arxiv.org/abs/2409.11321

Key features:
    - Kronecker-factored preconditioners (like Shampoo)
    - Adam-style momentum and bias correction
    - Improved stability over vanilla Shampoo
    - Memory-efficient compared to full second-order methods

Example:
    >>> optimizer = SOAP(
    ...     model.parameters(),
    ...     lr=1e-3,
    ...     betas=(0.9, 0.999),
    ...     precondition_frequency=10,
    ...     max_precond_dim=1024
    ... )
    >>> for batch in dataloader:
    ...     loss = model(batch)
    ...     loss.backward()
    ...     optimizer.step()
    ...     optimizer.zero_grad()
"""

import torch
from torch.optim import Optimizer
from typing import List, Optional, Tuple, Callable
import math


class SOAP(Optimizer):
    """SOAP: Shampoo with Adam Optimizer Preconditioning.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups.
        lr: Learning rate. Default: 1e-3.
        betas: Coefficients for computing running averages of gradient and its
            square. Default: (0.9, 0.999).
        eps: Term added to denominator for numerical stability. Default: 1e-8.
        weight_decay: Weight decay coefficient. Default: 0.0.
        precondition_frequency: Frequency (in steps) to update preconditioners.
            Higher values reduce computation but may slow convergence. Default: 10.
        max_precond_dim: Maximum dimension for preconditioning. Dimensions larger
            than this use standard Adam updates. Default: 1024.
        preconditioner_decay: Exponential moving average decay for preconditioner
            updates. Default: 0.95.
        merge_dims: Whether to merge dimensions for large tensors (memory optimization).
            Default: True.
        precondition_1d: Whether to precondition 1D parameters (e.g., biases).
            Default: False.

    Notes:
        - Memory overhead: ~2x parameter size for preconditioners (when applicable)
        - Compute overhead: Minimal, as preconditioning is infrequent
        - Works best for 2D parameters (linear layers). Falls back to Adam for others.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        precondition_frequency: int = 10,
        max_precond_dim: int = 1024,
        preconditioner_decay: float = 0.95,
        merge_dims: bool = True,
        precondition_1d: bool = False,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not precondition_frequency > 0:
            raise ValueError(f"Invalid precondition_frequency: {precondition_frequency}")
        if not max_precond_dim > 0:
            raise ValueError(f"Invalid max_precond_dim: {max_precond_dim}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            precondition_frequency=precondition_frequency,
            max_precond_dim=max_precond_dim,
            preconditioner_decay=preconditioner_decay,
            merge_dims=merge_dims,
            precondition_1d=precondition_1d,
        )
        super().__init__(params, defaults)

    def _matrix_power(self, matrix: torch.Tensor, power: float) -> torch.Tensor:
        """Compute matrix^power using eigendecomposition.

        Args:
            matrix: Symmetric matrix.
            power: Exponent.

        Returns:
            matrix^power.
        """
        # Eigendecomposition
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)

        # Clamp small eigenvalues for stability
        eigenvalues = torch.clamp(eigenvalues, min=1e-8)

        # Compute power
        powered_eigenvalues = eigenvalues.pow(power)

        # Reconstruct matrix
        return eigenvectors @ torch.diag(powered_eigenvalues) @ eigenvectors.T

    def _update_preconditioner(
        self,
        precond: torch.Tensor,
        grad: torch.Tensor,
        decay: float,
    ) -> torch.Tensor:
        """Update preconditioner with exponential moving average.

        Args:
            precond: Current preconditioner.
            grad: Gradient matrix.
            decay: EMA decay factor.

        Returns:
            Updated preconditioner.
        """
        # Compute outer product (or inner product for the other dimension)
        grad_cov = grad @ grad.T if precond.shape[0] == grad.shape[0] else grad.T @ grad

        # Exponential moving average
        precond.mul_(decay).add_(grad_cov, alpha=1.0 - decay)

        return precond

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Perform a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            Loss value if closure is provided, else None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            precondition_frequency = group["precondition_frequency"]
            max_precond_dim = group["max_precond_dim"]
            preconditioner_decay = group["preconditioner_decay"]
            merge_dims = group["merge_dims"]
            precondition_1d = group["precondition_1d"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # Apply weight decay
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Adam-style moment estimates
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                    # Determine if we should use preconditioning
                    use_precond = False
                    if p.dim() == 2:
                        dim1, dim2 = p.shape
                        if dim1 <= max_precond_dim and dim2 <= max_precond_dim:
                            use_precond = True
                    elif p.dim() == 1 and precondition_1d:
                        if p.shape[0] <= max_precond_dim:
                            use_precond = True

                    state["use_precond"] = use_precond

                    # Initialize preconditioners for 2D parameters
                    if use_precond and p.dim() == 2:
                        dim1, dim2 = p.shape
                        # Left and right preconditioners (Kronecker factorization)
                        state["precond_left"] = torch.eye(dim1, device=p.device, dtype=p.dtype)
                        state["precond_right"] = torch.eye(dim2, device=p.device, dtype=p.dtype)
                    elif use_precond and p.dim() == 1:
                        # Single preconditioner for 1D
                        state["precond"] = torch.eye(p.shape[0], device=p.device, dtype=p.dtype)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update biased second moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                bias_corrected_exp_avg = exp_avg / bias_correction1
                bias_corrected_exp_avg_sq = exp_avg_sq / bias_correction2

                # Apply preconditioning if applicable
                if state["use_precond"]:
                    # Update preconditioners at specified frequency
                    if step % precondition_frequency == 0:
                        if p.dim() == 2:
                            # Update left and right preconditioners
                            state["precond_left"] = self._update_preconditioner(
                                state["precond_left"],
                                grad,
                                preconditioner_decay,
                            )
                            state["precond_right"] = self._update_preconditioner(
                                state["precond_right"],
                                grad.T,
                                preconditioner_decay,
                            )
                        elif p.dim() == 1:
                            grad_2d = grad.unsqueeze(1)
                            state["precond"] = self._update_preconditioner(
                                state["precond"],
                                grad_2d,
                                preconditioner_decay,
                            )

                    # Apply preconditioning to the update
                    if p.dim() == 2:
                        # Compute P_L^{-1/4} @ grad @ P_R^{-1/4}
                        left_inv = self._matrix_power(state["precond_left"], -0.25)
                        right_inv = self._matrix_power(state["precond_right"], -0.25)

                        preconditioned_grad = left_inv @ bias_corrected_exp_avg @ right_inv

                        # Compute adaptive step size with preconditioning
                        preconditioned_sq = left_inv @ bias_corrected_exp_avg_sq @ right_inv
                        step_size = preconditioned_grad / (preconditioned_sq.sqrt() + eps)

                    elif p.dim() == 1:
                        # Single preconditioner
                        precond_inv = self._matrix_power(state["precond"], -0.5)
                        preconditioned_grad = precond_inv @ bias_corrected_exp_avg.unsqueeze(1)
                        preconditioned_sq = precond_inv @ bias_corrected_exp_avg_sq.unsqueeze(1)

                        step_size = preconditioned_grad / (preconditioned_sq.sqrt() + eps)
                        step_size = step_size.squeeze(1)

                else:
                    # Standard Adam update (no preconditioning)
                    step_size = bias_corrected_exp_avg / (bias_corrected_exp_avg_sq.sqrt() + eps)

                # Apply update
                p.add_(step_size, alpha=-lr)

        return loss
