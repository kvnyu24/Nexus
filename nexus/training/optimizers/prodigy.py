"""Prodigy Optimizer: Learning-Rate-Free Adaptive Optimization.

Reference: "Prodigy: An Expeditiously Adaptive Parameter-Free Learner"
(Mishchenko et al., 2023)

Prodigy is a learning-rate-free optimizer that automatically estimates
the distance to the solution (D) and adapts the effective learning rate
accordingly. It builds on D-Adaptation with improved convergence speed
by using Adagrad-like per-coordinate step sizes.

Key properties:
    - No learning rate tuning required (lr=1.0 is a scaling factor)
    - Automatically estimates distance D to the optimal solution
    - Combines D-Adaptation with Adagrad-like adaptive step sizes
    - Provably converges at the optimal rate for convex problems
    - Works well for both convex and non-convex optimization
"""

import torch
import math
from torch.optim import Optimizer
from typing import Tuple, Iterable, Optional, Callable


class Prodigy(Optimizer):
    """Implements the Prodigy optimizer (learning-rate-free).

    Prodigy automatically estimates the distance to the optimal solution
    and uses this estimate to set the effective learning rate. The `lr`
    parameter acts as a scaling factor rather than an absolute learning rate.

    Algorithm:
        1. Maintain running estimate of distance D to solution
        2. Update D based on inner product of gradient and parameter change
        3. Compute Adam-like update with effective lr = lr * D
        4. Use Adagrad-style denominator for per-coordinate adaptation

    Args:
        params: Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr: Scaling factor for the automatically estimated learning rate.
            Default: 1.0 (typically left at 1.0).
        betas: Coefficients for computing running averages of gradient
            and its square. Default: (0.9, 0.999).
        eps: Term added to denominator for numerical stability.
            Default: 1e-8.
        weight_decay: Weight decay (L2 penalty). Default: 0.0.
        d_coef: Coefficient for D estimate. Controls the aggressiveness
            of the distance estimation. Default: 1.0.
        growth_rate: Maximum multiplicative factor for D growth per step.
            Prevents D from growing too quickly. Default: float('inf').

    Example:
        >>> # No learning rate tuning needed
        >>> optimizer = Prodigy(model.parameters(), lr=1.0)
        >>> for batch in dataloader:
        ...     loss = model(batch)
        ...     loss.backward()
        ...     optimizer.step()
        ...     optimizer.zero_grad()
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1.0,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        d_coef: float = 1.0,
        growth_rate: float = float("inf"),
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if d_coef <= 0.0:
            raise ValueError(f"Invalid d_coef value: {d_coef}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            d_coef=d_coef,
            growth_rate=growth_rate,
        )
        super().__init__(params, defaults)

        # Global state for distance estimate
        self.d_estimate = 1e-6  # Initial small estimate
        self.d_numerator = 0.0
        self.d_denominator = 0.0

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[torch.Tensor]:
        """Perform a single optimization step.

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

        # Accumulate D-estimate updates across all parameter groups
        d_numerator_update = 0.0
        d_denominator_update = 0.0

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            d_coef = group["d_coef"]
            growth_rate = group["growth_rate"]

            dlr = lr * d_coef * self.d_estimate

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Prodigy does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["initial_param"] = p.data.clone()
                    # Running sum for the Adagrad-like denominator
                    state["s"] = torch.zeros_like(p)

                state["step"] += 1
                step = state["step"]

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                s = state["s"]
                initial_param = state["initial_param"]

                # Bias correction
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2_sq = math.sqrt(1.0 - beta2 ** step)

                # Update D-estimate numerator: inner product of grad and (x - x0)
                diff = p.data - initial_param
                d_numerator_update += dlr * torch.dot(
                    grad.flatten(), diff.flatten()
                ).item()

                # Update first and second moment estimates
                exp_avg.mul_(beta1).add_(grad, alpha=dlr * (1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(
                    grad, grad, value=(dlr ** 2) * (1.0 - beta2)
                )

                # Accumulate Adagrad-style sum of squared gradients
                s.addcmul_(grad, grad, value=(dlr ** 2))

                # D-estimate denominator accumulation
                d_denominator_update += s.sum().item()

                # Weight decay (decoupled)
                if weight_decay != 0.0:
                    p.mul_(1.0 - dlr * weight_decay)

                # Compute the adaptive denominator
                denom = (exp_avg_sq.sqrt() / bias_correction2_sq).add_(eps)

                # Parameter update
                step_size = 1.0 / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)

        # Update global distance estimate
        self.d_numerator = self.d_numerator + d_numerator_update
        self.d_denominator = max(self.d_denominator, d_denominator_update)

        if self.d_denominator > 0:
            new_d = abs(self.d_numerator) / math.sqrt(self.d_denominator)
            # Apply growth rate limit
            new_d = min(new_d, self.d_estimate * growth_rate)
            self.d_estimate = max(new_d, self.d_estimate)

        return loss

    def get_d_estimate(self) -> float:
        """Return the current distance estimate D.

        Returns:
            Current estimate of distance to the optimal solution.
        """
        return self.d_estimate
