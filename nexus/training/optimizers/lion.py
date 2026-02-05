"""Lion Optimizer: Evolved Sign Momentum.

Reference: "Symbolic Discovery of Optimization Algorithms" (Chen et al., 2023)

Lion (EvoLved Sign Momentum) was discovered through program search and is
remarkably simple: it uses only the sign of the momentum for parameter updates,
requiring only one momentum buffer (half the memory of AdamW). Despite its
simplicity, Lion matches or exceeds AdamW on a variety of tasks including
image classification, language modeling, and diffusion models.

Key properties:
    - Uses sign of interpolation between gradient and momentum for updates
    - Only maintains a single momentum buffer (vs. two for Adam)
    - Memory-efficient: roughly half the optimizer states of AdamW
    - Produces more uniform update magnitudes due to sign operation
    - Tends to prefer larger weight decay values than AdamW
"""

import torch
from torch.optim import Optimizer
from typing import Tuple, Iterable, Optional, Callable


class Lion(Optimizer):
    """Implements the Lion optimizer.

    Lion uses the sign of a momentum-gradient interpolation for updates,
    achieving competitive performance with significantly less memory than
    Adam-based optimizers.

    Algorithm:
        1. update = sign(beta1 * momentum + (1 - beta1) * gradient)
        2. parameter -= lr * (update + weight_decay * parameter)
        3. momentum = beta2 * momentum + (1 - beta2) * gradient

    Note: The update step uses beta1 for interpolation, but the momentum
    buffer is updated with beta2. This asymmetry is a key design choice
    discovered through the evolutionary search process.

    Args:
        params: Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr: Learning rate. Lion typically uses a smaller learning rate
            than AdamW (e.g., 3-10x smaller). Default: 1e-4.
        betas: Coefficients for computing the update direction and
            updating the momentum buffer. Default: (0.9, 0.99).
        weight_decay: Weight decay (L2 penalty). Lion often benefits
            from larger weight decay than AdamW. Default: 0.0.

    Example:
        >>> optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=1.0)
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

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

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]

                # Weight decay (decoupled)
                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)

                # Compute the update direction: sign(beta1 * m + (1 - beta1) * g)
                update = exp_avg.mul(beta1).add(grad, alpha=1.0 - beta1)
                p.add_(update.sign_(), alpha=-lr)

                # Update momentum buffer: m = beta2 * m + (1 - beta2) * g
                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)

        return loss
