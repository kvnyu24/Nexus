"""Schedule-Free AdamW Optimizer.

Reference: "The Road Less Scheduled" (Defazio et al., 2024)

Schedule-Free optimization eliminates the need for learning rate schedules
by using a novel interpolation technique between the iterate sequence and
its running average. This approach achieves the same theoretical convergence
guarantees as scheduled methods without requiring knowledge of the total
training duration.

Key properties:
    - No learning rate schedule needed (no cosine decay, etc.)
    - Uses interpolation between current iterate and Polyak average
    - Achieves same convergence as optimal schedule in theory
    - Must call optimizer.train() and optimizer.eval_mode() for proper behavior
    - During evaluation: uses the averaged parameters for better generalization
    - During training: uses the interpolated parameters for optimization
"""

import torch
import math
from torch.optim import Optimizer
from typing import Tuple, Iterable, Optional, Callable


class ScheduleFreeAdamW(Optimizer):
    """Implements Schedule-Free AdamW optimizer.

    This optimizer eliminates the need for learning rate schedules by
    maintaining two sequences of parameters: the optimization iterates
    (z) and a Polyak-style running average (x). During training, a
    specific interpolation (y) is used for forward passes, while
    during evaluation, the averaged parameters (x) provide better
    generalization.

    Important: You must call optimizer.train_mode() before training steps
    and optimizer.eval_mode() before evaluation to switch between the
    interpolated and averaged parameter views.

    Algorithm:
        1. y = (1 - beta1) * z + beta1 * x  (interpolation for training)
        2. z = z - lr * (grad / sqrt(v) + weight_decay * y)  (iterate update)
        3. x = (1 - 1/k) * x + (1/k) * z  (running average update)

    Args:
        params: Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr: Learning rate. Default: 0.025.
        betas: Coefficients used for computing running averages of
            gradient and its square. Default: (0.9, 0.999).
        eps: Term added to denominator for numerical stability.
            Default: 1e-8.
        weight_decay: Decoupled weight decay. Default: 0.0.
        warmup_steps: Number of warmup steps where the learning rate
            linearly increases from 0 to lr. Default: 0.

    Example:
        >>> optimizer = ScheduleFreeAdamW(model.parameters(), lr=0.025)
        >>> optimizer.train_mode()  # Must call before training
        >>> for batch in train_loader:
        ...     loss = model(batch)
        ...     loss.backward()
        ...     optimizer.step()
        ...     optimizer.zero_grad()
        >>> optimizer.eval_mode()  # Must call before evaluation
        >>> val_loss = validate(model, val_loader)
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.025,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        warmup_steps: int = 0,
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
        if warmup_steps < 0:
            raise ValueError(f"Invalid warmup_steps value: {warmup_steps}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
        )
        super().__init__(params, defaults)

        self._training_mode = True
        self._step_count = 0

    def train_mode(self) -> None:
        """Switch to training mode.

        Replaces the model parameters with the interpolated values (y)
        used for computing gradients during training. Must be called
        before each training phase.
        """
        if self._training_mode:
            return

        for group in self.param_groups:
            beta1 = group["betas"][0]
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    continue
                # Switch from x (averaged) to y (interpolated)
                z = state["z"]
                x = state["x"]
                # y = (1 - beta1) * z + beta1 * x
                p.data.copy_(z.lerp(x, beta1))

        self._training_mode = True

    def eval_mode(self) -> None:
        """Switch to evaluation mode.

        Replaces the model parameters with the averaged values (x)
        for evaluation. The averaged parameters typically provide
        better generalization. Must be called before each
        evaluation phase.
        """
        if not self._training_mode:
            return

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    continue
                # Switch from y (interpolated) to x (averaged)
                p.data.copy_(state["x"])

        self._training_mode = False

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

        self._step_count += 1
        k = self._step_count

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            warmup_steps = group["warmup_steps"]

            # Apply warmup
            if warmup_steps > 0 and k <= warmup_steps:
                warmup_factor = k / warmup_steps
                current_lr = lr * warmup_factor
            else:
                current_lr = lr

            # Averaging coefficient: increases over time
            ck = 1.0 / k

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        "ScheduleFreeAdamW does not support sparse gradients"
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["z"] = p.data.clone()
                    state["x"] = p.data.clone()
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                z = state["z"]
                x = state["x"]
                exp_avg_sq = state["exp_avg_sq"]

                # Update second moment estimate
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Bias correction for second moment
                bias_correction2 = 1.0 - beta2 ** state["step"]
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                # Compute the normalized gradient
                normalized_grad = grad / denom

                # Weight decay on the interpolated point y
                if weight_decay != 0.0:
                    # y is currently stored in p.data
                    normalized_grad.add_(p.data, alpha=weight_decay)

                # Update the iterate z
                z.add_(normalized_grad, alpha=-current_lr)

                # Update the running average x
                x.mul_(1.0 - ck).add_(z, alpha=ck)

                # Set p to the interpolated point y for next forward pass
                # y = (1 - beta1) * z + beta1 * x
                p.data.copy_(z.lerp(x, beta1))

        return loss
