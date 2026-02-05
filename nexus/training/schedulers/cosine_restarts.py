"""Cosine Annealing with Warm Restarts Learning Rate Schedule.

Reference: "SGDR: Stochastic Gradient Descent with Warm Restarts"
(Loshchilov and Hutter, 2017)

This schedule implements periodic cosine decay with restarts back to the
peak learning rate. Each cycle can optionally be longer than the previous
one (controlled by cycle_mult), allowing the optimizer to explore broader
regions of the loss landscape in later cycles.

Key properties:
    - Periodic cosine annealing with restarts to peak LR
    - Configurable cycle length with multiplicative increase
    - Optional warmup at the beginning of training
    - Minimum LR floor to prevent learning from stalling
    - Useful for exploring multiple loss basins during training
"""

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional


class CosineRestartScheduler(_LRScheduler):
    """Cosine annealing with warm restarts scheduler.

    At the start of each cycle, the learning rate is reset to peak_lr
    and then decays to min_lr following a cosine curve. The length of
    each subsequent cycle can be multiplied by cycle_mult.

    An optional warmup phase linearly increases the learning rate from
    min_lr to peak_lr at the beginning of training.

    The cycle length at cycle i is:
        T_i = cycle_length * cycle_mult^i

    Within each cycle, the learning rate follows:
        lr = min_lr + 0.5 * (peak_lr - min_lr) * (1 + cos(pi * t / T_i))

    where t is the step count within the current cycle.

    Args:
        optimizer: Wrapped optimizer.
        peak_lr: Maximum learning rate at the start of each cycle.
        min_lr: Minimum learning rate at the end of each cycle.
            Default: 0.0.
        cycle_length: Length of the first cycle in steps. Must be > 0.
        cycle_mult: Multiplicative factor for cycle length growth.
            Each cycle is cycle_mult times longer than the previous.
            Default: 1.0 (constant cycle length).
        warmup_steps: Number of steps for initial linear warmup.
            Default: 0.
        last_epoch: The index of the last epoch. Default: -1.

    Example:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        >>> scheduler = CosineRestartScheduler(
        ...     optimizer,
        ...     peak_lr=1e-3,
        ...     min_lr=1e-6,
        ...     cycle_length=10000,
        ...     cycle_mult=2.0,
        ...     warmup_steps=500
        ... )
        >>> for step in range(100000):
        ...     train_step()
        ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        peak_lr: float,
        min_lr: float = 0.0,
        cycle_length: int = 10000,
        cycle_mult: float = 1.0,
        warmup_steps: int = 0,
        last_epoch: int = -1,
    ):
        if peak_lr <= 0:
            raise ValueError(f"peak_lr must be > 0, got {peak_lr}")
        if min_lr < 0:
            raise ValueError(f"min_lr must be >= 0, got {min_lr}")
        if cycle_length <= 0:
            raise ValueError(f"cycle_length must be > 0, got {cycle_length}")
        if cycle_mult <= 0:
            raise ValueError(f"cycle_mult must be > 0, got {cycle_mult}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")

        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.cycle_length = cycle_length
        self.cycle_mult = cycle_mult
        self.warmup_steps = warmup_steps

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute learning rate for the current step.

        Returns:
            List of learning rates, one per parameter group.
        """
        step = self.last_epoch
        lr = self._compute_lr(step)
        return [lr for _ in self.base_lrs]

    def _compute_lr(self, step: int) -> float:
        """Compute the learning rate value for a given step.

        Args:
            step: Current training step.

        Returns:
            Learning rate value.
        """
        # Warmup phase
        if step < self.warmup_steps:
            if self.warmup_steps == 0:
                return self.peak_lr
            progress = step / self.warmup_steps
            return self.min_lr + progress * (self.peak_lr - self.min_lr)

        # Adjust step for warmup
        adjusted_step = step - self.warmup_steps

        # Determine which cycle we're in and position within cycle
        if self.cycle_mult == 1.0:
            # Constant cycle length - simple modular arithmetic
            cycle_idx = adjusted_step // self.cycle_length
            step_in_cycle = adjusted_step % self.cycle_length
            current_cycle_length = self.cycle_length
        else:
            # Variable cycle length - need to find which cycle
            # Total steps after n cycles: T_0 * (cycle_mult^n - 1) / (cycle_mult - 1)
            cycle_idx = 0
            cumulative_steps = 0
            current_cycle_length = self.cycle_length

            while cumulative_steps + current_cycle_length <= adjusted_step:
                cumulative_steps += current_cycle_length
                cycle_idx += 1
                current_cycle_length = int(
                    self.cycle_length * (self.cycle_mult ** cycle_idx)
                )

            step_in_cycle = adjusted_step - cumulative_steps

        # Cosine annealing within the current cycle
        progress = step_in_cycle / max(1, current_cycle_length)
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))

        return self.min_lr + cosine_factor * (self.peak_lr - self.min_lr)

    def get_cycle_info(self, step: Optional[int] = None) -> dict:
        """Get information about the current cycle.

        Args:
            step: Step to check. If None, uses current step.

        Returns:
            Dictionary with cycle_idx, step_in_cycle, and cycle_length.
        """
        if step is None:
            step = self.last_epoch

        adjusted_step = max(0, step - self.warmup_steps)

        if self.cycle_mult == 1.0:
            cycle_idx = adjusted_step // self.cycle_length
            step_in_cycle = adjusted_step % self.cycle_length
            current_cycle_length = self.cycle_length
        else:
            cycle_idx = 0
            cumulative_steps = 0
            current_cycle_length = self.cycle_length

            while cumulative_steps + current_cycle_length <= adjusted_step:
                cumulative_steps += current_cycle_length
                cycle_idx += 1
                current_cycle_length = int(
                    self.cycle_length * (self.cycle_mult ** cycle_idx)
                )

            step_in_cycle = adjusted_step - cumulative_steps

        return {
            "cycle_idx": cycle_idx,
            "step_in_cycle": step_in_cycle,
            "cycle_length": current_cycle_length,
        }
