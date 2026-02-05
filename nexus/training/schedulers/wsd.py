"""Warmup-Stable-Decay (WSD) Learning Rate Schedule.

Reference: "MiniCPM: Unveiling the Potential of Small Language Models
with Scalable Training Strategies" (Hu et al., 2024)

The WSD schedule divides training into three distinct phases:
    1. Warmup: Linear increase from 0 to peak learning rate
    2. Stable: Maintain peak learning rate (plateau)
    3. Decay: Controlled decrease from peak to minimum learning rate

Unlike cosine schedules, WSD does not require knowing the total number
of training steps upfront. The stable phase can be extended arbitrarily,
and the decay phase can be triggered when desired. This makes WSD
particularly suitable for large-scale training where the total compute
budget may not be known in advance.

Key properties:
    - Three clearly separated phases
    - No need to specify total training steps upfront
    - Stable phase allows model to train at peak LR as long as needed
    - Decay phase provides smooth transition to fine-tuning LR
    - Supports linear, cosine, and sqrt decay types
"""

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional


class WSDScheduler(_LRScheduler):
    """Warmup-Stable-Decay learning rate scheduler.

    Implements a three-phase learning rate schedule:
        1. Warmup (steps 0 to warmup_steps): linear increase to peak_lr
        2. Stable (steps warmup_steps to warmup_steps + stable_steps):
           constant at peak_lr
        3. Decay (steps warmup_steps + stable_steps to total): controlled
           decrease from peak_lr to min_lr

    Args:
        optimizer: Wrapped optimizer.
        warmup_steps: Number of steps for linear warmup. Must be >= 0.
        stable_steps: Number of steps to maintain peak learning rate.
            Must be >= 0.
        decay_steps: Number of steps for learning rate decay. Must be > 0.
        peak_lr: Peak learning rate reached after warmup. This overrides
            the optimizer's initial learning rate.
        min_lr: Minimum learning rate at the end of decay. Default: 0.0.
        decay_type: Type of decay curve. One of 'linear', 'cosine',
            or 'sqrt'. Default: 'linear'.
        last_epoch: The index of the last epoch. Default: -1.

    Example:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        >>> scheduler = WSDScheduler(
        ...     optimizer,
        ...     warmup_steps=1000,
        ...     stable_steps=50000,
        ...     decay_steps=10000,
        ...     peak_lr=1e-3,
        ...     min_lr=1e-5,
        ...     decay_type='cosine'
        ... )
        >>> for step in range(61000):
        ...     train_step()
        ...     scheduler.step()
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        stable_steps: int,
        decay_steps: int,
        peak_lr: float,
        min_lr: float = 0.0,
        decay_type: str = "linear",
        last_epoch: int = -1,
    ):
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        if stable_steps < 0:
            raise ValueError(f"stable_steps must be >= 0, got {stable_steps}")
        if decay_steps <= 0:
            raise ValueError(f"decay_steps must be > 0, got {decay_steps}")
        if peak_lr <= 0:
            raise ValueError(f"peak_lr must be > 0, got {peak_lr}")
        if min_lr < 0:
            raise ValueError(f"min_lr must be >= 0, got {min_lr}")
        if decay_type not in ("linear", "cosine", "sqrt"):
            raise ValueError(
                f"decay_type must be 'linear', 'cosine', or 'sqrt', got '{decay_type}'"
            )

        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.decay_steps = decay_steps
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.decay_type = decay_type

        # Total steps for all three phases
        self.total_steps = warmup_steps + stable_steps + decay_steps

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute learning rate for the current step.

        Returns:
            List of learning rates, one per parameter group.
        """
        step = self.last_epoch
        lr_scale = self._compute_lr_scale(step)
        return [lr_scale for _ in self.base_lrs]

    def _compute_lr_scale(self, step: int) -> float:
        """Compute the learning rate value for a given step.

        Args:
            step: Current training step.

        Returns:
            Learning rate value.
        """
        if step < self.warmup_steps:
            # Phase 1: Linear warmup
            if self.warmup_steps == 0:
                return self.peak_lr
            progress = step / self.warmup_steps
            return self.min_lr + progress * (self.peak_lr - self.min_lr)

        elif step < self.warmup_steps + self.stable_steps:
            # Phase 2: Stable (constant peak LR)
            return self.peak_lr

        else:
            # Phase 3: Decay
            decay_step = step - self.warmup_steps - self.stable_steps
            decay_progress = min(decay_step / self.decay_steps, 1.0)

            if self.decay_type == "linear":
                decay_factor = 1.0 - decay_progress
            elif self.decay_type == "cosine":
                decay_factor = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
            elif self.decay_type == "sqrt":
                decay_factor = 1.0 - math.sqrt(decay_progress)
            else:
                decay_factor = 1.0 - decay_progress

            return self.min_lr + decay_factor * (self.peak_lr - self.min_lr)

    def get_phase(self, step: Optional[int] = None) -> str:
        """Get the current training phase name.

        Args:
            step: Step to check. If None, uses current step.

        Returns:
            Phase name: 'warmup', 'stable', or 'decay'.
        """
        if step is None:
            step = self.last_epoch

        if step < self.warmup_steps:
            return "warmup"
        elif step < self.warmup_steps + self.stable_steps:
            return "stable"
        else:
            return "decay"
