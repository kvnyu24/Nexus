"""Gradient scaler for mixed precision training.

This module provides an enhanced gradient scaler with better overflow
detection and recovery mechanisms for stable mixed precision training.
"""

from typing import Dict, Optional, Any, List, Iterable
from enum import Enum
import math

import torch
import torch.nn as nn
from torch.optim import Optimizer


class ScalerState(Enum):
    """State of the gradient scaler."""
    READY = "ready"
    UNSCALED = "unscaled"
    STEPPED = "stepped"


class GradScaler:
    """Gradient scaler for mixed precision training.

    Enhanced version of PyTorch's GradScaler with better overflow detection
    and recovery. Supports dynamic scaling with configurable growth and
    backoff strategies.

    The scaler maintains a scale factor that multiplies the loss during
    backward pass to prevent gradient underflow in low-precision training.
    If gradients overflow, the scale is reduced (backoff). If training
    proceeds without overflow, the scale is periodically increased.

    Args:
        init_scale: Initial scale factor. Default: 65536.0.
        growth_factor: Factor to increase scale when no overflow detected.
                      Default: 2.0.
        backoff_factor: Factor to decrease scale on overflow. Default: 0.5.
        growth_interval: Number of consecutive steps without overflow before
                        increasing scale. Default: 2000.
        enabled: Whether scaling is enabled. Default: True.
        max_scale: Maximum allowed scale value. Default: 2^24.
        min_scale: Minimum allowed scale value. Default: 1.0.

    Example:
        >>> scaler = GradScaler(init_scale=65536.0)
        >>> for batch in dataloader:
        ...     optimizer.zero_grad()
        ...     with torch.cuda.amp.autocast():
        ...         loss = model(batch)
        ...     scaler.scale(loss).backward()
        ...     scaler.unscale_(optimizer)
        ...     torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        ...     scaler.step(optimizer)
        ...     scaler.update()
    """

    def __init__(
        self,
        init_scale: float = 65536.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled: bool = True,
        max_scale: float = 2 ** 24,
        min_scale: float = 1.0
    ):
        if init_scale <= 0:
            raise ValueError(f"init_scale must be positive, got {init_scale}")
        if growth_factor <= 1.0:
            raise ValueError(f"growth_factor must be > 1.0, got {growth_factor}")
        if not 0 < backoff_factor < 1.0:
            raise ValueError(f"backoff_factor must be in (0, 1), got {backoff_factor}")
        if growth_interval <= 0:
            raise ValueError(f"growth_interval must be positive, got {growth_interval}")

        self._init_scale = init_scale
        self._scale = init_scale
        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval
        self._enabled = enabled
        self._max_scale = max_scale
        self._min_scale = min_scale

        # Tracking state
        self._growth_tracker = 0
        self._state = ScalerState.READY

        # Overflow tracking
        self._found_inf_per_device: Dict[torch.device, torch.Tensor] = {}
        self._overflow_count = 0
        self._total_steps = 0

        # Per-optimizer state (for unscale tracking)
        self._per_optimizer_states: Dict[int, Dict[str, Any]] = {}

    @property
    def scale(self) -> float:
        """Current scale value."""
        return self._scale

    @property
    def is_enabled(self) -> bool:
        """Whether the scaler is enabled."""
        return self._enabled

    @property
    def overflow_rate(self) -> float:
        """Rate of overflow occurrences."""
        if self._total_steps == 0:
            return 0.0
        return self._overflow_count / self._total_steps

    def scale(self, outputs: torch.Tensor) -> torch.Tensor:
        """Scale the loss tensor.

        Args:
            outputs: Loss tensor to scale.

        Returns:
            Scaled loss tensor.
        """
        if not self._enabled:
            return outputs

        # Handle nested tensors (e.g., tuple of losses)
        if isinstance(outputs, (tuple, list)):
            return type(outputs)(self.scale(o) for o in outputs)

        if not outputs.is_floating_point():
            return outputs

        # Record device for this tensor
        device = outputs.device

        # Apply scale
        return outputs * self._scale

    def unscale_(self, optimizer: Optimizer) -> None:
        """Unscale gradients for the optimizer's parameters.

        This divides gradients by the scale factor, checking for inf/nan.

        Args:
            optimizer: Optimizer whose parameters' gradients should be unscaled.

        Raises:
            RuntimeError: If unscale_ is called more than once per step.
        """
        if not self._enabled:
            return

        optimizer_id = id(optimizer)
        optimizer_state = self._per_optimizer_states.get(optimizer_id)

        if optimizer_state is None:
            optimizer_state = {
                "stage": ScalerState.READY,
                "found_inf_per_device": {}
            }
            self._per_optimizer_states[optimizer_id] = optimizer_state

        if optimizer_state["stage"] == ScalerState.UNSCALED:
            raise RuntimeError(
                "unscale_() has already been called on this optimizer since the last update()."
            )

        optimizer_state["stage"] = ScalerState.UNSCALED

        # Get all parameter gradients
        grads = []
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    grads.append(param.grad)

        if not grads:
            return

        # Check for inf/nan and unscale
        inv_scale = 1.0 / self._scale
        found_inf = self._check_and_unscale_grads(grads, inv_scale)

        # Store found_inf for this optimizer
        if grads:
            device = grads[0].device
            optimizer_state["found_inf_per_device"][device] = found_inf

    def _check_and_unscale_grads(
        self,
        grads: List[torch.Tensor],
        inv_scale: float
    ) -> torch.Tensor:
        """Check for inf/nan in gradients and unscale.

        Args:
            grads: List of gradient tensors.
            inv_scale: Inverse of the scale factor.

        Returns:
            Tensor indicating if inf/nan was found (1.0) or not (0.0).
        """
        if not grads:
            return torch.tensor(0.0)

        device = grads[0].device
        found_inf = torch.zeros(1, device=device)

        for grad in grads:
            if grad is None:
                continue

            # Check for inf/nan
            if torch.isinf(grad).any() or torch.isnan(grad).any():
                found_inf.fill_(1.0)
                # Don't unscale if we found inf/nan
                return found_inf

            # Unscale gradient
            grad.mul_(inv_scale)

        return found_inf

    def step(self, optimizer: Optimizer, *args, **kwargs) -> Optional[float]:
        """Step the optimizer if gradients are finite.

        If gradients contain inf/nan, the optimizer step is skipped.

        Args:
            optimizer: Optimizer to step.
            *args: Arguments passed to optimizer.step().
            **kwargs: Keyword arguments passed to optimizer.step().

        Returns:
            Return value of optimizer.step() if gradients are finite, None otherwise.
        """
        if not self._enabled:
            return optimizer.step(*args, **kwargs)

        optimizer_id = id(optimizer)
        optimizer_state = self._per_optimizer_states.get(optimizer_id, {})

        # Check if unscale was called
        if optimizer_state.get("stage") != ScalerState.UNSCALED:
            # Auto-unscale if not done
            self.unscale_(optimizer)
            optimizer_state = self._per_optimizer_states[optimizer_id]

        # Check if we found inf in any device
        found_inf = False
        for device, inf_tensor in optimizer_state.get("found_inf_per_device", {}).items():
            if inf_tensor.item() > 0:
                found_inf = True
                break

        retval = None
        if not found_inf:
            retval = optimizer.step(*args, **kwargs)

        optimizer_state["stage"] = ScalerState.STEPPED
        return retval

    def update(self, new_scale: Optional[float] = None) -> None:
        """Update the scale factor.

        Should be called after optimizer.step() each iteration.

        Args:
            new_scale: If provided, sets the scale to this value directly.
                      Otherwise, uses dynamic scaling logic.
        """
        if not self._enabled:
            return

        self._total_steps += 1

        # Check if any optimizer found inf
        found_inf = False
        for opt_state in self._per_optimizer_states.values():
            for device, inf_tensor in opt_state.get("found_inf_per_device", {}).items():
                if inf_tensor.item() > 0:
                    found_inf = True
                    break
            if found_inf:
                break

        if new_scale is not None:
            # Use provided scale
            self._scale = new_scale
            self._growth_tracker = 0
        elif found_inf:
            # Backoff on overflow
            self._overflow_count += 1
            self._scale = max(
                self._min_scale,
                self._scale * self._backoff_factor
            )
            self._growth_tracker = 0
        else:
            # Potentially grow scale
            self._growth_tracker += 1
            if self._growth_tracker >= self._growth_interval:
                self._scale = min(
                    self._max_scale,
                    self._scale * self._growth_factor
                )
                self._growth_tracker = 0

        # Reset per-optimizer state
        self._per_optimizer_states.clear()
        self._state = ScalerState.READY

    def get_scale(self) -> float:
        """Get the current scale value.

        Returns:
            Current scale factor.
        """
        return self._scale

    def get_growth_factor(self) -> float:
        """Get the growth factor.

        Returns:
            Growth factor.
        """
        return self._growth_factor

    def get_backoff_factor(self) -> float:
        """Get the backoff factor.

        Returns:
            Backoff factor.
        """
        return self._backoff_factor

    def set_growth_factor(self, new_growth_factor: float) -> None:
        """Set the growth factor.

        Args:
            new_growth_factor: New growth factor (must be > 1.0).
        """
        if new_growth_factor <= 1.0:
            raise ValueError(f"growth_factor must be > 1.0, got {new_growth_factor}")
        self._growth_factor = new_growth_factor

    def set_backoff_factor(self, new_backoff_factor: float) -> None:
        """Set the backoff factor.

        Args:
            new_backoff_factor: New backoff factor (must be in (0, 1)).
        """
        if not 0 < new_backoff_factor < 1.0:
            raise ValueError(f"backoff_factor must be in (0, 1), got {new_backoff_factor}")
        self._backoff_factor = new_backoff_factor

    def state_dict(self) -> Dict[str, Any]:
        """Get state dict for checkpointing.

        Returns:
            State dictionary containing all scaler state.
        """
        return {
            "scale": self._scale,
            "growth_factor": self._growth_factor,
            "backoff_factor": self._backoff_factor,
            "growth_interval": self._growth_interval,
            "growth_tracker": self._growth_tracker,
            "overflow_count": self._overflow_count,
            "total_steps": self._total_steps,
            "enabled": self._enabled,
            "max_scale": self._max_scale,
            "min_scale": self._min_scale
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state from checkpoint.

        Args:
            state_dict: State dictionary to load.
        """
        self._scale = state_dict.get("scale", self._init_scale)
        self._growth_factor = state_dict.get("growth_factor", self._growth_factor)
        self._backoff_factor = state_dict.get("backoff_factor", self._backoff_factor)
        self._growth_interval = state_dict.get("growth_interval", self._growth_interval)
        self._growth_tracker = state_dict.get("growth_tracker", 0)
        self._overflow_count = state_dict.get("overflow_count", 0)
        self._total_steps = state_dict.get("total_steps", 0)
        self._enabled = state_dict.get("enabled", True)
        self._max_scale = state_dict.get("max_scale", 2 ** 24)
        self._min_scale = state_dict.get("min_scale", 1.0)

        # Reset per-optimizer state
        self._per_optimizer_states.clear()
        self._state = ScalerState.READY

    def __repr__(self) -> str:
        return (
            f"GradScaler("
            f"scale={self._scale}, "
            f"growth_factor={self._growth_factor}, "
            f"backoff_factor={self._backoff_factor}, "
            f"growth_interval={self._growth_interval}, "
            f"enabled={self._enabled})"
        )


class AdaptiveGradScaler(GradScaler):
    """Adaptive gradient scaler with automatic hyperparameter tuning.

    Extends GradScaler with automatic adjustment of growth and backoff
    factors based on training dynamics. This is useful for long training
    runs where optimal scaling parameters may change.

    Args:
        init_scale: Initial scale factor.
        growth_factor: Initial growth factor.
        backoff_factor: Initial backoff factor.
        growth_interval: Steps between potential scale increases.
        enabled: Whether scaling is enabled.
        target_overflow_rate: Target overflow rate (0-1). Default: 0.01.
        adaptation_interval: Steps between hyperparameter adaptations.
                            Default: 1000.
    """

    def __init__(
        self,
        init_scale: float = 65536.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled: bool = True,
        target_overflow_rate: float = 0.01,
        adaptation_interval: int = 1000
    ):
        super().__init__(
            init_scale=init_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            enabled=enabled
        )

        self._target_overflow_rate = target_overflow_rate
        self._adaptation_interval = adaptation_interval
        self._adaptation_step = 0

        # Window for recent overflow tracking
        self._recent_overflows: List[bool] = []
        self._window_size = 100

    def update(self, new_scale: Optional[float] = None) -> None:
        """Update scale and potentially adapt hyperparameters.

        Args:
            new_scale: Optional explicit new scale value.
        """
        # Track overflow before parent update clears state
        found_inf = False
        for opt_state in self._per_optimizer_states.values():
            for device, inf_tensor in opt_state.get("found_inf_per_device", {}).items():
                if inf_tensor.item() > 0:
                    found_inf = True
                    break
            if found_inf:
                break

        # Update recent overflow window
        self._recent_overflows.append(found_inf)
        if len(self._recent_overflows) > self._window_size:
            self._recent_overflows.pop(0)

        # Parent update
        super().update(new_scale)

        # Adaptation logic
        self._adaptation_step += 1
        if self._adaptation_step >= self._adaptation_interval:
            self._adapt_hyperparameters()
            self._adaptation_step = 0

    def _adapt_hyperparameters(self) -> None:
        """Adapt growth and backoff factors based on recent overflow rate."""
        if not self._recent_overflows:
            return

        recent_rate = sum(self._recent_overflows) / len(self._recent_overflows)

        if recent_rate > self._target_overflow_rate * 2:
            # Too many overflows - be more conservative
            self._growth_factor = max(1.1, self._growth_factor * 0.9)
            self._backoff_factor = max(0.1, self._backoff_factor * 0.9)
        elif recent_rate < self._target_overflow_rate * 0.5:
            # Few overflows - can be more aggressive
            self._growth_factor = min(4.0, self._growth_factor * 1.1)
            self._backoff_factor = min(0.9, self._backoff_factor * 1.1)

    def state_dict(self) -> Dict[str, Any]:
        """Get state dict including adaptive parameters."""
        state = super().state_dict()
        state.update({
            "target_overflow_rate": self._target_overflow_rate,
            "adaptation_interval": self._adaptation_interval,
            "adaptation_step": self._adaptation_step,
            "recent_overflows": self._recent_overflows
        })
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state including adaptive parameters."""
        super().load_state_dict(state_dict)
        self._target_overflow_rate = state_dict.get(
            "target_overflow_rate", self._target_overflow_rate
        )
        self._adaptation_interval = state_dict.get(
            "adaptation_interval", self._adaptation_interval
        )
        self._adaptation_step = state_dict.get("adaptation_step", 0)
        self._recent_overflows = state_dict.get("recent_overflows", [])


def create_grad_scaler(
    config: Optional["MixedPrecisionConfig"] = None,
    adaptive: bool = False
) -> GradScaler:
    """Factory function to create a gradient scaler.

    Args:
        config: Optional MixedPrecisionConfig for scaler parameters.
        adaptive: Whether to create an AdaptiveGradScaler.

    Returns:
        Configured GradScaler or AdaptiveGradScaler.
    """
    # Import here to avoid circular dependency
    from .config import MixedPrecisionConfig

    if config is None:
        config = MixedPrecisionConfig()

    scaler_class = AdaptiveGradScaler if adaptive else GradScaler

    return scaler_class(
        init_scale=config.loss_scale,
        growth_factor=config.scale_growth_factor,
        backoff_factor=config.scale_backoff_factor,
        growth_interval=config.scale_growth_interval,
        enabled=config.dynamic_loss_scale
    )
