import torch
import torch.nn as nn


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample for regularization.

    Randomly drops entire residual branches during training, which acts as
    a form of regularization similar to dropout but applied to entire paths
    rather than individual neurons.

    Args:
        drop_prob: Probability of dropping the path. Default: 0.0
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        if not 0.0 <= drop_prob <= 1.0:
            raise ValueError(f"drop_prob must be between 0 and 1, got {drop_prob}")
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        # Create binary mask with shape (batch_size, 1, 1, ...) for broadcasting
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        # Scale by keep_prob to maintain expected values
        return x * mask / keep_prob

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob}"


class DropPathMixin:
    """Mixin class providing drop path functionality.

    Classes using this mixin should have a `drop_path_prob` attribute
    and a `training` attribute (provided by nn.Module).

    Example:
        class MyBlock(nn.Module, DropPathMixin):
            def __init__(self):
                super().__init__()
                self.drop_path_prob = 0.1

            def forward(self, x):
                return self._drop_path(x)
    """

    drop_path_prob: float
    training: bool

    def _drop_path(self, x: torch.Tensor) -> torch.Tensor:
        """Apply drop path to input tensor.

        Args:
            x: Input tensor of any shape, typically (B, C, H, W) for images

        Returns:
            Tensor with drop path applied during training, unchanged during eval
        """
        if self.drop_path_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_path_prob
        # Create binary mask with shape (batch_size, 1, 1, ...) for broadcasting
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        # Scale by keep_prob to maintain expected values
        return x * mask / keep_prob
