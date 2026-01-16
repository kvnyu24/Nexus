import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention.

    This module implements the SE block from "Squeeze-and-Excitation Networks"
    (Hu et al., 2018). It adaptively recalibrates channel-wise feature responses
    by explicitly modeling interdependencies between channels.

    Args:
        channels: Number of input/output channels.
        reduction_ratio: Reduction ratio for the bottleneck. Default: 16.
        activation: Activation function for the bottleneck. Default: nn.ReLU.
    """

    def __init__(
        self,
        channels: int,
        reduction_ratio: int = 16,
        activation: nn.Module = None
    ):
        super().__init__()

        if channels < reduction_ratio:
            raise ValueError(
                f"channels ({channels}) must be >= reduction_ratio ({reduction_ratio})"
            )

        reduced_channels = max(1, channels // reduction_ratio)

        if activation is None:
            activation = nn.ReLU(inplace=True)

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=True),
            activation,
            nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply squeeze-and-excitation attention.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Tensor of shape (B, C, H, W) with channel attention applied.
        """
        scale = self.squeeze(x)
        scale = self.excitation(scale)
        return x * scale
