"""Depthwise Separable Convolution layer."""
import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from nexus.core.base import NexusModule


class DepthwiseSeparableConv(NexusModule):
    """Depthwise Separable Convolution.

    Factorizes standard convolution into depthwise and pointwise convolutions
    for improved efficiency. Used in MobileNet, EfficientNet, and other
    efficient architectures.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolving kernel
        stride: Stride of the convolution
        padding: Padding added to input
        dilation: Spacing between kernel elements
        bias: If True, adds learnable bias
        activation: Activation function (default: ReLU)
        use_bn: Whether to use batch normalization
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 1,
        dilation: Union[int, Tuple[int, int]] = 1,
        bias: bool = False,
        activation: Optional[nn.Module] = None,
        use_bn: bool = True
    ):
        super().__init__()

        # Depthwise convolution (groups=in_channels)
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False
        )

        # Batch norm after depthwise
        self.bn1 = nn.BatchNorm2d(in_channels) if use_bn else nn.Identity()

        # Pointwise convolution (1x1)
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias
        )

        # Batch norm after pointwise
        self.bn2 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()

        # Activation
        self.activation = activation if activation is not None else nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, in_channels, height, width)

        Returns:
            Output tensor of shape (batch, out_channels, height', width')
        """
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.activation(x)

        return x


class DepthwiseConv2d(NexusModule):
    """Pure depthwise convolution without pointwise.

    Args:
        channels: Number of input/output channels
        kernel_size: Size of the convolving kernel
        stride: Stride of the convolution
        padding: Padding added to input
        dilation: Spacing between kernel elements
        bias: If True, adds learnable bias
    """

    def __init__(
        self,
        channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 1,
        dilation: Union[int, Tuple[int, int]] = 1,
        bias: bool = False
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=channels,
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
