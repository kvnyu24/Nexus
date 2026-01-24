import torch
import torch.nn as nn
from typing import Optional
from nexus.core.base import NexusModule
from nexus.core.mixins import InputValidationMixin
from nexus.components.layers import DropPath, SEBlock

class ResidualBlock(InputValidationMixin, NexusModule):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[NexusModule] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[nn.Module] = None
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        # Validate inputs
        if groups < 1:
            raise ValueError('Number of groups must be positive')
        if base_width < 1:
            raise ValueError('Base width must be positive')
            
        width = int(out_channels * (base_width / 64.)) * groups
        
        # Enhanced convolution blocks with groups and dilation
        self.conv1 = nn.Conv2d(
            in_channels, width,
            kernel_size=3, stride=stride, padding=dilation,
            groups=groups, dilation=dilation, bias=False
        )
        self.bn1 = norm_layer(width)
        
        self.conv2 = nn.Conv2d(
            width, out_channels,
            kernel_size=3, stride=1, padding=1,
            groups=groups, bias=False
        )
        self.bn2 = norm_layer(out_channels)
        
        # Improved activation with PReLU
        self.relu = nn.PReLU()
        self.downsample = downsample
        
        # SE block for channel attention
        self.se = SEBlock(out_channels, reduction_ratio=16)
        
        # Stochastic depth for regularization
        self.drop_path = DropPath(drop_prob=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # Input validation
        self.validate_finite(x, name="input")
            
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply SE attention
        out = self.se(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        # Apply stochastic depth
        out = self.drop_path(out)
        
        # Residual connection with gradient checkpointing
        if torch.jit.is_scripting():
            out = out + identity
        else:
            out = out + identity  # Simple addition; checkpointing not beneficial here
            
        out = self.relu(out)
        
        return out

class InvertedResidualBlock(InputValidationMixin, NexusModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expansion_factor: int = 6,
        squeeze_factor: int = 4,
        norm_layer: Optional[nn.Module] = None
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = in_channels * expansion_factor

        self.use_residual = stride == 1 and in_channels == out_channels

        # Enhanced pointwise expansion
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            norm_layer(hidden_dim),
            nn.SiLU(inplace=True)  # Swish activation
        )

        # Improved depthwise with dilated convolutions
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                hidden_dim, hidden_dim, 3, stride,
                padding=2, dilation=2, groups=hidden_dim, bias=False
            ),
            norm_layer(hidden_dim),
            nn.SiLU(inplace=True)
        )

        # SE block for channel attention
        self.se = SEBlock(
            hidden_dim,
            reduction_ratio=squeeze_factor,
            activation=nn.SiLU(inplace=True)
        )

        # Enhanced pointwise projection
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            norm_layer(out_channels)
        )

        # Stochastic depth for regularization
        self.drop_path = DropPath(drop_prob=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.validate_finite(x, name="input")

        out = self.expand(x)
        out = self.depthwise(out)
        out = self.se(out)
        out = self.project(out)

        if self.use_residual:
            out = self.drop_path(out)
            return x + out
        return out