import torch
import torch.nn as nn
from typing import Optional
from nexus.core.base import NexusModule

class ResidualBlock(NexusModule):
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
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 16, out_channels, 1),
            nn.Sigmoid()
        )
        
        # Stochastic depth for regularization
        self.drop_path_prob = 0.1
        
    def _drop_path(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            keep_prob = 1 - self.drop_path_prob
            mask = torch.zeros_like(x[0, 0, 0, 0]).bernoulli_(keep_prob)
            mask = mask.view(1, 1, 1, 1).expand_as(x) / keep_prob
            return x * mask
        return x
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # Input validation
        if not torch.isfinite(x).all():
            raise ValueError("Input contains inf or nan values")
            
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Apply SE attention
        out = out * self.se(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        # Apply stochastic depth
        out = self._drop_path(out)
        
        # Residual connection with gradient checkpointing
        if torch.jit.is_scripting():
            out += identity
        else:
            torch.utils.checkpoint.checkpoint(lambda x, y: x + y, out, identity)
            
        out = self.relu(out)
        
        return out

class InvertedResidualBlock(NexusModule):
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
        squeeze_dim = max(1, in_channels // squeeze_factor)
        
        self.use_residual = stride == 1 and in_channels == out_channels
        
        layers = []
        # Enhanced pointwise with squeeze-excitation
        layers.extend([
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            norm_layer(hidden_dim),
            nn.SiLU(inplace=True)  # Swish activation
        ])
        
        # Improved depthwise with dilated convolutions
        layers.extend([
            nn.Conv2d(
                hidden_dim, hidden_dim, 3, stride, 
                padding=2, dilation=2, groups=hidden_dim, bias=False
            ),
            norm_layer(hidden_dim),
            nn.SiLU(inplace=True)
        ])
        
        # SE block
        layers.extend([
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, squeeze_dim, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(squeeze_dim, hidden_dim, 1),
            nn.Sigmoid()
        ])
        
        # Enhanced pointwise projection
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            norm_layer(out_channels)
        ])
        
        self.layers = nn.Sequential(*layers)
        self.drop_path_prob = 0.2
        
    def _drop_path(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            keep_prob = 1 - self.drop_path_prob
            mask = torch.zeros_like(x[0, 0, 0, 0]).bernoulli_(keep_prob)
            mask = mask.view(1, 1, 1, 1).expand_as(x) / keep_prob
            return x * mask
        return x
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.isfinite(x).all():
            raise ValueError("Input contains inf or nan values")
            
        out = self.layers(x)
        
        if self.use_residual:
            out = self._drop_path(out)
            return x + out
        return out