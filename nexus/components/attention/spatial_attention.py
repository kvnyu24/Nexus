import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .base_attention import BaseAttention

class SpatialAttention(BaseAttention):
    def __init__(self, kernel_size: int = 7, dropout: float = 0.1):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size should be odd to maintain spatial dimensions")
            
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.norm = nn.BatchNorm2d(1)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor (batch, channels, height, width), got shape {x.shape}")
            
        # Calculate attention weights using both average and max pooling
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_pool, max_pool], dim=1)
        
        # Apply convolution and normalization
        attention = self.conv(attention)
        attention = self.norm(attention)
        attention = self.dropout(attention)
        attention = self.sigmoid(attention)
        
        if mask is not None:
            attention = attention.masked_fill(mask.unsqueeze(1) == 0, 0)
            
        return x * attention

class ChannelAttention(BaseAttention):
    def __init__(self, channels: int, reduction: int = 16, dropout: float = 0.1):
        super().__init__()
        if channels % reduction != 0:
            raise ValueError(f"Channels ({channels}) must be divisible by reduction ({reduction})")
            
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        reduced_channels = channels // reduction
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.LayerNorm(reduced_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(reduced_channels, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor (batch, channels, height, width), got shape {x.shape}")
            
        b, c, _, _ = x.size()
        
        # Process through both pooling branches
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        # Combine and apply attention
        attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        
        if mask is not None:
            attention = attention.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, 0)
            
        return x * attention