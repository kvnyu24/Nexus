import torch
import torch.nn as nn
from typing import Dict, Any
from ....core.base import NexusModule

class HierarchicalPatchEmbedding(NexusModule):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        patch_size: int
    ):
        super().__init__({})
        
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        
        # Patch embedding
        self.proj = nn.Conv2d(
            in_channels,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Position embeddings will be added dynamically
        self.pos_embed = None
        
    def _get_position_embeddings(
        self,
        height: int,
        width: int,
        device: torch.device
    ) -> torch.Tensor:
        if self.pos_embed is None or self.pos_embed.shape[1] != height * width:
            # Generate new position embeddings
            pos_embed = nn.Parameter(
                torch.zeros(1, height * width, self.hidden_dim)
            )
            nn.init.trunc_normal_(pos_embed, std=0.02)
            self.pos_embed = pos_embed.to(device)
            
        return self.pos_embed
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project patches
        x = self.proj(x)
        
        # Reshape to sequence
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B, N, C
        
        # Add position embeddings
        pos_embed = self._get_position_embeddings(H, W, x.device)
        x = x + pos_embed
        
        # Layer norm
        x = self.norm(x)
        
        return x 