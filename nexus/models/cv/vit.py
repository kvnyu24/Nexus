import torch
import torch.nn as nn
from typing import Dict, Any
from ...components.attention import MultiHeadSelfAttention
from ...core.base import NexusModule

class PatchEmbedding(nn.Module):
    def __init__(self, image_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.projection(x)  # (B, E, H', W')
        x = x.flatten(2)  # (B, E, N)
        x = x.transpose(1, 2)  # (B, N, E)
        return x

class VisionTransformer(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.in_channels = config["in_channels"]
        self.num_classes = config["num_classes"]
        self.embed_dim = config["embed_dim"]
        self.num_layers = config["num_layers"]
        self.num_heads = config["num_heads"]
        self.mlp_ratio = config["mlp_ratio"]
        self.dropout = config["dropout"]
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            self.image_size, self.patch_size,
            self.in_channels, self.embed_dim
        )
        
        # Position embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, self.embed_dim)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout=self.dropout
            ) for _ in range(self.num_layers)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, self.num_classes)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add cls token and position embedding
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Classification
        x = self.norm(x)
        x = x[:, 0]  # Use CLS token
        logits = self.head(x)
        
        return {
            "logits": logits,
            "embeddings": x
        } 