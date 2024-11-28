from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
from ...core.base import NexusModule
from ...components.attention import MultiHeadSelfAttention

class TimeEmbedding(NexusModule):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        # Create sinusoidal embeddings
        half_dim = self.embedding.weight.shape[1] // 2
        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        
        # Project to desired dimensionality
        return self.embedding(embeddings)

class CrossAttentionBlock(NexusModule):
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(
            hidden_size=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.attention(self.norm(x), context)

class UNet(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Model dimensions
        self.hidden_dim = config["hidden_dim"]
        self.num_heads = config.get("num_heads", 8)
        self.attention_levels = config.get("attention_levels", [1, 2, 4, 8])
        
        # Time embedding
        self.time_embed = TimeEmbedding(self.hidden_dim)
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList([
            self._make_encoder_block(
                in_channels=self.hidden_dim * level,
                out_channels=self.hidden_dim * level * 2
            ) for level in self.attention_levels
        ])
        
        # Decoder blocks with skip connections
        self.decoder_blocks = nn.ModuleList([
            self._make_decoder_block(
                in_channels=self.hidden_dim * level * 2,
                out_channels=self.hidden_dim * level
            ) for level in reversed(self.attention_levels)
        ])
        
        # Output projection
        self.output_proj = nn.Conv2d(self.hidden_dim, 3, kernel_size=1)
        
    def _make_encoder_block(self, in_channels: int, out_channels: int) -> NexusModule:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            CrossAttentionBlock(out_channels, self.num_heads)
        )
        
    def _make_decoder_block(self, in_channels: int, out_channels: int) -> NexusModule:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            CrossAttentionBlock(out_channels, self.num_heads)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Time embedding
        t_emb = self.time_embed(timesteps)
        
        # Encoder path with skip connections
        skip_connections = []
        for encoder in self.encoder_blocks:
            x = encoder(x)
            skip_connections.append(x)
            
        # Decoder path
        for decoder, skip in zip(self.decoder_blocks, reversed(skip_connections)):
            x = decoder(torch.cat([x, skip], dim=1))
            
        return {
            "sample": self.output_proj(x),
            "hidden_states": x
        } 