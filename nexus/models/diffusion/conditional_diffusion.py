from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from .base_diffusion import BaseDiffusion

class ConditionalDiffusion(BaseDiffusion):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Model dimensions
        self.hidden_dim = config.get("hidden_dim", 256)
        self.condition_dim = config.get("condition_dim", 512)
        
        # Condition processor
        self.condition_processor = nn.Sequential(
            nn.Linear(self.condition_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim)
        )
        
    def get_time_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Convert timesteps to sinusoidal embeddings"""
        half_dim = self.hidden_dim // 2
        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return self.time_embed(embeddings)
        
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # Process condition
        cond_embedding = self.condition_processor(condition)
        
        # Get time embedding
        time_embedding = self.get_time_embedding(timesteps)
        
        # Combine embeddings
        combined_embedding = cond_embedding + time_embedding
        
        return {
            "embeddings": combined_embedding,
            "time_embedding": time_embedding,
            "condition_embedding": cond_embedding
        } 