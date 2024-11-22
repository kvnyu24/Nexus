from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from ...core.base import NexusModule

class BaseDiffusion(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Diffusion parameters
        self.num_timesteps = config.get("num_timesteps", 1000)
        self.beta_schedule = config.get("beta_schedule", "linear")
        self.beta_start = config.get("beta_start", 0.0001)
        self.beta_end = config.get("beta_end", 0.02)
        
        # Register diffusion schedule buffers
        self.register_schedule()
        
    def register_schedule(self):
        """Initialize and register diffusion schedule buffers"""
        if self.beta_schedule == "linear":
            betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)
        elif self.beta_schedule == "cosine":
            steps = torch.arange(self.num_timesteps + 1, dtype=torch.float32) / self.num_timesteps
            alpha_bar = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2) ** 2
            betas = torch.clip(1 - alpha_bar[1:] / alpha_bar[:-1], 0.0, 0.999)
        else:
            raise ValueError(f"Unknown beta schedule: {self.beta_schedule}")
        
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        
        # Register all buffers
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        
        # Calculations for diffusion q(x_t | x_{t-1})
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod)
        
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward diffusion process"""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].flatten()
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].flatten()
        
        return (
            sqrt_alphas_cumprod_t.view(-1, 1, 1, 1) * x_start +
            sqrt_one_minus_alphas_cumprod_t.view(-1, 1, 1, 1) * noise
        ) 