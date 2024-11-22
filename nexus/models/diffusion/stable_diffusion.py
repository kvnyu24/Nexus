from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
from ...core.base import NexusModule
from .unet import UNet
from ..nlp.t5 import EnhancedT5

class StableDiffusion(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Core components
        self.text_encoder = EnhancedT5(config)
        self.unet = UNet(config)
        
        # Diffusion parameters
        self.num_inference_steps = config.get("num_inference_steps", 50)
        self.beta_start = config.get("beta_start", 0.00085)
        self.beta_end = config.get("beta_end", 0.012)
        
        # Register buffers for diffusion schedule
        self.register_buffer(
            "betas",
            torch.linspace(self.beta_start, self.beta_end, self.num_inference_steps)
        )
        alphas = 1.0 - self.betas
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))
        
    def _get_noise_schedule(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha_cumprod = self.alphas_cumprod[t]
        alpha_cumprod_prev = torch.cat([
            torch.ones_like(alpha_cumprod[:1]),
            self.alphas_cumprod[:-1]
        ])[t]
        return alpha_cumprod, alpha_cumprod_prev
        
    def forward(
        self,
        prompt: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Encode text prompt
        text_outputs = self.text_encoder(
            input_ids=prompt,
            attention_mask=attention_mask
        )
        
        # Get noise prediction from UNet
        unet_outputs = self.unet(
            x=noise,
            timesteps=timesteps,
            context=text_outputs["encoder_states"]
        )
        
        return {
            "noise_pred": unet_outputs["sample"],
            "hidden_states": unet_outputs["hidden_states"]
        }
        
    def generate(
        self,
        prompt: torch.Tensor,
        image_size: Tuple[int, int] = (512, 512),
        num_inference_steps: Optional[int] = None,
        guidance_scale: float = 7.5
    ) -> torch.Tensor:
        batch_size = prompt.shape[0]
        device = prompt.device
        
        # Initialize image with random noise
        latents = torch.randn(
            (batch_size, 3, *image_size),
            device=device
        )
        
        # Setup timesteps
        timesteps = torch.linspace(
            0, self.num_inference_steps - 1,
            num_inference_steps or self.num_inference_steps,
            device=device
        ).long()
        
        # Diffusion process
        for t in timesteps:
            # Get noise schedule
            alpha, alpha_prev = self._get_noise_schedule(t)
            
            # Predict noise
            with torch.no_grad():
                noise_pred = self(
                    prompt=prompt,
                    noise=latents,
                    timesteps=t.expand(batch_size)
                )["noise_pred"]
            
            # Update latents
            latents = (
                alpha_prev.sqrt() * (latents - (1 - alpha).sqrt() * noise_pred) / 
                alpha.sqrt()
            )
            
        return latents 