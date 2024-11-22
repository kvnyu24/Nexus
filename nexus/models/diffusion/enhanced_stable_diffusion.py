from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from ...core.base import NexusModule
from .stable_diffusion import StableDiffusion
from ..cv.vae import EnhancedVAE

class EnhancedStableDiffusion(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Validate and prepare VAE config
        vae_config = {
            "architecture": "conv",  # Use convolutional architecture for images
            "in_channels": 3,  # RGB images
            "hidden_dims": [32, 64, 128, 256],  # Progressive dimension increase
            "latent_dim": config.get("latent_dim", 4),  # Default latent dimension
            "beta": config.get("vae_beta", 0.1)  # KL loss weight
        }
        
        # Core components
        self.stable_diffusion = StableDiffusion(config)
        self.vae = EnhancedVAE(vae_config)
        
        # Additional parameters
        self.guidance_scale = config.get("guidance_scale", 7.5)
        self.vae_scale = config.get("vae_scale", 0.18215)
        
    def encode_images(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Encode images to latent space using VAE"""
        vae_outputs = self.vae(images)
        
        # Scale latents by VAE scale factor
        scaled_latents = vae_outputs["z"] * self.vae_scale
        
        return {
            "latents": scaled_latents,
            "mu": vae_outputs["mu"],
            "log_var": vae_outputs["log_var"]
        }
        
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to images using VAE"""
        # Scale latents back to VAE range
        scaled_latents = latents / self.vae_scale
        
        # Use VAE decoder directly for efficiency
        images = self.vae.decoder(scaled_latents)
        return images
        
    def generate(
        self,
        prompt: torch.Tensor,
        negative_prompt: Optional[torch.Tensor] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        image_size: Tuple[int, int] = (512, 512)
    ) -> Dict[str, torch.Tensor]:
        # Use instance guidance scale if not provided
        guidance_scale = guidance_scale or self.guidance_scale
        
        # Generate latents using base StableDiffusion
        latents = self.stable_diffusion.generate(
            prompt=prompt,
            image_size=image_size,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        )
        
        # Decode latents to images
        images = self.decode_latents(latents)
        
        return {
            "images": images,
            "latents": latents
        }
        
    def forward(
        self,
        images: torch.Tensor,
        prompt: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Encode images to latent space
        encoded = self.encode_images(images)
        
        # Get noise prediction from base StableDiffusion
        diffusion_outputs = self.stable_diffusion(
            prompt=prompt,
            noise=encoded["latents"],
            timesteps=timesteps,
            attention_mask=attention_mask
        )
        
        return {
            "noise_pred": diffusion_outputs["noise_pred"],
            "hidden_states": diffusion_outputs["hidden_states"],
            "latents": encoded["latents"],
            "mu": encoded["mu"],
            "log_var": encoded["log_var"]
        }