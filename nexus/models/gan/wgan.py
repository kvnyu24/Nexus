import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from ...core.base import NexusModule
from ...core.initialization import WeightInitMixin
from .base_gan import BaseGenerator, BaseDiscriminator

class WGANGenerator(BaseGenerator):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Replace final activation for WGAN
        main_layers = list(self.main.children())[:-1]  # Remove Tanh
        self.main = nn.Sequential(*main_layers)

        # Re-initialize weights after modifying architecture
        self.init_weights_gan()

class WGANCritic(BaseDiscriminator):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Replace final sigmoid with linear layer for WGAN
        main_layers = list(self.main.children())[:-1]
        self.main = nn.Sequential(
            *main_layers,
            nn.Conv2d(self.hidden_dim * 8, 1, 4, 1, 0)
        )

        # Re-initialize weights after modifying architecture
        self.init_weights_gan()

class WGAN(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Validate configuration
        self._validate_config(config)
        
        # Initialize generator and critic
        self.generator = WGANGenerator(config)
        self.critic = WGANCritic(config)
        
        # WGAN specific configuration
        self.clip_value = config.get("clip_value", 0.01)
        self.n_critic = config.get("n_critic", 5)
        
    def _validate_config(self, config: Dict[str, Any]) -> None:
        required_keys = ["latent_dim", "hidden_dim"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")
                
        if config.get("clip_value", 0.01) <= 0:
            raise ValueError("clip_value must be positive")
            
    def clip_critic_weights(self):
        for p in self.critic.parameters():
            p.data.clamp_(-self.clip_value, self.clip_value)
            
    def get_critic_loss(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # Wasserstein loss
        real_validity = self.critic(real_images)["validity"]
        fake_validity = self.critic(fake_images.detach())["validity"]
        
        critic_loss = -torch.mean(real_validity) + torch.mean(fake_validity)
        
        return {
            "critic_loss": critic_loss,
            "wasserstein_distance": -critic_loss
        }
        
    def get_generator_loss(
        self,
        fake_images: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # Generator loss
        fake_validity = self.critic(fake_images)["validity"]
        generator_loss = -torch.mean(fake_validity)
        
        return {
            "generator_loss": generator_loss
        } 