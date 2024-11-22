import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from ...core.base import NexusModule

class BaseGenerator(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Core configuration
        self.latent_dim = config["latent_dim"]
        self.hidden_dim = config.get("hidden_dim", 512)
        self.output_channels = config.get("output_channels", 3)
        self.output_size = config.get("output_size", 64)
        
        # Build generator architecture
        self.main = nn.Sequential(
            # Initial projection
            nn.Linear(self.latent_dim, self.hidden_dim * 4 * 4),
            nn.BatchNorm1d(self.hidden_dim * 4 * 4),
            nn.ReLU(True),
            
            # Reshape
            nn.Unflatten(1, (self.hidden_dim, 4, 4)),
            
            # Upsampling layers
            *self._build_upsampling_layers()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _build_upsampling_layers(self) -> list:
        layers = []
        current_dim = self.hidden_dim
        current_size = 4
        
        while current_size < self.output_size:
            layers.extend([
                nn.ConvTranspose2d(current_dim, current_dim // 2, 4, 2, 1),
                nn.BatchNorm2d(current_dim // 2),
                nn.ReLU(True)
            ])
            current_dim //= 2
            current_size *= 2
            
        layers.append(
            nn.Conv2d(current_dim, self.output_channels, 3, 1, 1)
        )
        layers.append(nn.Tanh())
        return layers
        
    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight, 1.0, 0.02)
            nn.init.zeros_(module.bias)
            
    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        generated = self.main(z)
        return {"generated_images": generated}

class BaseDiscriminator(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Core configuration
        self.input_channels = config.get("input_channels", 3)
        self.hidden_dim = config.get("hidden_dim", 64)
        self.input_size = config.get("input_size", 64)
        
        # Build discriminator architecture
        self.main = nn.Sequential(
            # Initial layer
            nn.Conv2d(self.input_channels, self.hidden_dim, 4, 2, 1),
            nn.LeakyReLU(0.2, True),
            
            # Downsampling layers
            *self._build_downsampling_layers(),
            
            # Final layers
            nn.Conv2d(self.hidden_dim * 8, 1, 4, 1, 0),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _build_downsampling_layers(self) -> list:
        layers = []
        current_dim = self.hidden_dim
        
        while current_dim < self.hidden_dim * 8:
            layers.extend([
                nn.Conv2d(current_dim, current_dim * 2, 4, 2, 1),
                nn.BatchNorm2d(current_dim * 2),
                nn.LeakyReLU(0.2, True)
            ])
            current_dim *= 2
            
        return layers
        
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight, 1.0, 0.02)
            nn.init.zeros_(module.bias)
            
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        validity = self.main(x)
        return {
            "validity": validity,
            "features": x  # Return input features for feature matching if needed
        }

class BaseGAN(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Initialize generator and discriminator
        self.generator = BaseGenerator(config)
        self.discriminator = BaseDiscriminator(config)
        
        # Loss configuration
        self.adversarial_loss = nn.BCELoss()
        
    def forward(
        self,
        real_images: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
        mode: str = "generate"
    ) -> Dict[str, torch.Tensor]:
        if mode == "generate" and z is not None:
            return self.generator(z)
        elif mode == "discriminate" and real_images is not None:
            return self.discriminator(real_images)
        else:
            raise ValueError("Invalid mode or missing inputs")
            
    def get_generator_loss(
        self,
        fake_images: torch.Tensor,
        valid_labels: torch.Tensor
    ) -> torch.Tensor:
        return self.adversarial_loss(
            self.discriminator(fake_images)["validity"],
            valid_labels
        )
        
    def get_discriminator_loss(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
        valid_labels: torch.Tensor,
        fake_labels: torch.Tensor
    ) -> torch.Tensor:
        real_loss = self.adversarial_loss(
            self.discriminator(real_images)["validity"],
            valid_labels
        )
        fake_loss = self.adversarial_loss(
            self.discriminator(fake_images.detach())["validity"],
            fake_labels
        )
        return (real_loss + fake_loss) / 2 