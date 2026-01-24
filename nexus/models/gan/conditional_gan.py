import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from ...core.base import NexusModule
from ...core.mixins import ConfigValidatorMixin
from .base_gan import BaseGenerator, BaseDiscriminator

class ConditionalGenerator(BaseGenerator):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Additional conditional components
        self.num_classes = config["num_classes"]
        self.embedding_dim = config.get("embedding_dim", 32)
        
        self.class_embedding = nn.Embedding(
            num_embeddings=self.num_classes,
            embedding_dim=self.embedding_dim
        )
        
        # Modify initial projection to account for class embedding
        self.initial_proj = nn.Linear(
            self.latent_dim + self.embedding_dim,
            self.hidden_dim * 4 * 4
        )
        
    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Get class embeddings
        class_embed = self.class_embedding(labels)
        
        # Concatenate noise and class embedding
        combined = torch.cat([z, class_embed], dim=1)
        
        # Generate image
        x = self.initial_proj(combined)
        x = x.view(-1, self.hidden_dim, 4, 4)
        generated = self.main(x)
        
        return {
            "generated_images": generated,
            "class_embeddings": class_embed
        }

class ConditionalDiscriminator(BaseDiscriminator):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.num_classes = config["num_classes"]
        self.embedding_dim = config.get("embedding_dim", 32)
        
        # Class embedding
        self.class_embedding = nn.Embedding(
            num_embeddings=self.num_classes,
            embedding_dim=self.embedding_dim
        )
        
        # Additional projection for class conditioning
        self.class_proj = nn.Sequential(
            nn.Linear(self.embedding_dim, self.input_size * self.input_size),
            nn.LeakyReLU(0.2)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        batch_size = x.size(0)
        
        # Process class conditioning
        class_embed = self.class_embedding(labels)
        class_proj = self.class_proj(class_embed)
        class_proj = class_proj.view(
            batch_size, 1, self.input_size, self.input_size
        )
        
        # Concatenate image with class conditioning
        conditioned_input = torch.cat([x, class_proj], dim=1)
        
        # Get discriminator output
        validity = self.main(conditioned_input)
        
        return {
            "validity": validity,
            "class_embeddings": class_embed
        }

class ConditionalGAN(ConfigValidatorMixin, NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Validate config
        self.validate_config(config, required_keys=["num_classes", "latent_dim"])
        self.validate_positive(config["num_classes"], "num_classes")

        # Initialize generator and discriminator
        self.generator = ConditionalGenerator(config)
        self.discriminator = ConditionalDiscriminator(config)

        # Loss configuration
        self.adversarial_loss = nn.BCELoss()
        self.auxiliary_loss = nn.CrossEntropyLoss()
            
    def forward(
        self,
        z: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        mode: str = "generate"
    ) -> Dict[str, torch.Tensor]:
        if mode == "generate" and z is not None and labels is not None:
            return self.generator(z, labels)
        elif mode == "discriminate" and images is not None and labels is not None:
            return self.discriminator(images, labels)
        else:
            raise ValueError("Invalid mode or missing inputs")
            
    def get_generator_loss(
        self,
        fake_outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        valid: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # Adversarial loss
        gen_loss = self.adversarial_loss(
            fake_outputs["validity"],
            valid
        )
        
        return {
            "generator_loss": gen_loss,
            "total_loss": gen_loss
        }
        
    def get_discriminator_loss(
        self,
        real_outputs: Dict[str, torch.Tensor],
        fake_outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        valid: torch.Tensor,
        fake: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # Real and fake losses
        real_loss = self.adversarial_loss(
            real_outputs["validity"],
            valid
        )
        fake_loss = self.adversarial_loss(
            fake_outputs["validity"].detach(),
            fake
        )
        
        # Total discriminator loss
        total_loss = (real_loss + fake_loss) / 2
        
        return {
            "discriminator_loss": total_loss,
            "real_loss": real_loss,
            "fake_loss": fake_loss,
            "total_loss": total_loss
        } 