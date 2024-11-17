import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
from ....core.base import NexusModule
from .encoder import MLPEncoder, ConvEncoder
from .decoder import MLPDecoder, ConvDecoder

class EnhancedVAE(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.latent_dim = config["latent_dim"]
        self.beta = config.get("beta", 1.0)
        self.architecture = config.get("architecture", "mlp")
        
        if self.architecture == "mlp":
            self.encoder = MLPEncoder(
                input_dim=config["input_dim"],
                hidden_dim=config["hidden_dim"],
                latent_dim=self.latent_dim
            )
            self.decoder = MLPDecoder(
                latent_dim=self.latent_dim,
                hidden_dim=config["hidden_dim"],
                output_dim=config["input_dim"]
            )
        else:  # conv
            self.encoder = ConvEncoder(
                in_channels=config["in_channels"],
                hidden_dims=config["hidden_dims"],
                latent_dim=self.latent_dim
            )
            self.decoder = ConvDecoder(
                latent_dim=self.latent_dim,
                hidden_dims=config["hidden_dims"],
                out_channels=config["in_channels"]
            )
            
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        # Encode
        mu, log_var = self.encoder(x)
        
        # Reparameterize
        z = self.reparameterize(mu, log_var)
        
        # Decode
        reconstruction = self.decoder(z)
        
        return {
            "reconstruction": reconstruction,
            "mu": mu,
            "log_var": log_var,
            "z": z
        }
        
    def compute_loss(self, batch: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        outputs = self.forward(batch)
        
        # Reconstruction loss
        recon_loss = F.mse_loss(outputs["reconstruction"], batch, reduction="mean")
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(
            1 + outputs["log_var"] - outputs["mu"].pow(2) - outputs["log_var"].exp()
        )
        
        # Total loss (Beta-VAE formulation)
        total_loss = recon_loss + self.beta * kl_loss
        
        return {
            "loss": total_loss,
            "reconstruction_loss": recon_loss,
            "kl_loss": kl_loss
        }