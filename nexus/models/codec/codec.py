import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple, Union
from ...core.base import NexusModule
from ...core.mixins import ConfigValidatorMixin, FeatureBankMixin

class Codec(ConfigValidatorMixin, FeatureBankMixin, NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Validate config using ConfigValidatorMixin
        self.validate_config(config, required_keys=["hidden_dim", "input_dim"])
        self.validate_positive(config["hidden_dim"], "hidden_dim")
        self.validate_positive(config["input_dim"], "input_dim")
        
        # Core dimensions
        self.hidden_dim = config["hidden_dim"]
        self.latent_dim = config.get("latent_dim", self.hidden_dim // 2)
        self.num_layers = config.get("num_layers", 4)
        self.dropout = config.get("dropout", 0.1)
        self.activation = self._get_activation(config.get("activation", "gelu"))
        
        # Encoder network with residual connections
        encoder_layers = []
        in_dim = config["input_dim"]
        for i in range(self.num_layers):
            encoder_layers.extend([
                nn.Linear(in_dim if i == 0 else self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                self.activation,
                nn.Dropout(self.dropout)
            ])
            if i > 0:  # Add residual connection after first layer
                encoder_layers.append(ResidualConnection())
                
        encoder_layers.append(nn.Linear(self.hidden_dim, self.latent_dim * 2))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder network with residual connections
        decoder_layers = []
        for i in range(self.num_layers):
            decoder_layers.extend([
                nn.Linear(self.latent_dim if i == 0 else self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                self.activation,
                nn.Dropout(self.dropout)
            ])
            if i > 0:  # Add residual connection after first layer
                decoder_layers.append(ResidualConnection())
                
        decoder_layers.append(nn.Linear(self.hidden_dim, config["input_dim"]))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Feature bank with temperature scaling using FeatureBankMixin
        self.bank_size = config.get("bank_size", 10000)
        self.register_feature_bank("feature", self.bank_size, self.latent_dim)
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Enhanced quality assessment head
        self.quality_head = nn.Sequential(
            nn.Linear(self.latent_dim * 2, self.hidden_dim // 2),  # Takes both z and relative similarity
            nn.LayerNorm(self.hidden_dim // 2),
            self.activation,
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name"""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "mish": nn.Mish()
        }
        if name not in activations:
            raise ValueError(f"Unsupported activation: {name}")
        return activations[name]

    def compute_similarity(self, z: torch.Tensor) -> torch.Tensor:
        """Compute similarity to feature bank entries"""
        # Normalize features
        z_norm = F.normalize(z, dim=1)
        bank_norm = F.normalize(self.get_feature_bank("feature"), dim=1)
        
        # Compute cosine similarity with temperature scaling
        similarity = torch.mm(z_norm, bank_norm.t()) * self.temperature
        return similarity

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick with numerical stability"""
        std = torch.exp(0.5 * torch.clamp(log_var, min=-10, max=10))
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: torch.Tensor, return_similarity: bool = True) -> Dict[str, torch.Tensor]:
        """Enhanced encoding with similarity computation"""
        encoder_output = self.encoder(x)
        mu, log_var = torch.chunk(encoder_output, 2, dim=-1)
        z = self.reparameterize(mu, log_var)
        
        # Update feature bank using FeatureBankMixin
        self.update_feature_bank("feature", z)
        
        # Compute similarity if requested
        similarity = self.compute_similarity(z) if return_similarity else None
        
        # Assess encoding quality using both z and similarity
        quality_input = torch.cat([z, similarity.max(dim=1, keepdim=True)[0]], dim=1) if return_similarity else z
        quality = self.quality_head(quality_input)
        
        outputs = {
            "z": z,
            "mu": mu,
            "log_var": log_var,
            "quality": quality
        }
        if return_similarity:
            outputs["similarity"] = similarity
            
        return outputs

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation with gradient clipping"""
        return self.decoder(z)

    def forward(
        self,
        x: torch.Tensor,
        return_latents: bool = False,
        return_similarity: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Enhanced forward pass with additional features"""
        # Input validation
        if not torch.isfinite(x).all():
            raise ValueError("Input contains inf or nan")
            
        # Encode
        encode_outputs = self.encode(x, return_similarity=return_similarity)
        
        # Decode
        reconstruction = self.decode(encode_outputs["z"])
        
        outputs = {
            "reconstruction": reconstruction,
            "quality": encode_outputs["quality"]
        }
        
        if return_similarity:
            outputs["similarity"] = encode_outputs["similarity"]
            
        if return_latents:
            outputs.update({
                "z": encode_outputs["z"],
                "mu": encode_outputs["mu"],
                "log_var": encode_outputs["log_var"]
            })
            
        return outputs

class ResidualConnection(NexusModule):
    """Residual connection with scaling."""
    def __init__(self, scale: float = 0.1):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * x