import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from ...core.base import NexusModule
from ...core.mixins import ConfigValidatorMixin
from .base_gan import BaseGenerator, BaseDiscriminator
from ...components import ResidualBlock

class CycleGANGenerator(BaseGenerator):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Add residual blocks for better style transfer
        self.n_residual = config.get("n_residual_blocks", 9)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(self.hidden_dim)
            for _ in range(self.n_residual)
        ])
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Initial convolution
        out = self.main(x)
        
        # Residual blocks
        for block in self.residual_blocks:
            out = block(out)
            
        return {"generated_images": out}

class CycleGAN(ConfigValidatorMixin, NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Validate configuration
        self.validate_config(config, required_keys=["input_channels", "hidden_dim"])

        # Initialize generators and discriminators
        self.G_AB = CycleGANGenerator(config)
        self.G_BA = CycleGANGenerator(config)
        self.D_A = BaseDiscriminator(config)
        self.D_B = BaseDiscriminator(config)

        # Loss weights
        self.lambda_cycle = config.get("lambda_cycle", 10.0)
        self.lambda_identity = config.get("lambda_identity", 0.5)

        # Loss functions
        self.criterion_gan = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
                
    def get_cycle_consistency_loss(
        self,
        real_A: torch.Tensor,
        real_B: torch.Tensor,
        fake_A: torch.Tensor,
        fake_B: torch.Tensor,
        rec_A: torch.Tensor,
        rec_B: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # Cycle consistency loss
        cycle_A = self.criterion_cycle(rec_A, real_A)
        cycle_B = self.criterion_cycle(rec_B, real_B)
        
        # Identity loss
        identity_A = self.criterion_identity(
            self.G_BA(real_A)["generated_images"],
            real_A
        )
        identity_B = self.criterion_identity(
            self.G_AB(real_B)["generated_images"],
            real_B
        )
        
        return {
            "cycle_loss": cycle_A + cycle_B,
            "identity_loss": identity_A + identity_B
        } 