from nexus.models.cv.nerf import HierarchicalNeRF
from nexus.training import Trainer
import torch
import torch.nn.functional as F
from typing import Dict


class HierarchicalNeRFTrainer(Trainer):
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.optimizer.zero_grad()
        
        # Move batch data to device
        ray_origins = batch["ray_origins"].to(self.device)
        ray_directions = batch["ray_directions"].to(self.device)
        target_rgb = batch["target_rgb"].to(self.device)
        
        # Process rays in chunks
        chunk_size = 4096
        total_coarse_loss = 0
        total_fine_loss = 0
        
        for i in range(0, ray_origins.shape[0], chunk_size):
            chunk_origins = ray_origins[i:i+chunk_size]
            chunk_directions = ray_directions[i:i+chunk_size]
            chunk_target = target_rgb[i:i+chunk_size]
            
            # Render with hierarchical sampling
            outputs = self.model(
                ray_origins=chunk_origins,
                ray_directions=chunk_directions,
                near=2.0,
                far=6.0,
                num_coarse=64,
                num_fine=128,
                noise_std=0.0
            )
            
            # Compute losses
            coarse_loss = F.mse_loss(outputs["coarse"]["rgb"], chunk_target)
            fine_loss = F.mse_loss(outputs["fine"]["rgb"], chunk_target)
            
            # Combined loss with higher weight on fine network
            loss = coarse_loss + 2.0 * fine_loss
            loss.backward()
            
            total_coarse_loss += coarse_loss.item()
            total_fine_loss += fine_loss.item()
            
        self.optimizer.step()
        
        return {
            "coarse_loss": total_coarse_loss,
            "fine_loss": total_fine_loss,
            "total_loss": total_coarse_loss + total_fine_loss
        } 