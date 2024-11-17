from nexus.models.vision.nerf import EnhancedNeRF, NeRFRenderer
from nexus.training import Trainer, CosineWarmupScheduler
import torch
import numpy as np
from typing import Dict
# Configure NeRF model
config = {
    "pos_encoding_dims": 10,
    "dir_encoding_dims": 4,
    "hidden_dim": 256
}

# Initialize model
nerf = EnhancedNeRF(config)

# Create custom trainer for NeRF
class NeRFTrainer(Trainer):
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.optimizer.zero_grad()
        
        # Extract batch data
        ray_origins = batch["ray_origins"].to(self.device)
        ray_directions = batch["ray_directions"].to(self.device)
        target_rgb = batch["target_rgb"].to(self.device)
        
        # Render rays
        outputs = self.model.render_rays(
            ray_origins=ray_origins,
            ray_directions=ray_directions,
            near=2.0,
            far=6.0,
            num_samples=64,
            noise_std=0.0
        )
        
        # Compute loss
        loss = F.mse_loss(outputs["rgb"], target_rgb)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return {"loss": loss.item()}

# Initialize trainer
trainer = NeRFTrainer(
    model=nerf,
    optimizer="adam",
    learning_rate=5e-4,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Train model (assuming you have a dataset)
trainer.train(
    train_dataset=nerf_dataset,
    batch_size=1024,
    num_epochs=100
) 