from nexus.models.cv.nerf import NeRFNetwork, NeRFRenderer
from nexus.training import Trainer
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict
from nexus.data import NeRFDataset

# Configure NeRF model
config = {
    "pos_encoding_dims": 10,
    "dir_encoding_dims": 4,
    "hidden_dim": 256
}

# Initialize model
nerf = NeRFNetwork(config)

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

# Initialize dataset
nerf_dataset = NeRFDataset(
    root_dir='./data/nerf_synthetic/lego',
    split='train',
    img_wh=(400, 400)  # Reduce resolution for faster training
)

# Train model (assuming you have a dataset)
trainer.train(
    train_dataset=nerf_dataset,
    batch_size=1024,
    num_epochs=100
) 