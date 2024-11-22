from nexus.models.cv.nerf import NeRFNetwork, NeRFRenderer
from nexus.training import Trainer
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict
from nexus.data import NeRFDataset, DataLoader
import torch.multiprocessing as mp
from nexus.training.distributed import DistributedTrainer



# Configure NeRF model
config = {
    "pos_encoding_dims": 10,
    "dir_encoding_dims": 4,
    "hidden_dim": 256
}

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

def train_nerf():
    # Initialize model
    nerf = NeRFNetwork(config)

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

    # Create data loader using Nexus DataLoader
    train_loader = DataLoader(
        dataset=nerf_dataset,
        batch_size=1,  # Process one image at a time
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Define number of training epochs
    num_epochs = 100

    # Training loop
    for epoch in range(num_epochs):
        for batch in train_loader:
            loss_dict = trainer.train_step(batch)
            print(f"Epoch {epoch}, Loss: {loss_dict['loss']:.4f}")

def train_distributed_nerf(rank, world_size):
    # Initialize distributed trainer
    nerf = NeRFNetwork(config)
    trainer = DistributedTrainer(
        model=nerf,
        rank=rank,
        world_size=world_size,
        checkpoint_dir="checkpoints/nerf"
    )
    
    # Initialize dataset with distributed sampler
    nerf_dataset = NeRFDataset(
        root_dir='./data/nerf_synthetic/lego',
        split='train',
        img_wh=(400, 400)
    )
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        nerf_dataset,
        num_replicas=world_size,
        rank=rank
    )
    
    train_loader = torch.utils.data.DataLoader(
        nerf_dataset,
        batch_size=1,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # Training loop
    for epoch in range(100):
        train_sampler.set_epoch(epoch)
        for batch in train_loader:
            metrics = trainer.train_step(batch)
            if rank == 0:
                print(f"Epoch {epoch}, Loss: {metrics['loss']:.4f}")
    
    trainer.cleanup()


def main():
    # Optional: Set up distributed training if needed
    if torch.cuda.device_count() > 1:
        # Reference distributed training setup from examples/distributed_training.py
        world_size = torch.cuda.device_count()
        mp.spawn(train_distributed_nerf, args=(world_size,), nprocs=world_size)
    else:
        train_nerf()

if __name__ == '__main__':
    main() 