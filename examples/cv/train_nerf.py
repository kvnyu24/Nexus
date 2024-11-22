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
        
        # Move batch data to device after loading
        ray_origins = batch["ray_origins"].to(self.device)
        ray_directions = batch["ray_directions"].to(self.device)
        target_rgb = batch["target_rgb"].to(self.device)
        
        # Reshape ray directions to remove extra dimensions
        ray_directions = ray_directions.squeeze(0).squeeze(0)
        ray_origins = ray_origins.squeeze(0).squeeze(0)
        
        # Ensure target_rgb has correct shape and channels
        # Remove alpha channel if present and reshape
        if target_rgb.shape[-1] == 4:
            target_rgb = target_rgb[..., :3]  # Keep only RGB channels
        target_rgb = target_rgb.squeeze(0)  # Remove batch dimension if present
        
        # Split rays into chunks to avoid OOM
        chunk_size = 4096
        total_loss = 0
        num_chunks = ray_origins.shape[0] // chunk_size + (1 if ray_origins.shape[0] % chunk_size != 0 else 0)
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, ray_origins.shape[0])
            
            chunk_origins = ray_origins[start_idx:end_idx]
            chunk_directions = ray_directions[start_idx:end_idx]
            chunk_target = target_rgb[start_idx:end_idx]
            
            # Render rays for this chunk
            outputs = self.model.render_rays(
                ray_origins=chunk_origins,
                ray_directions=chunk_directions,
                near=2.0,
                far=6.0,
                num_samples=64,
                noise_std=0.0
            )
            
            # Ensure output and target shapes match
            chunk_target = chunk_target.view(-1, 3)  # Reshape to [N, 3]
            output_rgb = outputs["rgb"].view(-1, 3)  # Ensure output is [N, 3]
            
            # Compute loss for this chunk
            loss = F.mse_loss(output_rgb, chunk_target)
            total_loss += loss.item() * (end_idx - start_idx)
            
            # Backward pass for this chunk
            loss.backward()
        
        # Average loss and update parameters
        avg_loss = total_loss / ray_origins.shape[0]
        self.optimizer.step()
        
        return {"loss": avg_loss}

def train_nerf():
    # Initialize model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    nerf = NeRFNetwork(config).to(device)

    # Initialize trainer with device and checkpoint directory
    trainer = NeRFTrainer(
        model=nerf,
        optimizer="adam",
        learning_rate=5e-4,
        device=device,
        checkpoint_dir="checkpoints/nerf"  # Add checkpoint directory
    )

    # Initialize dataset (removed device parameter)
    nerf_dataset = NeRFDataset(
        root_dir='./data/nerf_synthetic/lego',
        split='train',
        img_wh=(400, 400),
        precache_rays=True,
        num_workers=8
    )

    # Create data loader
    train_loader = DataLoader(
        dataset=nerf_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=8 if device.type == "cpu" else 0,
        pin_memory=True if device.type != "cpu" else False,
        collate_fn=NeRFDataset.collate_fn,
        persistent_workers=True if device.type == "cpu" else False
    )

    # Define number of training epochs and checkpoint frequency
    num_epochs = 100
    checkpoint_frequency = 1  # Save checkpoint every 5 epochs

    # Training loop with checkpointing
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0

        for batch in train_loader:
            loss_dict = trainer.train_step(batch)
            epoch_loss += loss_dict['loss']
            num_batches += 1
            print(f"Epoch {epoch}, Loss: {loss_dict['loss']:.4f}")

        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / num_batches

        # Save checkpoint if needed
        if (epoch + 1) % checkpoint_frequency == 0:
            metrics = {
                "epoch": epoch + 1,
                "train_loss": avg_epoch_loss,
                "learning_rate": trainer.optimizer.param_groups[0]['lr']
            }
            
            checkpoint_path = trainer.save_checkpoint(
                trainer.checkpoint_dir,
                epoch + 1,
                metrics
            )
            print(f"Saved checkpoint to {checkpoint_path}")

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