import torch
import torch.multiprocessing as mp
from nexus.training.distributed import DistributedTrainer
from nexus.training.mixed_precision import MixedPrecisionTrainer
from nexus.utils.gpu import GPUManager
from torch.utils.data import DataLoader
import torch.distributed as dist

def train(rank, world_size, model, dataset, num_epochs, batch_size):
    # Initialize process group
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:12355',
        world_size=world_size,
        rank=rank
    )
    
    # Initialize distributed trainer
    trainer = DistributedTrainer(model, rank, world_size)
    
    # Create distributed sampler and dataloader
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize mixed precision trainer
    mp_trainer = MixedPrecisionTrainer(
        trainer.model,
        trainer.optimizer
    )
    
    # Training loop with mixed precision
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.cuda(rank) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Training step
            metrics = mp_trainer.training_step(batch)
            
            # Reduce metrics across processes
            reduced_metrics = trainer.all_reduce_dict(metrics)
            
            if rank == 0:
                print(f"Epoch {epoch}, Loss: {reduced_metrics['loss']:.4f}")
    
    # Cleanup
    dist.destroy_process_group()
    trainer.cleanup()

def main(model, dataset, num_epochs=10, batch_size=32):
    # Initialize GPU manager
    gpu_manager = GPUManager()
    world_size = torch.cuda.device_count()
    
    if world_size < 1:
        raise RuntimeError("No CUDA devices available for distributed training")
    
    # Spawn processes for distributed training
    mp.spawn(
        train,
        args=(world_size, model, dataset, num_epochs, batch_size),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    # Example usage:
    from nexus.models.cv import CompactCNN
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import Compose, ToTensor, Normalize
    
    # Create model
    model = CompactCNN({"num_classes": 10})
    
    # Create dataset
    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = CIFAR10(
        root='./data',
        train=True,
        transform=transform,
        download=True
    )
    
    # Run distributed training
    main(model, dataset) 