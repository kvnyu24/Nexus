import torch.multiprocessing as mp
from nexus.training.distributed import DistributedTrainer
from nexus.training.mixed_precision import MixedPrecisionTrainer
from nexus.utils.gpu import GPUManager

def train(rank, world_size, model, dataset):
    # Initialize distributed trainer
    trainer = DistributedTrainer(model, rank, world_size)
    
    # Initialize mixed precision trainer
    mp_trainer = MixedPrecisionTrainer(
        trainer.model,
        trainer.optimizer
    )
    
    # Training loop with mixed precision
    for epoch in range(num_epochs):
        for batch in dataloader:
            metrics = mp_trainer.training_step(batch)
            
            # Reduce metrics across processes
            reduced_metrics = trainer.all_reduce_dict(metrics)
            
    trainer.cleanup()

def main():
    gpu_manager = GPUManager()
    world_size = torch.cuda.device_count()
    
    # Spawn processes for distributed training
    mp.spawn(
        train,
        args=(world_size, model, dataset),
        nprocs=world_size,
        join=True
    ) 