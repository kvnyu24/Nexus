import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from typing import Optional, Dict, Any
from .checkpointing import CheckpointMixin
from nexus.core.base import NexusModule

class DistributedTrainer(CheckpointMixin):
    def __init__(
        self,
        model: NexusModule,
        rank: int,
        world_size: int,
        backend: str = "nccl",
        checkpoint_dir: Optional[str] = None
    ):
        self.rank = rank
        self.world_size = world_size
        self.backend = backend
        self.checkpoint_dir = checkpoint_dir or "checkpoints"
        
        # Initialize distributed process group
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size
        )
        
        # Set device
        self.device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(self.device)
        
        # Wrap model
        self.model = DistributedDataParallel(
            model.to(self.device),
            device_ids=[rank]
        )
        
    def all_reduce_dict(self, data: Dict[str, float]) -> Dict[str, float]:
        """Reduce metrics across all processes."""
        reduced_data = {}
        for key, value in data.items():
            tensor = torch.tensor(value).to(self.device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            reduced_data[key] = (tensor / self.world_size).item()
        return reduced_data
        
    def cleanup(self):
        dist.destroy_process_group() 
        
    def save_checkpoint(self, *args, **kwargs):
        """Override to only save on rank 0."""
        if self.rank == 0:
            return super().save_checkpoint(*args, **kwargs)
        return None
        
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint on all ranks."""
        checkpoint = super().load_checkpoint(checkpoint_path)
        dist.barrier()  # Ensure all processes load the checkpoint
        return checkpoint 