from torch.utils.data import DataLoader as TorchDataLoader
from typing import Optional, Callable, Any
from .dataset import Dataset

class DataLoader(TorchDataLoader):
    """Custom DataLoader that extends PyTorch's DataLoader with additional functionality"""
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        collate_fn: Optional[Callable] = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        **kwargs: Any
    ):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            **kwargs
        ) 