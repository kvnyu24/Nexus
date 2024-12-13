import logging
from torch.utils.data import DataLoader as TorchDataLoader
from typing import Optional, Callable, Any, Union
from .dataset import Dataset

class DataLoader(TorchDataLoader):
    """Custom DataLoader that extends PyTorch's DataLoader with additional functionality.
    
    Provides robust error handling, validation of inputs, and compatibility with streaming datasets.
    """
    
    def __init__(
        self,
        dataset: Union[Dataset, Any],
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        collate_fn: Optional[Callable] = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: Optional[float] = None,
        prefetch_factor: Optional[int] = None,
        **kwargs: Any
    ):
        # Input validation
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if prefetch_factor is not None and prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be positive")
            
        # Configure logging
        self.logger = logging.getLogger(__name__)
        
        try:
            # Handle streaming datasets
            if hasattr(dataset, 'buffer_size'):
                if timeout is None:
                    timeout = 30.0  # Default timeout for streaming
                if prefetch_factor is None:
                    prefetch_factor = 2  # Default prefetch factor
                kwargs['timeout'] = timeout
                kwargs['prefetch_factor'] = prefetch_factor
                
            super().__init__(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle and not hasattr(dataset, 'buffer_size'),  # Disable shuffle for streaming
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=pin_memory,
                drop_last=drop_last,
                **kwargs
            )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize DataLoader: {str(e)}")
            raise