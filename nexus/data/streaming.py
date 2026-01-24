import torch
from torch.utils.data import IterableDataset
from typing import Iterator, Optional, Dict, Any, Union
import queue
import threading
import time
from nexus.utils.logging import Logger
import numpy as np

from .validation import DataValidation

class StreamingDataset(IterableDataset):
    def __init__(
        self,
        data_source: Iterator,
        buffer_size: int = 1000,
        prefetch_factor: int = 2,
        timeout: float = 1.0
    ):
        """Initialize streaming dataset with prefetching.
        
        Args:
            data_source: Iterator providing the data
            buffer_size: Size of prefetch buffer
            prefetch_factor: Number of items to try prefetching at once
            timeout: Timeout in seconds for queue operations
        """
        DataValidation.validate_type(buffer_size, int, "buffer_size")
        DataValidation.validate_positive(buffer_size, "buffer_size")
        DataValidation.validate_type(prefetch_factor, int, "prefetch_factor")
        DataValidation.validate_positive(prefetch_factor, "prefetch_factor")
            
        self.data_source = data_source
        self.buffer_size = buffer_size
        self.prefetch_factor = prefetch_factor
        self.timeout = timeout
        self.buffer = queue.Queue(maxsize=buffer_size)
        self.stop_event = threading.Event()
        self.prefetch_thread = None
        self.logger = Logger(self.__class__.__name__)
        
    def _convert_item(self, item: Any) -> Any:
        """Convert item to appropriate format."""
        if isinstance(item, torch.Tensor):
            return item
        elif isinstance(item, np.ndarray):
            return torch.from_numpy(item)
        elif isinstance(item, (list, tuple)):
            return type(item)(self._convert_item(x) for x in item)
        elif isinstance(item, dict):
            return {k: self._convert_item(v) for k, v in item.items()}
        return item
        
    def _prefetch_data(self):
        """Prefetch data in background thread."""
        try:
            while not self.stop_event.is_set():
                try:
                    # Try to prefetch multiple items
                    for _ in range(self.prefetch_factor):
                        if self.stop_event.is_set():
                            break
                        data = next(self.data_source)
                        data = self._convert_item(data)
                        self.buffer.put(data, timeout=self.timeout)
                except StopIteration:
                    break
                except queue.Full:
                    time.sleep(0.1)  # Brief pause before retrying
                    continue
                except Exception as e:
                    self.logger.error(f"Error in prefetch thread: {str(e)}")
                    break
        finally:
            # Ensure end signal is sent
            try:
                self.buffer.put(None, timeout=self.timeout)
            except queue.Full:
                pass
            
    def __iter__(self):
        """Create iterator with prefetching."""
        self.stop_event.clear()
        if self.prefetch_thread is not None and self.prefetch_thread.is_alive():
            self.stop_event.set()
            self.prefetch_thread.join(timeout=self.timeout)
            
        self.prefetch_thread = threading.Thread(
            target=self._prefetch_data,
            daemon=True
        )
        self.prefetch_thread.start()
        
        while True:
            try:
                item = self.buffer.get(timeout=self.timeout)
                if item is None:
                    break
                yield item
            except queue.Empty:
                if not self.prefetch_thread.is_alive():
                    break
                continue
            except Exception as e:
                self.logger.error(f"Error while iterating: {str(e)}")
                break
                
    def __del__(self):
        """Cleanup when object is deleted."""
        self.stop_event.set()
        if self.prefetch_thread is not None and self.prefetch_thread.is_alive():
            self.prefetch_thread.join(timeout=self.timeout)