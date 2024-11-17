import torch
from torch.utils.data import IterableDataset
from typing import Iterator, Optional, Dict, Any
import queue
import threading
import time

class StreamingDataset(IterableDataset):
    def __init__(
        self,
        data_source: Iterator,
        buffer_size: int = 1000,
        prefetch_factor: int = 2
    ):
        self.data_source = data_source
        self.buffer_size = buffer_size
        self.prefetch_factor = prefetch_factor
        self.buffer = queue.Queue(maxsize=buffer_size)
        self.stop_event = threading.Event()
        
    def _prefetch_data(self):
        try:
            while not self.stop_event.is_set():
                try:
                    data = next(self.data_source)
                    self.buffer.put(data, timeout=1)
                except StopIteration:
                    break
                except queue.Full:
                    continue
        finally:
            self.buffer.put(None)  # Signal end of dataset
            
    def __iter__(self):
        self.stop_event.clear()
        self.prefetch_thread = threading.Thread(
            target=self._prefetch_data,
            daemon=True
        )
        self.prefetch_thread.start()
        
        while True:
            item = self.buffer.get()
            if item is None:
                break
            yield item
            
    def __del__(self):
        self.stop_event.set() 