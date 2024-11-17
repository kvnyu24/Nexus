import time
import psutil
import torch
from typing import Dict, Optional, List
from collections import deque

class PerformanceMonitor:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.batch_times = deque(maxlen=window_size)
        self.memory_usage = deque(maxlen=window_size)
        self.start_time = None
        
    def start_batch(self):
        self.start_time = time.time()
        
    def end_batch(self) -> Dict[str, float]:
        if self.start_time is None:
            raise RuntimeError("start_batch() must be called before end_batch()")
            
        batch_time = time.time() - self.start_time
        self.batch_times.append(batch_time)
        
        # Get memory stats
        memory = psutil.Process().memory_info()
        gpu_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        self.memory_usage.append((memory.rss, gpu_memory))
        
        return {
            "batch_time": batch_time,
            "avg_batch_time": sum(self.batch_times) / len(self.batch_times),
            "ram_usage_gb": memory.rss / (1024 ** 3),
            "gpu_memory_gb": gpu_memory / (1024 ** 3)
        } 