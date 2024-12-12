import time
import psutil
import torch
import numpy as np
from typing import Dict, Optional, List, Tuple
from collections import deque
from .logging import Logger

class PerformanceMonitor:
    def __init__(self, window_size: int = 100, log_warnings: bool = True):
        """Initialize performance monitor with configurable window size and warning logging.
        
        Args:
            window_size: Number of batches to keep statistics for
            log_warnings: Whether to log performance warnings
        """
        self.window_size = max(1, window_size)  # Ensure positive window size
        self.batch_times = deque(maxlen=window_size)
        self.memory_usage = deque(maxlen=window_size)
        self.start_time = None
        self.total_batches = 0
        self.logger = Logger("PerformanceMonitor") if log_warnings else None
        
        # Performance thresholds
        self.batch_time_threshold = 1.0  # seconds
        self.memory_usage_threshold = 0.9  # 90% of system memory
        
    def start_batch(self):
        """Start timing a new batch."""
        if self.start_time is not None:
            self.logger and self.logger.warning("Previous batch was not ended properly")
        self.start_time = time.perf_counter()  # More precise than time.time()
        
    def end_batch(self) -> Dict[str, float]:
        """End batch timing and collect performance metrics."""
        if self.start_time is None:
            raise RuntimeError("start_batch() must be called before end_batch()")
            
        batch_time = time.perf_counter() - self.start_time
        self.batch_times.append(batch_time)
        self.total_batches += 1
        self.start_time = None
        
        # Get detailed memory stats
        memory = psutil.Process().memory_info()
        system_memory = psutil.virtual_memory()
        
        # Get GPU memory for all available devices
        gpu_memory_allocated = 0
        gpu_memory_reserved = 0
        if torch.cuda.is_available():
            for device in range(torch.cuda.device_count()):
                gpu_memory_allocated += torch.cuda.memory_allocated(device)
                gpu_memory_reserved += torch.cuda.memory_reserved(device)
        
        self.memory_usage.append((memory.rss, gpu_memory_allocated, gpu_memory_reserved))
        
        # Calculate statistics
        batch_times_arr = np.array(self.batch_times)
        memory_usage_arr = np.array(self.memory_usage)
        
        stats = {
            "batch_time": batch_time,
            "avg_batch_time": np.mean(batch_times_arr),
            "std_batch_time": np.std(batch_times_arr),
            "min_batch_time": np.min(batch_times_arr),
            "max_batch_time": np.max(batch_times_arr),
            "ram_usage_gb": memory.rss / (1024 ** 3),
            "ram_usage_percent": system_memory.percent,
            "gpu_allocated_gb": gpu_memory_allocated / (1024 ** 3),
            "gpu_reserved_gb": gpu_memory_reserved / (1024 ** 3),
            "total_batches": self.total_batches,
            "batches_per_second": 1.0 / batch_time if batch_time > 0 else float('inf')
        }
        
        # Log warnings for concerning metrics
        if self.logger:
            if batch_time > self.batch_time_threshold:
                self.logger.warning(f"Batch time ({batch_time:.2f}s) exceeded threshold")
            if system_memory.percent > self.memory_usage_threshold * 100:
                self.logger.warning(f"High system memory usage: {system_memory.percent}%")
        
        return stats
        
    def reset(self):
        """Reset all monitoring state."""
        self.batch_times.clear()
        self.memory_usage.clear()
        self.start_time = None
        self.total_batches = 0