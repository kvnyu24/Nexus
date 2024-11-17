import torch
import time
from collections import defaultdict
import numpy as np
from typing import Dict

class ModelProfiler:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.hooks = []
        
    def profile_memory(self, model: torch.nn.Module) -> Dict[str, float]:
        memory_stats = {}
        
        # Get model size
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            
        memory_stats['model_size_mb'] = (param_size + buffer_size) / 1024**2
        
        # Get CUDA memory if available
        if torch.cuda.is_available():
            memory_stats['cuda_allocated_mb'] = torch.cuda.memory_allocated() / 1024**2
            memory_stats['cuda_cached_mb'] = torch.cuda.memory_reserved() / 1024**2
            
        return memory_stats
        
    def profile_forward_pass(
        self,
        model: torch.nn.Module,
        input_size: tuple,
        num_runs: int = 100
    ) -> Dict[str, float]:
        device = next(model.parameters()).device
        dummy_input = torch.randn(input_size).to(device)
        
        # Warmup
        for _ in range(10):
            model(dummy_input)
            
        # Profile
        timings = []
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        with torch.no_grad():
            for _ in range(num_runs):
                start_event.record()
                model(dummy_input)
                end_event.record()
                
                torch.cuda.synchronize()
                timings.append(start_event.elapsed_time(end_event))
                
        return {
            'mean_time_ms': np.mean(timings),
            'std_time_ms': np.std(timings),
            'min_time_ms': np.min(timings),
            'max_time_ms': np.max(timings)
        } 