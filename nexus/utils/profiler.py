import torch
import time
from collections import defaultdict
import numpy as np
from typing import Dict, Optional, Union, Tuple, List
from nexus.core.base import NexusModule
from .logging import Logger

class ModelProfiler:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.hooks = []
        self.logger = Logger("ModelProfiler")
        
    def profile_memory(self, model: NexusModule, detailed: bool = False) -> Dict[str, float]:
        """Profile model memory usage.
        
        Args:
            model: Model to profile
            detailed: Whether to include per-layer memory breakdown
            
        Returns:
            Dict containing memory statistics
        """
        memory_stats = {}
        
        try:
            # Get model size
            param_size = 0
            buffer_size = 0
            grad_size = 0
            
            for name, param in model.named_parameters():
                param_size += param.nelement() * param.element_size()
                if param.grad is not None:
                    grad_size += param.grad.nelement() * param.grad.element_size()
                
                if detailed:
                    memory_stats[f'param_size_mb_{name}'] = (
                        param.nelement() * param.element_size() / 1024**2
                    )
            
            for name, buffer in model.named_buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
                if detailed:
                    memory_stats[f'buffer_size_mb_{name}'] = (
                        buffer.nelement() * buffer.element_size() / 1024**2
                    )
                    
            memory_stats['model_size_mb'] = (param_size + buffer_size) / 1024**2
            memory_stats['gradient_size_mb'] = grad_size / 1024**2
            
            # Get CUDA memory if available
            if torch.cuda.is_available():
                memory_stats['cuda_allocated_mb'] = torch.cuda.memory_allocated() / 1024**2
                memory_stats['cuda_cached_mb'] = torch.cuda.memory_reserved() / 1024**2
                memory_stats['cuda_max_allocated_mb'] = torch.cuda.max_memory_allocated() / 1024**2
                
                # Clear memory peaks
                torch.cuda.reset_peak_memory_stats()
                
            # Get CPU memory
            import psutil
            process = psutil.Process()
            memory_stats['cpu_memory_mb'] = process.memory_info().rss / 1024**2
                
        except Exception as e:
            self.logger.error(f"Error profiling memory: {str(e)}")
            
        return memory_stats
        
    def profile_forward_pass(
        self,
        model: NexusModule,
        input_size: Union[tuple, List[tuple]],
        num_runs: int = 100,
        warmup: int = 10,
        device: Optional[torch.device] = None
    ) -> Dict[str, float]:
        """Profile model forward pass performance.
        
        Args:
            model: Model to profile
            input_size: Input size(s) - can be tuple or list of tuples for multiple inputs
            num_runs: Number of profiling runs
            warmup: Number of warmup runs
            device: Device to run on (defaults to model's device)
            
        Returns:
            Dict containing timing statistics
        """
        try:
            if device is None:
                device = next(model.parameters()).device
                
            # Handle single or multiple inputs
            if isinstance(input_size[0], int):
                dummy_inputs = torch.randn(input_size).to(device)
            else:
                dummy_inputs = [torch.randn(size).to(device) for size in input_size]
            
            # Warmup
            model.eval()
            for _ in range(warmup):
                if isinstance(dummy_inputs, list):
                    model(*dummy_inputs)
                else:
                    model(dummy_inputs)
                    
            # Profile
            timings = []
            memory_peaks = []
            
            if device.type == 'cuda':
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                with torch.no_grad():
                    for _ in range(num_runs):
                        torch.cuda.reset_peak_memory_stats()
                        start_event.record()
                        
                        if isinstance(dummy_inputs, list):
                            model(*dummy_inputs)
                        else:
                            model(dummy_inputs)
                            
                        end_event.record()
                        torch.cuda.synchronize()
                        
                        timings.append(start_event.elapsed_time(end_event))
                        memory_peaks.append(torch.cuda.max_memory_allocated() / 1024**2)
            else:
                with torch.no_grad():
                    for _ in range(num_runs):
                        start_time = time.perf_counter()
                        
                        if isinstance(dummy_inputs, list):
                            model(*dummy_inputs)
                        else:
                            model(dummy_inputs)
                            
                        end_time = time.perf_counter()
                        timings.append((end_time - start_time) * 1000)  # Convert to ms
                
            stats = {
                'mean_time_ms': np.mean(timings),
                'std_time_ms': np.std(timings),
                'min_time_ms': np.min(timings),
                'max_time_ms': np.max(timings),
                'p95_time_ms': np.percentile(timings, 95)
            }
            
            if device.type == 'cuda':
                stats['mean_memory_mb'] = np.mean(memory_peaks)
                stats['max_memory_mb'] = np.max(memory_peaks)
                
            return stats
            
        except Exception as e:
            self.logger.error(f"Error profiling forward pass: {str(e)}")
            return {}