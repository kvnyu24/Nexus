import torch
import time
from typing import Dict, Any, Optional, List
from ..utils.metrics import MetricsCalculator
from ..utils.gpu import GPUManager
from dataclasses import dataclass
import numpy as np

@dataclass
class BenchmarkConfig:
    batch_sizes: List[int] = None
    num_iterations: int = 100
    warmup_iterations: int = 10
    measure_memory: bool = True
    measure_flops: bool = True
    
class ModelBenchmark:
    def __init__(self, model: torch.nn.Module, config: BenchmarkConfig):
        self.model = model
        self.config = config
        self.gpu_manager = GPUManager()
        self.metrics = MetricsCalculator()
        self.device = self.gpu_manager.get_optimal_device()
        
    def run_latency_benchmark(self, input_shape: tuple) -> Dict[str, float]:
        self.model.to(self.device)
        self.model.eval()
        
        latencies = []
        memory_usage = []
        
        # Warmup
        for _ in range(self.config.warmup_iterations):
            input_data = torch.randn(input_shape).to(self.device)
            with torch.no_grad():
                _ = self.model(input_data)
                
        # Measure performance
        for _ in range(self.config.num_iterations):
            input_data = torch.randn(input_shape).to(self.device)
            
            if self.config.measure_memory:
                start_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                
            start_time = time.perf_counter()
            with torch.no_grad():
                _ = self.model(input_data)
            end_time = time.perf_counter()
            
            if self.config.measure_memory:
                end_mem = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                memory_usage.append(end_mem - start_mem)
                
            latencies.append(end_time - start_time)
            
        return {
            "mean_latency": np.mean(latencies),
            "std_latency": np.std(latencies),
            "p90_latency": np.percentile(latencies, 90),
            "p95_latency": np.percentile(latencies, 95),
            "p99_latency": np.percentile(latencies, 99),
            "mean_memory": np.mean(memory_usage) if memory_usage else 0,
            "peak_memory": max(memory_usage) if memory_usage else 0
        }
        
    def run_throughput_benchmark(self) -> Dict[str, float]:
        results = {}
        for batch_size in self.config.batch_sizes:
            input_shape = (batch_size,) + self.model.input_shape[1:]
            batch_results = self.run_latency_benchmark(input_shape)
            
            # Calculate throughput
            throughput = batch_size / batch_results["mean_latency"]
            results[f"batch_size_{batch_size}"] = {
                **batch_results,
                "throughput": throughput
            }
            
        return results 