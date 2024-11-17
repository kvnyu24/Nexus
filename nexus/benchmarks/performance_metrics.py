import numpy as np
from typing import Dict, List, Optional
import torch
from dataclasses import dataclass

@dataclass
class BenchmarkMetrics:
    """Class for calculating and storing benchmark performance metrics."""
    
    def calculate_latency_metrics(self, latencies: List[float]) -> Dict[str, float]:
        """Calculate standard latency metrics from a list of measurements."""
        if not latencies:
            return {}
            
        return {
            "mean_latency": np.mean(latencies),
            "std_latency": np.std(latencies),
            "min_latency": np.min(latencies),
            "max_latency": np.max(latencies),
            "p50_latency": np.percentile(latencies, 50),
            "p90_latency": np.percentile(latencies, 90),
            "p95_latency": np.percentile(latencies, 95),
            "p99_latency": np.percentile(latencies, 99)
        }
        
    def calculate_throughput_metrics(
        self,
        batch_sizes: List[int],
        latencies: List[float]
    ) -> Dict[str, float]:
        """Calculate throughput metrics for different batch sizes."""
        throughput_metrics = {}
        
        for batch_size, latency in zip(batch_sizes, latencies):
            throughput = batch_size / latency
            throughput_metrics[f"throughput_batch_{batch_size}"] = throughput
            
        if throughput_metrics:
            throughput_metrics["mean_throughput"] = np.mean(list(throughput_metrics.values()))
            
        return throughput_metrics
        
    def calculate_memory_metrics(
        self,
        memory_measurements: List[float],
        device: str = "cuda"
    ) -> Dict[str, float]:
        """Calculate memory usage metrics."""
        if not memory_measurements:
            return {}
            
        metrics = {
            "mean_memory_mb": np.mean(memory_measurements) / (1024 * 1024),
            "peak_memory_mb": np.max(memory_measurements) / (1024 * 1024),
            "std_memory_mb": np.std(memory_measurements) / (1024 * 1024)
        }
        
        if device == "cuda" and torch.cuda.is_available():
            metrics["total_gpu_memory_mb"] = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            
        return metrics
        
    def calculate_efficiency_metrics(
        self,
        throughput: float,
        memory_usage: float,
        power_usage: Optional[float] = None
    ) -> Dict[str, float]:
        """Calculate efficiency metrics combining throughput and resource usage."""
        metrics = {
            "throughput_per_memory": throughput / max(memory_usage, 1e-7)
        }
        
        if power_usage is not None:
            metrics["throughput_per_watt"] = throughput / max(power_usage, 1e-7)
            
        return metrics
        
    def aggregate_metrics(
        self,
        latency_metrics: Dict[str, float],
        throughput_metrics: Dict[str, float],
        memory_metrics: Dict[str, float],
        efficiency_metrics: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Combine all metrics into a single dictionary."""
        metrics = {
            **latency_metrics,
            **throughput_metrics,
            **memory_metrics
        }
        
        if efficiency_metrics:
            metrics.update(efficiency_metrics)
            
        return metrics
