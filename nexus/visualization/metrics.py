from typing import Dict, Any, List, Optional
import torch
import matplotlib.pyplot as plt
import numpy as np
from ..visualization.base import BaseVisualizer

class MetricsVisualizer(BaseVisualizer):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.line_styles = config.get("line_styles", ['-', '--', ':', '-.'])
        self.colors = config.get("colors", ['#2ecc71', '#3498db', '#e74c3c', '#f1c40f'])
        self.moving_avg_window = config.get("moving_avg_window", 10)
        self.grid = config.get("grid", True)
        
    def visualize_training_metrics(
        self,
        metrics: Dict[str, List[float]],
        names: Optional[List[str]] = None,
        include_moving_avg: bool = True
    ) -> Dict[str, Any]:
        """Visualize training metrics over time"""
        if not metrics:
            raise ValueError("metrics dictionary cannot be empty")
            
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for idx, (metric_name, values) in enumerate(metrics.items()):
            values = np.array(values)
            color = self.colors[idx % len(self.colors)]
            label = names[idx] if names else metric_name
            
            # Plot raw values
            ax.plot(values, 
                   label=f"{label} (raw)",
                   color=color,
                   linestyle=self.line_styles[0],
                   alpha=0.5)
            
            # Plot moving average
            if include_moving_avg and len(values) > self.moving_avg_window:
                moving_avg = np.convolve(values, 
                                       np.ones(self.moving_avg_window)/self.moving_avg_window,
                                       mode='valid')
                ax.plot(np.arange(len(moving_avg)) + self.moving_avg_window - 1,
                       moving_avg,
                       label=f"{label} (MA)",
                       color=color,
                       linestyle=self.line_styles[1])
        
        ax.set_xlabel('Steps')
        ax.set_ylabel('Value')
        ax.grid(self.grid)
        ax.legend()
        
        plt.tight_layout()
        self.save_figure(fig, "training_metrics.png")
        plt.close()
        
        return {"figure": fig} 