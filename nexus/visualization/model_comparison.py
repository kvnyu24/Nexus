from typing import Dict, Any, List, Optional
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from ..visualization.base import BaseVisualizer

class ModelComparisonVisualizer(BaseVisualizer):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.plot_type = config.get("plot_type", "bar")
        self.palette = config.get("palette", "Set2")
        self.figsize = config.get("figsize", (12, 6))
        self.rotation = config.get("label_rotation", 45)
        
    def visualize_model_comparison(
        self,
        metrics: Dict[str, Dict[str, float]],
        metric_names: Optional[List[str]] = None,
        model_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Visualize performance comparison across different models"""
        # Prepare data for plotting
        df_data = []
        for model_name, model_metrics in metrics.items():
            for metric_name, value in model_metrics.items():
                df_data.append({
                    'Model': model_names[model_name] if model_names else model_name,
                    'Metric': metric_names[metric_name] if metric_names else metric_name,
                    'Value': value
                })
        
        df = pd.DataFrame(df_data)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if self.plot_type == "bar":
            sns.barplot(
                data=df,
                x='Model',
                y='Value',
                hue='Metric',
                palette=self.palette,
                ax=ax
            )
        elif self.plot_type == "heatmap":
            pivot_df = df.pivot(index='Model', columns='Metric', values='Value')
            sns.heatmap(
                pivot_df,
                annot=True,
                fmt='.3f',
                cmap=self.palette,
                ax=ax
            )
            
        plt.xticks(rotation=self.rotation)
        plt.tight_layout()
        
        # Save visualization
        self.save_figure(fig, f"model_comparison_{self.plot_type}.png")
        plt.close()
        
        return {
            "figure": fig,
            "comparison_data": df
        } 