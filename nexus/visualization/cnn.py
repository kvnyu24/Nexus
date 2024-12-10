from typing import Dict, Any, List, Optional, Tuple
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from ..visualization.base import BaseVisualizer

class CNNVisualizer(BaseVisualizer):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.kernel_cmap = config.get("kernel_cmap", "viridis")
        self.activation_cmap = config.get("activation_cmap", "RdBu")
        self.max_filters = config.get("max_filters", 16)
        self.figsize = config.get("figsize", (15, 8))
        
    def visualize_kernels(
        self,
        layer: torch.nn.Conv2d,
        layer_name: str = "conv"
    ) -> Dict[str, Any]:
        """Visualize convolutional kernels"""
        # Get kernel weights
        kernels = layer.weight.detach().cpu()
        n_kernels = min(kernels.size(0), self.max_filters)
        
        # Create subplot grid
        n_rows = int(n_kernels ** 0.5)
        n_cols = (n_kernels + n_rows - 1) // n_rows
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=self.figsize)
        axes = axes.flatten()
        
        # Plot each kernel
        for i in range(n_kernels):
            kernel = kernels[i].mean(dim=0)  # Average across input channels
            axes[i].imshow(kernel, cmap=self.kernel_cmap)
            axes[i].axis('off')
            axes[i].set_title(f'Kernel {i}')
            
        # Remove empty subplots
        for i in range(n_kernels, len(axes)):
            fig.delaxes(axes[i])
            
        plt.tight_layout()
        self.save_figure(fig, f"kernels_{layer_name}.png")
        plt.close()
        
        return {"figure": fig}
        
    def visualize_feature_maps(
        self,
        activation_maps: torch.Tensor,
        layer_name: str = "activation"
    ) -> Dict[str, Any]:
        """Visualize CNN activation maps"""
        # Input validation (following AttentionVisualizer pattern)
        if not isinstance(activation_maps, torch.Tensor):
            raise ValueError("activation_maps must be a torch.Tensor")
            
        # Handle batched input
        if activation_maps.dim() == 4:
            activation_maps = activation_maps[0]
            
        n_maps = min(activation_maps.size(0), self.max_filters)
        n_rows = int(n_maps ** 0.5)
        n_cols = (n_maps + n_rows - 1) // n_rows
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=self.figsize)
        axes = axes.flatten()
        
        for i in range(n_maps):
            im = axes[i].imshow(
                activation_maps[i].cpu(),
                cmap=self.activation_cmap
            )
            axes[i].axis('off')
            axes[i].set_title(f'Channel {i}')
            
        plt.colorbar(im, ax=axes.ravel().tolist())
        plt.tight_layout()
        
        self.save_figure(fig, f"activations_{layer_name}.png")
        plt.close()
        
        return {"figure": fig} 