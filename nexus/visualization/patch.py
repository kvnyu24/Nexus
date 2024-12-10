from typing import Dict, Any, Tuple, Optional
import torch
import matplotlib.pyplot as plt
import numpy as np
from ..visualization.base import BaseVisualizer

class PatchVisualizer(BaseVisualizer):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.patch_cmap = config.get("patch_cmap", "viridis")
        self.attention_alpha = config.get("attention_alpha", 0.5)
        self.grid_size = config.get("grid_size", (14, 14))
        
    def visualize_patches(
        self,
        image: torch.Tensor,
        patch_embeddings: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
        patch_size: Tuple[int, int] = (16, 16)
    ) -> Dict[str, Any]:
        """Visualize image patches and optional attention weights"""
        if image.dim() == 4:
            image = image[0]  # Take first image if batched
            
        # Convert image to numpy and normalize
        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Plot original image with patch grid
        ax1.imshow(img_np)
        
        # Draw patch grid
        for i in range(0, img_np.shape[0], patch_size[0]):
            ax1.axhline(y=i, color='white', linestyle='-', alpha=0.3)
        for j in range(0, img_np.shape[1], patch_size[1]):
            ax1.axvline(x=j, color='white', linestyle='-', alpha=0.3)
            
        ax1.set_title("Image with Patch Grid")
        
        # Visualize patch embeddings
        patch_grid = patch_embeddings.reshape(*self.grid_size, -1)
        patch_viz = torch.norm(patch_grid, dim=-1).cpu()
        
        im = ax2.imshow(patch_viz, cmap=self.patch_cmap)
        plt.colorbar(im, ax=ax2)
        ax2.set_title("Patch Embedding Magnitudes")
        
        # Overlay attention if provided
        if attention_weights is not None:
            attention = attention_weights.reshape(*self.grid_size)
            ax2.imshow(
                attention.cpu(),
                cmap='Reds',
                alpha=self.attention_alpha
            )
            
        plt.tight_layout()
        
        # Save visualization
        self.save_figure(fig, "patch_visualization.png")
        plt.close()
        
        return {"figure": fig} 