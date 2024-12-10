from typing import Dict, Any, Optional
import torch
import matplotlib.pyplot as plt
import math
from ..visualization.base import BaseVisualizer

class FeatureMapVisualizer(BaseVisualizer):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.feature_cmap = config.get("feature_cmap", "viridis")
        self.max_features = config.get("max_features", 16)
        self.normalize = config.get("normalize", True)
        
    def visualize_feature_maps(
        self,
        feature_maps: Dict[str, torch.Tensor],
        layer_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Visualize CNN/Transformer feature maps"""
        import matplotlib.pyplot as plt
        import math
        
        # Handle different feature map formats
        if isinstance(feature_maps, torch.Tensor):
            feature_maps = {"features": feature_maps}
            
        figures = {}
        for name, features in feature_maps.items():
            # Take first example if batched
            if features.dim() == 4:
                features = features[0]
                
            # Get number of channels to visualize
            num_channels = min(features.size(0), self.max_features)
            grid_size = math.ceil(math.sqrt(num_channels))
            
            # Create subplot grid
            fig, axes = plt.subplots(
                grid_size, grid_size,
                figsize=(2*grid_size, 2*grid_size)
            )
            
            # Normalize if requested
            if self.normalize:
                features = (features - features.min()) / (features.max() - features.min())
            
            # Plot each feature channel
            for idx in range(num_channels):
                row = idx // grid_size
                col = idx % grid_size
                ax = axes[row, col] if grid_size > 1 else axes
                
                im = ax.imshow(
                    features[idx].cpu(),
                    cmap=self.feature_cmap
                )
                ax.axis('off')
                
            # Remove empty subplots
            for idx in range(num_channels, grid_size**2):
                row = idx // grid_size
                col = idx % grid_size
                if grid_size > 1:
                    fig.delaxes(axes[row, col])
                    
            plt.tight_layout()
            
            # Save visualization
            filename = f"features_{name}"
            if layer_name:
                filename += f"_{layer_name}"
            self.save_figure(fig, f"{filename}.png")
            
            figures[name] = fig
            plt.close()
            
        return figures 