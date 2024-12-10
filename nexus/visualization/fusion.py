from typing import Dict, Any, List, Optional
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from ..visualization.base import BaseVisualizer
import umap

class FusionVisualizer(BaseVisualizer):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.modality_colors = config.get("modality_colors", {
            "text": "#2ecc71",
            "image": "#3498db",
            "audio": "#e74c3c"
        })
        self.feature_dim = config.get("feature_dim", 2)
        
    def visualize_modality_features(
        self,
        features: Dict[str, torch.Tensor],
        labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Visualize features from different modalities"""
        
        # Reduce dimensionality if needed
        if next(iter(features.values())).shape[-1] > self.feature_dim:
            reducer = umap.UMAP(n_components=self.feature_dim)
            reduced_features = {}
            for modality, feat in features.items():
                reduced = reducer.fit_transform(feat.cpu().numpy())
                reduced_features[modality] = torch.from_numpy(reduced)
        else:
            reduced_features = features
            
        # Create scatter plot
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot each modality
        for modality, feat in reduced_features.items():
            feat_np = feat.cpu().numpy()
            ax.scatter(
                feat_np[:, 0],
                feat_np[:, 1],
                c=self.modality_colors.get(modality, "#000000"),
                label=modality,
                alpha=0.7
            )
            
        ax.legend()
        ax.set_title("Modality Feature Distribution")
        
        # Save visualization
        self.save_figure(fig, "modality_features.png")
        plt.close()
        
        return {
            "figure": fig,
            "reduced_features": reduced_features
        } 