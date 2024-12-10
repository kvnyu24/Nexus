from typing import Dict, Any, List, Optional
import torch
import matplotlib.pyplot as plt
import numpy as np
from ..visualization.base import BaseVisualizer

class EmbeddingVisualizer(BaseVisualizer):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.perplexity = config.get("perplexity", 30)
        self.n_iter = config.get("n_iter", 1000)
        self.marker_size = config.get("marker_size", 100)
        
    def visualize_embedding_space(
        self,
        embeddings: torch.Tensor,
        labels: Optional[List[str]] = None,
        method: str = "tsne"
    ) -> Dict[str, Any]:
        """Visualize high-dimensional embeddings in 2D"""
        # Input validation
        if not isinstance(embeddings, torch.Tensor):
            raise ValueError("embeddings must be a torch.Tensor")
            
        # Convert to numpy
        embed_np = embeddings.cpu().numpy()
        
        # Dimensionality reduction
        if method == "tsne":
            from sklearn.manifold import TSNE
            reducer = TSNE(
                n_components=2,
                perplexity=self.perplexity,
                n_iter=self.n_iter
            )
        elif method == "umap":
            import umap
            reducer = umap.UMAP(n_components=2)
        else:
            raise ValueError(f"Unknown reduction method: {method}")
            
        # Reduce dimensionality
        reduced = reducer.fit_transform(embed_np)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 10))
        
        scatter = ax.scatter(
            reduced[:, 0],
            reduced[:, 1],
            c=np.arange(len(reduced)) if labels is None else labels,
            cmap='viridis',
            s=self.marker_size,
            alpha=0.6
        )
        
        if labels is not None:
            plt.colorbar(scatter, label='Class')
            
        ax.set_title(f"Embedding Space ({method.upper()})")
        
        # Save visualization
        self.save_figure(fig, f"embedding_space_{method}.png")
        plt.close()
        
        return {
            "figure": fig,
            "reduced_embeddings": reduced
        } 