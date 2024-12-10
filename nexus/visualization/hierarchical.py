from typing import Dict, Any, List, Optional
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from ..visualization.base import BaseVisualizer

class HierarchicalVisualizer(BaseVisualizer):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.attention_cmap = config.get("attention_cmap", "viridis")
        self.local_color = config.get("local_color", "#2ecc71")
        self.global_color = config.get("global_color", "#3498db")
        self.alpha = config.get("alpha", 0.7)
        
    def visualize_hierarchical_attention(
        self,
        local_attention: Dict[str, torch.Tensor],
        global_attention: Dict[str, torch.Tensor],
        tokens: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Visualize hierarchical attention patterns"""
        # Input validation (following AttentionVisualizer pattern)
        for attn_dict in [local_attention, global_attention]:
            for key, attn in attn_dict.items():
                if not isinstance(attn, torch.Tensor):
                    raise ValueError(f"Attention weights for {key} must be torch.Tensor")
                
        # Create figure with subplots for each attention type
        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(2, 2)
        
        # Local attention visualization
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_attention_map(
            local_attention["vision_to_text"],
            ax1,
            title="Local Vision-Text Attention",
            tokens=tokens
        )
        
        # Global attention visualization
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_attention_map(
            global_attention["vision_to_text"],
            ax2,
            title="Global Vision-Text Attention",
            tokens=tokens
        )
        
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_attention_map(
            global_attention["text_to_vision"],
            ax3,
            title="Global Text-Vision Attention",
            tokens=tokens
        )
        
        plt.tight_layout()
        
        # Save visualization
        self.save_figure(fig, "hierarchical_attention.png")
        plt.close()
        
        return {"figure": fig}
        
    def _plot_attention_map(
        self,
        attention: torch.Tensor,
        ax: plt.Axes,
        title: str,
        tokens: Optional[List[str]] = None
    ) -> None:
        """Helper method to plot attention maps"""
        if attention.dim() == 3:
            attention = attention.mean(0)  # Average over heads
            
        sns.heatmap(
            attention.cpu().detach(),
            ax=ax,
            cmap=self.attention_cmap,
            xticklabels=tokens,
            yticklabels=tokens,
            square=True
        )
        ax.set_title(title) 