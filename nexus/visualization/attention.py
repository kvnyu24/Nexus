from typing import Dict, Any, List, Optional
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from nexus.visualization.base import BaseVisualizer

class AttentionVisualizer(BaseVisualizer):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.attention_cmap = config.get("attention_cmap", "viridis")
        self.max_heads = config.get("max_heads", 4)
        
    def visualize_attention(
        self,
        attention_weights: torch.Tensor,
        tokens: Optional[List[str]] = None,
        layer_name: str = "layer_0"
    ) -> Dict[str, Any]:
        """Visualize multi-head attention patterns"""
        # Input validation
        if not isinstance(attention_weights, torch.Tensor):
            raise ValueError("attention_weights must be a torch.Tensor")
            
        # Handle batched input
        if attention_weights.dim() == 4:
            attention_weights = attention_weights[0]  # Take first example
        elif attention_weights.dim() != 3:
            raise ValueError("attention_weights must have 3 or 4 dimensions")
            
        num_heads = min(attention_weights.size(0), self.max_heads)
        seq_len = attention_weights.size(1)
        
        # Validate tokens if provided
        if tokens is not None:
            if len(tokens) != seq_len:
                raise ValueError(f"Number of tokens ({len(tokens)}) must match sequence length ({seq_len})")
        
        # Create subplot grid
        fig, axes = plt.subplots(
            1, num_heads,
            figsize=(4 * num_heads, 4),
            squeeze=False
        )
        
        # Plot each attention head
        for head in range(num_heads):
            ax = axes[0, head]
            attn_map = attention_weights[head].cpu().detach()
            
            # Ensure attention weights sum to 1
            if not torch.allclose(attn_map.sum(dim=-1), torch.ones_like(attn_map.sum(dim=-1)), rtol=1e-3):
                attn_map = torch.nn.functional.softmax(attn_map, dim=-1)
                
            sns.heatmap(
                attn_map,
                ax=ax,
                cmap=self.attention_cmap,
                xticklabels=tokens,
                yticklabels=tokens,
                square=True,
                vmin=0,
                vmax=1
            )
            ax.set_title(f"Head {head}")
            
        plt.tight_layout()
        
        # Save visualization
        self.save_figure(fig, f"attention_{layer_name}.png")
        plt.close()
        
        return {"figure": fig}