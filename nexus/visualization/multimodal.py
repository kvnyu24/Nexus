from typing import Dict, Any, List, Tuple
import torch
from ..visualization.base import BaseVisualizer

class MultiModalVisualizer(BaseVisualizer):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.max_tokens = config.get("max_tokens", 50)
        self.attention_cmap = config.get("attention_cmap", "RdBu")
        self.image_size = config.get("image_size", (14, 14))
        
    def visualize_cross_attention(
        self,
        vision_text_attention: torch.Tensor,
        text_vision_attention: torch.Tensor,
        tokens: List[str],
        patch_size: Tuple[int, int] = (16, 16)
    ) -> Dict[str, Any]:
        """Visualize cross-modal attention patterns"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Truncate tokens if needed
        if len(tokens) > self.max_tokens:
            tokens = tokens[:self.max_tokens]
            vision_text_attention = vision_text_attention[:self.max_tokens]
            text_vision_attention = text_vision_attention[:self.max_tokens]
            
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Vision->Text attention
        sns.heatmap(
            vision_text_attention.cpu(),
            ax=ax1,
            cmap=self.attention_cmap,
            xticklabels=tokens,
            yticklabels=[f"Patch {i}" for i in range(vision_text_attention.size(0))],
            square=True
        )
        ax1.set_title("Vision -> Text Attention")
        
        # Text->Vision attention
        attention_map = text_vision_attention.view(-1, *self.image_size).cpu()
        im = ax2.imshow(attention_map.mean(0), cmap=self.attention_cmap)
        ax2.set_title("Text -> Vision Attention")
        plt.colorbar(im, ax=ax2)
        
        plt.tight_layout()
        
        # Save visualization
        self.save_figure(fig, "cross_modal_attention.png")
        plt.close()
        
        return {
            "figure": fig,
            "attention_maps": {
                "vision_text": vision_text_attention,
                "text_vision": text_vision_attention
            }
        } 