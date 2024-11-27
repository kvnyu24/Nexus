from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from ...core.base import NexusModule
from ..cv.vit import VisionTransformer
from ..nlp.t5 import EnhancedT5
from ..fusion.enhanced_fusion import EnhancedFusionModule

class HierarchicalViLTransformer(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Core components
        self.vision_encoder = VisionTransformer(config.get("vision_config", {}))
        self.text_encoder = EnhancedT5(config.get("text_config", {}))
        
        # Hierarchical fusion layers
        self.local_fusion = EnhancedFusionModule(config)
        self.global_fusion = EnhancedFusionModule(config)
        
        # Cross-modal attention
        self.cross_attention = nn.ModuleDict({
            'vision_to_text': nn.MultiheadAttention(
                embed_dim=config["hidden_dim"],
                num_heads=config.get("num_heads", 8)
            ),
            'text_to_vision': nn.MultiheadAttention(
                embed_dim=config["hidden_dim"],
                num_heads=config.get("num_heads", 8)
            )
        })
        
        # Output projection
        self.output_proj = nn.Linear(
            config["hidden_dim"],
            config.get("num_classes", 1000)
        )
        
    def forward(
        self,
        images: torch.Tensor,
        text: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Encode modalities
        vision_features = self.vision_encoder(images)
        text_features = self.text_encoder(text)
        
        # Local fusion
        local_fusion = self.local_fusion({
            'vision': vision_features,
            'text': text_features
        })
        
        # Global fusion
        global_fusion = self.global_fusion({
            'local': local_fusion['fused_features'],
            'vision': vision_features,
            'text': text_features
        })
        
        # Cross-modal attention
        vision_attended = self.cross_attention['text_to_vision'](
            global_fusion['fused_features'],
            vision_features,
            vision_features,
            attn_mask=attention_mask
        )[0]
        
        text_attended = self.cross_attention['vision_to_text'](
            global_fusion['fused_features'],
            text_features,
            text_features,
            attn_mask=attention_mask
        )[0]
        
        # Final prediction
        output = self.output_proj(vision_attended + text_attended)
        
        return {
            'output': output,
            'vision_features': vision_features,
            'text_features': text_features,
            'local_fusion': local_fusion['fused_features'],
            'global_fusion': global_fusion['fused_features']
        } 