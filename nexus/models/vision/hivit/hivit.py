import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
from ....core.base import NexusModule
from .patch_embed import HierarchicalPatchEmbedding
from .attention import MultiScaleAttention

class HiViT(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Validate config
        self._validate_config(config)
        
        # Core dimensions
        self.hidden_dim = config["hidden_dim"]
        self.num_heads = config.get("num_heads", 8)
        self.num_layers = config.get("num_layers", 12)
        self.patch_size = config.get("patch_size", 16)
        
        # Hierarchical patch embeddings
        self.patch_embeddings = nn.ModuleDict({
            f'scale_{i}': HierarchicalPatchEmbedding(
                in_channels=config.get("in_channels", 3),
                hidden_dim=self.hidden_dim // (2 ** i),
                patch_size=self.patch_size * (2 ** i)
            ) for i in range(3)  # 3 scales
        })
        
        # Multi-scale attention blocks
        self.attention_blocks = nn.ModuleList([
            MultiScaleAttention(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                dropout=config.get("dropout", 0.1)
            ) for _ in range(self.num_layers)
        ])
        
        # Feature fusion
        self.fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim * 3, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU()
            ) for _ in range(self.num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(
            self.hidden_dim,
            config.get("num_classes", 1000)
        )
        
        # Feature bank (following EnhancedReID pattern)
        self.register_buffer(
            "feature_bank",
            torch.zeros(
                config.get("bank_size", 10000),
                self.hidden_dim
            )
        )
        self.register_buffer("bank_ptr", torch.zeros(1, dtype=torch.long))
        
    def _validate_config(self, config: Dict[str, Any]) -> None:
        required = ["hidden_dim", "num_classes"]
        for key in required:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
                
    def forward(
        self,
        images: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        batch_size = images.shape[0]
        
        # Get multi-scale embeddings
        scale_embeddings = {
            scale: embedder(images)
            for scale, embedder in self.patch_embeddings.items()
        }
        
        # Process through attention blocks
        attention_outputs = []
        scale_features = []
        
        for block_idx, (attn_block, fusion_layer) in enumerate(
            zip(self.attention_blocks, self.fusion_layers)
        ):
            # Multi-scale attention
            scale_attn = attn_block(
                list(scale_embeddings.values()),
                attention_mask=attention_mask
            )
            attention_outputs.append(scale_attn["attention_weights"])
            
            # Fuse features across scales
            concat_features = torch.cat(
                [feat for feat in scale_attn["scale_features"]], 
                dim=-1
            )
            fused = fusion_layer(concat_features)
            scale_features.append(fused)
            
            # Update embeddings for next layer
            for scale in scale_embeddings.keys():
                scale_embeddings[scale] = scale_attn["scale_features"][int(scale[-1])]
        
        # Get final representation
        final_features = scale_features[-1].mean(dim=1)  # Global average pooling
        
        # Update feature bank
        self._update_feature_bank(final_features)
        
        # Generate class predictions
        logits = self.output_proj(final_features)
        
        return {
            "logits": logits,
            "features": final_features,
            "attention_weights": attention_outputs,
            "scale_features": scale_features
        }
        
    def _update_feature_bank(self, features: torch.Tensor):
        batch_size = features.size(0)
        ptr = int(self.bank_ptr)
        
        if ptr + batch_size > self.feature_bank.size(0):
            ptr = 0
            
        self.feature_bank[ptr:ptr + batch_size] = features.detach()
        self.bank_ptr[0] = (ptr + batch_size) % self.feature_bank.size(0) 