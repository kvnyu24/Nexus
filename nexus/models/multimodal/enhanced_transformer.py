import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from ...core.base import NexusModule
from ...core.mixins import ConfigValidatorMixin, FeatureBankMixin
from ..nlp.rag import EnhancedRAGModule
from ..cv.vae import EnhancedVAE

class EnhancedMultiModalTransformer(ConfigValidatorMixin, FeatureBankMixin, NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Validate configuration using ConfigValidatorMixin
        self.validate_config(config, required_keys=["hidden_size", "vision_config"])
        
        # Core dimensions
        self.hidden_size = config["hidden_size"]
        self.num_heads = config.get("num_heads", 8)
        self.num_layers = config.get("num_layers", 6)
        
        # Modality-specific encoders
        self.vision_encoder = EnhancedVAE(config.get("vision_config", {}))
        self.text_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=self.num_heads
            ),
            num_layers=self.num_layers
        )
        
        # Cross-modal fusion (following EnhancedFusionModule pattern)
        self.modal_fusion = nn.ModuleDict({
            'vision_text': nn.MultiheadAttention(
                embed_dim=self.hidden_size,
                num_heads=self.num_heads
            ),
            'text_vision': nn.MultiheadAttention(
                embed_dim=self.hidden_size,
                num_heads=self.num_heads
            )
        })
        
        # Knowledge enhancement (following RAGModule pattern)
        self.knowledge_enhancer = EnhancedRAGModule(config)
        
        # Feature bank using FeatureBankMixin
        self.bank_size = config.get("bank_size", 10000)
        self.register_feature_bank("multimodal", self.bank_size, self.hidden_size)
        
    def forward(
        self,
        vision_input: torch.Tensor,
        text_input: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Encode modalities
        vision_features = self.vision_encoder(vision_input)["z"]
        text_features = self.text_encoder(text_input)
        
        # Cross-modal attention
        vision_attended = self.modal_fusion['vision_text'](
            vision_features,
            text_features,
            text_features
        )[0]
        
        text_attended = self.modal_fusion['text_vision'](
            text_features,
            vision_features,
            vision_features
        )[0]
        
        # Fuse features
        fused_features = torch.cat([vision_attended, text_attended], dim=-1)
        
        # Knowledge enhancement
        enhanced_features = self.knowledge_enhancer(
            fused_features,
            attention_mask=attention_mask
        )["output"]
        
        # Update feature bank using FeatureBankMixin
        self.update_feature_bank("multimodal", enhanced_features)
        
        return {
            "enhanced_features": enhanced_features,
            "vision_features": vision_features,
            "text_features": text_features,
            "attention_weights": {
                "vision_text": vision_attended,
                "text_vision": text_attended
            }
        } 