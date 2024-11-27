import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from ...core.base import NexusModule
from ..nlp.rag import EnhancedRAGModule
from ..cv.vae import EnhancedVAE

class EnhancedMultiModalTransformer(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Validate configuration
        self._validate_config(config)
        
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
        
        # Feature bank (following EnhancedReID pattern)
        self.register_buffer(
            "multimodal_bank",
            torch.zeros(
                config.get("bank_size", 10000),
                self.hidden_size
            )
        )
        self.register_buffer("bank_ptr", torch.zeros(1, dtype=torch.long))
        
    def _validate_config(self, config: Dict[str, Any]) -> None:
        required = ["hidden_size", "vision_config"]
        for key in required:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
                
    def update_feature_bank(self, features: torch.Tensor):
        """Update feature bank following EnhancedReID pattern"""
        batch_size = features.size(0)
        ptr = int(self.bank_ptr)
        
        if ptr + batch_size > self.multimodal_bank.size(0):
            ptr = 0
            
        self.multimodal_bank[ptr:ptr + batch_size] = features.detach()
        self.bank_ptr[0] = (ptr + batch_size) % self.multimodal_bank.size(0)
        
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
        
        # Update feature bank
        self.update_feature_bank(enhanced_features)
        
        return {
            "enhanced_features": enhanced_features,
            "vision_features": vision_features,
            "text_features": text_features,
            "attention_weights": {
                "vision_text": vision_attended,
                "text_vision": text_attended
            }
        } 