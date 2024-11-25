from typing import Dict, Any, Optional, List, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...core.base import NexusModule
from ..cv.vit import VisionTransformer
from ..nlp.llm import LlamaModel
from ..nlp.rag import EnhancedRAGModule
from ..nlp.hallucination_reducer import HallucinationReducer
from .llava_rlhf_config import LLaVARLHFConfig
from ...training.losses import EnhancedSFTLoss

class LLaVARLHF(NexusModule):
    def __init__(self, config: Union[Dict[str, Any], LLaVARLHFConfig]):
        super().__init__(config)
        
        # Convert dict config to LLaVARLHFConfig if needed
        if not isinstance(config, LLaVARLHFConfig):
            config = LLaVARLHFConfig(**config)
        self.config = config
        
        # Core components
        self.vision_encoder = VisionTransformer(config.vision_config)
        self.language_model = LlamaModel(config.language_config)
        
        # Enhanced components following EnhancedSFTModule pattern
        if config.use_rag:
            self.rag_module = EnhancedRAGModule(config)
        if config.use_hallucination_reducer:
            self.hallucination_reducer = HallucinationReducer(config)
            
        # Cross-modal fusion following EnhancedFusionModule pattern
        self.vision_projection = nn.Linear(
            config.vision_config["hidden_size"],
            config.hidden_size
        )
        self.cross_attention = nn.MultiheadAttention(
            config.hidden_size,
            config.num_heads,
            batch_first=True
        )
        
        # Quality assessment following AlphaFold pattern
        self.quality_head = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Experience bank following EnhancedReID pattern
        self.register_buffer(
            "experience_bank",
            torch.zeros(config.bank_size, config.hidden_size)
        )
        self.register_buffer("rewards", torch.zeros(config.bank_size))
        self.register_buffer("bank_ptr", torch.zeros(1, dtype=torch.long))
        
        # Loss function
        self.loss_fn = EnhancedSFTLoss(config)
        
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration following FasterRCNN pattern"""
        required = ["vision_config", "language_config", "hidden_size"]
        for key in required:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
                
    def update_experience_bank(
        self,
        states: torch.Tensor,
        rewards: torch.Tensor
    ) -> None:
        """Update experience bank following EnhancedReID pattern"""
        batch_size = states.size(0)
        ptr = int(self.bank_ptr)
        
        if ptr + batch_size > self.experience_bank.size(0):
            ptr = 0
            
        self.experience_bank[ptr:ptr + batch_size] = states.detach()
        self.rewards[ptr:ptr + batch_size] = rewards.detach()
        self.bank_ptr[0] = (ptr + batch_size) % self.experience_bank.size(0)
        
    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_loss: bool = True,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        # Vision encoding
        vision_features = self.vision_encoder(images)
        vision_features = self.vision_projection(vision_features)
        
        # Language features
        language_outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        language_features = language_outputs["hidden_states"]
        
        # Cross-modal fusion
        fused_features, attention_weights = self.cross_attention(
            language_features,
            vision_features,
            vision_features
        )
        
        # Apply RAG if enabled
        if self.config.use_rag:
            rag_outputs = self.rag_module(
                query_states=fused_features,
                document_ids=input_ids,
                attention_mask=attention_mask
            )
            fused_features = rag_outputs["fused_states"]
            
        # Apply hallucination reduction if enabled
        if self.config.use_hallucination_reducer:
            verification_outputs = self.hallucination_reducer(
                input_ids=input_ids,
                hidden_states=fused_features,
                attention_mask=attention_mask
            )
            confidence_scores = verification_outputs["confidence_scores"]
            
        # Quality assessment
        quality_scores = self.quality_head(fused_features)
        
        # Generate outputs
        outputs = {
            "fused_features": fused_features,
            "quality_scores": quality_scores,
            "logits": language_outputs["logits"]
        }
        
        if return_attention:
            outputs["attention_weights"] = attention_weights
            
        if return_loss and labels is not None:
            # Compute losses following EnhancedSFTLoss pattern
            loss_inputs = {
                "logits": outputs["logits"],
                "labels": labels,
                "quality_scores": quality_scores,
                "hallucination_scores": confidence_scores if self.config.use_hallucination_reducer else None,
                "attention_mask": attention_mask
            }
            losses = self.loss_fn(**loss_inputs)
            outputs.update(losses)
            
            # Update experience bank
            self.update_experience_bank(fused_features, quality_scores)
            
        return outputs 