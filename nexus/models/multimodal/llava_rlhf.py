from typing import Dict, Any, Optional, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...core.base import NexusModule
from ..cv.vit import VisionTransformer
from ..nlp.llm import LlamaModel
from ..nlp.rag import EnhancedRAGModule
from ..nlp.hallucination_reducer import HallucinationReducer

class LLaVARLHFConfig:
    """Configuration class for LLaVA-RLHF"""
    def __init__(
        self,
        vision_config: Dict[str, Any],
        language_config: Dict[str, Any],
        hidden_size: int = 768,
        num_heads: int = 12,
        max_seq_length: int = 1024,
        reward_scale: float = 0.1,
        kl_coef: float = 0.1,
        **kwargs
    ):
        self.vision_config = vision_config
        self.language_config = language_config
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.max_seq_length = max_seq_length
        self.reward_scale = reward_scale
        self.kl_coef = kl_coef
        
        for key, value in kwargs.items():
            setattr(self, key, value)

class LLaVARLHF(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Convert dict config to LLaVARLHFConfig
        if not isinstance(config, LLaVARLHFConfig):
            config = LLaVARLHFConfig(**config)
        self.config = config
        
        # Core components
        self.vision_encoder = VisionTransformer(config.vision_config)
        self.language_model = LlamaModel(config.language_config)
        self.reward_model = self._build_reward_model()
        
        # Cross-modal fusion
        self.vision_projection = nn.Linear(
            config.vision_config["hidden_size"],
            config.hidden_size
        )
        self.cross_attention = nn.MultiheadAttention(
            config.hidden_size,
            config.num_heads,
            batch_first=True
        )
        
        # Quality assessment components
        self.rag_module = EnhancedRAGModule(config)
        self.hallucination_reducer = HallucinationReducer(config)
        
        # Experience replay buffer
        self.register_buffer(
            "experience_bank",
            torch.zeros(
                config.get("bank_size", 10000),
                config.hidden_size
            )
        )
        self.register_buffer("rewards", torch.zeros(config.get("bank_size", 10000)))
        self.register_buffer("bank_ptr", torch.zeros(1, dtype=torch.long))
        
    def _build_reward_model(self) -> nn.Module:
        """Build reward model following RLHF architecture"""
        return nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.config.hidden_size, 1)
        )
        
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration parameters"""
        required = [
            "vision_config",
            "language_config",
            "hidden_size",
            "num_heads"
        ]
        for key in required:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
                
    def update_experience_bank(
        self,
        states: torch.Tensor,
        rewards: torch.Tensor
    ) -> None:
        """Update experience replay buffer"""
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
        return_loss: bool = True
    ) -> Dict[str, torch.Tensor]:
        # Encode images
        vision_features = self.vision_encoder(images)
        vision_features = self.vision_projection(vision_features)
        
        # Get language features
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
        
        # Get reward predictions
        rewards = self.reward_model(fused_features).squeeze(-1)
        
        # Verify factual consistency
        verification_outputs = self.hallucination_reducer(
            input_ids=input_ids,
            hidden_states=fused_features,
            attention_mask=attention_mask
        )
        
        outputs = {
            "rewards": rewards,
            "fused_features": fused_features,
            "attention_weights": attention_weights,
            "verification_scores": verification_outputs["confidence_scores"]
        }
        
        if return_loss and labels is not None:
            # Compute RLHF loss
            policy_loss = -rewards.mean()
            kl_loss = F.kl_div(
                F.log_softmax(language_outputs["logits"], dim=-1),
                F.softmax(labels, dim=-1),
                reduction='batchmean'
            )
            
            # Combine losses
            total_loss = (
                policy_loss +
                self.config.kl_coef * kl_loss +
                verification_outputs["loss"]
            )
            
            outputs.update({
                "policy_loss": policy_loss,
                "kl_loss": kl_loss,
                "total_loss": total_loss
            })
            
        # Update experience bank
        self.update_experience_bank(fused_features, rewards)
        
        return outputs 