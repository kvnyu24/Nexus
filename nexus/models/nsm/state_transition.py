import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from ...core.base import NexusModule

class StateTransitionModule(NexusModule):
    def __init__(
        self,
        hidden_dim: int,
        num_states: int,
        num_heads: int = 8
    ):
        super().__init__({})
        
        # Core dimensions
        self.hidden_dim = hidden_dim
        self.num_states = num_states
        
        # Transition network (following EnhancedGNN pattern)
        self.transition_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_states)
        )
        
        # Memory attention (following EnhancedReID pattern)
        self.memory_attention = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1
            ),
            num_layers=2
        )
        
        # Uncertainty estimation (following PlanningModule pattern)
        self.uncertainty_head = nn.Linear(hidden_dim, num_states)
        
    def forward(
        self,
        state_embed: torch.Tensor,
        input_embed: torch.Tensor,
        memory_bank: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Process memory context (following EnhancedPlanningModule pattern)
        memory_context = self.memory_attention(
            memory_bank.unsqueeze(0),
            src_key_padding_mask=attention_mask
        ).squeeze(0)
        
        # Concatenate inputs (following EnhancedGNN pattern)
        transition_input = torch.cat([
            state_embed,
            input_embed,
            memory_context
        ], dim=-1)
        
        # Generate transition logits and uncertainty
        transition_logits = self.transition_net(transition_input)
        uncertainty = self.uncertainty_head(state_embed)
        
        return {
            "logits": transition_logits,
            "uncertainty": uncertainty,
            "memory_context": memory_context
        }