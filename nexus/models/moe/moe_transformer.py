import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from ...core.base import NexusModule
from .expert_layer import ExpertTransformerLayer
from .router import BalancedTopKRouter

class EnhancedMoETransformer(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Validate config
        self._validate_config(config)
        
        # Core dimensions
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.num_experts = config["num_experts"]
        self.num_heads = config.get("num_heads", 8)
        
        # Input embeddings
        self.token_embedding = nn.Embedding(
            config["vocab_size"],
            self.hidden_size
        )
        self.position_embedding = nn.Embedding(
            config.get("max_position_embeddings", 512),
            self.hidden_size
        )
        
        # Expert transformer layers
        self.layers = nn.ModuleList([
            ExpertTransformerLayer(config)
            for _ in range(self.num_layers)
        ])
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        
        # Output projection
        self.output_proj = nn.Linear(
            self.hidden_size,
            config["vocab_size"]
        )
        
    def _validate_config(self, config: Dict[str, Any]) -> None:
        required = [
            "hidden_size",
            "num_layers",
            "num_experts",
            "vocab_size"
        ]
        for key in required:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Get sequence length and batch size
        batch_size, seq_length = input_ids.shape
        
        # Generate position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(
                seq_length,
                dtype=torch.long,
                device=input_ids.device
            ).unsqueeze(0).expand(batch_size, -1)
        
        # Get embeddings
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(position_ids)
        hidden_states = token_embeds + pos_embeds
        
        # Track expert routing patterns
        all_router_logits = []
        all_expert_patterns = []
        
        # Process through expert transformer layers
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask
            )
            hidden_states = layer_outputs["hidden_states"]
            
            # Track routing information
            all_router_logits.append(layer_outputs["router_logits"])
            all_expert_patterns.append(layer_outputs["expert_patterns"])
        
        # Final layer norm
        hidden_states = self.layer_norm(hidden_states)
        
        # Generate logits
        logits = self.output_proj(hidden_states)
        
        return {
            "logits": logits,
            "hidden_states": hidden_states,
            "router_logits": all_router_logits,
            "expert_patterns": all_expert_patterns
        }
