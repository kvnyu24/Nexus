from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from .base_llm import BaseLLM, BaseLLMConfig
from nexus.core.base import NexusModule
from ....core.initialization import WeightInitMixin

class EdgeLLMConfig(BaseLLMConfig):
    """Configuration class for EdgeLLM"""
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 384,
        num_layers: int = 6,
        num_heads: int = 6,
        intermediate_size: int = 1024,
        max_seq_length: int = 512,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            max_seq_length=max_seq_length,
            dropout=dropout,
            **kwargs
        )

class EdgeTransformerBlock(NexusModule):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        dropout: float
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_size,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with residual
        normed_states = self.norm1(hidden_states)
        attention_output, _ = self.attention(
            normed_states,
            normed_states,
            normed_states,
            key_padding_mask=attention_mask,
            need_weights=False
        )
        hidden_states = hidden_states + attention_output
        
        # Feed-forward with residual
        hidden_states = hidden_states + self.feed_forward(self.norm2(hidden_states))
        return hidden_states

class EdgeLLM(BaseLLM):
    def __init__(self, config: Dict[str, Any]):
        # Convert dict config to EdgeLLMConfig if needed
        if not isinstance(config, EdgeLLMConfig):
            config = EdgeLLMConfig(**config)
        super().__init__(config)

        # Override transformer layers with EdgeLLM-specific blocks
        self.layers = nn.ModuleList([
            EdgeTransformerBlock(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                intermediate_size=config.intermediate_size,
                dropout=config.dropout
            ) for _ in range(config.num_layers)
        ])

        # Initialize weights using LLM preset
        self.init_weights_llm()
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        B, L = input_ids.shape
        
        # Get embeddings
        x = self.token_embedding(input_ids)
        x = x + self.position_embedding[:, :L, :]
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)
            
        # Output
        x = self.norm(x)
        logits = self.output(x)
        
        return {
            "logits": logits,
            "hidden_states": x
        } 