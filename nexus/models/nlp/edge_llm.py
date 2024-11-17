import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
from ...core.base import NexusModule
from ...components.attention import FlashAttention

class EdgeTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Efficient attention using Flash Attention
        self.attention = FlashAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            batch_size=1  # Optimized for edge inference
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # Efficient feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm architecture for better training stability
        attn_output = self.attention(self.norm1(x))
        x = x + attn_output
        
        ffn_output = self.ffn(self.norm2(x))
        x = x + ffn_output
        
        return x

class EdgeLLM(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Model configuration
        self.vocab_size = config["vocab_size"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.num_heads = config["num_heads"]
        self.intermediate_size = config["intermediate_size"]
        self.max_seq_length = config["max_seq_length"]
        self.dropout = config.get("dropout", 0.1)
        
        # Token embeddings
        self.token_embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.max_seq_length, self.hidden_size)
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            EdgeTransformerBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                intermediate_size=self.intermediate_size,
                dropout=self.dropout
            ) for _ in range(self.num_layers)
        ])
        
        # Output head
        self.norm = nn.LayerNorm(self.hidden_size)
        self.output = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        
        # Tie weights
        self.output.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
            
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