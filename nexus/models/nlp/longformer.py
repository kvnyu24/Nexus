import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from ...core.base import NexusModule

class LongSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, window_size: int = 512):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, _ = x.shape
        
        # Project queries, keys, values
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim)
        
        # Compute sliding window attention
        output = torch.zeros_like(q)
        for i in range(0, L, self.window_size):
            end_idx = min(i + self.window_size, L)
            
            # Local attention
            scores = torch.matmul(
                q[:, i:end_idx],
                k[:, max(0, i-self.window_size):end_idx].transpose(-2, -1)
            )
            
            if attention_mask is not None:
                scores = scores.masked_fill(
                    attention_mask[:, i:end_idx, max(0, i-self.window_size):end_idx].unsqueeze(2) == 0,
                    float('-inf')
                )
            
            attn_probs = torch.softmax(scores / (self.head_dim ** 0.5), dim=-1)
            output[:, i:end_idx] = torch.matmul(
                attn_probs,
                v[:, max(0, i-self.window_size):end_idx]
            )
        
        return self.o_proj(output.reshape(B, L, -1))

class Longformer(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.vocab_size = config["vocab_size"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.num_heads = config["num_heads"]
        self.window_size = config.get("window_size", 512)
        
        # Embeddings
        self.token_embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config["max_seq_length"], self.hidden_size)
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': LongSelfAttention(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    window_size=self.window_size
                ),
                'feed_forward': nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size * 4),
                    nn.GELU(),
                    nn.Linear(self.hidden_size * 4, self.hidden_size)
                ),
                'norm1': nn.LayerNorm(self.hidden_size),
                'norm2': nn.LayerNorm(self.hidden_size)
            }) for _ in range(self.num_layers)
        ])
        
        self.output = nn.Linear(self.hidden_size, self.vocab_size)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Get embeddings
        x = self.token_embedding(input_ids)
        x = x + self.position_embedding[:, :input_ids.size(1)]
        
        # Apply transformer layers
        for layer in self.layers:
            # Self attention
            norm_x = layer['norm1'](x)
            x = x + layer['attention'](norm_x, attention_mask)
            
            # Feed forward
            norm_x = layer['norm2'](x)
            x = x + layer['feed_forward'](norm_x)
            
        return {
            "logits": self.output(x),
            "hidden_states": x
        }
