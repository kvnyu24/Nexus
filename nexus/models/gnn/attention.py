import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from ...core.base import NexusModule
import math

class GraphAttention(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Core dimensions
        self.hidden_dim = config.get("hidden_dim", 256)
        self.num_heads = config.get("num_heads", 8)
        self.head_dim = self.hidden_dim // self.num_heads
        self.dropout = config.get("dropout", 0.1)
        
        # Multi-head attention components
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.o_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        # Normalization and dropout
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(self.dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Project queries, keys, values
        q = self.q_proj(x).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(-1, self.num_heads, self.head_dim)
        
        # Get source and target node indices
        row, col = edge_index
        
        # Compute attention scores
        scores = torch.matmul(q[row], k[col].transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)
        
        # Apply edge attributes if provided
        if edge_attr is not None:
            edge_weights = self.edge_proj(edge_attr).view(-1, self.num_heads, 1)
            scores = scores * edge_weights
            
        # Normalize attention weights
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        
        # Compute weighted values
        out = torch.matmul(weights, v[col])
        out = out.view(-1, self.hidden_dim)
        out = self.o_proj(out)
        
        return {
            "node_features": self.layer_norm(x + out),
            "attention_weights": weights
        } 