import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from ...core.base import NexusModule
from .attention import GraphAttention
from torch_scatter import scatter_mean

class EnhancedGNNLayer(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Core dimensions
        self.hidden_dim = config.get("hidden_dim", 256)
        self.intermediate_dim = config.get("intermediate_dim", self.hidden_dim * 4)
        
        # Multi-head attention
        self.attention = GraphAttention(config)
        
        # Message passing components
        self.message_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.intermediate_dim),
            nn.GELU(),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(self.intermediate_dim, self.hidden_dim)
        )
        
        # Node update
        self.node_update = nn.GRUCell(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim
        )
        
        # Layer norm and dropout
        self.layer_norm1 = nn.LayerNorm(self.hidden_dim)
        self.layer_norm2 = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(config.get("dropout", 0.1))
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Attention mechanism
        attention_out = self.attention(x, edge_index, edge_attr)
        x = attention_out["node_features"]
        
        # Message passing
        row, col = edge_index
        edge_features = torch.cat([x[row], x[col]], dim=-1)
        messages = self.message_mlp(edge_features)
        
        # Aggregate messages
        aggregated = scatter_mean(messages, row, dim=0, dim_size=x.size(0))
        
        # Update node features
        updated = self.node_update(aggregated, x)
        
        # Residual connection and normalization
        out = self.layer_norm2(x + self.dropout(updated))
        
        return {
            "node_features": out,
            "attention_weights": attention_out["attention_weights"],
            "messages": messages
        } 