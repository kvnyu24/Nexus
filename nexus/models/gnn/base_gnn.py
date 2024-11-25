import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union
from ...core.base import NexusModule

class BaseGNNLayer(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Validate config
        self._validate_config(config)
        
        # Core dimensions
        self.input_dim = config["input_dim"]
        self.hidden_dim = config.get("hidden_dim", 256)
        self.output_dim = config.get("output_dim", self.hidden_dim)
        
        # Message passing components
        self.message_mlp = nn.Sequential(
            nn.Linear(self.input_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Node update
        self.node_update = nn.GRUCell(
            input_size=self.hidden_dim,
            hidden_size=self.output_dim
        )
        
        # Optional components
        self.dropout = nn.Dropout(config.get("dropout", 0.1))
        self.layer_norm = nn.LayerNorm(self.output_dim)
        
    def _validate_config(self, config: Dict[str, Any]) -> None:
        required = ["input_dim"]
        for key in required:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
                
    def message_fn(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        """Compute messages between nodes"""
        inputs = torch.cat([x_i, x_j], dim=-1)
        return self.message_mlp(inputs)
        
    def aggregate_fn(self, messages: torch.Tensor) -> torch.Tensor:
        """Aggregate messages from neighbors"""
        return torch.mean(messages, dim=1)
        
    def update_fn(self, nodes: torch.Tensor, messages: torch.Tensor) -> torch.Tensor:
        """Update node features"""
        return self.node_update(messages, nodes)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Get source and target node features
        row, col = edge_index
        x_i, x_j = x[row], x[col]
        
        # Compute messages
        messages = self.message_fn(x_i, x_j)
        
        # Aggregate messages for each node
        aggregated = self.aggregate_fn(messages)
        
        # Update node features
        updated = self.update_fn(x, aggregated)
        
        # Apply normalization and dropout
        out = self.dropout(self.layer_norm(updated))
        
        return {
            "node_features": out,
            "messages": messages,
            "aggregated_messages": aggregated
        } 