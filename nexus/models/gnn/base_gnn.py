import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union, Tuple
from ...core.base import NexusModule
from ...core.mixins import ConfigValidatorMixin
from ...visualization.hierarchical import HierarchicalVisualizer

class BaseGNNLayer(ConfigValidatorMixin, NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Validate config using ConfigValidatorMixin
        self.validate_config(config, required_keys=["input_dim"])
        if config.get("hidden_dim", 256) % config.get("num_heads", 4) != 0:
            raise ValueError("Hidden dimension must be divisible by number of heads")
        
        # Core dimensions
        self.input_dim = config["input_dim"]
        self.hidden_dim = config.get("hidden_dim", 256)
        self.output_dim = config.get("output_dim", self.hidden_dim)
        self.num_heads = config.get("num_heads", 4)
        
        # Message passing components with multi-head attention
        self.message_mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_dim * 2, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim // self.num_heads)
            ) for _ in range(self.num_heads)
        ])
        
        # Edge feature processing
        self.edge_proj = nn.Linear(config.get("edge_dim", 1), self.hidden_dim)
        
        # Node update with residual connection
        self.node_update = nn.GRUCell(
            input_size=self.hidden_dim,
            hidden_size=self.output_dim
        )
        
        # Output projection
        self.output_proj = nn.Linear(self.output_dim, self.output_dim)
        
        # Regularization
        dropout_rate = config.get("dropout", 0.1)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(self.output_dim)
        
        # Visualization
        self.visualizer = HierarchicalVisualizer(config)
        
    def message_fn(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute messages between nodes with multi-head attention"""
        inputs = torch.cat([x_i, x_j], dim=-1)
        
        # Multi-head message computation
        messages = []
        attention_weights = []
        
        for head in self.message_mlp:
            head_message = head(inputs)
            
            # Compute attention scores
            scores = torch.matmul(head_message, head_message.transpose(-2, -1))
            scores = scores / torch.sqrt(torch.tensor(self.hidden_dim / self.num_heads))
            
            # Apply edge features if provided
            if edge_attr is not None:
                edge_weights = self.edge_proj(edge_attr)
                scores = scores * edge_weights.unsqueeze(-1)
                
            weights = torch.softmax(scores, dim=-1)
            attention_weights.append(weights)
            
            # Apply attention
            messages.append(head_message * weights)
            
        return torch.cat(messages, dim=-1), torch.stack(attention_weights)
        
    def aggregate_fn(self, messages: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Aggregate messages with optional masking"""
        if mask is not None:
            messages = messages.masked_fill(~mask.unsqueeze(-1), 0)
        return torch.mean(messages, dim=1)
        
    def update_fn(self, nodes: torch.Tensor, messages: torch.Tensor) -> torch.Tensor:
        """Update node features with residual connection"""
        updated = self.node_update(messages, nodes)
        return updated + nodes  # Residual connection
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Input validation
        if x.dim() != 2:
            raise ValueError(f"Node features must be 2-dimensional, got shape {x.shape}")
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(f"Edge index must have shape [2, E], got {edge_index.shape}")
            
        # Get source and target node features
        row, col = edge_index
        x_i, x_j = x[row], x[col]
        
        # Compute messages and attention
        messages, attention_weights = self.message_fn(x_i, x_j, edge_attr)
        
        # Aggregate messages for each node
        aggregated = self.aggregate_fn(messages, mask)
        
        # Update node features
        updated = self.update_fn(x, aggregated)
        
        # Final processing
        out = self.output_proj(self.dropout(self.layer_norm(updated)))
        
        return {
            "node_features": out,
            "messages": messages,
            "aggregated_messages": aggregated,
            "attention_weights": attention_weights
        }