import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union, Tuple
from ...core.base import NexusModule
from .attention import GraphAttention
from torch_scatter import scatter_mean, scatter_max, scatter_add
from ...visualization.hierarchical import HierarchicalVisualizer

class AdaptiveMessagePassingLayer(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Core dimensions
        self.hidden_dim = config.get("hidden_dim", 256)
        self.intermediate_dim = config.get("intermediate_dim", self.hidden_dim * 4)
        
        # Reuse attention module from GraphAttention
        self.attention = GraphAttention(config)
        
        # Focused message passing with edge features
        self.message_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim * 2 + config.get("edge_dim", 0), self.intermediate_dim),
            nn.GELU(),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(self.intermediate_dim, self.hidden_dim)
        )
        
        # Adaptive node update mechanism
        self.node_update = nn.GRUCell(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim
        )
        
        # Dynamic feature gating
        self.feature_gate = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Sigmoid()
        )
        
        # Normalization and regularization
        norm_eps = config.get("layer_norm_eps", 1e-5)
        self.pre_norm = nn.LayerNorm(self.hidden_dim, eps=norm_eps)
        self.post_norm = nn.LayerNorm(self.hidden_dim, eps=norm_eps)
        self.dropout = nn.Dropout(config.get("dropout", 0.1))
        
        # Visualization
        self.visualizer = HierarchicalVisualizer(config)
        
    def _validate_input(self, x: torch.Tensor) -> None:
        if x.dim() != 2:
            raise ValueError(f"Node features must be 2-dimensional, got {x.dim()}")
        if x.size(-1) != self.hidden_dim:
            raise ValueError(f"Feature dimension mismatch. Expected {self.hidden_dim}, got {x.size(-1)}")
            
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with enhanced message passing and attention
        
        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Optional edge features [num_edges, edge_dim]
            batch: Optional batch assignment for nodes
            
        Returns:
            Dictionary containing updated features and intermediate values
        """
        self._validate_input(x)
        identity = x
        
        # Pre-normalize and apply attention
        x = self.pre_norm(x)
        attention_out = self.attention(x, edge_index, edge_attr, batch=batch)
        x = x + self.dropout(attention_out["node_features"])
        
        # Process messages with edge features
        row, col = edge_index
        message_inputs = [x[row], x[col]]
        if edge_attr is not None:
            message_inputs.append(edge_attr)
        messages = self.message_encoder(torch.cat(message_inputs, dim=-1))
        
        # Aggregate messages using attention module's aggregation
        aggregated = self.attention._aggregate_neighbors(messages, row)
        
        # Update node states with gating
        gate_weights = self.feature_gate(torch.cat([x, aggregated], dim=-1))
        updated = self.node_update(aggregated, x)
        x = gate_weights * updated + (1 - gate_weights) * x
        
        # Post-process with residual
        out = self.post_norm(identity + self.dropout(x))
        
        return {
            "node_features": out,
            "attention_weights": attention_out.get("attention_weights"),
            "gate_values": gate_weights
        }