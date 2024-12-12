import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, Union, List
from ...core.base import NexusModule
from ...visualization.hierarchical import HierarchicalVisualizer
import math
from torch_scatter import scatter_mean, scatter_max, scatter_add

class GraphAttention(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Core dimensions with validation
        self.hidden_dim = config.get("hidden_dim", 256)
        self.num_heads = config.get("num_heads", 8)
        self.head_dim = self.hidden_dim // self.num_heads
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("Hidden dim must be divisible by num heads")
        
        # Enhanced attention config
        self.dropout_rate = config.get("dropout", 0.1)
        self.use_bias = config.get("use_bias", True)
        self.attention_type = config.get("attention_type", "scaled_dot_product")
        self.edge_dim = config.get("edge_dim", None)
        self.use_layer_norm = config.get("use_layer_norm", True)
        self.aggregation_type = config.get("aggregation", "combined")
        
        # Multi-head attention components with initialization
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.use_bias)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.use_bias)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.use_bias)
        self.o_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.use_bias)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
        
        # Enhanced edge feature processing
        if self.edge_dim is not None:
            self.edge_proj = nn.Sequential(
                nn.Linear(self.edge_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.num_heads),
                nn.Dropout(self.dropout_rate)
            )
            
        # Multiple aggregation functions
        self.aggregation_fns = {
            "mean": lambda x, idx: scatter_mean(x, idx, dim=0),
            "max": lambda x, idx: scatter_max(x, idx, dim=0)[0],
            "sum": lambda x, idx: scatter_add(x, idx, dim=0)
        }
        if self.aggregation_type == "combined":
            self.agg_weights = nn.Parameter(torch.ones(3))
        
        # Enhanced normalization and regularization
        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(self.hidden_dim)
            self.pre_norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Optional visualization with config
        if config.get("enable_visualization", False):
            vis_config = config.get("vis_config", {})
            vis_config.update({
                "attention_cmap": config.get("attention_cmap", "viridis"),
                "alpha": config.get("vis_alpha", 0.7)
            })
            self.visualizer = HierarchicalVisualizer(vis_config)
        
    def _compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        edge_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Enhanced attention computation with edge weights"""
        if self.attention_type == "scaled_dot_product":
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        elif self.attention_type == "additive":
            scores = torch.tanh(q.unsqueeze(-2) + k.unsqueeze(-3))
            scores = scores.mean(dim=-1) / math.sqrt(self.head_dim)
        else:
            raise ValueError(f"Unknown attention type: {self.attention_type}")
            
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        if edge_weights is not None:
            scores = scores * edge_weights.unsqueeze(-1)
            
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        
        return torch.matmul(weights, v), weights
        
    def _aggregate_neighbors(self, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """Aggregate node features using specified method"""
        if self.aggregation_type == "combined":
            weights = torch.softmax(self.agg_weights, dim=0)
            outputs = []
            for i, fn in enumerate(self.aggregation_fns.values()):
                outputs.append(weights[i] * fn(x, index))
            return sum(outputs)
        return self.aggregation_fns[self.aggregation_type](x, index)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Enhanced forward pass with additional features
        
        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Optional edge features [num_edges, edge_dim]
            attention_mask: Optional mask for attention weights
            return_attention: Whether to return attention weights
            batch: Optional batch assignment for nodes
            
        Returns:
            Dictionary containing updated node features and optionally attention weights
        """
        batch_size = x.size(0)
        
        # Pre-normalization
        if self.use_layer_norm:
            x = self.pre_norm(x)
        
        # Project queries, keys, values with dropout
        q = self.dropout(self.q_proj(x)).view(batch_size, -1, self.num_heads, self.head_dim)
        k = self.dropout(self.k_proj(x)).view(batch_size, -1, self.num_heads, self.head_dim)
        v = self.dropout(self.v_proj(x)).view(batch_size, -1, self.num_heads, self.head_dim)
        
        # Get source and target node indices
        row, col = edge_index
        
        # Process edge features
        edge_weights = None
        if edge_attr is not None and self.edge_dim is not None:
            edge_weights = self.edge_proj(edge_attr)
        
        # Compute attention scores with edge attributes
        out, weights = self._compute_attention(
            q[row], k[col], v[col],
            mask=attention_mask,
            edge_weights=edge_weights
        )
        
        # Aggregate neighborhood information
        if batch is not None:
            out = self._aggregate_neighbors(out, batch)
            
        # Reshape and project output
        out = out.view(batch_size, -1, self.hidden_dim)
        out = self.o_proj(out)
        out = self.dropout(out)
        
        # Residual connection and normalization
        output = x + out
        if self.use_layer_norm:
            output = self.layer_norm(output)
        
        result = {"node_features": output}
        if return_attention:
            result["attention_weights"] = weights
            
        return result