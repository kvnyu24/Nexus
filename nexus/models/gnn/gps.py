"""GPS (Graph Transformer) - Modular graph transformer framework.

GPS (General, Powerful, Scalable Graph Transformer) is a modular framework that combines
message-passing GNNs with global attention mechanisms to achieve strong performance
on a wide range of graph learning tasks.

Key features:
- Modular architecture combining local MPNN and global attention
- Flexible design allowing different MPNN and attention variants
- Positional/structural encodings (LapPE, RWSE)
- Scalable to large graphs
- Strong empirical performance across diverse benchmarks

References:
    - GPS: "Recipe for a General, Powerful, Scalable Graph Transformer" (NeurIPS 2022)
    - Paper: https://arxiv.org/abs/2205.12454
    - Code: https://github.com/rampasek/GraphGPS

Authors: Rampášek et al. (2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math
from nexus.core.base import NexusModule


class LaplacianPositionalEncoding(NexusModule):
    """Laplacian Positional Encoding for graphs.

    Uses eigenvectors of the graph Laplacian as positional features.

    Args:
        num_eigenvectors: Number of eigenvectors to use
        hidden_dim: Output dimension
    """

    def __init__(
        self,
        num_eigenvectors: int = 8,
        hidden_dim: int = 64,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)
        self.num_eigenvectors = num_eigenvectors
        self.hidden_dim = hidden_dim

        # Linear projection of eigenvectors
        self.linear = nn.Linear(num_eigenvectors, hidden_dim)

    def forward(self, eigenvectors: torch.Tensor) -> torch.Tensor:
        """Encode positional information from Laplacian eigenvectors.

        Args:
            eigenvectors: [num_nodes, num_eigenvectors]

        Returns:
            Positional encodings [num_nodes, hidden_dim]
        """
        pe = self.linear(eigenvectors)
        return pe


class RandomWalkStructuralEncoding(NexusModule):
    """Random Walk Structural Encoding for graphs.

    Uses random walk landing probabilities as structural features.

    Args:
        walk_length: Length of random walks
        hidden_dim: Output dimension
    """

    def __init__(
        self,
        walk_length: int = 16,
        hidden_dim: int = 64,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)
        self.walk_length = walk_length
        self.hidden_dim = hidden_dim

        # MLP to process random walk features
        self.mlp = nn.Sequential(
            nn.Linear(walk_length, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, rw_features: torch.Tensor) -> torch.Tensor:
        """Encode structural information from random walk features.

        Args:
            rw_features: [num_nodes, walk_length] landing probabilities

        Returns:
            Structural encodings [num_nodes, hidden_dim]
        """
        se = self.mlp(rw_features)
        return se


class LocalMessagePassing(NexusModule):
    """Local message passing layer (MPNN).

    Standard message passing over edges for local neighborhood aggregation.

    Args:
        hidden_dim: Hidden dimension
        dropout: Dropout rate
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)
        self.hidden_dim = hidden_dim

        # Message function
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Update function
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Local message passing.

        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim] (optional)

        Returns:
            Updated node features [num_nodes, hidden_dim]
        """
        num_nodes = x.shape[0]
        src, dst = edge_index

        # Compute messages
        messages = torch.cat([x[src], x[dst]], dim=-1)
        messages = self.message_mlp(messages)

        # Aggregate messages
        aggregated = torch.zeros(num_nodes, self.hidden_dim, device=x.device, dtype=x.dtype)
        aggregated.index_add_(0, dst, messages)

        # Update nodes
        updated = torch.cat([x, aggregated], dim=-1)
        updated = self.update_mlp(updated)

        return updated


class GlobalAttention(NexusModule):
    """Global attention layer for graphs.

    Multi-head attention over all nodes in the graph.

    Args:
        hidden_dim: Hidden dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.0,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Multi-head attention
        self.qkv_proj = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Global attention over all nodes.

        Args:
            x: Node features [num_nodes, hidden_dim]
            attention_mask: Optional mask [num_nodes, num_nodes]

        Returns:
            Updated node features [num_nodes, hidden_dim]
        """
        num_nodes = x.shape[0]

        # Compute Q, K, V
        qkv = self.qkv_proj(x)  # [num_nodes, hidden_dim * 3]
        qkv = qkv.reshape(num_nodes, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(1, 2, 0, 3)  # [3, num_heads, num_nodes, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [num_heads, num_nodes, num_nodes]

        # Apply mask if provided
        if attention_mask is not None:
            attn = attn.masked_fill(attention_mask == 0, float('-inf'))

        # Softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)  # [num_heads, num_nodes, head_dim]
        out = out.permute(1, 0, 2).reshape(num_nodes, self.hidden_dim)

        # Output projection
        out = self.out_proj(out)

        return out


class GPSLayer(NexusModule):
    """GPS layer combining local MPNN and global attention.

    Each GPS layer consists of:
    1. Local message passing (MPNN)
    2. Global attention
    3. Feed-forward network
    All with residual connections and layer normalization.

    Args:
        hidden_dim: Hidden dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
        use_mpnn: Whether to use local message passing
        use_attention: Whether to use global attention
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_mpnn: bool = True,
        use_attention: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)
        self.use_mpnn = use_mpnn
        self.use_attention = use_attention

        # Local MPNN
        if use_mpnn:
            self.mpnn = LocalMessagePassing(hidden_dim, dropout)
            self.norm1 = nn.LayerNorm(hidden_dim)

        # Global attention
        if use_attention:
            self.attention = GlobalAttention(hidden_dim, num_heads, dropout)
            self.norm2 = nn.LayerNorm(hidden_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """GPS layer forward pass.

        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features (optional)
            attention_mask: Attention mask (optional)

        Returns:
            Updated node features [num_nodes, hidden_dim]
        """
        # Local MPNN
        if self.use_mpnn:
            x_mpnn = self.mpnn(x, edge_index, edge_attr)
            x = x + x_mpnn
            x = self.norm1(x)

        # Global attention
        if self.use_attention:
            x_attn = self.attention(x, attention_mask)
            x = x + x_attn
            x = self.norm2(x)

        # Feed-forward
        x_ffn = self.ffn(x)
        x = x + x_ffn
        x = self.norm3(x)

        return x


class GPS(NexusModule):
    """GPS: General, Powerful, Scalable Graph Transformer.

    A modular graph transformer that combines local message passing with
    global attention for strong performance on diverse graph tasks.

    Key components:
    - Positional/structural encodings (LapPE, RWSE)
    - Local message passing (MPNN)
    - Global attention mechanism
    - Residual connections and normalization

    Args:
        in_channels: Input node feature dimension
        hidden_dim: Hidden dimension
        out_channels: Output dimension
        num_layers: Number of GPS layers
        num_heads: Number of attention heads
        dropout: Dropout rate
        use_laplacian_pe: Whether to use Laplacian positional encoding
        use_rwse: Whether to use random walk structural encoding
        num_eigenvectors: Number of Laplacian eigenvectors
        walk_length: Random walk length

    Example:
        >>> model = GPS(
        ...     in_channels=32,
        ...     hidden_dim=256,
        ...     out_channels=10,
        ...     num_layers=4,
        ...     num_heads=8
        ... )
        >>> x = torch.randn(100, 32)  # 100 nodes, 32 features
        >>> edge_index = torch.randint(0, 100, (2, 300))  # 300 edges
        >>> # Optional: provide Laplacian eigenvectors
        >>> laplacian_eigvec = torch.randn(100, 8)
        >>> output = model(x, edge_index, laplacian_eigvec=laplacian_eigvec)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
        out_channels: int = 1,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_laplacian_pe: bool = True,
        use_rwse: bool = True,
        num_eigenvectors: int = 8,
        walk_length: int = 16,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.num_layers = num_layers

        # Input embedding
        self.node_embed = nn.Linear(in_channels, hidden_dim)

        # Positional/structural encodings
        self.use_laplacian_pe = use_laplacian_pe
        self.use_rwse = use_rwse

        if use_laplacian_pe:
            self.laplacian_pe = LaplacianPositionalEncoding(
                num_eigenvectors=num_eigenvectors,
                hidden_dim=hidden_dim
            )

        if use_rwse:
            self.rwse = RandomWalkStructuralEncoding(
                walk_length=walk_length,
                hidden_dim=hidden_dim
            )

        # GPS layers
        self.layers = nn.ModuleList([
            GPSLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_mpnn=True,
                use_attention=True
            )
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_channels)
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        laplacian_eigvec: Optional[torch.Tensor] = None,
        rw_features: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through GPS.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim] (optional)
            laplacian_eigvec: Laplacian eigenvectors [num_nodes, num_eigenvectors] (optional)
            rw_features: Random walk features [num_nodes, walk_length] (optional)
            batch: Batch assignment [num_nodes] (optional, for graph-level tasks)

        Returns:
            Node or graph embeddings [num_nodes, out_channels] or [batch_size, out_channels]
        """
        # Embed input features
        h = self.node_embed(x)

        # Add positional encodings
        if self.use_laplacian_pe and laplacian_eigvec is not None:
            h = h + self.laplacian_pe(laplacian_eigvec)

        # Add structural encodings
        if self.use_rwse and rw_features is not None:
            h = h + self.rwse(rw_features)

        # Apply GPS layers
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)

        # Output projection
        out = self.output_proj(h)

        # Graph-level pooling if batch is provided
        if batch is not None:
            # Global mean pooling
            batch_size = batch.max().item() + 1
            graph_out = torch.zeros(
                batch_size, self.out_channels,
                device=x.device, dtype=x.dtype
            )

            for i in range(batch_size):
                mask = (batch == i)
                graph_out[i] = out[mask].mean(dim=0)

            return graph_out

        return out


# Export
__all__ = [
    'GPS',
    'GPSLayer',
    'LocalMessagePassing',
    'GlobalAttention',
    'LaplacianPositionalEncoding',
    'RandomWalkStructuralEncoding'
]
