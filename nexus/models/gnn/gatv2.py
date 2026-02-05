"""GATv2: Graph Attention Network v2 with dynamic attention.

GATv2 fixes a limitation of the original GAT where attention mechanisms were
essentially static. GATv2 introduces dynamic attention that can attend to any node
in the neighborhood regardless of the query node.

Key improvements over GAT:
- Dynamic attention mechanism (more expressive)
- Better performance on various graph learning tasks
- Can learn attention patterns that depend on both query and key nodes
- Addresses theoretical limitations of original GAT

References:
    - GATv2: "How Attentive are Graph Attention Networks?" (ICLR 2022)
    - Paper: https://arxiv.org/abs/2105.14491
    - Original GAT: https://arxiv.org/abs/1710.10903

Authors: Brody et al. (2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import math
from nexus.core.base import NexusModule


class GATv2Conv(NexusModule):
    """Graph Attention Network v2 Convolution Layer.

    Implements dynamic attention mechanism that addresses the static
    attention limitation of original GAT.

    The key difference from GAT:
    - GAT: a(Wh_i, Wh_j) - attention computed on linearly transformed features
    - GATv2: LeakyReLU(a^T [W[h_i || h_j]]) - attention computed after concatenation

    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension
        heads: Number of attention heads
        concat: Whether to concatenate head outputs (True) or average (False)
        dropout: Dropout rate
        add_self_loops: Whether to add self-loops
        bias: Whether to use bias
        share_weights: Whether to share weight matrix for source and target nodes
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        bias: bool = True,
        share_weights: bool = False,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.share_weights = share_weights

        # Weight matrices
        if not share_weights:
            # Separate weights for source and target nodes
            self.weight_src = nn.Parameter(
                torch.Tensor(in_channels, heads * out_channels)
            )
            self.weight_dst = nn.Parameter(
                torch.Tensor(in_channels, heads * out_channels)
            )
        else:
            # Shared weight matrix
            self.weight = nn.Parameter(
                torch.Tensor(in_channels, heads * out_channels)
            )

        # Attention mechanism (key difference from GAT)
        # Single attention weight vector applied after concatenation
        self.att = nn.Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        if hasattr(self, 'weight'):
            nn.init.xavier_uniform_(self.weight)
        else:
            nn.init.xavier_uniform_(self.weight_src)
            nn.init.xavier_uniform_(self.weight_dst)

        nn.init.xavier_uniform_(self.att)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of GATv2 convolution.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features (optional, not used in standard GATv2)
            return_attention_weights: Whether to return attention weights

        Returns:
            Tuple of:
                - Updated node features [num_nodes, heads * out_channels] if concat
                  else [num_nodes, out_channels]
                - Attention weights [num_edges, heads] if return_attention_weights=True
        """
        num_nodes = x.shape[0]

        # Add self-loops
        if self.add_self_loops:
            self_loop_edges = torch.arange(num_nodes, device=x.device).repeat(2, 1)
            edge_index = torch.cat([edge_index, self_loop_edges], dim=1)

        # Linear transformation
        if self.share_weights:
            h_src = h_dst = x @ self.weight
        else:
            h_src = x @ self.weight_src
            h_dst = x @ self.weight_dst

        # Reshape for multi-head attention
        h_src = h_src.view(-1, self.heads, self.out_channels)
        h_dst = h_dst.view(-1, self.heads, self.out_channels)

        # Get source and destination nodes
        src, dst = edge_index

        # GATv2 dynamic attention mechanism
        # Key difference: concatenate THEN apply non-linearity and attention
        h_i = h_dst[dst]  # [num_edges, heads, out_channels]
        h_j = h_src[src]  # [num_edges, heads, out_channels]

        # Add features then apply attention (GATv2 formulation)
        alpha = (h_i + h_j) * self.att  # [num_edges, heads, out_channels]
        alpha = alpha.sum(dim=-1)  # [num_edges, heads]
        alpha = F.leaky_relu(alpha, negative_slope=0.2)

        # Softmax over incoming edges for each node
        alpha = self._softmax_per_node(alpha, dst, num_nodes)

        # Apply dropout
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Aggregate messages
        out = torch.zeros(
            num_nodes, self.heads, self.out_channels,
            device=x.device, dtype=x.dtype
        )

        # Weight messages by attention and aggregate
        weighted_h_j = alpha.unsqueeze(-1) * h_j  # [num_edges, heads, out_channels]
        out.index_add_(0, dst, weighted_h_j)

        # Concatenate or average heads
        if self.concat:
            out = out.view(num_nodes, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        # Add bias
        if self.bias is not None:
            out = out + self.bias

        if return_attention_weights:
            return out, alpha
        return out, None

    def _softmax_per_node(
        self,
        alpha: torch.Tensor,
        dst: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """Apply softmax per destination node.

        Args:
            alpha: Attention logits [num_edges, heads]
            dst: Destination node indices [num_edges]
            num_nodes: Total number of nodes

        Returns:
            Normalized attention weights [num_edges, heads]
        """
        # Compute max per node for numerical stability
        alpha_max = torch.zeros(num_nodes, self.heads, device=alpha.device, dtype=alpha.dtype)
        alpha_max = alpha_max.fill_(float('-inf'))
        alpha_max.scatter_reduce_(0, dst.unsqueeze(1).expand_as(alpha), alpha, reduce='amax')
        alpha_max = alpha_max[dst]

        # Subtract max and exponentiate
        alpha = torch.exp(alpha - alpha_max)

        # Compute sum per node
        alpha_sum = torch.zeros(num_nodes, self.heads, device=alpha.device, dtype=alpha.dtype)
        alpha_sum.index_add_(0, dst, alpha)
        alpha_sum = alpha_sum[dst]

        # Normalize
        alpha = alpha / (alpha_sum + 1e-16)

        return alpha


class GATv2(NexusModule):
    """Graph Attention Network v2 with dynamic attention.

    Multi-layer GATv2 for node-level or graph-level predictions.
    Fixes theoretical limitations of original GAT by using dynamic
    attention mechanism.

    Key advantages:
    - More expressive attention than GAT
    - Dynamic attention patterns
    - Better performance on various benchmarks
    - Can handle both homophilic and heterophilic graphs

    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden layer dimensions
        out_channels: Output dimension
        num_layers: Number of GATv2 layers
        heads: Number of attention heads per layer
        concat_heads: Whether to concatenate (True) or average (False) heads
        dropout: Dropout rate
        add_self_loops: Whether to add self-loops to edges

    Example:
        >>> model = GATv2(
        ...     in_channels=16,
        ...     hidden_channels=64,
        ...     out_channels=7,
        ...     num_layers=2,
        ...     heads=4
        ... )
        >>> x = torch.randn(100, 16)  # 100 nodes, 16 features
        >>> edge_index = torch.randint(0, 100, (2, 300))
        >>> output = model(x, edge_index)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 1,
        num_layers: int = 2,
        heads: int = 4,
        concat_heads: bool = True,
        dropout: float = 0.6,
        add_self_loops: bool = True,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # Input layer
        self.convs.append(
            GATv2Conv(
                in_channels,
                hidden_channels,
                heads=heads,
                concat=concat_heads,
                dropout=dropout,
                add_self_loops=add_self_loops
            )
        )
        self.norms.append(
            nn.LayerNorm(hidden_channels * heads if concat_heads else hidden_channels)
        )

        # Hidden layers
        for _ in range(num_layers - 2):
            in_dim = hidden_channels * heads if concat_heads else hidden_channels
            self.convs.append(
                GATv2Conv(
                    in_dim,
                    hidden_channels,
                    heads=heads,
                    concat=concat_heads,
                    dropout=dropout,
                    add_self_loops=add_self_loops
                )
            )
            self.norms.append(
                nn.LayerNorm(hidden_channels * heads if concat_heads else hidden_channels)
            )

        # Output layer
        if num_layers > 1:
            in_dim = hidden_channels * heads if concat_heads else hidden_channels
            self.convs.append(
                GATv2Conv(
                    in_dim,
                    out_channels,
                    heads=1,  # Single head for output
                    concat=False,
                    dropout=dropout,
                    add_self_loops=add_self_loops
                )
            )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """Forward pass through GATv2.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]
            batch: Batch assignment [num_nodes] for graph-level tasks
            return_attention_weights: Whether to return attention weights from each layer

        Returns:
            Tuple of:
                - Output features [num_nodes, out_channels] or [batch_size, out_channels]
                - List of attention weights per layer (if return_attention_weights=True)
        """
        attention_weights = [] if return_attention_weights else None

        # Apply GATv2 layers
        for i, conv in enumerate(self.convs):
            h, attn = conv(x, edge_index, return_attention_weights=return_attention_weights)

            if return_attention_weights:
                attention_weights.append(attn)

            # Apply normalization and activation (except last layer)
            if i < len(self.convs) - 1:
                h = self.norms[i](h)
                h = F.elu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

            x = h

        # Graph-level pooling if batch is provided
        if batch is not None:
            batch_size = batch.max().item() + 1
            graph_out = torch.zeros(
                batch_size, self.out_channels,
                device=x.device, dtype=x.dtype
            )

            for i in range(batch_size):
                mask = (batch == i)
                graph_out[i] = x[mask].mean(dim=0)

            return graph_out, attention_weights

        return x, attention_weights


# Export
__all__ = ['GATv2', 'GATv2Conv']
