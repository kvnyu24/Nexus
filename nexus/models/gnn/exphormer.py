"""Exphormer: Sparse graph transformer with expander graphs.

Exphormer uses expander graphs to enable efficient sparse global attention on graphs,
achieving linear complexity while maintaining expressive power of full attention.

Key features:
- Sparse attention via expander graphs (O(N) complexity)
- Virtual global nodes for graph-level information
- Combines local MPNN with sparse global attention
- Scalable to very large graphs
- Strong performance on long-range dependencies

References:
    - Exphormer: "Exphormer: Sparse Transformers for Graphs" (ICML 2023)
    - Paper: https://arxiv.org/abs/2303.06147
    - Uses expander graph theory for sparse connectivity

Authors: Shirzad et al. (ICML 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import math
from nexus.core.base import NexusModule


class ExpanderGraphGenerator(NexusModule):
    """Generates expander graph edges for sparse attention.

    Expander graphs are sparse graphs with strong connectivity properties,
    enabling efficient global communication.

    Args:
        num_virt_nodes: Number of virtual global nodes
        expansion_degree: Degree of the expander graph
    """

    def __init__(
        self,
        num_virt_nodes: int = 8,
        expansion_degree: int = 6,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)
        self.num_virt_nodes = num_virt_nodes
        self.expansion_degree = expansion_degree

    def generate_expander_edges(
        self,
        num_nodes: int,
        device: torch.device
    ) -> torch.Tensor:
        """Generate expander graph edges.

        For simplicity, we use a random regular graph as an approximation
        to expander graphs. In practice, could use explicit constructions.

        Args:
            num_nodes: Number of nodes in the graph
            device: Device to create edges on

        Returns:
            Edge indices [2, num_edges] for expander connections
        """
        edges = []

        # Each node connects to expansion_degree random nodes
        for node in range(num_nodes):
            # Random neighbors (simple approximation)
            neighbors = torch.randint(
                0, num_nodes, (self.expansion_degree,),
                device=device
            )
            # Avoid self-loops
            neighbors = neighbors[neighbors != node]

            for neighbor in neighbors:
                edges.append([node, neighbor.item()])

        if len(edges) == 0:
            # Fallback for small graphs
            return torch.zeros(2, 0, dtype=torch.long, device=device)

        edge_index = torch.tensor(edges, dtype=torch.long, device=device).T

        return edge_index


class VirtualGlobalNodes(NexusModule):
    """Virtual global nodes for graph-level information exchange.

    Args:
        num_virt_nodes: Number of virtual nodes
        hidden_dim: Hidden dimension
    """

    def __init__(
        self,
        num_virt_nodes: int = 8,
        hidden_dim: int = 256,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)
        self.num_virt_nodes = num_virt_nodes

        # Initialize virtual node features
        self.virt_node_embed = nn.Parameter(
            torch.randn(num_virt_nodes, hidden_dim) * 0.02
        )

    def get_virtual_nodes(self, batch_size: int) -> torch.Tensor:
        """Get virtual node embeddings for a batch.

        Args:
            batch_size: Batch size (number of graphs)

        Returns:
            Virtual node features [batch_size * num_virt_nodes, hidden_dim]
        """
        virt_nodes = self.virt_node_embed.unsqueeze(0).expand(batch_size, -1, -1)
        virt_nodes = virt_nodes.reshape(-1, self.virt_node_embed.shape[-1])
        return virt_nodes


class SparseAttention(NexusModule):
    """Sparse multi-head attention using expander graph connectivity.

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

        assert hidden_dim % num_heads == 0

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Sparse attention over edges.

        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Sparse attention edges [2, num_edges]

        Returns:
            Updated features [num_nodes, hidden_dim]
        """
        num_nodes = x.shape[0]

        # Compute Q, K, V
        q = self.q_proj(x).view(num_nodes, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(num_nodes, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(num_nodes, self.num_heads, self.head_dim)

        # Compute attention only on edges
        src, dst = edge_index
        q_i = q[dst]  # [num_edges, num_heads, head_dim]
        k_j = k[src]  # [num_edges, num_heads, head_dim]
        v_j = v[src]  # [num_edges, num_heads, head_dim]

        # Attention scores
        attn = (q_i * k_j).sum(dim=-1) * self.scale  # [num_edges, num_heads]
        attn = F.softmax(attn, dim=0)  # Normalize over incoming edges
        attn = self.dropout(attn)

        # Apply attention to values
        attn_v = attn.unsqueeze(-1) * v_j  # [num_edges, num_heads, head_dim]

        # Aggregate
        out = torch.zeros(num_nodes, self.num_heads, self.head_dim, device=x.device, dtype=x.dtype)
        out.index_add_(0, dst, attn_v)

        out = out.reshape(num_nodes, self.hidden_dim)
        out = self.out_proj(out)

        return out


class ExphormerLayer(NexusModule):
    """Single Exphormer layer with local MPNN and sparse global attention.

    Args:
        hidden_dim: Hidden dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
        expansion_degree: Degree of expander graph
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.0,
        expansion_degree: int = 6,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)
        self.hidden_dim = hidden_dim
        self.expansion_degree = expansion_degree

        # Local message passing
        self.local_mpnn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Sparse global attention
        self.sparse_attention = SparseAttention(
            hidden_dim, num_heads, dropout
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

    def local_message_passing(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Local MPNN on graph edges.

        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Graph edges [2, num_edges]

        Returns:
            Updated features [num_nodes, hidden_dim]
        """
        num_nodes = x.shape[0]
        src, dst = edge_index

        # Message computation
        messages = torch.cat([x[src], x[dst]], dim=-1)
        messages = self.local_mpnn(messages)

        # Aggregation
        aggregated = torch.zeros_like(x)
        aggregated.index_add_(0, dst, messages)

        return aggregated

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        expander_edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Exphormer layer forward pass.

        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Original graph edges [2, num_edges]
            expander_edge_index: Expander graph edges [2, num_expander_edges]

        Returns:
            Updated features [num_nodes, hidden_dim]
        """
        # Local MPNN
        x_local = self.local_message_passing(x, edge_index)
        x = x + x_local
        x = self.norm1(x)

        # Sparse global attention via expander graph
        x_global = self.sparse_attention(x, expander_edge_index)
        x = x + x_global
        x = self.norm2(x)

        # Feed-forward
        x_ffn = self.ffn(x)
        x = x + x_ffn
        x = self.norm3(x)

        return x


class Exphormer(NexusModule):
    """Exphormer: Sparse graph transformer using expander graphs.

    Achieves linear complexity while maintaining strong expressiveness
    through sparse expander graph connectivity and virtual global nodes.

    Key advantages:
    - O(N) complexity (linear in number of nodes)
    - Scalable to very large graphs
    - Strong long-range dependencies
    - Combines local and global information flow

    Args:
        in_channels: Input node feature dimension
        hidden_dim: Hidden dimension
        out_channels: Output dimension
        num_layers: Number of Exphormer layers
        num_heads: Number of attention heads
        num_virt_nodes: Number of virtual global nodes
        expansion_degree: Degree of expander graph connections
        dropout: Dropout rate

    Example:
        >>> model = Exphormer(
        ...     in_channels=32,
        ...     hidden_dim=256,
        ...     out_channels=10,
        ...     num_layers=4,
        ...     num_heads=8,
        ...     num_virt_nodes=8
        ... )
        >>> x = torch.randn(1000, 32)  # Large graph with 1000 nodes
        >>> edge_index = torch.randint(0, 1000, (2, 3000))
        >>> output = model(x, edge_index)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 256,
        out_channels: int = 1,
        num_layers: int = 4,
        num_heads: int = 8,
        num_virt_nodes: int = 8,
        expansion_degree: int = 6,
        dropout: float = 0.0,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(config)
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.num_virt_nodes = num_virt_nodes

        # Input embedding
        self.node_embed = nn.Linear(in_channels, hidden_dim)

        # Virtual global nodes
        self.virtual_nodes = VirtualGlobalNodes(
            num_virt_nodes=num_virt_nodes,
            hidden_dim=hidden_dim
        )

        # Expander graph generator
        self.expander_gen = ExpanderGraphGenerator(
            num_virt_nodes=num_virt_nodes,
            expansion_degree=expansion_degree
        )

        # Exphormer layers
        self.layers = nn.ModuleList([
            ExphormerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                expansion_degree=expansion_degree
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

    def add_virtual_nodes(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Add virtual global nodes to the graph.

        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Graph edges [2, num_edges]
            batch: Batch assignment (optional)

        Returns:
            Tuple of:
                - Extended features [num_nodes + num_virt, hidden_dim]
                - Extended edges [2, num_edges + connection_edges]
                - num_virt: Number of virtual nodes added
        """
        num_nodes = x.shape[0]
        num_graphs = 1 if batch is None else batch.max().item() + 1

        # Get virtual node embeddings
        virt_nodes = self.virtual_nodes.get_virtual_nodes(num_graphs)
        num_virt = virt_nodes.shape[0]

        # Concatenate features
        x_extended = torch.cat([x, virt_nodes], dim=0)

        # Connect virtual nodes to real nodes
        # Each virtual node connects to all nodes in its graph
        virt_edges = []
        for graph_id in range(num_graphs):
            if batch is not None:
                node_mask = (batch == graph_id)
                graph_nodes = torch.where(node_mask)[0]
            else:
                graph_nodes = torch.arange(num_nodes, device=x.device)

            virt_node_ids = torch.arange(
                num_nodes + graph_id * self.num_virt_nodes,
                num_nodes + (graph_id + 1) * self.num_virt_nodes,
                device=x.device
            )

            # Bidirectional connections
            for virt_id in virt_node_ids:
                for node_id in graph_nodes:
                    virt_edges.append([virt_id, node_id])
                    virt_edges.append([node_id, virt_id])

        if len(virt_edges) > 0:
            virt_edge_index = torch.tensor(
                virt_edges, dtype=torch.long, device=x.device
            ).T
            edge_index_extended = torch.cat([edge_index, virt_edge_index], dim=1)
        else:
            edge_index_extended = edge_index

        return x_extended, edge_index_extended, num_virt

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through Exphormer.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph edges [2, num_edges]
            batch: Batch assignment [num_nodes] (optional)

        Returns:
            Node or graph embeddings [num_nodes, out_channels] or [batch_size, out_channels]
        """
        num_nodes = x.shape[0]

        # Embed input
        h = self.node_embed(x)

        # Add virtual nodes
        h, edge_index_extended, num_virt = self.add_virtual_nodes(h, edge_index, batch)
        total_nodes = h.shape[0]

        # Generate expander graph edges
        expander_edges = self.expander_gen.generate_expander_edges(
            total_nodes, x.device
        )

        # Apply Exphormer layers
        for layer in self.layers:
            h = layer(h, edge_index_extended, expander_edges)

        # Remove virtual nodes
        h = h[:num_nodes]

        # Output projection
        out = self.output_proj(h)

        # Graph-level pooling if batch provided
        if batch is not None:
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
    'Exphormer',
    'ExphormerLayer',
    'SparseAttention',
    'VirtualGlobalNodes',
    'ExpanderGraphGenerator'
]
