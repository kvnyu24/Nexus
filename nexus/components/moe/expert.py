"""
Expert layer implementations for Mixture of Experts.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from nexus.core.base import NexusModule


class ExpertLayer(NexusModule):
    """Single expert MLP layer.

    A standard feed-forward network used as an expert in MoE layers.

    Args:
        dim: Input/output dimension
        hidden_dim: Hidden dimension
        activation: Activation type ('relu', 'gelu', 'swiglu')
        dropout: Dropout probability
        bias: Whether to use bias
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        activation: str = 'swiglu',
        dropout: float = 0.0,
        bias: bool = False
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.activation_type = activation

        if activation == 'swiglu':
            # SwiGLU uses two up projections
            self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
            self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
            self.w3 = nn.Linear(dim, hidden_dim, bias=bias)
            self.activation = F.silu
        else:
            self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
            self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
            self.w3 = None
            if activation == 'relu':
                self.activation = F.relu
            elif activation == 'gelu':
                self.activation = F.gelu
            else:
                self.activation = F.silu

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through expert."""
        if self.w3 is not None:
            # SwiGLU: gate * up
            return self.dropout(self.w2(self.activation(self.w1(x)) * self.w3(x)))
        else:
            return self.dropout(self.w2(self.activation(self.w1(x))))


class SharedExpert(NexusModule):
    """Shared expert that's always activated.

    Used in DeepSeek V3 to provide a baseline computation that all
    tokens go through, complementing the routed experts.

    Args:
        dim: Input/output dimension
        hidden_dim: Hidden dimension
        num_shared: Number of shared experts
        activation: Activation type
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_shared: int = 1,
        activation: str = 'swiglu'
    ):
        super().__init__()
        self.num_shared = num_shared

        self.experts = nn.ModuleList([
            ExpertLayer(dim, hidden_dim, activation)
            for _ in range(num_shared)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through all shared experts and average."""
        outputs = [expert(x) for expert in self.experts]
        return sum(outputs) / self.num_shared


class MoELayer(NexusModule):
    """Complete Mixture of Experts layer.

    Combines routing with expert computation for a full MoE block.

    Args:
        dim: Model dimension
        num_experts: Number of routed experts
        top_k: Number of experts per token
        expert_hidden_dim: Hidden dimension for each expert
        shared_expert: Whether to include shared expert(s)
        num_shared_experts: Number of shared experts
        activation: Expert activation type
        dropout: Dropout probability
        router_jitter: Jitter noise for router
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int = 2,
        expert_hidden_dim: Optional[int] = None,
        shared_expert: bool = False,
        num_shared_experts: int = 1,
        activation: str = 'swiglu',
        dropout: float = 0.0,
        router_jitter: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_hidden_dim = expert_hidden_dim or (dim * 4)

        # Router
        from .router import ExpertRouter
        self.router = ExpertRouter(
            dim=dim,
            num_experts=num_experts,
            top_k=top_k,
            jitter_noise=router_jitter
        )

        # Routed experts
        self.experts = nn.ModuleList([
            ExpertLayer(dim, self.expert_hidden_dim, activation, dropout)
            for _ in range(num_experts)
        ])

        # Optional shared expert
        self.shared_expert = None
        if shared_expert:
            self.shared_expert = SharedExpert(
                dim, self.expert_hidden_dim, num_shared_experts, activation
            )

    def forward(
        self,
        x: torch.Tensor,
        return_aux_loss: bool = True
    ) -> tuple:
        """
        Forward pass through MoE layer.

        Args:
            x: Input of shape (batch, seq_len, dim)
            return_aux_loss: Whether to return auxiliary loss

        Returns:
            output: MoE output of shape (batch, seq_len, dim)
            aux_loss: Auxiliary load balancing loss
        """
        batch_size, seq_len, dim = x.shape

        # Get routing weights and indices
        expert_weights, expert_indices, aux_loss = self.router(x, return_aux_loss)

        # Flatten for easier indexing
        x_flat = x.view(-1, dim)  # (batch*seq, dim)
        expert_weights_flat = expert_weights.view(-1, self.top_k)
        expert_indices_flat = expert_indices.view(-1, self.top_k)

        # Compute output
        output = torch.zeros_like(x_flat)

        for k in range(self.top_k):
            # Get indices and weights for k-th expert choice
            indices_k = expert_indices_flat[:, k]  # (batch*seq,)
            weights_k = expert_weights_flat[:, k]  # (batch*seq,)

            # Process tokens through their assigned experts
            for expert_idx in range(self.num_experts):
                # Find tokens routed to this expert
                mask = indices_k == expert_idx
                if not mask.any():
                    continue

                # Get tokens for this expert
                expert_input = x_flat[mask]

                # Process through expert
                expert_output = self.experts[expert_idx](expert_input)

                # Weight and accumulate
                output[mask] += weights_k[mask].unsqueeze(-1) * expert_output

        # Reshape output
        output = output.view(batch_size, seq_len, dim)

        # Add shared expert contribution
        if self.shared_expert is not None:
            output = output + self.shared_expert(x)

        return output, aux_loss


class SparseMoE(MoELayer):
    """Alias for MoELayer with sparse routing."""
    pass
