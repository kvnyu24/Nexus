"""
DeepSeek MoE: Shared + Routed Experts with Fine-Grained Segmentation.

DeepSeek-V2/V3 introduces an enhanced MoE architecture with:
1. Shared experts that all tokens pass through
2. Routed experts with fine-grained segmentation
3. Loss-free load balancing via bias adjustment
4. Device-limited expert placement for efficiency

Key innovations:
- Shared experts provide a stable baseline computation
- Fine-grained expert segmentation reduces parameter redundancy
- Multi-token prediction for training efficiency
- Auxiliary-loss-free load balancing

Reference:
    DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model
    https://arxiv.org/abs/2405.04434
    DeepSeek-V3 Technical Report (2024)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from nexus.core.base import NexusModule


class FineGrainedExpert(NexusModule):
    """Fine-grained expert with smaller intermediate dimension.

    Instead of one large expert, uses multiple smaller "sub-experts"
    to reduce parameter redundancy while maintaining capacity.

    Args:
        dim: Input/output dimension
        expert_dim: Expert intermediate dimension
        num_segments: Number of fine-grained segments
        activation: Activation function
    """

    def __init__(
        self,
        dim: int,
        expert_dim: int,
        num_segments: int = 1,
        activation: str = 'swiglu',
    ):
        super().__init__()
        self.dim = dim
        self.expert_dim = expert_dim
        self.num_segments = num_segments
        self.segment_dim = expert_dim // num_segments

        # Fine-grained expert segments
        if activation == 'swiglu':
            self.gate_proj = nn.Linear(dim, expert_dim, bias=False)
            self.up_proj = nn.Linear(dim, expert_dim, bias=False)
            self.down_proj = nn.Linear(expert_dim, dim, bias=False)
            self.act_fn = F.silu
        else:
            self.w1 = nn.Linear(dim, expert_dim, bias=False)
            self.w2 = nn.Linear(expert_dim, dim, bias=False)
            self.act_fn = F.gelu if activation == 'gelu' else F.relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through fine-grained expert.

        Args:
            x: Input tensor (*, dim)

        Returns:
            Output tensor (*, dim)
        """
        if hasattr(self, 'gate_proj'):
            # SwiGLU variant
            gate = self.act_fn(self.gate_proj(x))
            up = self.up_proj(x)
            return self.down_proj(gate * up)
        else:
            # Standard FFN
            return self.w2(self.act_fn(self.w1(x)))


class DeepSeekMoELayer(NexusModule):
    """DeepSeek MoE Layer with shared + routed experts.

    Combines:
    - Shared experts: Always activated, provide stable baseline
    - Routed experts: Sparsely activated via top-k routing
    - Fine-grained segmentation: Reduced parameter redundancy

    The final output is:
        output = shared_output + routed_output

    Args:
        dim: Model dimension
        num_shared_experts: Number of shared experts (always active)
        num_routed_experts: Number of routed experts (sparse)
        top_k_experts: Number of routed experts to activate per token
        expert_dim: Intermediate dimension for experts
        shared_expert_dim: Intermediate dimension for shared experts
        num_segments: Fine-grained segmentation count
        activation: Activation function type
        router_aux_loss_coef: Auxiliary loss coefficient (0 for loss-free)
        use_expert_choice: Whether to use expert-choice routing
    """

    def __init__(
        self,
        dim: int,
        num_shared_experts: int = 2,
        num_routed_experts: int = 64,
        top_k_experts: int = 6,
        expert_dim: Optional[int] = None,
        shared_expert_dim: Optional[int] = None,
        num_segments: int = 4,
        activation: str = 'swiglu',
        router_aux_loss_coef: float = 0.001,
        use_expert_choice: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_shared_experts = num_shared_experts
        self.num_routed_experts = num_routed_experts
        self.top_k_experts = top_k_experts
        self.expert_dim = expert_dim or (dim * 4)
        self.shared_expert_dim = shared_expert_dim or (dim * 4)
        self.num_segments = num_segments
        self.router_aux_loss_coef = router_aux_loss_coef
        self.use_expert_choice = use_expert_choice

        # Shared experts (always active)
        self.shared_experts = nn.ModuleList([
            FineGrainedExpert(
                dim=dim,
                expert_dim=self.shared_expert_dim,
                num_segments=1,  # Shared experts don't use segmentation
                activation=activation,
            )
            for _ in range(num_shared_experts)
        ])

        # Routed experts (sparsely activated)
        self.routed_experts = nn.ModuleList([
            FineGrainedExpert(
                dim=dim,
                expert_dim=self.expert_dim,
                num_segments=num_segments,
                activation=activation,
            )
            for _ in range(num_routed_experts)
        ])

        # Router for routed experts
        self.gate = nn.Linear(dim, num_routed_experts, bias=False)

        # Loss-free balancing (DeepSeek-V3 approach)
        if router_aux_loss_coef == 0:
            # Learnable bias for load balancing without auxiliary loss
            self.balance_bias = nn.Parameter(torch.zeros(num_routed_experts))
        else:
            self.register_parameter('balance_bias', None)

    def _compute_routing_weights(
        self,
        hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute routing weights for experts.

        Args:
            hidden_states: Input tensor (batch, seq_len, dim)

        Returns:
            routing_weights: Normalized weights (batch, seq_len, top_k)
            selected_experts: Expert indices (batch, seq_len, top_k)
            router_logits: Raw logits for aux loss (batch, seq_len, num_experts)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Compute router logits
        router_logits = self.gate(hidden_states)  # (B, S, E)

        # Apply loss-free balancing bias if enabled
        if self.balance_bias is not None:
            router_logits = router_logits + self.balance_bias

        # Top-k routing
        routing_weights, selected_experts = torch.topk(
            router_logits, self.top_k_experts, dim=-1
        )

        # Normalize routing weights
        routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float32).to(
            hidden_states.dtype
        )

        return routing_weights, selected_experts, router_logits

    def _compute_aux_loss(
        self,
        router_logits: torch.Tensor,
        selected_experts: torch.Tensor
    ) -> torch.Tensor:
        """Compute auxiliary load balancing loss.

        Args:
            router_logits: Raw router logits (batch, seq_len, num_experts)
            selected_experts: Selected expert indices (batch, seq_len, top_k)

        Returns:
            Auxiliary loss scalar
        """
        if self.router_aux_loss_coef == 0:
            return torch.tensor(0.0, device=router_logits.device)

        # Standard load balancing loss
        router_probs = F.softmax(router_logits, dim=-1)
        router_probs_mean = router_probs.mean(dim=(0, 1))  # (num_experts,)

        # Token assignment per expert
        expert_mask = F.one_hot(
            selected_experts, num_classes=self.num_routed_experts
        ).float()
        expert_mask = expert_mask.sum(dim=2)  # (B, S, E)
        tokens_per_expert = expert_mask.sum(dim=(0, 1))  # (E,)
        tokens_per_expert = tokens_per_expert / tokens_per_expert.sum()

        # Load balancing loss
        aux_loss = (router_probs_mean * tokens_per_expert).sum()
        aux_loss = aux_loss * self.num_routed_experts * self.router_aux_loss_coef

        return aux_loss

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_aux_loss: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through DeepSeek MoE layer.

        Args:
            hidden_states: Input tensor (batch, seq_len, dim)
            return_aux_loss: Whether to compute auxiliary loss

        Returns:
            output: Combined output from shared + routed experts
            aux_loss: Auxiliary loss (if return_aux_loss=True and training)
        """
        batch_size, seq_len, dim = hidden_states.shape

        # 1. Shared experts (all tokens)
        shared_output = torch.zeros_like(hidden_states)
        for expert in self.shared_experts:
            shared_output = shared_output + expert(hidden_states)
        # Average shared expert outputs
        shared_output = shared_output / self.num_shared_experts

        # 2. Routed experts (sparse)
        routing_weights, selected_experts, router_logits = self._compute_routing_weights(
            hidden_states
        )

        # Flatten for efficient routing
        hidden_states_flat = hidden_states.view(-1, dim)  # (B*S, D)
        routing_weights_flat = routing_weights.view(-1, self.top_k_experts)
        selected_experts_flat = selected_experts.view(-1, self.top_k_experts)

        routed_output = torch.zeros_like(hidden_states_flat)

        # Process each top-k choice
        for k in range(self.top_k_experts):
            expert_indices = selected_experts_flat[:, k]  # (B*S,)
            expert_weights = routing_weights_flat[:, k]  # (B*S,)

            # Process tokens for each expert
            for expert_idx in range(self.num_routed_experts):
                # Find tokens routed to this expert
                token_mask = expert_indices == expert_idx
                if not token_mask.any():
                    continue

                # Get tokens for this expert
                expert_input = hidden_states_flat[token_mask]

                # Process through expert
                expert_output = self.routed_experts[expert_idx](expert_input)

                # Accumulate weighted output
                routed_output[token_mask] += (
                    expert_weights[token_mask].unsqueeze(-1) * expert_output
                )

        # Reshape routed output
        routed_output = routed_output.view(batch_size, seq_len, dim)

        # 3. Combine shared + routed
        final_output = shared_output + routed_output

        # 4. Compute auxiliary loss
        aux_loss = None
        if return_aux_loss and self.training:
            aux_loss = self._compute_aux_loss(router_logits, selected_experts)

        return final_output, aux_loss


class DeepSeekMoE(NexusModule):
    """Complete DeepSeek MoE module with optional normalization and residual.

    Wraps DeepSeekMoELayer with layer norm and residual connection
    for easy integration into transformer blocks.

    Args:
        dim: Model dimension
        num_shared_experts: Number of shared experts
        num_routed_experts: Number of routed experts
        top_k_experts: Experts to activate per token
        expert_dim: Expert intermediate dimension
        shared_expert_dim: Shared expert intermediate dimension
        num_segments: Fine-grained segmentation
        activation: Activation type
        norm_type: Normalization type ('layer', 'rms', None)
        use_residual: Whether to add residual connection
        router_aux_loss_coef: Auxiliary loss coefficient

    Example:
        >>> moe = DeepSeekMoE(
        ...     dim=2048,
        ...     num_shared_experts=2,
        ...     num_routed_experts=160,
        ...     top_k_experts=6,
        ...     num_segments=4,
        ... )
        >>> x = torch.randn(2, 100, 2048)
        >>> output, aux_loss = moe(x)
    """

    def __init__(
        self,
        dim: int,
        num_shared_experts: int = 2,
        num_routed_experts: int = 64,
        top_k_experts: int = 6,
        expert_dim: Optional[int] = None,
        shared_expert_dim: Optional[int] = None,
        num_segments: int = 4,
        activation: str = 'swiglu',
        norm_type: Optional[str] = 'rms',
        use_residual: bool = True,
        router_aux_loss_coef: float = 0.001,
    ):
        super().__init__()
        self.use_residual = use_residual

        # Pre-normalization
        if norm_type == 'layer':
            self.norm = nn.LayerNorm(dim)
        elif norm_type == 'rms':
            from nexus.components.normalization import RMSNorm
            self.norm = RMSNorm(dim)
        else:
            self.norm = None

        # MoE layer
        self.moe_layer = DeepSeekMoELayer(
            dim=dim,
            num_shared_experts=num_shared_experts,
            num_routed_experts=num_routed_experts,
            top_k_experts=top_k_experts,
            expert_dim=expert_dim,
            shared_expert_dim=shared_expert_dim,
            num_segments=num_segments,
            activation=activation,
            router_aux_loss_coef=router_aux_loss_coef,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_aux_loss: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with optional normalization and residual.

        Args:
            hidden_states: Input tensor
            return_aux_loss: Whether to return auxiliary loss

        Returns:
            output: MoE output
            aux_loss: Auxiliary loss if requested
        """
        residual = hidden_states

        # Pre-norm
        if self.norm is not None:
            hidden_states = self.norm(hidden_states)

        # MoE
        output, aux_loss = self.moe_layer(hidden_states, return_aux_loss)

        # Residual
        if self.use_residual:
            output = output + residual

        return output, aux_loss
