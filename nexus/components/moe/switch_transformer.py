"""
Switch Transformer: Simplified Expert Routing with Top-1 Gating.

Switch Transformer simplifies MoE by routing each token to exactly one expert
(top-1 routing), making it more hardware-efficient while maintaining quality.
Uses capacity factor and expert capacity to handle load balancing.

Key innovations:
- Top-1 routing: Each token goes to exactly one expert
- Expert capacity: Hard limit on tokens per expert
- Simplified routing: No softmax normalization needed
- Load balancing loss: Auxiliary loss encourages uniform distribution

Reference:
    Switch Transformers: Scaling to Trillion Parameter Models with Simple
    and Efficient Sparsity
    https://arxiv.org/abs/2101.03961
    Google Research, 2021
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from nexus.core.base import NexusModule


class SwitchRouter(NexusModule):
    """Top-1 Router for Switch Transformer.

    Routes each token to exactly one expert using a simple argmax.
    Includes capacity factor to limit tokens per expert and handles
    overflow tokens.

    Args:
        dim: Input dimension
        num_experts: Number of experts
        capacity_factor: Expert capacity = (tokens / num_experts) * capacity_factor
        jitter_eps: Add small noise to router logits for load balancing
        ignore_overflow: How to handle tokens exceeding expert capacity
            - 'drop': Drop overflow tokens (they pass through unchanged)
            - 'random': Route overflow to random expert
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        capacity_factor: float = 1.25,
        jitter_eps: float = 0.0,
        ignore_overflow: str = 'drop',
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.jitter_eps = jitter_eps
        self.ignore_overflow = ignore_overflow

        # Router weights
        self.gate = nn.Linear(dim, num_experts, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts using top-1 selection.

        Args:
            hidden_states: Input tensor (batch, seq_len, dim)

        Returns:
            expert_indices: Selected expert for each token (batch, seq_len)
            expert_weights: Gate values for selected experts (batch, seq_len)
            router_logits: Raw router logits (batch, seq_len, num_experts)
            expert_mask: Binary mask indicating which tokens are routed (batch, seq_len)
        """
        batch_size, seq_len, _ = hidden_states.shape
        total_tokens = batch_size * seq_len

        # Compute router logits
        router_logits = self.gate(hidden_states)  # (B, S, E)

        # Add jitter noise during training
        if self.training and self.jitter_eps > 0:
            noise = torch.rand_like(router_logits)
            noise = (noise * 2 - 1) * self.jitter_eps  # Uniform[-eps, eps]
            router_logits = router_logits + noise

        # Top-1 routing (no softmax needed)
        expert_weights, expert_indices = torch.max(router_logits, dim=-1)
        # expert_indices: (B, S)
        # expert_weights: (B, S)

        # Normalize expert weights (optional, for stability)
        expert_weights = F.softmax(expert_weights.unsqueeze(-1), dim=-1).squeeze(-1)

        # Compute expert capacity
        expert_capacity = int(
            (total_tokens / self.num_experts) * self.capacity_factor
        )

        # Track tokens per expert and apply capacity constraint
        expert_mask = torch.ones_like(expert_indices, dtype=torch.bool)

        if expert_capacity > 0:
            # Flatten for capacity checking
            expert_indices_flat = expert_indices.view(-1)

            # Count tokens per expert
            # We need to mask overflow tokens
            position_in_expert = torch.zeros_like(expert_indices_flat)
            for i in range(self.num_experts):
                expert_mask_i = (expert_indices_flat == i)
                # Cumulative count of tokens for this expert
                cumsum = torch.cumsum(expert_mask_i.float(), dim=0)
                position_in_expert[expert_mask_i] = cumsum[expert_mask_i]

            # Mask out tokens exceeding capacity
            overflow_mask = position_in_expert > expert_capacity
            expert_mask = expert_mask.view(-1)
            expert_mask[overflow_mask] = False
            expert_mask = expert_mask.view(batch_size, seq_len)

            if self.ignore_overflow == 'random' and overflow_mask.any():
                # Route overflow tokens to random experts
                num_overflow = overflow_mask.sum()
                random_experts = torch.randint(
                    0, self.num_experts, (num_overflow,),
                    device=expert_indices.device
                )
                expert_indices_flat = expert_indices.view(-1)
                expert_indices_flat[overflow_mask] = random_experts
                expert_indices = expert_indices_flat.view(batch_size, seq_len)

        return expert_indices, expert_weights, router_logits, expert_mask


class SwitchFFN(NexusModule):
    """Switch Transformer Feed-Forward layer with top-1 expert routing.

    Each token is routed to exactly one expert FFN. Tokens that exceed
    expert capacity can be dropped or handled specially.

    Args:
        dim: Model dimension
        num_experts: Number of expert FFNs
        expert_dim: Hidden dimension for each expert (default: 4 * dim)
        capacity_factor: Expert capacity factor
        activation: Activation function ('relu', 'gelu', 'swiglu')
        dropout: Dropout probability
        load_balance_loss_coef: Coefficient for load balancing auxiliary loss
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        expert_dim: Optional[int] = None,
        capacity_factor: float = 1.25,
        activation: str = 'relu',
        dropout: float = 0.0,
        load_balance_loss_coef: float = 0.01,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.expert_dim = expert_dim or (4 * dim)
        self.load_balance_loss_coef = load_balance_loss_coef

        # Router
        self.router = SwitchRouter(
            dim=dim,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
        )

        # Expert FFNs
        from .expert import ExpertLayer
        self.experts = nn.ModuleList([
            ExpertLayer(
                dim=dim,
                hidden_dim=self.expert_dim,
                activation=activation,
                dropout=dropout,
            )
            for _ in range(num_experts)
        ])

    def _compute_load_balance_loss(
        self,
        router_logits: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute load balancing auxiliary loss.

        Encourages uniform distribution of tokens across experts.

        Loss = num_experts * sum_i (f_i * P_i)
        where:
            f_i = fraction of tokens assigned to expert i
            P_i = average router probability for expert i

        Args:
            router_logits: Router logits (batch, seq_len, num_experts)
            expert_indices: Selected expert indices (batch, seq_len)

        Returns:
            Scalar auxiliary loss
        """
        # Router probabilities
        router_probs = F.softmax(router_logits, dim=-1)
        # Average probability per expert across all tokens
        avg_probs = router_probs.mean(dim=(0, 1))  # (num_experts,)

        # Fraction of tokens routed to each expert
        expert_mask = F.one_hot(expert_indices, self.num_experts).float()
        tokens_per_expert = expert_mask.sum(dim=(0, 1))  # (num_experts,)
        total_tokens = expert_indices.numel()
        frac_per_expert = tokens_per_expert / total_tokens

        # Load balance loss
        loss = (frac_per_expert * avg_probs).sum() * self.num_experts
        return loss * self.load_balance_loss_coef

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_aux_loss: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with top-1 expert routing.

        Args:
            hidden_states: Input tensor (batch, seq_len, dim)
            return_aux_loss: Whether to compute auxiliary loss

        Returns:
            output: Expert outputs (batch, seq_len, dim)
            aux_loss: Load balancing loss (if requested and training)
        """
        batch_size, seq_len, dim = hidden_states.shape

        # Route tokens
        expert_indices, expert_weights, router_logits, expert_mask = self.router(
            hidden_states
        )

        # Flatten for efficient expert computation
        hidden_flat = hidden_states.view(-1, dim)  # (B*S, D)
        expert_indices_flat = expert_indices.view(-1)  # (B*S,)
        expert_weights_flat = expert_weights.view(-1)  # (B*S,)
        expert_mask_flat = expert_mask.view(-1)  # (B*S,)

        # Initialize output
        output = torch.zeros_like(hidden_flat)

        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find tokens assigned to this expert
            token_mask = (expert_indices_flat == expert_idx) & expert_mask_flat

            if not token_mask.any():
                continue

            # Get inputs for this expert
            expert_input = hidden_flat[token_mask]

            # Forward through expert
            expert_output = self.experts[expert_idx](expert_input)

            # Weight by gate value
            expert_output = expert_output * expert_weights_flat[token_mask].unsqueeze(-1)

            # Assign to output
            output[token_mask] = expert_output

        # Handle dropped tokens (those that exceeded capacity)
        dropped_mask = ~expert_mask_flat
        if dropped_mask.any():
            # Pass through unchanged (residual connection handles this)
            output[dropped_mask] = hidden_flat[dropped_mask]

        # Reshape output
        output = output.view(batch_size, seq_len, dim)

        # Compute auxiliary loss
        aux_loss = None
        if return_aux_loss and self.training:
            aux_loss = self._compute_load_balance_loss(router_logits, expert_indices)

        return output, aux_loss


class SwitchTransformerLayer(NexusModule):
    """Complete Switch Transformer layer with attention and Switch FFN.

    A standard transformer layer where the FFN is replaced with Switch FFN.

    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        num_experts: Number of experts in Switch FFN
        expert_dim: Expert hidden dimension
        capacity_factor: Expert capacity factor
        attention_type: Type of attention ('standard', 'flash')
        dropout: Dropout probability
        load_balance_loss_coef: Load balancing loss coefficient
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_experts: int,
        expert_dim: Optional[int] = None,
        capacity_factor: float = 1.25,
        attention_type: str = 'standard',
        dropout: float = 0.1,
        load_balance_loss_coef: float = 0.01,
    ):
        super().__init__()

        # Attention sublayer
        if attention_type == 'flash':
            from nexus.components.attention import FlashAttention
            self.attn = FlashAttention(d_model=dim, num_heads=num_heads, dropout=dropout)
        else:
            from nexus.components.attention import MultiHeadSelfAttention
            self.attn = MultiHeadSelfAttention(d_model=dim, num_heads=num_heads, dropout=dropout)

        self.attn_norm = nn.LayerNorm(dim)

        # Switch FFN sublayer
        self.switch_ffn = SwitchFFN(
            dim=dim,
            num_experts=num_experts,
            expert_dim=expert_dim,
            capacity_factor=capacity_factor,
            dropout=dropout,
            load_balance_loss_coef=load_balance_loss_coef,
        )

        self.ffn_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_aux_loss: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through transformer layer with Switch FFN.

        Args:
            hidden_states: Input tensor (batch, seq_len, dim)
            attention_mask: Optional attention mask
            return_aux_loss: Whether to return auxiliary loss

        Returns:
            output: Layer output
            aux_loss: Switch FFN auxiliary loss
        """
        # Attention sublayer (with Pre-LN)
        normed = self.attn_norm(hidden_states)
        attn_out = self.attn(normed, attention_mask=attention_mask)
        if isinstance(attn_out, tuple):
            attn_out = attn_out[0]
        hidden_states = hidden_states + self.dropout(attn_out)

        # Switch FFN sublayer (with Pre-LN)
        normed = self.ffn_norm(hidden_states)
        ffn_out, aux_loss = self.switch_ffn(normed, return_aux_loss=return_aux_loss)
        hidden_states = hidden_states + self.dropout(ffn_out)

        return hidden_states, aux_loss


class SwitchTransformer(NexusModule):
    """Switch Transformer: Transformer with Switch FFN layers.

    A complete transformer model where each FFN is replaced with a
    Switch FFN using top-1 expert routing.

    Args:
        num_layers: Number of transformer layers
        dim: Model dimension
        num_heads: Number of attention heads
        num_experts: Number of experts per layer
        vocab_size: Vocabulary size (for embedding)
        max_seq_len: Maximum sequence length
        expert_dim: Expert hidden dimension
        capacity_factor: Expert capacity factor
        dropout: Dropout probability
        load_balance_loss_coef: Load balancing coefficient

    Example:
        >>> model = SwitchTransformer(
        ...     num_layers=12,
        ...     dim=768,
        ...     num_heads=12,
        ...     num_experts=128,
        ...     vocab_size=50000,
        ... )
        >>> input_ids = torch.randint(0, 50000, (2, 100))
        >>> output, aux_loss = model(input_ids)
    """

    def __init__(
        self,
        num_layers: int,
        dim: int,
        num_heads: int,
        num_experts: int,
        vocab_size: int,
        max_seq_len: int = 2048,
        expert_dim: Optional[int] = None,
        capacity_factor: float = 1.25,
        dropout: float = 0.1,
        load_balance_loss_coef: float = 0.01,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dim = dim

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_seq_len, dim)

        # Switch Transformer layers
        self.layers = nn.ModuleList([
            SwitchTransformerLayer(
                dim=dim,
                num_heads=num_heads,
                num_experts=num_experts,
                expert_dim=expert_dim,
                capacity_factor=capacity_factor,
                dropout=dropout,
                load_balance_loss_coef=load_balance_loss_coef,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.token_embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_aux_loss: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through Switch Transformer.

        Args:
            input_ids: Input token IDs (batch, seq_len)
            attention_mask: Optional attention mask
            return_aux_loss: Whether to return auxiliary loss

        Returns:
            logits: Output logits (batch, seq_len, vocab_size)
            total_aux_loss: Sum of auxiliary losses from all layers
        """
        batch_size, seq_len = input_ids.shape

        # Embeddings
        token_emb = self.token_embedding(input_ids)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)
        hidden_states = token_emb + pos_emb

        # Process through layers
        total_aux_loss = 0.0 if return_aux_loss else None
        for layer in self.layers:
            hidden_states, aux_loss = layer(
                hidden_states,
                attention_mask=attention_mask,
                return_aux_loss=return_aux_loss
            )
            if aux_loss is not None:
                total_aux_loss = total_aux_loss + aux_loss

        # Final norm and LM head
        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits, total_aux_loss
