"""
Mixture-of-Depths (MoD) for conditional computation in transformers.

Mixture-of-Depths allows a transformer to dynamically allocate
computation per token: a learned router decides which tokens receive
the full transformer block computation and which tokens simply pass
through via the residual connection.  This yields significant
throughput improvements with minimal quality degradation, since many
tokens (especially in later layers) do not require deep processing.

Key ideas:
- A lightweight router produces a scalar score per token.
- Only the top-k tokens (determined by ``capacity_ratio``) are
  processed by the transformer sub-layer.
- Remaining tokens bypass the sub-layer via the residual skip.
- The router is trained end-to-end with a straight-through estimator
  or auxiliary load-balancing loss.

Reference:
    Mixture-of-Depths: Dynamically allocating compute in
    transformer-based language models
    https://arxiv.org/abs/2404.02258

Usage example::

    block = MoDBlock(
        transformer_block=TransformerBlock(dim=512, ...),
        dim=512,
        capacity_ratio=0.5,
    )
    output = block(hidden_states)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from nexus.core.base import NexusModule


class MoDRouter(NexusModule):
    """Learned router for Mixture-of-Depths token selection.

    Produces a scalar routing score for every token.  During the
    forward pass the top-k tokens (based on ``capacity_ratio``) are
    selected for full computation; the rest are skipped.

    The router is a single linear projection followed by a sigmoid
    (or softmax across the sequence) that produces a gating weight
    between 0 and 1.

    Args:
        dim: Hidden dimension of the input tokens.
        capacity_ratio: Fraction of tokens that receive full
            computation.  For example, ``0.5`` means half the tokens
            are processed by the sub-layer.
        jitter_noise: Standard deviation of Gaussian noise added to
            router logits during training to encourage exploration.
        straight_through: Use straight-through estimator for the
            top-k selection to allow gradient flow.
    """

    def __init__(
        self,
        dim: int,
        capacity_ratio: float = 0.5,
        jitter_noise: float = 0.01,
        straight_through: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.capacity_ratio = capacity_ratio
        self.jitter_noise = jitter_noise
        self.straight_through = straight_through

        # Single-layer routing projection: hidden -> scalar
        self.router_proj = nn.Linear(dim, 1, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute routing decisions for each token.

        Args:
            hidden_states: Input tensor ``(batch, seq_len, dim)``.
            attention_mask: Optional boolean mask ``(batch, seq_len)``
                where ``True`` indicates a valid (non-padding) token.
                Padding tokens are never routed.

        Returns:
            routing_weights: Per-token gating weights
                ``(batch, seq_len, 1)``.  Selected tokens have weight
                close to 1; skipped tokens have weight 0.
            routing_mask: Boolean mask ``(batch, seq_len)`` with
                ``True`` for tokens selected for computation.
            aux_info: Dictionary with auxiliary information for
                logging and loss computation (e.g. router logits,
                fraction of tokens selected).
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Router logits: (batch, seq_len, 1)
        router_logits = self.router_proj(hidden_states)

        # Add jitter noise during training
        if self.training and self.jitter_noise > 0.0:
            noise = torch.randn_like(router_logits) * self.jitter_noise
            router_logits = router_logits + noise

        # Sigmoid gating
        router_scores = torch.sigmoid(router_logits)  # (batch, seq_len, 1)

        # Determine capacity (number of tokens to route)
        if attention_mask is not None:
            valid_counts = attention_mask.sum(dim=1)  # (batch,)
            capacities = (valid_counts.float() * self.capacity_ratio).long()
            capacities = capacities.clamp(min=1)
        else:
            capacity = max(1, int(seq_len * self.capacity_ratio))
            capacities = torch.full(
                (batch_size,), capacity, dtype=torch.long, device=hidden_states.device
            )

        # Top-k selection per batch item
        scores_flat = router_scores.squeeze(-1)  # (batch, seq_len)

        # Mask out padding tokens with -inf before top-k
        if attention_mask is not None:
            scores_flat = scores_flat.masked_fill(~attention_mask, -float("inf"))

        routing_mask = torch.zeros(
            batch_size, seq_len, dtype=torch.bool, device=hidden_states.device
        )

        for b in range(batch_size):
            k = capacities[b].item()
            _, topk_indices = torch.topk(scores_flat[b], k, dim=-1)
            routing_mask[b, topk_indices] = True

        # Build gating weights: selected tokens get their router score,
        # others get zero.
        routing_weights = router_scores.squeeze(-1) * routing_mask.float()
        routing_weights = routing_weights.unsqueeze(-1)  # (batch, seq_len, 1)

        # Straight-through estimator: in the backward pass, gradients
        # flow through all tokens as if they were selected.
        if self.straight_through and self.training:
            routing_weights = (
                routing_weights
                + (router_scores - router_scores.detach())
            )

        # Auxiliary info for logging / load balancing
        aux_info = {
            "router_logits": router_logits.squeeze(-1),
            "fraction_selected": routing_mask.float().mean(),
            "router_scores_mean": router_scores.mean(),
        }

        return routing_weights, routing_mask, aux_info


class MoDBlock(NexusModule):
    """Mixture-of-Depths transformer block wrapper.

    Wraps an arbitrary transformer sub-layer (attention or FFN) with
    Mixture-of-Depths routing.  Only the selected (top-k) tokens are
    processed by the sub-layer; the remaining tokens pass through
    unchanged via the residual connection.

    This is fully compatible with any ``nn.Module`` that takes
    ``(batch, seq_len, dim)`` input and returns the same shape.

    Args:
        transformer_block: The transformer sub-layer to wrap.
        dim: Hidden dimension.
        capacity_ratio: Fraction of tokens to route through the
            sub-layer (default 0.5).
        jitter_noise: Jitter noise for the router.
        straight_through: Use straight-through gradient estimator.
        use_residual: Whether the wrapped block already includes a
            residual connection.  If ``False`` (default), MoDBlock
            adds the residual.  If ``True``, the block's output
            is used directly for selected tokens.
    """

    def __init__(
        self,
        transformer_block: nn.Module,
        dim: int,
        capacity_ratio: float = 0.5,
        jitter_noise: float = 0.01,
        straight_through: bool = True,
        use_residual: bool = False,
    ):
        super().__init__()
        self.block = transformer_block
        self.dim = dim
        self.use_residual = use_residual

        self.router = MoDRouter(
            dim=dim,
            capacity_ratio=capacity_ratio,
            jitter_noise=jitter_noise,
            straight_through=straight_through,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **block_kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with conditional computation.

        Args:
            hidden_states: Input tensor ``(batch, seq_len, dim)``.
            attention_mask: Optional boolean mask ``(batch, seq_len)``.
            **block_kwargs: Extra keyword arguments forwarded to the
                wrapped transformer block (e.g. ``position_ids``,
                ``kv_cache``).

        Returns:
            output: Output tensor ``(batch, seq_len, dim)``.
            aux_info: Routing auxiliary information.
        """
        residual = hidden_states

        # Get routing decisions
        routing_weights, routing_mask, aux_info = self.router(
            hidden_states, attention_mask
        )

        # Gather selected tokens for efficient computation
        batch_size, seq_len, dim = hidden_states.shape
        selected_count = routing_mask.sum(dim=1)  # (batch,)

        # Process selected tokens through the block.
        # For simplicity and generality we process the full tensor
        # and mask the output.  For large-scale deployment the
        # selected tokens would be gathered / scattered for true
        # compute savings.
        block_output = self.block(hidden_states, **block_kwargs)

        # Handle blocks that return tuples (e.g. attention with KV)
        if isinstance(block_output, tuple):
            block_output = block_output[0]

        # Apply routing weights: selected tokens get block output,
        # skipped tokens get the residual
        if self.use_residual:
            # Block already includes residual
            output = routing_weights * block_output + (1.0 - routing_weights) * residual
        else:
            # Add residual for skipped tokens; selected tokens get
            # residual + weighted block output
            output = residual + routing_weights * block_output

        # Track how many FLOPs were "saved"
        aux_info["compute_fraction"] = routing_mask.float().mean()
        aux_info["tokens_computed"] = selected_count.float().mean()
        aux_info["tokens_skipped"] = (
            (seq_len - selected_count).float().mean()
        )

        return output, aux_info

    @staticmethod
    def compute_load_balancing_loss(
        router_logits: torch.Tensor,
        routing_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute auxiliary load-balancing loss.

        Encourages the router to distribute computation evenly across
        tokens rather than always selecting the same positions.

        The loss is the dot product of the mean routing probability
        and the fraction of tokens routed at each position (analogous
        to the Switch Transformer auxiliary loss).

        Args:
            router_logits: Raw router logits ``(batch, seq_len)``.
            routing_mask: Boolean mask ``(batch, seq_len)``.

        Returns:
            Scalar load-balancing loss.
        """
        router_probs = torch.sigmoid(router_logits)

        # Fraction of tokens routed (per position across batch)
        f = routing_mask.float().mean(dim=0)  # (seq_len,)
        # Mean routing probability (per position across batch)
        p = router_probs.mean(dim=0)  # (seq_len,)

        # Dot product as load-balancing signal
        num_positions = f.shape[0]
        return (f * p).sum() * num_positions
