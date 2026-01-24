"""
Mixture of Experts (MoE) routing and load balancing components.

MoE allows scaling model capacity without proportionally increasing compute
by activating only a subset of experts per token.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Literal
from nexus.core.base import NexusModule


class ExpertRouter(NexusModule):
    """Top-K Expert Router for Mixture of Experts.

    Routes tokens to the top-k experts based on gating scores.
    Supports various gating mechanisms and load balancing.

    Args:
        dim: Input dimension
        num_experts: Total number of experts
        top_k: Number of experts to activate per token
        gating_type: Gating mechanism ('softmax', 'sigmoid', 'noisy_top_k')
        capacity_factor: Expert capacity for load balancing (1.0 = exact top-k)
        jitter_noise: Noise for load balancing during training
        normalize_weights: Whether to normalize expert weights to sum to 1
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int = 2,
        gating_type: Literal['softmax', 'sigmoid', 'noisy_top_k'] = 'softmax',
        capacity_factor: float = 1.25,
        jitter_noise: float = 0.0,
        normalize_weights: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.gating_type = gating_type
        self.capacity_factor = capacity_factor
        self.jitter_noise = jitter_noise
        self.normalize_weights = normalize_weights

        # Gating network
        self.gate = nn.Linear(dim, num_experts, bias=False)

        # For noisy top-k
        if gating_type == 'noisy_top_k':
            self.noise_weight = nn.Linear(dim, num_experts, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        return_aux_loss: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Route tokens to experts.

        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            return_aux_loss: Whether to compute auxiliary load balancing loss

        Returns:
            expert_weights: Weights for selected experts (batch, seq, top_k)
            expert_indices: Indices of selected experts (batch, seq, top_k)
            aux_loss: Auxiliary loss for load balancing (if requested)
        """
        batch_size, seq_len, _ = x.shape

        # Compute gating logits
        logits = self.gate(x)  # (batch, seq, num_experts)

        # Apply jitter noise during training
        if self.training and self.jitter_noise > 0:
            logits = logits + torch.randn_like(logits) * self.jitter_noise

        # Apply gating mechanism
        if self.gating_type == 'noisy_top_k':
            # Add learned noise for load balancing
            noise = torch.randn_like(logits) * F.softplus(self.noise_weight(x))
            logits = logits + noise

        # Get top-k experts
        top_k_logits, expert_indices = torch.topk(logits, self.top_k, dim=-1)

        # Compute expert weights
        if self.gating_type == 'sigmoid':
            expert_weights = torch.sigmoid(top_k_logits)
        else:
            expert_weights = F.softmax(top_k_logits, dim=-1)

        # Normalize weights if requested
        if self.normalize_weights:
            expert_weights = expert_weights / (expert_weights.sum(dim=-1, keepdim=True) + 1e-6)

        # Compute auxiliary loss
        aux_loss = None
        if return_aux_loss and self.training:
            aux_loss = self._compute_aux_loss(logits, expert_indices)

        return expert_weights, expert_indices, aux_loss

    def _compute_aux_loss(
        self,
        logits: torch.Tensor,
        expert_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute auxiliary load balancing loss.

        Uses the standard formulation: loss = α * Σ(f_i * P_i)
        where f_i is the fraction of tokens routed to expert i
        and P_i is the average probability for expert i.
        """
        # Compute routing probabilities
        probs = F.softmax(logits, dim=-1)  # (batch, seq, num_experts)

        # Flatten for statistics
        probs_flat = probs.view(-1, self.num_experts)
        indices_flat = expert_indices.view(-1, self.top_k)

        # Compute fraction of tokens routed to each expert
        expert_mask = F.one_hot(indices_flat, self.num_experts).sum(dim=1).float()
        tokens_per_expert = expert_mask.sum(dim=0)
        total_tokens = tokens_per_expert.sum()
        f = tokens_per_expert / (total_tokens + 1e-6)

        # Compute average probability for each expert
        P = probs_flat.mean(dim=0)

        # Auxiliary loss (encourages uniform distribution)
        aux_loss = (f * P).sum() * self.num_experts

        return aux_loss


class LoadBalancingLoss(NexusModule):
    """Auxiliary loss functions for MoE load balancing.

    Implements various load balancing strategies:
    - Standard: f_i * P_i based loss (Switch Transformer)
    - Z-loss: Regularizes router logits to prevent divergence
    - Expert-choice: Entropy-based loss for expert selection

    Args:
        num_experts: Number of experts
        loss_type: Type of loss ('standard', 'z_loss', 'entropy')
        loss_weight: Weight for the auxiliary loss
    """

    def __init__(
        self,
        num_experts: int,
        loss_type: Literal['standard', 'z_loss', 'entropy'] = 'standard',
        loss_weight: float = 0.01
    ):
        super().__init__()
        self.num_experts = num_experts
        self.loss_type = loss_type
        self.loss_weight = loss_weight

    def forward(
        self,
        router_logits: torch.Tensor,
        expert_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute load balancing loss.

        Args:
            router_logits: Raw logits from router (batch, seq, num_experts)
            expert_indices: Selected expert indices (batch, seq, top_k)

        Returns:
            Scaled auxiliary loss
        """
        if self.loss_type == 'standard':
            return self._standard_loss(router_logits, expert_indices) * self.loss_weight
        elif self.loss_type == 'z_loss':
            return self._z_loss(router_logits) * self.loss_weight
        elif self.loss_type == 'entropy':
            return self._entropy_loss(router_logits) * self.loss_weight
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def _standard_loss(
        self,
        logits: torch.Tensor,
        expert_indices: torch.Tensor
    ) -> torch.Tensor:
        """Standard load balancing loss from Switch Transformer."""
        probs = F.softmax(logits, dim=-1)
        probs_flat = probs.view(-1, self.num_experts)

        # Count tokens per expert
        indices_flat = expert_indices.view(-1, expert_indices.shape[-1])
        mask = F.one_hot(indices_flat, self.num_experts).sum(dim=1).float()
        tokens_per_expert = mask.sum(dim=0)

        # Normalize
        f = tokens_per_expert / (tokens_per_expert.sum() + 1e-6)
        P = probs_flat.mean(dim=0)

        return (f * P).sum() * self.num_experts

    def _z_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Router Z-loss to prevent logit divergence."""
        # Penalize large router logits
        return torch.logsumexp(logits, dim=-1).pow(2).mean()

    def _entropy_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Entropy-based loss for diverse expert selection."""
        probs = F.softmax(logits, dim=-1)
        # Negative entropy (maximize entropy for uniform distribution)
        entropy = -(probs * torch.log(probs + 1e-6)).sum(dim=-1)
        return -entropy.mean()


class LossFreeBalancing(NexusModule):
    """Loss-Free Load Balancing using bias terms.

    DeepSeek V3's approach that adjusts routing without auxiliary loss
    by maintaining per-expert bias terms updated based on usage statistics.

    Args:
        num_experts: Number of experts
        update_rate: Rate for exponential moving average of usage stats
        balance_factor: Strength of balancing adjustment
    """

    def __init__(
        self,
        num_experts: int,
        update_rate: float = 0.01,
        balance_factor: float = 0.1
    ):
        super().__init__()
        self.num_experts = num_experts
        self.update_rate = update_rate
        self.balance_factor = balance_factor

        # Running statistics of expert usage
        self.register_buffer('expert_usage', torch.zeros(num_experts))
        self.register_buffer('total_tokens', torch.tensor(0.0))

        # Bias terms for routing adjustment
        self.bias = nn.Parameter(torch.zeros(num_experts))

    def forward(
        self,
        router_logits: torch.Tensor,
        expert_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply bias adjustment and update statistics.

        Args:
            router_logits: Raw router logits
            expert_indices: Selected expert indices

        Returns:
            Adjusted router logits
        """
        # Apply learned bias
        adjusted_logits = router_logits + self.bias

        # Update usage statistics during training
        if self.training:
            with torch.no_grad():
                # Count tokens per expert
                batch_size = expert_indices.numel() // expert_indices.shape[-1]
                indices_flat = expert_indices.view(-1)
                usage = torch.bincount(indices_flat, minlength=self.num_experts).float()

                # Exponential moving average update
                self.expert_usage = (
                    (1 - self.update_rate) * self.expert_usage +
                    self.update_rate * usage
                )
                self.total_tokens = (
                    (1 - self.update_rate) * self.total_tokens +
                    self.update_rate * batch_size
                )

                # Compute target (uniform) vs actual usage
                target_usage = self.total_tokens / self.num_experts
                usage_diff = target_usage - self.expert_usage

                # Update bias to encourage underused experts
                # (This is a simplified version; actual implementation may vary)
                self.bias.data += self.balance_factor * usage_diff / (self.total_tokens + 1)

        return adjusted_logits


class ExpertChoiceRouter(NexusModule):
    """Expert-Choice routing where experts select tokens.

    Instead of tokens selecting experts, each expert selects its top-k tokens.
    This naturally achieves load balancing.

    Reference: https://arxiv.org/abs/2202.09368

    Args:
        dim: Input dimension
        num_experts: Number of experts
        capacity_factor: Tokens per expert = (total_tokens / num_experts) * capacity_factor
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        capacity_factor: float = 1.0
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor

        self.gate = nn.Linear(dim, num_experts, bias=False)

    def forward(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route using expert-choice mechanism.

        Args:
            x: Input of shape (batch, seq_len, dim)

        Returns:
            expert_weights: Weights (num_experts, capacity, batch*seq)
            token_indices: Which tokens selected (num_experts, capacity)
            combine_weights: Weights for combining expert outputs
        """
        batch_size, seq_len, _ = x.shape
        total_tokens = batch_size * seq_len
        capacity = int(total_tokens / self.num_experts * self.capacity_factor)

        # Compute scores
        scores = self.gate(x)  # (batch, seq, num_experts)
        scores = scores.view(total_tokens, self.num_experts)  # (tokens, experts)

        # Each expert selects its top tokens
        # Transpose so experts are first dimension
        scores_t = scores.t()  # (experts, tokens)

        # Top-k tokens per expert
        expert_weights, token_indices = torch.topk(scores_t, capacity, dim=-1)
        expert_weights = F.softmax(expert_weights, dim=-1)

        return expert_weights, token_indices, scores
