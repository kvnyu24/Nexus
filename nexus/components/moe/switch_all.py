"""
SwitchAll: Fully Mixture-of-Experts Attention + FFN.

SwitchAll extends the MoE paradigm to both attention and feed-forward layers,
routing tokens through expert attention heads and expert FFNs. This provides
maximum sparsity and capacity scaling.

Key features:
- Expert routing for both attention and FFN sublayers
- Separate routers for attention and FFN
- Independent expert selection per sublayer
- Accumulated auxiliary losses for load balancing

This combines:
- SwitchHead (MoE attention) from switch_attention.py
- Switch FFN from switch_transformer.py

Reference:
    SwitchHead: Accelerating Transformers with Mixture-of-Experts Attention Heads
    https://arxiv.org/abs/2312.07987
    NeurIPS 2024
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from nexus.core.base import NexusModule


class SwitchAllLayer(NexusModule):
    """Transformer layer with MoE in both attention and FFN.

    Both the attention and feed-forward sublayers use expert routing,
    making the entire layer sparse and highly scalable.

    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        num_attn_experts: Number of attention experts
        num_ffn_experts: Number of FFN experts
        top_k_attn: Number of attention experts to activate per token
        top_k_ffn: Number of FFN experts to activate per token
        head_dim: Dimension per attention head
        ffn_dim: FFN hidden dimension
        dropout: Dropout probability
        attn_aux_loss_coef: Attention load balancing coefficient
        ffn_aux_loss_coef: FFN load balancing coefficient
        use_flash_attn: Whether to use flash attention implementation
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_attn_experts: int = 4,
        num_ffn_experts: int = 8,
        top_k_attn: int = 1,
        top_k_ffn: int = 2,
        head_dim: Optional[int] = None,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        attn_aux_loss_coef: float = 0.01,
        ffn_aux_loss_coef: float = 0.01,
        use_flash_attn: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim or (dim // num_heads)
        self.ffn_dim = ffn_dim or (dim * 4)

        # MoE Attention (SwitchHead)
        from nexus.components.attention.switch_attention import SwitchHeadAttention
        self.moe_attn = SwitchHeadAttention(
            d_model=dim,
            num_heads=num_heads,
            num_experts=num_attn_experts,
            top_k=top_k_attn,
            head_dim=self.head_dim,
            dropout=dropout,
            aux_loss_coeff=attn_aux_loss_coef,
        )

        self.attn_norm = nn.LayerNorm(dim)

        # MoE FFN (Switch FFN)
        from nexus.components.moe.switch_transformer import SwitchFFN
        self.moe_ffn = SwitchFFN(
            dim=dim,
            num_experts=num_ffn_experts,
            expert_dim=self.ffn_dim,
            capacity_factor=1.25,
            dropout=dropout,
            load_balance_loss_coef=ffn_aux_loss_coef,
        )

        self.ffn_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_aux_loss: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass through SwitchAll layer.

        Args:
            hidden_states: Input tensor (batch, seq_len, dim)
            attention_mask: Optional attention mask
            position_embeddings: Optional (cos, sin) for RoPE
            return_aux_loss: Whether to return auxiliary losses

        Returns:
            output: Layer output (batch, seq_len, dim)
            attn_aux_loss: Attention load balancing loss
            ffn_aux_loss: FFN load balancing loss
        """
        # MoE Attention sublayer (Pre-LN)
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)

        attn_output, attn_weights, attn_aux_loss = self.moe_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            output_attentions=False,
        )

        hidden_states = residual + self.dropout(attn_output)

        # MoE FFN sublayer (Pre-LN)
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)

        ffn_output, ffn_aux_loss = self.moe_ffn(
            hidden_states,
            return_aux_loss=return_aux_loss,
        )

        hidden_states = residual + self.dropout(ffn_output)

        if not return_aux_loss:
            attn_aux_loss = None
            ffn_aux_loss = None

        return hidden_states, attn_aux_loss, ffn_aux_loss


class SwitchAll(NexusModule):
    """Complete SwitchAll Transformer with full MoE in all layers.

    A transformer where both attention and FFN use expert routing in every layer,
    maximizing sparsity and scaling capacity.

    Args:
        num_layers: Number of transformer layers
        dim: Model dimension
        num_heads: Number of attention heads
        num_attn_experts: Number of attention experts per layer
        num_ffn_experts: Number of FFN experts per layer
        top_k_attn: Attention experts to activate per token
        top_k_ffn: FFN experts to activate per token
        vocab_size: Vocabulary size
        max_seq_len: Maximum sequence length
        head_dim: Dimension per attention head
        ffn_dim: FFN hidden dimension
        dropout: Dropout probability
        attn_aux_loss_coef: Attention load balancing coefficient
        ffn_aux_loss_coef: FFN load balancing coefficient

    Example:
        >>> model = SwitchAll(
        ...     num_layers=12,
        ...     dim=768,
        ...     num_heads=12,
        ...     num_attn_experts=4,
        ...     num_ffn_experts=128,
        ...     vocab_size=50000,
        ... )
        >>> input_ids = torch.randint(0, 50000, (2, 100))
        >>> logits, attn_loss, ffn_loss = model(input_ids)
    """

    def __init__(
        self,
        num_layers: int,
        dim: int,
        num_heads: int,
        num_attn_experts: int = 4,
        num_ffn_experts: int = 128,
        top_k_attn: int = 1,
        top_k_ffn: int = 2,
        vocab_size: int = 50000,
        max_seq_len: int = 2048,
        head_dim: Optional[int] = None,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        attn_aux_loss_coef: float = 0.01,
        ffn_aux_loss_coef: float = 0.01,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dim = dim
        self.vocab_size = vocab_size

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_seq_len, dim)
        self.dropout = nn.Dropout(dropout)

        # SwitchAll layers
        self.layers = nn.ModuleList([
            SwitchAllLayer(
                dim=dim,
                num_heads=num_heads,
                num_attn_experts=num_attn_experts,
                num_ffn_experts=num_ffn_experts,
                top_k_attn=top_k_attn,
                top_k_ffn=top_k_ffn,
                head_dim=head_dim,
                ffn_dim=ffn_dim,
                dropout=dropout,
                attn_aux_loss_coef=attn_aux_loss_coef,
                ffn_aux_loss_coef=ffn_aux_loss_coef,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_aux_loss: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass through SwitchAll model.

        Args:
            input_ids: Input token IDs (batch, seq_len)
            attention_mask: Optional attention mask
            return_aux_loss: Whether to return auxiliary losses

        Returns:
            logits: Output logits (batch, seq_len, vocab_size)
            total_attn_aux_loss: Sum of attention auxiliary losses
            total_ffn_aux_loss: Sum of FFN auxiliary losses
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Embeddings
        token_emb = self.token_embedding(input_ids)
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)
        hidden_states = self.dropout(token_emb + pos_emb)

        # Accumulate auxiliary losses
        total_attn_aux_loss = torch.tensor(0.0, device=device) if return_aux_loss else None
        total_ffn_aux_loss = torch.tensor(0.0, device=device) if return_aux_loss else None

        # Process through layers
        for layer in self.layers:
            hidden_states, attn_aux_loss, ffn_aux_loss = layer(
                hidden_states,
                attention_mask=attention_mask,
                return_aux_loss=return_aux_loss,
            )

            if return_aux_loss:
                if attn_aux_loss is not None:
                    total_attn_aux_loss = total_attn_aux_loss + attn_aux_loss
                if ffn_aux_loss is not None:
                    total_ffn_aux_loss = total_ffn_aux_loss + ffn_aux_loss

        # Final norm and projection
        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits, total_attn_aux_loss, total_ffn_aux_loss

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Get the number of parameters in the model.

        Args:
            non_embedding: Whether to exclude embedding parameters

        Returns:
            Number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
            n_params -= self.position_embedding.weight.numel()
        return n_params

    def estimate_flops_per_token(self) -> int:
        """
        Estimate FLOPs per token for this sparse model.

        SwitchAll is much more efficient than dense transformers due to
        sparse expert routing in both attention and FFN.

        Returns:
            Approximate FLOPs per token
        """
        # This is a rough estimate
        # Attention: num_heads * top_k_attn experts
        # FFN: top_k_ffn experts
        # Both are sparse so we only count activated parameters

        # For a full implementation, you'd count:
        # - Q,K,V projections (shared or expert-specific)
        # - Attention computation
        # - Output projection (expert-specific)
        # - FFN computation (expert-specific)

        # Placeholder estimate
        return self.dim * self.dim * 6 * self.num_layers


class SwitchAllConfig:
    """Configuration for SwitchAll model.

    Provides default configurations for different model sizes.
    """

    @staticmethod
    def base(vocab_size: int = 50000) -> dict:
        """Base configuration (similar to BERT-base)."""
        return {
            'num_layers': 12,
            'dim': 768,
            'num_heads': 12,
            'num_attn_experts': 4,
            'num_ffn_experts': 64,
            'top_k_attn': 1,
            'top_k_ffn': 2,
            'vocab_size': vocab_size,
            'dropout': 0.1,
        }

    @staticmethod
    def large(vocab_size: int = 50000) -> dict:
        """Large configuration (similar to BERT-large)."""
        return {
            'num_layers': 24,
            'dim': 1024,
            'num_heads': 16,
            'num_attn_experts': 8,
            'num_ffn_experts': 128,
            'top_k_attn': 2,
            'top_k_ffn': 4,
            'vocab_size': vocab_size,
            'dropout': 0.1,
        }

    @staticmethod
    def xlarge(vocab_size: int = 50000) -> dict:
        """Extra-large configuration for research."""
        return {
            'num_layers': 32,
            'dim': 2048,
            'num_heads': 32,
            'num_attn_experts': 16,
            'num_ffn_experts': 256,
            'top_k_attn': 2,
            'top_k_ffn': 4,
            'vocab_size': vocab_size,
            'dropout': 0.1,
        }
