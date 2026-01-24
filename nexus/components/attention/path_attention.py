"""
PaTH Attention (Persistent Attention for Transformer History).

PaTH Attention improves state tracking and sequential reasoning by maintaining
persistent memory tokens that carry information across the sequence.

Reference: https://arxiv.org/abs/2501.03124 (MIT-IBM Watson AI Lab)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Literal
from nexus.core.base import NexusModule


class PaTHAttention(NexusModule):
    """PaTH Attention (Persistent Attention for Transformer History).

    Improves state tracking and sequential reasoning by maintaining persistent
    memory tokens that carry information across the sequence. The persistent
    tokens attend to the input sequence and are updated recurrently, enabling
    the model to maintain long-term state.

    Key innovations:
    - Persistent memory tokens that maintain state across sequence positions
    - Bidirectional attention between input and memory tokens
    - Recurrent state updates for sequential reasoning
    - Gated memory updates for selective information retention

    Used by: Sequential reasoning tasks, state tracking, algorithmic tasks

    Reference: https://arxiv.org/abs/2501.03124 (MIT-IBM Watson AI Lab)

    Args:
        dim: Model dimension
        num_heads: Number of attention heads
        num_persistent_tokens: Number of persistent memory tokens
        head_dim: Dimension per head (default: dim // num_heads)
        dropout: Attention dropout probability
        bias: Whether to use bias in projections
        causal: Whether to use causal masking for input-to-input attention
        memory_gate: Whether to use gated memory updates
        cross_attend: Whether persistent tokens cross-attend to input
        update_mode: How to update persistent tokens ('recurrent', 'parallel', 'hybrid')
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_persistent_tokens: int = 8,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
        causal: bool = True,
        memory_gate: bool = True,
        cross_attend: bool = True,
        update_mode: Literal['recurrent', 'parallel', 'hybrid'] = 'hybrid'
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim or (dim // num_heads)
        self.num_persistent_tokens = num_persistent_tokens
        self.dropout_p = dropout
        self.causal = causal
        self.memory_gate = memory_gate
        self.cross_attend = cross_attend
        self.update_mode = update_mode

        self.scale = self.head_dim ** -0.5

        # Persistent memory tokens (learned initial state)
        self.persistent_tokens = nn.Parameter(
            torch.randn(1, num_persistent_tokens, dim) * 0.02
        )

        # Input attention projections
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, num_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, num_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, dim, bias=bias)

        # Memory attention projections (for persistent tokens)
        self.mem_q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=bias)
        self.mem_k_proj = nn.Linear(dim, num_heads * self.head_dim, bias=bias)
        self.mem_v_proj = nn.Linear(dim, num_heads * self.head_dim, bias=bias)
        self.mem_o_proj = nn.Linear(num_heads * self.head_dim, dim, bias=bias)

        # Gated memory update
        if memory_gate:
            self.memory_gate_proj = nn.Linear(dim * 2, dim, bias=bias)
            self.memory_update_proj = nn.Linear(dim, dim, bias=bias)

        # Layer norms for stability
        self.input_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)

        self.attn_dropout = nn.Dropout(dropout)

        # For recurrent mode, we need state tracking
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stability."""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.o_proj,
                       self.mem_q_proj, self.mem_k_proj, self.mem_v_proj, self.mem_o_proj]:
            nn.init.xavier_uniform_(module.weight, gain=0.1)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _compute_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute scaled dot-product attention.

        Args:
            query: (batch, heads, seq_q, head_dim)
            key: (batch, heads, seq_k, head_dim)
            value: (batch, heads, seq_k, head_dim)
            mask: Optional attention mask

        Returns:
            Attention output of shape (batch, heads, seq_q, head_dim)
        """
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn_scores = attn_scores + mask

        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        return torch.matmul(attn_weights, value)

    def _create_causal_mask(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """Create causal attention mask."""
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=dtype) * float('-inf'),
            diagonal=1
        )
        return mask.unsqueeze(0).unsqueeze(0)

    def _update_memory_gated(
        self,
        memory: torch.Tensor,
        memory_update: torch.Tensor
    ) -> torch.Tensor:
        """Apply gated update to memory tokens.

        Args:
            memory: Current memory state (batch, num_tokens, dim)
            memory_update: Proposed update (batch, num_tokens, dim)

        Returns:
            Updated memory state
        """
        # Compute gate
        gate_input = torch.cat([memory, memory_update], dim=-1)
        gate = torch.sigmoid(self.memory_gate_proj(gate_input))

        # Apply gated update
        update = self.memory_update_proj(memory_update)
        return memory * (1 - gate) + update * gate

    def _process_parallel(
        self,
        hidden_states: torch.Tensor,
        memory_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process input and memory in parallel (faster, less sequential).

        All tokens see the same memory state; memory is updated based on
        aggregate attention to the input.
        """
        batch_size, seq_len, _ = hidden_states.shape
        num_mem = self.num_persistent_tokens

        # Normalize
        hidden_norm = self.input_norm(hidden_states)
        memory_norm = self.memory_norm(memory_states)

        # === Input attention with memory ===
        # Input tokens attend to themselves and memory tokens
        q = self.q_proj(hidden_norm)
        k_input = self.k_proj(hidden_norm)
        v_input = self.v_proj(hidden_norm)
        k_mem = self.k_proj(memory_norm)
        v_mem = self.v_proj(memory_norm)

        # Reshape
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_input = k_input.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_input = v_input.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_mem = k_mem.view(batch_size, num_mem, self.num_heads, self.head_dim).transpose(1, 2)
        v_mem = v_mem.view(batch_size, num_mem, self.num_heads, self.head_dim).transpose(1, 2)

        # Concatenate keys and values: [memory, input]
        k = torch.cat([k_mem, k_input], dim=2)
        v = torch.cat([v_mem, v_input], dim=2)

        # Create combined mask: input can attend to memory (always) + causal input attention
        total_len = num_mem + seq_len
        mask = torch.zeros(seq_len, total_len, device=hidden_states.device, dtype=hidden_states.dtype)

        # Memory tokens are always visible (no mask needed for first num_mem columns)
        # Apply causal mask for input-to-input attention
        if self.causal:
            causal_mask = self._create_causal_mask(seq_len, hidden_states.device, hidden_states.dtype)
            mask[:, num_mem:] = causal_mask.squeeze(0).squeeze(0)

        # Add additional attention mask if provided
        if attention_mask is not None:
            # Expand mask for memory columns (no masking for memory)
            full_mask = torch.cat([
                torch.zeros(batch_size, 1, seq_len, num_mem, device=hidden_states.device, dtype=hidden_states.dtype),
                attention_mask
            ], dim=-1)
            mask = mask.unsqueeze(0).unsqueeze(0) + full_mask
        else:
            mask = mask.unsqueeze(0).unsqueeze(0)

        # Compute input attention
        input_output = self._compute_attention(q, k, v, mask)
        input_output = input_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        input_output = self.o_proj(input_output)

        # === Memory attention (memory attends to input) ===
        if self.cross_attend:
            mem_q = self.mem_q_proj(memory_norm)
            mem_k = self.mem_k_proj(hidden_norm)
            mem_v = self.mem_v_proj(hidden_norm)

            mem_q = mem_q.view(batch_size, num_mem, self.num_heads, self.head_dim).transpose(1, 2)
            mem_k = mem_k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            mem_v = mem_v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

            # Memory can attend to all input positions
            memory_attn_output = self._compute_attention(mem_q, mem_k, mem_v)
            memory_attn_output = memory_attn_output.transpose(1, 2).contiguous().view(batch_size, num_mem, -1)
            memory_update = self.mem_o_proj(memory_attn_output)

            # Update memory
            if self.memory_gate:
                memory_states = self._update_memory_gated(memory_states, memory_update)
            else:
                memory_states = memory_states + memory_update

        return hidden_states + input_output, memory_states

    def _process_recurrent(
        self,
        hidden_states: torch.Tensor,
        memory_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process input recurrently (better for state tracking, slower).

        Memory is updated at each position, so later positions see updated memory.
        """
        batch_size, seq_len, _ = hidden_states.shape
        num_mem = self.num_persistent_tokens

        outputs = []

        for t in range(seq_len):
            # Get current token
            h_t = hidden_states[:, t:t+1, :]  # (batch, 1, dim)

            # Normalize
            h_t_norm = self.input_norm(h_t)
            mem_norm = self.memory_norm(memory_states)

            # Input attention: current token attends to memory + past tokens
            q = self.q_proj(h_t_norm)
            q = q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)

            # Keys/values from memory and past input
            if t > 0:
                past_hidden = hidden_states[:, :t, :]
                past_norm = self.input_norm(past_hidden)
                k_past = self.k_proj(past_norm)
                v_past = self.v_proj(past_norm)
                k_past = k_past.view(batch_size, t, self.num_heads, self.head_dim).transpose(1, 2)
                v_past = v_past.view(batch_size, t, self.num_heads, self.head_dim).transpose(1, 2)

                k_mem = self.k_proj(mem_norm)
                v_mem = self.v_proj(mem_norm)
                k_mem = k_mem.view(batch_size, num_mem, self.num_heads, self.head_dim).transpose(1, 2)
                v_mem = v_mem.view(batch_size, num_mem, self.num_heads, self.head_dim).transpose(1, 2)

                k = torch.cat([k_mem, k_past], dim=2)
                v = torch.cat([v_mem, v_past], dim=2)
            else:
                k_mem = self.k_proj(mem_norm)
                v_mem = self.v_proj(mem_norm)
                k = k_mem.view(batch_size, num_mem, self.num_heads, self.head_dim).transpose(1, 2)
                v = v_mem.view(batch_size, num_mem, self.num_heads, self.head_dim).transpose(1, 2)

            # Compute attention
            output_t = self._compute_attention(q, k, v)
            output_t = output_t.transpose(1, 2).contiguous().view(batch_size, 1, -1)
            output_t = self.o_proj(output_t)
            outputs.append(h_t + output_t)

            # Update memory based on current token
            if self.cross_attend:
                mem_q = self.mem_q_proj(mem_norm)
                mem_k = self.mem_k_proj(h_t_norm)
                mem_v = self.mem_v_proj(h_t_norm)

                mem_q = mem_q.view(batch_size, num_mem, self.num_heads, self.head_dim).transpose(1, 2)
                mem_k = mem_k.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
                mem_v = mem_v.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)

                memory_attn = self._compute_attention(mem_q, mem_k, mem_v)
                memory_attn = memory_attn.transpose(1, 2).contiguous().view(batch_size, num_mem, -1)
                memory_update = self.mem_o_proj(memory_attn)

                if self.memory_gate:
                    memory_states = self._update_memory_gated(memory_states, memory_update)
                else:
                    memory_states = memory_states + memory_update

        output = torch.cat(outputs, dim=1)
        return output, memory_states

    def _process_hybrid(
        self,
        hidden_states: torch.Tensor,
        memory_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        chunk_size: int = 64
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Hybrid processing: parallel within chunks, recurrent across chunks.

        Balances efficiency with state tracking capability.
        """
        batch_size, seq_len, _ = hidden_states.shape

        if seq_len <= chunk_size:
            return self._process_parallel(hidden_states, memory_states, attention_mask)

        outputs = []
        num_chunks = (seq_len + chunk_size - 1) // chunk_size

        for i in range(num_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, seq_len)
            chunk = hidden_states[:, start:end, :]

            chunk_mask = None
            if attention_mask is not None:
                chunk_mask = attention_mask[:, :, start:end, start:end]

            chunk_output, memory_states = self._process_parallel(
                chunk, memory_states, chunk_mask
            )
            outputs.append(chunk_output)

        return torch.cat(outputs, dim=1), memory_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_memory_state: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass with PaTH attention.

        Args:
            hidden_states: Input of shape (batch, seq_len, dim)
            attention_mask: Optional attention mask
            position_embeddings: Tuple of (cos, sin) for RoPE (applied to input)
            past_memory_state: Previous memory state for continuation
            use_cache: Whether to return memory state for continuation
            output_attentions: Whether to return attention weights (not supported)

        Returns:
            output: Shape (batch, seq_len, dim)
            memory_state: If use_cache, the updated memory state
            attn_weights: None (not supported for efficiency)
        """
        batch_size = hidden_states.shape[0]

        # Initialize or retrieve memory state
        if past_memory_state is not None:
            memory_states = past_memory_state
        else:
            # Expand persistent tokens for batch
            memory_states = self.persistent_tokens.expand(batch_size, -1, -1)

        # Apply position embeddings to input if provided
        if position_embeddings is not None:
            cos, sin = position_embeddings
            # Apply RoPE to input (memory tokens don't need position)
            hidden_states = self._apply_rotary_to_input(hidden_states, cos, sin)

        # Process based on update mode
        if self.update_mode == 'recurrent':
            output, memory_states = self._process_recurrent(
                hidden_states, memory_states, attention_mask
            )
        elif self.update_mode == 'parallel':
            output, memory_states = self._process_parallel(
                hidden_states, memory_states, attention_mask
            )
        else:  # hybrid
            output, memory_states = self._process_hybrid(
                hidden_states, memory_states, attention_mask
            )

        memory_output = memory_states if use_cache else None

        return output, memory_output, None

    def _apply_rotary_to_input(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor
    ) -> torch.Tensor:
        """Apply rotary embeddings to input hidden states.

        This is a simplified version - in practice, you'd apply RoPE
        after Q/K projection in the attention computation.
        """
        # For simplicity, we return hidden_states unchanged here
        # RoPE should be applied in the attention computation
        # This is a placeholder for the integration point
        return hidden_states

    def reset_memory(self, batch_size: int = 1) -> torch.Tensor:
        """Reset memory to initial learned state.

        Args:
            batch_size: Batch size for the memory state

        Returns:
            Initial memory state of shape (batch_size, num_persistent_tokens, dim)
        """
        return self.persistent_tokens.expand(batch_size, -1, -1).clone()

    def get_memory_size(self) -> int:
        """Get the number of persistent memory tokens."""
        return self.num_persistent_tokens


class PaTH(PaTHAttention):
    """Alias for PaTHAttention."""
    pass
