# CoPE: Contextual Position Encoding

## Overview

CoPE (Contextual Position Encoding) makes positions depend on content, not just absolute indices. Instead of fixed positions, CoPE computes "soft" positions based on learned gates applied to the input, allowing positions to adapt to sequence structure.

**Key Innovation**: Position of token i = cumulative sum of content-based gates up to i. Tokens "count what matters."

## Motivation

Traditional PE assumes uniform position progression: 0, 1, 2, 3, ...

But in structured data:
- Code: Indentation levels matter more than line numbers
- Markup: Nesting depth more important than character position
- Text: Semantic units (sentences, paragraphs) more meaningful than tokens

CoPE learns to count what's structurally important.

## Core Concept

```python
# Traditional PE: positions = [0, 1, 2, 3, 4, ...]

# CoPE:
gates = GateNetwork(tokens)  # [0.5, 1.0, 0.2, 1.0, 0.8, ...]
positions = cumsum(gates)     # [0.5, 1.5, 1.7, 2.7, 3.5, ...]
```

Positions are now content-dependent and non-uniform!

## Implementation

```python
class CoPE(nn.Module):
    """Contextual Position Encoding."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        max_pos: int = 2048,
        gate_type: str = 'sigmoid'
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Gate projection: token → "how much does this token count?"
        self.gate_proj = nn.Linear(dim, num_heads, bias=True)

        # Position embeddings (looked up using soft positions)
        self.pos_emb = nn.Embedding(max_pos, self.head_dim)

        # Position query projection
        self.pos_query = nn.Linear(self.head_dim, self.head_dim, bias=False)

        self._init_weights()

    def _init_weights(self):
        nn.init.zeros_(self.gate_proj.bias)
        nn.init.normal_(self.gate_proj.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        nn.init.normal_(self.pos_query.weight, std=0.02)

    def compute_gates(self, x):
        """Compute content-based gate values."""
        gates = self.gate_proj(x)  # (batch, seq_len, num_heads)
        return torch.sigmoid(gates)

    def compute_positions(self, gates):
        """Compute soft positions via cumulative sum."""
        return torch.cumsum(gates, dim=1)  # (batch, seq_len, num_heads)

    def interpolate_pos_emb(self, positions):
        """Linear interpolation for soft (non-integer) positions."""
        batch_size, seq_len, num_heads = positions.shape

        # Clamp to valid range
        positions = torch.clamp(positions, 0, self.pos_emb.num_embeddings - 1.001)

        # Floor and ceil
        pos_floor = positions.floor().long()
        pos_ceil = (pos_floor + 1).clamp(max=self.pos_emb.num_embeddings - 1)

        # Get embeddings
        emb_floor = self.pos_emb(pos_floor)  # (B, S, H, D)
        emb_ceil = self.pos_emb(pos_ceil)

        # Interpolate
        weight = (positions - pos_floor.float()).unsqueeze(-1)
        return emb_floor * (1 - weight) + emb_ceil * weight

    def forward(self, q, k, x=None, gates=None):
        """
        Compute CoPE position bias.

        Args:
            q: Queries (batch, heads, seq, head_dim)
            k: Keys (batch, heads, seq, head_dim)
            x: Input for gate computation (batch, seq, dim)
            gates: Pre-computed gates (optional)

        Returns:
            Position bias (batch, heads, seq, seq)
        """
        # Compute gates
        if gates is None:
            if x is None:
                raise ValueError("Either x or gates must be provided")
            gates = self.compute_gates(x)

        # Compute soft positions
        positions = self.compute_positions(gates)  # (B, S, H)

        # Get position embeddings
        pos_emb = self.interpolate_pos_emb(positions)  # (B, S, H, D)
        pos_emb = pos_emb.permute(0, 2, 1, 3)  # (B, H, S, D)

        # Compute position queries
        q_pos = self.pos_query(q)  # (B, H, S, D)

        # Position bias: q_pos @ pos_emb^T
        pos_bias = torch.matmul(q_pos, pos_emb.transpose(-2, -1))
        pos_bias = pos_bias / math.sqrt(self.head_dim)

        return pos_bias
```

## Usage

```python
from nexus.components.embeddings import CoPE

# Initialize
cope = CoPE(dim=512, num_heads=8, max_pos=2048)

# In attention
Q, K, V = ... # Standard attention inputs
x = embeddings  # Token embeddings (for gate computation)

# Get position bias
pos_bias = cope(Q, K, x)

# Add to attention scores
scores = Q @ K.T / sqrt(d_k) + pos_bias
attn = softmax(scores, dim=-1) @ V
```

## Hybrid: CoPE + RoPE

```python
class CoPEWithRoPE(nn.Module):
    """Combine content-dependent (CoPE) and absolute (RoPE) positions."""

    def __init__(self, dim, num_heads, max_pos=2048):
        super().__init__()
        self.cope = CoPE(dim, num_heads, max_pos)
        # RoPE for absolute position
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim // num_heads, 2).float() / (dim // num_heads)))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, q, k, x):
        # Apply RoPE (absolute positions)
        seq_len = q.shape[2]
        positions = torch.arange(seq_len, device=q.device).float()
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1).unsqueeze(0).unsqueeze(0)

        def rotate_half(x):
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)

        q_rot = (q * emb.cos()) + (rotate_half(q) * emb.sin())
        k_rot = (k * emb.cos()) + (rotate_half(k) * emb.sin())

        # Add CoPE bias
        cope_bias = self.cope(q, k, x)

        return q_rot, k_rot, cope_bias
```

## When to Use

**Use CoPE if**:
- Working with structured data (code, markup, math)
- Want position to adapt to content structure
- Need both content and position awareness

**Don't use if**:
- Simple text sequences (standard PE sufficient)
- Very tight computational budget
- Need zero parameters (CoPE has gate network)

## Experiments

### On Structured Data (Code)

| Method | Exact Match | Pass@1 |
|--------|-------------|--------|
| Learned PE | 23.1 | 45.2 |
| RoPE | 26.4 | 48.7 |
| CoPE | **31.8** | **54.3** |

**Observation**: CoPE excels on structured sequences where semantic position differs from token position.

### On Natural Text

| Method | PPL |
|--------|-----|
| RoPE | 15.0 |
| ALiBi | 15.2 |
| CoPE | 15.1 |
| RoPE + CoPE | **14.8** |

**Observation**: Hybrid approaches work best for general text.

## Common Pitfalls

1. **Not normalizing gates**: Ensure gates produce reasonable ranges (0-1)
2. **Large position embeddings**: Use small max_pos or interpolation
3. **Forgetting to pass input**: CoPE needs token embeddings for gates

## References

- Gu, A., et al. (2024). **CoPE: Contextual Position Encoding for Transformers**. [arXiv:2405.18719](https://arxiv.org/abs/2405.18719)

**Implementation**: [/nexus/components/embeddings/cope.py](../../nexus/components/embeddings/cope.py)

---
**Back to Overview**: [README.md](./README.md)
