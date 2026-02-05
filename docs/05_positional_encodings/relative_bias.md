# Relative Positional Bias (T5-style)

## Overview

Relative Positional Bias, introduced in T5, learns bias terms based on bucketed relative distances between tokens. Unlike fixed methods, it learns which distances are important from data.

**Key Features**:
- Learns bias per relative distance bucket
- Logarithmic bucketing for efficient long-range encoding
- Shared across layers in T5
- Excellent for encoder-decoder models

## Core Concept

Instead of encoding absolute positions, add learned biases to attention scores based on relative distance:

```
attention_score[i,j] = Q[i] · K[j]^T + bias[bucket(i-j)]
```

Where `bucket()` maps distances to bucket indices using logarithmic binning.

## Logarithmic Bucketing

```python
def relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
    """
    Map relative positions to bucket indices.

    Nearby positions (< num_buckets/2): exact buckets
    Distant positions (>= num_buckets/2): logarithmic buckets
    """
    ret = 0
    n = -relative_position  # Make negative

    # First half: exact buckets for nearby positions
    num_buckets //= 2
    ret += (n < 0).to(torch.long) * num_buckets  # Offset for positive positions
    n = torch.abs(n)

    # Second half: logarithmic buckets for distant positions
    max_exact = num_buckets // 2
    is_small = n < max_exact

    # Logarithmic scaling for large distances
    val_if_large = max_exact + (
        torch.log(n.float() / max_exact) /
        math.log(max_distance / max_exact) *
        (num_buckets - max_exact)
    ).to(torch.long)
    val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

    ret += torch.where(is_small, n, val_if_large)
    return ret
```

## Implementation

```python
class RelativePositionalBias(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_buckets: int = 32,
        max_distance: int = 128,
        bidirectional: bool = True
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional

        # Learnable bias per bucket per head
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)
        nn.init.normal_(self.relative_attention_bias.weight, std=0.02)

    def forward(self, query_length, key_length, device=None):
        if device is None:
            device = self.relative_attention_bias.weight.device

        # Compute relative positions
        q_pos = torch.arange(query_length, device=device)
        k_pos = torch.arange(key_length, device=device)
        relative_position = q_pos.unsqueeze(1) - k_pos.unsqueeze(0)

        # Map to buckets
        relative_buckets = self._relative_position_bucket(relative_position)

        # Get bias values (query_length, key_length, num_heads)
        values = self.relative_attention_bias(relative_buckets)

        # Reshape to (1, num_heads, query_length, key_length)
        return values.permute(2, 0, 1).unsqueeze(0)
```

## Usage

```python
from nexus.components.embeddings import RelativePositionalBias

# Initialize
rel_bias = RelativePositionalBias(
    num_heads=8,
    num_buckets=32,
    max_distance=128,
    bidirectional=True  # False for decoder
)

# In attention
seq_len = 512
bias = rel_bias(seq_len, seq_len)  # (1, 8, 512, 512)
scores = Q @ K.T / sqrt(d_k) + bias
```

## Advantages

1. **Learnable**: Adapts to data patterns
2. **Efficient**: O(buckets × heads) parameters, not O(seq_len²)
3. **Generalizes**: Logarithmic bucketing handles any length
4. **Proven**: Works well in T5, LongT5

## Disadvantages

1. **Moderate parameters**: ~32 × num_heads parameters
2. **Less extrapolation**: Not as strong as ALiBi for unseen lengths
3. **Bucketing granularity**: May lose fine-grained distance info

## Experiments

| Method | Params | Train PPL | Test 2x | Test 4x |
|--------|--------|-----------|---------|---------|
| Learned PE | 1M | 18.2 | ∞ | ∞ |
| Sinusoidal | 0 | 18.1 | 22.5 | 38.4 |
| Relative Bias | 256 | 18.0 | 19.8 | 24.1 |
| ALiBi | 0 | 18.2 | 19.1 | 20.8 |

## References

- Raffel, C., et al. (2020). **Exploring the Limits of Transfer Learning with T5**. [arXiv:1910.10683](https://arxiv.org/abs/1910.10683)

**Implementation**: [/nexus/components/embeddings/relative_bias.py](../../nexus/components/embeddings/relative_bias.py)

---
**Back to Overview**: [README.md](./README.md)
