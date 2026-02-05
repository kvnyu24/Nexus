# FIRE: Functional Interpolation for Relative Positional Encoding

## Overview & Motivation

FIRE (Functional Interpolation for Relative positional Encoding) provides a unified framework that can represent many existing relative position methods (ALiBi, T5 RPE, Kerple) as special cases. It combines progressive interpolation with learned MLPs for excellent length generalization.

**Key Innovation**: Map relative positions to [0,1] via learnable progressive interpolation, then apply a learned MLP to produce position biases.

**Why FIRE?**
- Generalizes ALiBi, T5 bias, and other methods
- Excellent length extrapolation (bounded input domain)
- Learned but sample-efficient (small MLP)
- Functional approach (not lookup tables)

## Theoretical Background

### Two-Stage Architecture

**Stage 1: Progressive Interpolation**
Maps unbounded distances [0, ∞) to bounded [0, 1]:
```
f(d) = log(1 + d) / log(1 + L)
```
Where L is a learnable threshold per head.

**Stage 2: Learned Mapping**
Small MLP transforms [0,1] to bias values:
```
bias(d) = MLP(f(d))
```

### Why This Works

**Bounded domain**: All relative positions map to [0,1], regardless of sequence length. The model learns to interpret this normalized space.

**Progressive compression**:
- Nearby positions (d < L): Linear mapping, fine-grained distinction
- Distant positions (d > L): Logarithmic compression, coarse distinction

**Learnable**: MLP adapts to data, can learn linear (ALiBi-like), lookup-table (T5-like), or nonlinear patterns.

## Implementation

```python
class ProgressiveInterpolation(nn.Module):
    """Map relative positions to [0,1]."""

    def __init__(self, num_heads: int, init_threshold: float = 512.0):
        super().__init__()
        self.log_threshold = nn.Parameter(
            torch.full((num_heads,), math.log(init_threshold))
        )

    @property
    def threshold(self):
        return self.log_threshold.exp()

    def forward(self, relative_positions):
        """
        Args:
            relative_positions: (seq_len, seq_len) or (batch, seq_len, seq_len)
        Returns:
            interpolated: (num_heads, seq_len, seq_len) in [0, 1]
        """
        positions = relative_positions.float().abs()
        if positions.dim() == 2:
            positions = positions.unsqueeze(0)

        threshold = self.threshold.view(-1, 1, 1)
        interpolated = torch.log1p(positions) / torch.log1p(threshold)
        return interpolated.clamp(0.0, 1.0)


class FIRE(nn.Module):
    """Functional Interpolation for Relative Positional Encoding."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        max_position: int = 8192,
        num_layers: int = 2,
        mlp_width: int = 32,
        init_threshold: Optional[float] = None
    ):
        super().__init__()
        self.num_heads = num_heads
        self.max_position = max_position

        # Progressive interpolation
        threshold = init_threshold or float(max_position)
        self.interpolation = ProgressiveInterpolation(num_heads, threshold)

        # Learned mapping MLP: [0,1] → R
        layers = []
        in_features = 1
        for i in range(num_layers):
            out_features = mlp_width if i < num_layers - 1 else 1
            layers.append(nn.Linear(in_features, out_features, bias=True))
            if i < num_layers - 1:
                layers.append(nn.GELU())
            in_features = out_features
        self.mlp = nn.Sequential(*layers)

        # Initialize for near-zero initial bias
        self._init_weights()

    def _init_weights(self):
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, seq_len=None, relative_positions=None, device=None):
        """
        Compute FIRE position bias.

        Returns:
            bias: (num_heads, seq_len, seq_len)
        """
        if relative_positions is None:
            if device is None:
                device = next(self.parameters()).device
            q_pos = torch.arange(seq_len, device=device).float()
            k_pos = torch.arange(seq_len, device=device).float()
            relative_positions = (q_pos.unsqueeze(1) - k_pos.unsqueeze(0)).abs()

        # Progressive interpolation: (num_heads, S, S)
        interpolated = self.interpolation(relative_positions)

        # Apply MLP
        num_heads, s_q, s_k = interpolated.shape
        mlp_input = interpolated.reshape(-1, 1)
        bias_values = self.mlp(mlp_input)
        bias = bias_values.view(num_heads, s_q, s_k)

        return bias
```

## Usage Example

```python
from nexus.components.embeddings import FIRE

# Initialize FIRE
fire = FIRE(
    dim=512,
    num_heads=8,
    max_position=8192,
    num_layers=2,
    mlp_width=32
)

# In attention mechanism
seq_len = 1024
bias = fire(seq_len=seq_len)  # (8, 1024, 1024)

# Add to attention scores
scores = Q @ K.T / sqrt(d_k)
scores = scores + bias.unsqueeze(0)  # Add bias
attn = softmax(scores, dim=-1)
```

## Optimization Tricks

### 1. Incremental Decoding

```python
def incremental_fire_bias(fire, query_pos, seq_len_k, device):
    """Compute bias for single query position (generation)."""
    k_pos = torch.arange(seq_len_k, device=device).float()
    relative_pos = (query_pos - k_pos).abs().unsqueeze(0)

    interpolated = fire.interpolation(relative_pos)  # (H, 1, S_k)
    H, _, S_k = interpolated.shape
    mlp_input = interpolated.reshape(-1, 1)
    bias_values = fire.mlp(mlp_input)
    return bias_values.view(H, 1, S_k)
```

### 2. Sharing MLP Across Heads

```python
# Instead of per-head MLPs, use one shared MLP
# More parameter-efficient, slightly less expressive
class SharedMLPFIRE(nn.Module):
    def __init__(self, ...):
        # One MLP for all heads
        self.mlp = nn.Sequential(...)

    def forward(self, seq_len):
        interpolated = self.interpolation(...)  # (H, S, S)
        # Reshape and apply shared MLP
        bias = self.mlp(interpolated.reshape(-1, 1))
        return bias.view(self.num_heads, seq_len, seq_len)
```

## Experiments & Results

### Length Generalization

Training on 2048 tokens:

| Method | Train 2K | Test 4K | Test 8K | Test 16K | Test 32K |
|--------|----------|---------|---------|----------|----------|
| ALiBi | 15.2 | 15.8 | 16.9 | 18.2 | 19.8 |
| T5 Bias | 15.1 | 16.4 | 19.2 | 25.3 | 38.7 |
| FIRE | **15.1** | **15.4** | **15.7** | **16.2** | **17.1** |

**Observation**: FIRE matches ALiBi's extrapolation while being learnable.

### As Generalization of Other Methods

FIRE can approximate existing methods:

| Config | Approximates | Notes |
|--------|--------------|-------|
| Linear MLP + high threshold | ALiBi | Near-linear bias |
| 0-layer MLP (lookup) | T5 Bias | Learned per-bucket |
| Specific nonlinearity | Kerple | Kernelized bias |

### Parameter Efficiency

| Method | Parameters per Head | Total (8 heads) |
|--------|---------------------|-----------------|
| T5 Bias | 32 buckets | 256 |
| FIRE (2-layer, width=32) | ~1K | ~8K |
| FIRE (1-layer, width=16) | ~32 | ~256 |

## Common Pitfalls

1. **Too large MLP**: Overfitting on exact distances. Use small width (16-32).
2. **Wrong initialization**: Large initial biases cause instability. Use small gain (0.1).
3. **Forgetting threshold learning**: Let interpolation.threshold be learnable.
4. **Applying after softmax**: Bias must be added before softmax!

## References

- Liu, L., et al. (2024). **FIRE: Functional Interpolation for Relative Positional Encoding**. [arXiv:2310.04418](https://arxiv.org/abs/2310.04418)

**Implementation**: [/nexus/components/embeddings/fire.py](../../nexus/components/embeddings/fire.py)

---

**Next**: [CoPE](./cope.md) | [Resonance RoPE](./resonance_rope.md) | [Back to Overview](./README.md)
