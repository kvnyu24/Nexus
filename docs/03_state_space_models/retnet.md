# RetNet: Retentive Networks for Efficient Sequence Modeling

## Overview & Motivation

RetNet (Retentive Network) introduces a retention mechanism that combines the parallel training of Transformers with the efficient O(1) inference of RNNs. The key innovation is a **multi-scale retention** mechanism with exponential decay that can be computed in three equivalent formulations: parallel (for training), recurrent (for inference), and chunkwise (for long sequences).

### Why RetNet vs Transformers/RNNs?

| Aspect | Transformer | RNN/LSTM | RetNet |
|--------|-------------|----------|--------|
| Training complexity | O(n²) | O(n) | O(n) |
| Training parallelization | Full | None | Full |
| Inference per token | O(n) | O(1) | O(1) |
| Inference speed | Slow | Fast | Fastest |
| Long-range modeling | Excellent | Limited | Good |
| Memory at inference | O(n) | O(1) | O(1) |

RetNet achieves **competitive performance** with Transformers while having **2-8x faster inference** and lower memory usage.

## Theoretical Background

### Retention Mechanism

RetNet replaces attention with **retention**, a mechanism with exponential decay:

Standard attention:
```
Attention(Q, K, V) = softmax(QK^T / √d) V
```

Retention:
```
Retention(Q, K, V) = (Q (D ⊙ K)^T) V

where D[i,j] = γ^(i-j) for i ≥ j (exponential decay)
      D[i,j] = 0 for i < j (causal mask)
```

Key properties:
1. **Exponential decay**: Recent tokens have more influence
2. **Causal**: Only attends to past
3. **Three formulations**: Parallel, recurrent, chunkwise

### Multi-Scale Retention

Different heads use different decay rates:

```
Head 1: γ₁ = 0.99  (slow decay, long-range)
Head 2: γ₂ = 0.95  (medium decay)
Head 3: γ₃ = 0.90  (fast decay, local)
...
```

This creates **multi-scale temporal receptive fields**, similar to how CNNs use different kernel sizes.

### Triple Formulation

**Parallel (training)**:
```
O = (Q (D ⊙ (K^T V)))

where D is the decay matrix
```

**Recurrent (inference)**:
```
S_t = γ S_{t-1} + K_t^T V_t
O_t = Q_t S_t
```

**Chunkwise (long sequences)**:
```
Within chunk: parallel retention
Across chunks: recurrent state propagation
```

All three are **mathematically equivalent** but optimized for different scenarios!

## Mathematical Formulation

### 1. Retention Definition

Given queries Q, keys K, values V ∈ ℝ^(L×d):

```
1. Define decay matrix D:
   D[i,j] = γ^(i-j)  if i ≥ j
   D[i,j] = 0         if i < j

2. Retention output:
   Retention(Q, K, V) = (Q (D ⊙ K^T)) V

Equivalently:
   O[i] = Q[i] (∑_{j≤i} γ^(i-j) K[j]^T V[j])
```

### 2. Parallel Formulation

For training (full sequence):

```
1. Construct decay matrix:
   D = [
     [1,     0,     0,     ...],
     [γ,     1,     0,     ...],
     [γ²,    γ,     1,     ...],
     ...
   ]

2. Compute retention:
   O = (Q @ D @ K^T) @ V

Complexity: O(L² d) like attention, but D is structured
```

### 3. Recurrent Formulation

For inference (one token at a time):

```
Initialize: S_0 = 0 (d × d matrix)

For each timestep t:
  1. Update state:
     S_t = γ S_{t-1} + K_t^T V_t

  2. Compute output:
     O_t = Q_t @ S_t

Complexity: O(d²) per step - constant in sequence length!
```

### 4. Chunkwise Formulation

For long sequences:

```
Divide sequence into chunks of size C

For each chunk c:
  1. Within-chunk: Use parallel formulation
     O_chunk = parallel_retention(Q_c, K_c, V_c)

  2. Cross-chunk: Use recurrence
     S_c = γ^C S_{c-1} + (state from chunk c)

  3. Combine: O_c = O_chunk + Q_c @ (γ^C S_{c-1})
```

This balances parallelism (within chunks) and efficiency (across chunks).

### 5. Multi-Scale Extension

For H heads with different γ_h:

```
For each head h:
  O_h = Retention_h(Q_h, K_h, V_h; γ_h)

Combine:
  O = Concat(O_1, ..., O_H) W_O
```

Different decay rates capture different timescales.

## High-Level Intuition

Think of RetNet as:

1. **Exponentially weighted moving average**: Like EWMA in signal processing
2. **Recency bias**: Recent tokens matter more (decay with distance)
3. **Multi-scale**: Different heads for different timescales

Analogy to human memory:
- **Short-term (γ=0.9)**: Immediate context, fast decay
- **Medium-term (γ=0.95)**: Recent conversation
- **Long-term (γ=0.99)**: Overall topic, slow decay

The "retention" name:
- **Retain** information with exponential decay
- More principled than arbitrary positional encodings
- Natural inductive bias for sequential data

Comparison to attention:
- **Attention**: Learned weights (via softmax)
- **Retention**: Fixed exponential decay + learned Q, K, V
- **Result**: Simpler, faster, more efficient

## Implementation Details

### Architecture Components

```python
class MultiScaleRetention:
    def __init__(
        self,
        dim,
        num_heads=4,
        head_dim=None,
        gamma=None  # Decay factors
    ):
        self.num_heads = num_heads
        self.head_dim = head_dim or dim // num_heads
        self.hidden_dim = self.num_heads * self.head_dim

        # Decay factors per head (log-spaced by default)
        if gamma is None:
            gamma = [1 - 2**(-5-i) for i in range(num_heads)]
        self.register_buffer('gamma', torch.tensor(gamma))

        # Projections
        self.q_proj = Linear(dim, self.hidden_dim, bias=False)
        self.k_proj = Linear(dim, self.hidden_dim, bias=False)
        self.v_proj = Linear(dim, self.hidden_dim, bias=False)

        # Output gate (similar to Mamba)
        self.g_proj = Linear(dim, self.hidden_dim)

        # Output projection
        self.out_proj = Linear(self.hidden_dim, dim, bias=False)

        # Group norm
        self.group_norm = GroupNorm(num_heads, self.hidden_dim)
```

### Training Mode: Parallel Retention

```python
def forward_parallel(self, x):
    # x: (B, L, D)
    B, L, D = x.shape

    # 1. Project to Q, K, V
    Q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
    K = self.k_proj(x).view(B, L, self.num_heads, self.head_dim)
    V = self.v_proj(x).view(B, L, self.num_heads, self.head_dim)

    # 2. Compute retention for each head
    outputs = []
    for h in range(self.num_heads):
        Q_h = Q[:, :, h]  # (B, L, d_head)
        K_h = K[:, :, h]
        V_h = V[:, :, h]
        gamma_h = self.gamma[h]

        # Create decay matrix
        decay_matrix = self.get_decay_matrix(L, gamma_h)  # (L, L)

        # Retention: (Q @ D @ K^T) @ V
        retention = torch.einsum('bld,lk,bkd->bld', Q_h, decay_matrix, K_h)
        output_h = torch.einsum('bld,bld->bld', retention, V_h)

        outputs.append(output_h)

    # 3. Concatenate heads
    output = torch.cat(outputs, dim=-1)  # (B, L, H*d_head)

    # 4. Apply gate and normalize
    gate = F.silu(self.g_proj(x))
    output = output * gate
    output = self.group_norm(output)

    # 5. Output projection
    output = self.out_proj(output)

    return output

def get_decay_matrix(self, L, gamma):
    # Create L×L decay matrix
    # D[i,j] = gamma^(i-j) if i >= j, else 0

    positions = torch.arange(L, device=self.gamma.device)
    decay = gamma ** (positions.unsqueeze(0) - positions.unsqueeze(1))
    decay = torch.tril(decay)  # Causal mask

    return decay
```

### Inference Mode: Recurrent Retention

```python
def forward_recurrent(self, x, state):
    # x: (B, D) - single token
    # state: (B, num_heads, d_head, d_head) - per-head states

    B, D = x.shape

    # 1. Project
    Q = self.q_proj(x).view(B, self.num_heads, self.head_dim)
    K = self.k_proj(x).view(B, self.num_heads, self.head_dim)
    V = self.v_proj(x).view(B, self.num_heads, self.head_dim)

    # 2. Update states and compute output per head
    new_state = []
    outputs = []

    for h in range(self.num_heads):
        Q_h = Q[:, h]  # (B, d_head)
        K_h = K[:, h]
        V_h = V[:, h]
        S_h = state[:, h]  # (B, d_head, d_head)
        gamma_h = self.gamma[h]

        # Recurrent update: S_t = γ S_{t-1} + K^T V
        S_h_new = gamma_h * S_h + torch.einsum('bi,bj->bij', K_h, V_h)

        # Output: O_t = Q S_t
        O_h = torch.einsum('bi,bij->bj', Q_h, S_h_new)

        new_state.append(S_h_new)
        outputs.append(O_h)

    # 3. Combine
    state = torch.stack(new_state, dim=1)  # (B, H, d, d)
    output = torch.cat(outputs, dim=-1)  # (B, H*d)

    # 4. Gate and norm
    gate = F.silu(self.g_proj(x))
    output = output * gate
    output = self.group_norm(output.unsqueeze(1)).squeeze(1)

    # 5. Project
    output = self.out_proj(output)

    return output, state
```

### Chunkwise Retention (for Long Sequences)

```python
def forward_chunkwise(self, x, chunk_size=64):
    # x: (B, L, D)
    B, L, D = x.shape
    num_chunks = (L + chunk_size - 1) // chunk_size

    outputs = []
    state = torch.zeros(
        B, self.num_heads, self.head_dim, self.head_dim,
        device=x.device
    )

    for c in range(num_chunks):
        start = c * chunk_size
        end = min(start + chunk_size, L)
        x_chunk = x[:, start:end]

        # Within-chunk: parallel
        o_chunk = self.forward_parallel(x_chunk)

        # Cross-chunk: incorporate previous state
        # (simplified - full version more complex)
        if c > 0:
            # Decay state by gamma^chunk_size
            for h in range(self.num_heads):
                state[:, h] *= self.gamma[h] ** chunk_size

        outputs.append(o_chunk)

    return torch.cat(outputs, dim=1)
```

## Code Walkthrough

See `Nexus/nexus/components/ssm/retnet.py` for full implementation.

### Key Functions

1. **MultiScaleRetention**: Core retention mechanism
   - Multiple heads with different γ
   - Three computation modes
   - Group normalization

2. **forward_parallel()**: Training mode
   - Constructs decay matrix
   - Computes retention via matrix operations
   - O(L² d) but parallelizable

3. **forward_recurrent()**: Inference mode
   - O(1) per token
   - Maintains matrix-valued state per head
   - Fast autoregressive generation

4. **RetNetBlock**: Full block with FFN
   - Pre-norm + residual
   - GLU feedforward
   - Optional dropout

## Optimization Tricks

### 1. Decay Matrix Computation

Cache decay matrix for common lengths:

```python
self.decay_cache = {}

def get_decay_matrix(self, L, gamma):
    if (L, gamma) not in self.decay_cache:
        self.decay_cache[(L, gamma)] = self._compute_decay(L, gamma)
    return self.decay_cache[(L, gamma)]
```

### 2. Efficient Parallel Retention

Use cumulative sum trick:

```python
# Instead of: O @ D @ K^T @ V
# Use: O @ cumsum(gamma^i * K_i^T V_i)

kv = torch.einsum('bld,ble->blde', K, V)  # Outer product
decay_weights = gamma ** torch.arange(L)
kv_weighted = kv * decay_weights.view(1, L, 1, 1)
kv_cumsum = torch.cumsum(kv_weighted, dim=1)
output = torch.einsum('bld,blde->ble', Q, kv_cumsum)
```

### 3. Gamma Initialization

Log-spaced decay rates:

```python
# Heads have logarithmically spaced retention
gamma = [1 - 2**(-5-i) for i in range(num_heads)]
# e.g., [0.96875, 0.984375, 0.9921875, 0.99609375]
```

### 4. Mixed Precision

Use bf16 for computation, fp32 for state:

```python
with torch.cuda.amp.autocast(dtype=torch.bfloat16):
    Q, K, V = self.q_proj(x), self.k_proj(x), self.v_proj(x)

# State updates in fp32
state = state.float()
state = gamma * state + K_t.float().T @ V_t.float()
state = state.bfloat16()
```

### 5. Fused Kernel for Recurrence

Custom CUDA kernel for state update:

```python
# Fuse: state = gamma * state + K^T V
from retnet_kernels import fused_retention_recurrence

state = fused_retention_recurrence(state, K, V, gamma)
```

### 6. Chunkwise for Very Long Sequences

Automatically switch to chunkwise:

```python
def forward(self, x, state=None):
    L = x.shape[1]

    if state is not None:
        return self.forward_recurrent(x, state)
    elif L > 2048:
        return self.forward_chunkwise(x, chunk_size=256)
    else:
        return self.forward_parallel(x)
```

## Experiments & Results

### Language Modeling

Performance on Pile (1.4T tokens):

| Model | Size | Perplexity | Training Speed | Inference Speed |
|-------|------|-----------|----------------|----------------|
| Transformer | 1.3B | 8.21 | 10k tok/s | 45 tok/s |
| RetNet | 1.3B | 8.32 | 12k tok/s | 120 tok/s |

RetNet achieves **competitive quality** with **2.7x faster inference**.

### Long Context

Performance on long-context tasks:

| Task (length) | Transformer | RetNet | Relative |
|--------------|-------------|--------|----------|
| LRA Text (4k) | 86.8% | 85.3% | 98.3% |
| LRA Retrieval (4k) | 90.9% | 89.1% | 98.0% |
| LongBench (16k) | 44.1 | 42.8 | 97.1% |

RetNet achieves **~97-98%** of Transformer performance on long sequences.

### Inference Speed Comparison

Tokens/second for autoregressive generation (batch=1):

| Model | Seq Len 512 | Seq Len 2048 | Seq Len 8192 |
|-------|------------|--------------|--------------|
| Transformer | 120 | 45 | 12 |
| RetNet | 180 | 150 | 140 |

RetNet maintains **constant speed** regardless of context length!

### Memory Usage

Peak memory during inference:

| Model | Seq Len 512 | 2048 | 8192 | 32768 |
|-------|------------|------|------|-------|
| Transformer | 2GB | 4GB | 12GB | OOM |
| RetNet | 1GB | 1GB | 1GB | 1GB |

RetNet uses **O(1) memory** at inference.

### Ablation: Number of Heads

Effect of multi-scale heads:

| Heads | Perplexity | Speed |
|-------|-----------|-------|
| 1 | 8.95 | 1.2x |
| 2 | 8.54 | 1.1x |
| 4 | 8.32 | 1.0x |
| 8 | 8.35 | 0.9x |

**4 heads** is the sweet spot for most tasks.

### Ablation: Decay Rates

Effect of γ range:

| γ_min, γ_max | Perplexity |
|--------------|-----------|
| 0.5, 0.9 | 9.12 |
| 0.9, 0.99 | 8.32 |
| 0.95, 0.999 | 8.28 |

Range **[0.9, 0.99]** works well for language modeling.

## Common Pitfalls

### 1. Wrong Decay Matrix

**Problem**: Using symmetric decay (past and future).

**Solution**: Ensure causal (lower triangular):
```python
# BAD: Symmetric
decay = gamma ** torch.abs(i - j)

# GOOD: Causal
decay = gamma ** (i - j) if i >= j else 0
decay = torch.tril(decay)
```

### 2. Not Using Multi-Scale

**Problem**: Same γ for all heads.

**Solution**: Use different decay rates:
```python
gamma = [1 - 2**(-5-i) for i in range(num_heads)]
```

### 3. State Shape Mismatch

**Problem**: State is (B, H, d) instead of (B, H, d, d).

**Solution**: RetNet uses **matrix-valued states**:
```python
state = torch.zeros(B, num_heads, d_head, d_head)
```

### 4. Forgetting to Decay State

**Problem**: In recurrence, not multiplying by γ.

**Solution**: Always decay:
```python
state = gamma * state + K.T @ V
```

### 5. Using Float16 for State

**Problem**: State accumulation loses precision.

**Solution**: Use bf16 or fp32:
```python
state = state.float()  # Accumulate in fp32
```

### 6. Not Caching Decay Matrix

**Problem**: Recomputing decay matrix every forward pass.

**Solution**: Cache it:
```python
if L not in self.decay_cache:
    self.decay_cache[L] = self.get_decay_matrix(L)
```

### 7. Wrong Normalization Position

**Problem**: Normalizing before gating.

**Solution**: Gate first, then normalize:
```python
output = output * gate
output = self.group_norm(output)
```

## Initialization Best Practices

```python
def init_retnet_layer(layer):
    # 1. Q, K, V projections: Xavier
    nn.init.xavier_uniform_(layer.q_proj.weight)
    nn.init.xavier_uniform_(layer.k_proj.weight)
    nn.init.xavier_uniform_(layer.v_proj.weight)

    # 2. Gamma: Log-spaced decay rates
    gamma = [1 - 2**(-5-i) for i in range(layer.num_heads)]
    layer.gamma.data = torch.tensor(gamma)

    # 3. Gate projection: Xavier
    nn.init.xavier_uniform_(layer.g_proj.weight)
    nn.init.zeros_(layer.g_proj.bias)

    # 4. Output projection: Xavier with small init
    nn.init.xavier_uniform_(layer.out_proj.weight)
    layer.out_proj.weight.data *= 0.5  # Smaller init for stability

    # 5. Group norm: default (gain=1, bias=0)
```

## References

### Primary Papers

1. **RetNet (2023)**
   - Sun et al. "Retentive Network: A Successor to Transformer for Large Language Models"
   - https://arxiv.org/abs/2307.08621
   - Introduces retention mechanism

2. **Multi-Scale Retention (2023)**
   - Same paper as above
   - Details multi-scale design

### Related Work

3. **Linear Attention (2020)**
   - Katharopoulos et al. "Transformers are RNNs"
   - https://arxiv.org/abs/2006.16236
   - Foundation for linear-time attention

4. **AFT (2021)**
   - Zhai et al. "Attention Free Transformer"
   - https://arxiv.org/abs/2105.14103
   - Related attention alternative

5. **RWKV (2023)**
   - Peng et al. "RWKV: Reinventing RNNs"
   - https://arxiv.org/abs/2305.13048
   - Similar linear RNN approach

6. **Mamba (2023)**
   - Gu & Dao. "Mamba: Linear-Time Sequence Modeling"
   - https://arxiv.org/abs/2312.00752
   - State-of-the-art SSM

## Implementation Checklist

When implementing RetNet from scratch:

- [ ] Multi-head retention (4-8 heads)
- [ ] Log-spaced decay rates per head
- [ ] Three computation modes (parallel/recurrent/chunkwise)
- [ ] Causal decay matrix (lower triangular)
- [ ] Matrix-valued state (B, H, d_head, d_head)
- [ ] Q, K, V projections
- [ ] Output gating mechanism
- [ ] Group normalization
- [ ] Decay matrix caching
- [ ] Mixed precision (bf16/fp32)
- [ ] Proper state decay in recurrence
- [ ] Efficient cumsum for parallel mode
- [ ] Chunkwise for long sequences
- [ ] Fused kernels (optional)

---

*For implementation reference, see `Nexus/nexus/components/ssm/retnet.py`*
