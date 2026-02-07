# Mamba-2: State Space Duality (SSD)

## Overview & Motivation

Mamba-2 introduces the State Space Duality (SSD) framework, revealing a fundamental connection between State Space Models (SSMs) and structured masked attention. This insight enables Mamba-2 to leverage efficient matrix multiplication kernels (like Flash Attention) while maintaining the linear-time complexity and efficient inference of SSMs.

### Why Mamba-2 vs Mamba?

| Aspect | Mamba | Mamba-2 (SSD) |
|--------|-------|---------------|
| Training speed | 1x | 2-8x faster |
| State dimension | 16 (typical) | 64-128 |
| Computation | Custom selective scan | Matrix multiply (tensor cores) |
| Hardware utilization | ~40% | ~80% (tensor cores) |
| Multi-head structure | No | Yes (8-16 heads) |
| Theoretical framework | Selective SSM | SSM-attention duality |

Mamba-2 achieves 2-8x speedup over Mamba while maintaining the same quality, making it the preferred choice for large-scale training.

## Theoretical Background

### State Space Duality

The key insight is that SSMs can be reformulated as a special case of **structured masked attention**:

**SSM formulation:**
```
h_t = A h_{t-1} + B x_t
y_t = C h_t
```

**Attention formulation (with mask):**
```
y_t = ∑_{i=1}^t (C_t A^{t-i} B_i) x_i
```

This shows that SSMs compute a **cumulative weighted sum** of past inputs, which is exactly what masked attention does, but with:
1. **Exponential decay**: A^{t-i} creates structured decay
2. **Low-rank projections**: B, C are low-rank
3. **Causal mask**: Only attend to past (i ≤ t)

### Multi-Head SSD

Mamba-2 extends this to multi-head structure:

```
For each head h:
  h_t^(h) = A^(h) h_{t-1}^(h) + B^(h) x_t
  y_t^(h) = C^(h) h_t^(h)

Output: y_t = ∑_h y_t^(h)
```

This allows:
- Different heads to capture different time scales
- Larger total state capacity (num_heads × state_dim)
- Better parallelization on modern GPUs

### Chunk-wise Processing

Mamba-2 processes sequences in chunks for efficiency:

```
Chunk size = 64 or 128 tokens

Within chunk: Use matrix multiply (parallel attention-like)
Across chunks: Use SSM recurrence to propagate state
```

This hybrid approach combines:
- **Intra-chunk parallelism**: Fast matrix ops
- **Inter-chunk efficiency**: Linear complexity

## Mathematical Formulation

### 1. SSD Layer Definition

Given input sequence x ∈ ℝ^(L×D):

```
1. Project to Q, K, V:
   Q = x W_Q  (L, H, P)
   K = x W_K  (L, H, P)
   V = x W_V  (L, H, P)

2. Compute per-head:
   For head h:
     A^(h) = exp(-softplus(log_A^(h)))  (scalar decay per head)

     h_t^(h) = A^(h) h_{t-1}^(h) + K_t^(h) ⊗ V_t^(h)
     y_t^(h) = Q_t^(h) ⊙ h_t^(h)

3. Combine heads:
   y = Concat(y^(1), ..., y^(H)) W_O
```

where:
- H is number of heads
- P is head dimension
- ⊗ is outer product (creates rank-1 state update)
- ⊙ is element-wise product

### 2. Chunk-wise Computation

For chunk of size C:

```
Within chunk [t, t+C):
  # Compute attention-like scores
  S[i,j] = A^(j-i) for j >= i  (upper triangular decay matrix)

  # Compute chunk output
  Y_chunk = S @ (K ⊗ V)  (matrix multiply!)
  Y_chunk = Q ⊙ Y_chunk

Across chunks:
  # Propagate state
  h_{t+C} = A^C h_t + (state from chunk)
```

The key: within-chunk computation is just **matrix multiply with structured mask**, enabling tensor cores!

### 3. State Update Formula

The state update can be written compactly:

```
h_t = A ⊙ h_{t-1} + K_t ⊗ V_t

Expanded:
h_t[i,j] = A[i,j] * h_{t-1}[i,j] + K_t[i] * V_t[j]
```

This is a **rank-1 update** with **element-wise decay**, allowing efficient computation.

### 4. Multi-Scale Decay

Different heads use different decay rates:

```
Head 1: A = 0.9   (slow decay, long-range)
Head 2: A = 0.5   (medium decay)
Head 3: A = 0.1   (fast decay, local)
...
```

This creates a **multi-scale** temporal receptive field, similar to multi-scale retention in RetNet.

## High-Level Intuition

Think of Mamba-2 as:

1. **Multi-head SSM**: Each head maintains its own state with its own decay rate
2. **Attention-like computation**: Within chunks, computation looks like masked attention
3. **Best of both worlds**:
   - Training: Fast matrix multiplies (like attention)
   - Inference: O(1) state updates (like RNN)

The "duality" means:
- **SSM view**: Recurrent state updates with exponential decay
- **Attention view**: Structured masked attention with low-rank projections

Both views are **mathematically equivalent**, but enable different optimizations!

## Implementation Details

### Architecture Components

```python
class Mamba2Block:
    def __init__(
        self,
        d_model=1024,
        d_state=128,      # State dimension per head
        num_heads=8,      # Number of parallel heads
        chunk_size=64,    # Chunk size for hybrid processing
        expand=2          # Expansion ratio
    ):
        self.d_inner = d_model * expand
        self.d_head = self.d_inner // num_heads

        # Input projection
        self.in_proj = Linear(d_model, self.d_inner * 2)

        # SSM parameters (per head)
        self.A_log = Parameter(torch.randn(num_heads))

        # Conv1d for local context (like Mamba)
        self.conv1d = Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=4,
            padding=3,
            groups=self.d_inner
        )

        # Projections for SSD
        self.x_proj = Linear(self.d_inner, num_heads * d_state * 2)  # K, V
        self.dt_proj = Linear(d_state, num_heads)

        # Output projection
        self.out_proj = Linear(self.d_inner, d_model)
```

### Training Mode: Chunk-wise SSD

```python
def forward_ssd(self, x):
    # x: (B, L, D)
    B, L, D = x.shape

    # 1. Input projection and activation
    xz = self.in_proj(x)  # (B, L, 2*d_inner)
    x, z = xz.chunk(2, dim=-1)  # Split for gating

    # 2. Local convolution
    x = rearrange(x, 'b l d -> b d l')
    x = self.conv1d(x)[:, :, :L]  # Causal
    x = rearrange(x, 'b d l -> b l d')
    x = F.silu(x)

    # 3. SSM computation
    x = rearrange(x, 'b l (h p) -> b l h p', h=self.num_heads)

    # Compute K, V
    kv = self.x_proj(x)  # (B, L, H, 2*P)
    k, v = kv.chunk(2, dim=-1)

    # Compute A (decay per head)
    A = -torch.exp(self.A_log)  # Ensure stability

    # Process in chunks
    y = self.chunk_ssd(k, v, A, chunk_size=self.chunk_size)

    # 4. Gating and output
    y = rearrange(y, 'b l h p -> b l (h p)')
    y = y * F.silu(z)
    y = self.out_proj(y)

    return y

def chunk_ssd(self, k, v, A, chunk_size=64):
    B, L, H, P = k.shape
    num_chunks = (L + chunk_size - 1) // chunk_size

    y = []
    state = torch.zeros(B, H, P, P, device=k.device)

    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, L)
        chunk_len = end - start

        k_chunk = k[:, start:end]  # (B, C, H, P)
        v_chunk = v[:, start:end]

        # Within-chunk attention-like computation
        # Create decay matrix: S[i,j] = A^(j-i) for j >= i
        positions = torch.arange(chunk_len, device=k.device)
        decay_matrix = A.view(1, 1, H, 1, 1) ** (
            positions.view(1, chunk_len, 1, 1) -
            positions.view(chunk_len, 1, 1, 1)
        )
        decay_matrix = decay_matrix.triu()  # Causal mask

        # Compute: Y = Q ⊙ (S @ (K ⊗ V))
        # Efficiently: Y[t] = Q[t] ⊙ (∑_{i≤t} A^(t-i) K[i] ⊗ V[i])
        kv_states = torch.einsum('bthi,bthj->bthij', k_chunk, v_chunk)

        # Apply decay and accumulate
        y_chunk = torch.zeros_like(k_chunk)
        for t in range(chunk_len):
            # Combine initial state and within-chunk contributions
            state_contrib = torch.einsum('bhij,bhj->bhi', state, v_chunk[:, t])
            chunk_contrib = torch.einsum(
                'i,bhiij->bhj',
                decay_matrix[t, :t+1].squeeze(),
                kv_states[:, :t+1]
            ).sum(dim=1)

            y_chunk[:, t] = state_contrib + chunk_contrib

            # Update state for next step
            if t == chunk_len - 1:
                state = A.view(1, H, 1, 1) * state
                state = state + kv_states[:, t]

        y.append(y_chunk)

    return torch.cat(y, dim=1)
```

### Inference Mode: Recurrent State Updates

```python
def forward_recurrent(self, x, state):
    # x: (B, D) - single token
    # state: (B, H, P, P) - per-head matrix state

    B, D = x.shape

    # 1. Project and activate
    xz = self.in_proj(x)
    x, z = xz.chunk(2, dim=-1)

    # 2. Conv (use cached conv state)
    x, conv_state = self.step_conv(x, state['conv'])
    x = F.silu(x)

    # 3. SSM step
    x = rearrange(x, 'b (h p) -> b h p', h=self.num_heads)

    kv = self.x_proj(x)
    k, v = kv.chunk(2, dim=-1)  # (B, H, P)

    # Decay matrix
    A = -torch.exp(self.A_log).view(1, -1, 1, 1)

    # State update: h_t = A ⊙ h_{t-1} + k ⊗ v
    ssm_state = state['ssm']  # (B, H, P, P)
    ssm_state = A * ssm_state + torch.einsum('bhk,bhv->bhkv', k, v)

    # Output: y = sum over state weighted by k (query-like)
    y = torch.einsum('bhkv,bhk->bhv', ssm_state, k)

    # 4. Gating and output
    y = rearrange(y, 'b h p -> b (h p)')
    y = y * F.silu(z)
    y = self.out_proj(y)

    new_state = {'conv': conv_state, 'ssm': ssm_state}
    return y, new_state
```

### Efficient Matrix Multiply Kernel

The key optimization is using **block-sparse attention kernels** (like FlashAttention):

```python
# Pseudo-code for optimized SSD kernel
def ssd_chunk_fused(Q, K, V, A, chunk_size):
    # Fused kernel that:
    # 1. Computes decay matrix on-the-fly
    # 2. Uses tensor cores for K⊗V and attention
    # 3. Accumulates in shared memory

    # This achieves 80%+ tensor core utilization
    # vs 40% for custom selective scan

    return flash_attn_with_decay(Q, K, V, A, chunk_size)
```

## Code Walkthrough

See `Nexus/nexus/components/ssm/mamba2.py` for full implementation.

### Key Functions

1. **Mamba2Layer**: Main SSD layer
   - Handles both training (chunk-wise) and inference (recurrent)
   - Multi-head structure with per-head decay
   - Conv1d for local context

2. **chunk_ssd()**: Chunk-wise SSD computation
   - Processes sequence in chunks (default 64)
   - Uses matrix multiply within chunks
   - Propagates state across chunks

3. **ssd_minimal_discrete()**: Core SSD algorithm
   - Discretizes continuous SSM
   - Computes structured decay matrix
   - Applies to K⊗V states

4. **Mamba2Block**: Full block with normalization
   - Pre-normalization
   - Residual connections
   - Optional MLP

### State Structure

```python
state = {
    'conv': conv_state,  # (B, D, kernel_size-1)
    'ssm': ssm_state     # (B, H, P, P) - matrix per head
}
```

## Optimization Tricks

### 1. Tensor Core Utilization

Reshape operations to align with tensor core dimensions (16×16):

```python
# Ensure dimensions are multiples of 16
d_head = ((d_model * expand) // num_heads // 16) * 16
```

### 2. Chunk Size Selection

Optimal chunk size depends on hardware:
- A100: 64-128
- H100: 128-256
- Larger chunks = more parallelism but more memory

```python
chunk_size = 64 if seq_len < 2048 else 128
```

### 3. Memory-Efficient State Updates

Store state in low-rank form when P is large:

```python
# Instead of full P×P matrix:
state_full = k ⊗ v  # (P, P)

# Store as outer product factors:
state_k, state_v = k, v  # 2P instead of P²
```

### 4. Decay Computation

Pre-compute powers of A for common positions:

```python
# Cache A^0, A^1, A^2, ..., A^63 for chunk_size=64
A_powers = A.unsqueeze(-1) ** torch.arange(64)
```

### 5. Fused Operations

Fuse projection and activation:

```python
# Instead of: x = Linear(x); x = silu(x)
# Use: x = linear_silu_fused(x)
@torch.jit.script
def linear_silu_fused(x, weight, bias):
    return F.silu(F.linear(x, weight, bias))
```

### 6. Mixed Precision

Use bf16 for most ops, fp32 for state accumulation:

```python
with torch.autocast('cuda', dtype=torch.bfloat16):
    y = ssd_layer(x)

# State updates in fp32
state = state.float()
state = A * state + k.float() ⊗ v.float()
state = state.bfloat16()
```

## Experiments & Results

### Training Speed Comparison

Benchmark on A100 GPU (d_model=2048, seq_len=2048):

| Model | Time/Iteration | Speedup | Memory |
|-------|----------------|---------|---------|
| Transformer | 180ms | 1.0x | 24GB |
| Mamba | 120ms | 1.5x | 16GB |
| Mamba-2 | 45ms | 4.0x | 14GB |

Mamba-2 is **4x faster** than Transformer, **2.7x faster** than Mamba!

### Scaling to Large Models

| Model Size | Mamba Throughput | Mamba-2 Throughput | Speedup |
|-----------|------------------|-------------------|---------|
| 350M | 32k tok/s | 85k tok/s | 2.7x |
| 1.3B | 18k tok/s | 52k tok/s | 2.9x |
| 2.7B | 9k tok/s | 31k tok/s | 3.4x |

Speedup **increases** with model size due to better tensor core utilization.

### Language Modeling Performance

Comparison on Pile (1.4T tokens):

| Model | Params | Perplexity | Throughput |
|-------|--------|-----------|------------|
| GPT-3 (Transformer) | 2.7B | 8.21 | 10k tok/s |
| Mamba | 2.7B | 8.23 | 15k tok/s |
| Mamba-2 | 2.7B | 8.19 | 42k tok/s |

Mamba-2 matches quality while being **4x faster** than Transformer, **2.8x faster** than Mamba.

### Long Context Benchmarks

Performance on sequences up to 32k tokens:

| Task | Mamba | Mamba-2 | Transformer |
|------|-------|---------|-------------|
| Passkey Retrieval (32k) | 97.2% | 98.1% | 95.3% |
| LongBench (avg) | 45.3 | 46.8 | 44.1 |
| Multi-Document QA | 71.2 | 73.4 | 69.8 |

Mamba-2 **slightly outperforms** Mamba, likely due to larger state capacity.

### State Dimension Analysis

Effect of state dimension on quality (2.7B model):

| State Dim | Perplexity | Memory/Layer | Speed |
|-----------|-----------|--------------|-------|
| 16 | 8.45 | 8MB | Fast |
| 64 | 8.24 | 32MB | Medium |
| 128 | 8.19 | 64MB | Slower |
| 256 | 8.18 | 128MB | Slowest |

Sweet spot: **64-128** state dimension per head.

## Common Pitfalls

### 1. Not Using Multi-Head Structure

**Problem**: Using single-head SSD (like Mamba).

**Solution**: Always use multi-head (8-16 heads) for Mamba-2:
```python
# BAD: Single head
ssd = Mamba2(num_heads=1)

# GOOD: Multi-head
ssd = Mamba2(num_heads=8)
```

Multi-head enables tensor cores and larger state capacity.

### 2. Wrong Chunk Size

**Problem**: Using chunk_size=1 (fully sequential) or chunk_size=seq_len (fully parallel).

**Solution**: Use chunk_size=64-128:
```python
chunk_size = 64  # Good balance
```

### 3. Incorrect Decay Initialization

**Problem**: Initializing A close to 0 or 1.

**Solution**: Initialize log_A for moderate decay:
```python
# A in range [0.9, 0.999] for slow decay
log_A = torch.randn(num_heads) * 0.5 - 3.0
A = -torch.exp(log_A)  # A ≈ -0.95
```

### 4. Forgetting Causal Mask in Chunks

**Problem**: Full attention within chunks (sees future).

**Solution**: Apply causal mask to decay matrix:
```python
decay_matrix = decay_matrix.triu()  # Upper triangular only
```

### 5. State Shape Mismatch

**Problem**: State is (B, H, P) instead of (B, H, P, P).

**Solution**: Mamba-2 uses **matrix-valued states**:
```python
state = torch.zeros(B, num_heads, d_state, d_state)
```

Not vector-valued like Mamba!

### 6. Not Propagating State Across Chunks

**Problem**: Resetting state at each chunk boundary.

**Solution**: Carry state forward:
```python
state = A**chunk_size * state + chunk_contribution
```

### 7. Using Float16 for State

**Problem**: State accumulation loses precision in fp16.

**Solution**: Use bf16 or fp32 for states:
```python
state = state.float()  # Accumulate in fp32
# ... compute ...
state = state.bfloat16()  # Convert back
```

### 8. Not Utilizing Tensor Cores

**Problem**: Arbitrary dimensions that don't align with tensor cores.

**Solution**: Make dimensions multiples of 16:
```python
d_head = (d_inner // num_heads // 16) * 16
d_state = (d_state // 16) * 16
```

## Initialization Best Practices

```python
def init_mamba2(layer):
    # 1. Decay initialization (per head)
    # Heads should have varying decay rates
    log_A = torch.linspace(-4, -1, layer.num_heads)
    layer.A_log.data = log_A

    # 2. Projection initialization
    nn.init.xavier_uniform_(layer.in_proj.weight)
    nn.init.xavier_uniform_(layer.out_proj.weight)
    nn.init.zeros_(layer.out_proj.bias)

    # 3. Conv initialization
    nn.init.xavier_uniform_(layer.conv1d.weight)
    nn.init.zeros_(layer.conv1d.bias)

    # 4. SSM projection
    nn.init.xavier_uniform_(layer.x_proj.weight)
    nn.init.zeros_(layer.x_proj.bias)
```

## References

### Primary Papers

1. **Mamba-2 / SSD (2024)**
   - Dao & Gu. "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality"
   - https://arxiv.org/abs/2405.21060
   - Introduces SSD framework and Mamba-2

2. **Mamba (2023)**
   - Gu & Dao. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
   - https://arxiv.org/abs/2312.00752
   - Foundation for selective SSMs

3. **S4 (2022)**
   - Gu et al. "Efficiently Modeling Long Sequences with Structured State Spaces"
   - https://arxiv.org/abs/2111.00396
   - Original SSM foundation

### Implementation References

4. **Official Mamba-2 Repository**
   - https://github.com/state-spaces/mamba
   - Reference CUDA kernels and PyTorch implementation

5. **FlashAttention-2**
   - Dao. "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
   - https://arxiv.org/abs/2307.08691
   - Kernel design inspiration for SSD

### Related Work

6. **Structured Attention**
   - Child et al. "Generating Long Sequences with Sparse Transformers" (2019)
   - https://arxiv.org/abs/1904.10509
   - Structured sparsity in attention

7. **Linear Attention**
   - Katharopoulos et al. "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention" (2020)
   - https://arxiv.org/abs/2006.16236
   - Linear complexity attention variants

## Implementation Checklist

When implementing Mamba-2 from scratch:

- [ ] Multi-head structure (8-16 heads)
- [ ] Per-head decay parameters (different rates)
- [ ] Chunk-wise processing (chunk_size=64-128)
- [ ] Matrix-valued states (B, H, P, P)
- [ ] Causal masking within chunks
- [ ] State propagation across chunks
- [ ] Conv1d for local context
- [ ] Gating mechanism (SiLU activation)
- [ ] Tensor core alignment (dims divisible by 16)
- [ ] Mixed precision (bf16/fp32)
- [ ] Efficient K⊗V computation
- [ ] Recurrent mode for inference
- [ ] Proper state initialization
- [ ] Fused kernels for critical paths

---

*For implementation reference, see `Nexus/nexus/components/ssm/mamba2.py`*
