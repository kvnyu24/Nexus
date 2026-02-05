# Mamba: Linear-Time Sequence Modeling with Selective State Spaces

## Overview & Motivation

Mamba introduces **selective state space models (SSMs)**, where the fundamental SSM parameters (A, B, C, Δ) are functions of the input rather than fixed matrices. This seemingly simple change has profound implications, enabling Mamba to match or exceed Transformer performance while maintaining linear complexity.

### The Selective SSM Insight

Traditional SSMs (S4, S4D, S5) use **content-independent** parameters:
```python
# Traditional SSM
A, B, C = fixed_parameters  # Same for all inputs
y = SSM(A, B, C, u)
```

Mamba makes parameters **content-aware**:
```python
# Selective SSM
A, B, C, Δ = f(u)  # Functions of input
y = SSM(A(u), B(u), C(u), Δ(u), u)
```

This allows the model to:
- **Selectively remember** important information
- **Selectively forget** irrelevant details
- **Focus** on specific parts of the context
- **Adapt** recurrence to input characteristics

### Why Mamba vs Other SSMs?

| Feature | S4/S4D | Mamba | Transformer |
|---------|--------|-------|-------------|
| Time complexity | O(n log n) | O(n) | O(n²) |
| Space complexity | O(n) | O(n) | O(n²) |
| Input-dependent | No | Yes | Yes |
| Selective attention | No | Yes | Yes |
| Hardware efficient | Medium | High | High |
| Long-range modeling | Excellent | Excellent | Good |

Mamba combines the best of both worlds: SSM efficiency + Transformer selectivity.

## Theoretical Background

### From Fixed to Selective SSMs

The standard SSM equations are:
```
h'(t) = Ah(t) + Bu(t)
y(t) = Ch(t)
```

After discretization with step Δ:
```
h_t = exp(ΔA)h_{t-1} + ΔBu_t
y_t = Ch_t
```

In S4, `A`, `B`, `C`, `Δ` are fixed parameters. In Mamba:
```
Δ(t), B(t), C(t) = sB(u_t), sC(u_t), τ_Δ(u_t)
A = -exp(Parameter)  # Still fixed but with special initialization
```

where `sB`, `sC`, `τ_Δ` are linear projections of the input.

### Selection Mechanism

The input-dependent Δ (discretization step) is the key to selectivity:
- **Large Δ**: Fast forgetting, focus on current input
- **Small Δ**: Slow forgetting, remember long history

The model learns to modulate Δ based on input importance:
```python
Δ = softplus(Linear(u))  # Adaptive timestep
h_t = exp(ΔA)h_{t-1} + ΔBu_t  # Selective integration
```

### Hardware-Aware Selective Scan

Computing selective SSM naively is O(BLD) where B=batch, L=length, D=dimension. Mamba uses a hardware-aware algorithm that:

1. **Fuses operations**: Combines scan and element-wise ops in one kernel
2. **Avoids materialization**: Doesn't store full attention-like matrix
3. **Uses SRAM efficiently**: Keeps working set in fast memory
4. **Parallelizes across batch and channels**: Maximizes GPU utilization

This achieves 2-3x speedup over naive implementation.

## Mathematical Formulation

### 1. Input Projections

Given input `u ∈ ℝ^{B×L×D}`:

```python
x, z = split(Linear(u), dim=-1)  # Each: (B, L, D_inner)
x = conv1d(x)  # Local context
x = silu(x)    # Activation
```

where `D_inner = expand * D` (typically expand=2).

### 2. Selective SSM Parameters

For each position t:
```python
Δ, B, C = split(Linear(x), [dt_rank, N, N])
Δ = softplus(Linear(Δ))  # Project dt_rank → D_inner
# Δ: (B, L, D_inner)
# B, C: (B, L, N) where N is state dimension
```

### 3. Discretization

Convert continuous parameters to discrete:
```python
A = -exp(log_A)  # (D_inner, N)
Ā = exp(Δ[:,:,:,None] * A[None,None,:,:])  # (B, L, D_inner, N)
B̄ = Δ[:,:,:,None] * B[:,:,None,:]  # (B, L, D_inner, N)
```

### 4. Selective Scan

The core recurrence:
```python
h_0 = zeros(B, D_inner, N)
for t in range(L):
    h_t = Ā[:,t] * h_{t-1} + B̄[:,t] * x[:,t,:,None]
    y_t = (C[:,t,:,None].T @ h_t).squeeze()
y = stack(y_t for t in range(L))
```

### 5. Gated Output

```python
y = y * silu(z)  # Gate with parallel branch
output = Linear(y)  # Project back to D
```

## High-Level Intuition

### The Selective Memory Analogy

Think of Mamba as a person reading a document:

1. **Convolution**: Quick glance at local context (3-4 words)
2. **Selective Scan**: Decide what to remember in detail
   - Important sentences: Small Δ (remember longer)
   - Filler words: Large Δ (forget quickly)
3. **State**: Compressed memory of everything read so far
4. **Output**: What to say based on memory and current input

Example:
```
Input: "The capital of France is Paris. The weather is nice. What's the capital?"
        ↓
Δ:     [small........, large......, small]  # Remember capitals, forget weather
        ↓
Memory: Stores "France → Paris"
        ↓
Output: "Paris"
```

### Comparison with Attention

Attention:
```python
# Explicit comparison of query with all keys
scores = Q @ K.T  # O(n²)
weights = softmax(scores)
output = weights @ V
```

Mamba:
```python
# Implicit compression via selective integration
h = selective_scan(x, Δ(x), B(x), C(x))  # O(n)
output = C @ h
```

Mamba achieves similar expressivity by selectively compressing context instead of explicitly attending.

## Implementation Details

### Architecture: Mamba Block

```
Input (B, L, D)
    ↓
Expand: Linear → 2D_inner (split into x and z)
    ↓
Convolution: 1D depthwise conv on x
    ↓
Activation: SiLU
    ↓
Selective SSM:
    x → Δ, B, C
    Discretize: Ā, B̄
    Scan: h_t = Ā_t h_{t-1} + B̄_t x_t
    Output: y_t = C_t h_t + D x_t
    ↓
Gating: y = y * SiLU(z)
    ↓
Contract: Linear → D
    ↓
Output (B, L, D)
```

### Selective SSM Module

```python
class SelectiveSSM(nn.Module):
    def __init__(self, d_model, d_state=16, dt_rank='auto'):
        self.d_state = d_state
        self.dt_rank = d_model // 16 if dt_rank == 'auto' else dt_rank

        # Initialize A with special initialization (HiPPO-inspired)
        A = arange(1, d_state + 1).repeat(d_model, 1)
        self.A_log = Parameter(log(A))  # Log for positivity

        # Skip connection
        self.D = Parameter(ones(d_model))

        # Project input to Δ, B, C
        self.x_proj = Linear(d_model, dt_rank + 2 * d_state)
        self.dt_proj = Linear(dt_rank, d_model)

        # Initialize dt projection bias
        dt = exp(rand(d_model) * (log(0.1) - log(0.001)) + log(0.001))
        inv_dt = dt + log(-expm1(-dt))  # Inverse initialization
        self.dt_proj.bias.data = inv_dt
```

### Selective Scan Implementation

```python
def selective_scan(u, delta, A, B, C, D, state=None):
    """
    Args:
        u: (B, L, D) input
        delta: (B, L, D) discretization step
        A: (D, N) state matrix
        B: (B, L, N) input matrix
        C: (B, L, N) output matrix
        D: (D,) skip connection
        state: (B, D, N) initial state

    Returns:
        y: (B, L, D) output
        state: (B, D, N) final state
    """
    B, L, D = u.shape
    N = A.shape[1]

    if state is None:
        state = zeros(B, D, N)

    outputs = []
    for t in range(L):
        # Discretize
        delta_t = delta[:, t, :]  # (B, D)
        deltaA = exp(delta_t.unsqueeze(-1) * A)  # (B, D, N)
        deltaB = delta_t.unsqueeze(-1) * B[:, t, :].unsqueeze(1)  # (B, D, N)

        # Update state
        state = deltaA * state + deltaB * u[:, t, :].unsqueeze(-1)

        # Output
        y_t = einsum('bdn,bn->bd', state, C[:, t, :]) + D * u[:, t, :]
        outputs.append(y_t)

    return stack(outputs, dim=1), state
```

### Hardware-Aware Optimizations

The reference implementation uses a CUDA kernel that:

1. **Scans in parallel** across batch and D dimensions
2. **Fuses** discretization, scan, and output in one kernel
3. **Recomputes** activations in backward pass (memory-time tradeoff)
4. **Uses** warp-level primitives for efficiency

Pseudo-code for optimized kernel:
```python
@cuda.jit
def selective_scan_fwd_kernel(u, delta, A, B, C, D, out, state):
    # Each thread handles one (batch, channel) pair
    b = cuda.blockIdx.x
    d = cuda.threadIdx.x

    # Load into registers/shared memory
    local_state = 0.0  # N-dimensional state per thread

    for t in range(L):
        # Load inputs
        u_t = u[b, t, d]
        delta_t = delta[b, t, d]
        B_t = B[b, t, :]
        C_t = C[b, t, :]

        # Discretize and scan (fused)
        for n in range(N):
            dA = exp(delta_t * A[d, n])
            dB = delta_t * B_t[n]
            local_state[n] = dA * local_state[n] + dB * u_t

        # Output
        y_t = sum(C_t[n] * local_state[n] for n in range(N))
        out[b, t, d] = y_t + D[d] * u_t

    # Store final state
    state[b, d, :] = local_state
```

## Code Walkthrough

See `/Users/kevinyu/Projects/Nexus/nexus/components/ssm/mamba.py`.

### Key Components

1. **SelectiveSSM** (lines 16-156): Core selective scan module
   - Manages A, B, C, Δ parameters
   - Handles discretization
   - Implements scan algorithm

2. **MambaBlock** (lines 158-259): Complete Mamba block
   - Input projection and splitting
   - 1D convolution for local context
   - Selective SSM
   - Gated output and projection

3. **Mamba2Block** (lines 261-392): Extended version with multi-head
   - Multiple SSM heads (like multi-head attention)
   - Enhanced expressivity
   - State space duality formulation

### Usage Example

```python
from nexus.components.ssm import MambaBlock

# Create Mamba block
block = MambaBlock(
    d_model=512,
    d_state=16,      # State dimension (default: 16)
    d_conv=4,        # Convolution kernel size
    expand=2,        # Expansion factor
)

# Forward pass
x = torch.randn(2, 100, 512)  # (batch, length, dim)
output, state = block(x)
# output: (2, 100, 512)
# state: (2, 1024, 16)  # 1024 = 512 * expand

# Autoregressive generation
state = None
for token in input_tokens:
    token_emb = embedding(token).unsqueeze(1)  # (1, 1, 512)
    logits, state = block(token_emb, state)
```

## Optimization Tricks

### 1. Selective Scan Recomputation

**Problem**: Storing all intermediate states for backward pass uses O(BLD N) memory.

**Solution**: Recompute states in backward pass (from stored inputs):
```python
class SelectiveScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, delta, A, B, C, D):
        y, final_state = selective_scan(u, delta, A, B, C, D)
        # Only save inputs (not intermediate states)
        ctx.save_for_backward(u, delta, A, B, C, D)
        return y

    @staticmethod
    def backward(ctx, grad_out):
        u, delta, A, B, C, D = ctx.saved_tensors
        # Recompute forward pass to get states
        states = recompute_states(u, delta, A, B, C)
        # Compute gradients
        ...
```

### 2. Parallel Scan Algorithm

For very long sequences, use parallel scan (associative scan):
```python
def parallel_selective_scan(u, delta, A, B, C, D):
    # Represent as associative operator
    # (A1, B1) ⊕ (A2, B2) = (A1*A2, A2*B1 + B2)

    # Binary tree reduction
    for level in range(log2(L)):
        # Combine pairs
        ...

    return outputs
```

### 3. Mixed Precision

Use fp16/bf16 for speed, fp32 for stability:
```python
with torch.cuda.amp.autocast():
    # SSM computation in fp16
    y = selective_scan(u, delta, A, B, C, D)

# dt projection in fp32 for numerical stability
delta = delta.float()
delta = softplus(self.dt_proj(delta))
delta = delta.half()
```

### 4. Kernel Fusion

Fuse operations to reduce memory bandwidth:
```python
# BAD: Multiple kernel launches
x = conv1d(x)
x = silu(x)
Δ, B, C = split(linear(x))
Δ = softplus(linear(Δ))

# GOOD: Fused kernel
x = fused_conv_silu(x)
Δ, B, C = fused_linear_split_softplus(x)
```

### 5. State Caching for Generation

Cache conv state for autoregressive generation:
```python
def forward_with_cache(self, x, state, conv_state):
    # Incremental convolution
    conv_state = torch.cat([conv_state, x], dim=1)
    conv_state = conv_state[:, -self.d_conv:, :]  # Keep last d_conv
    x_conv = self.conv1d(conv_state)[:, -1:, :]  # Only compute last

    # SSM with cached state
    y, state = self.ssm(x_conv, state)

    return y, state, conv_state
```

## Experiments & Results

### Language Modeling

Mamba matches or exceeds Transformers on language modeling:

| Model | Params | Pile (ppl) | Training Speed | Inference Speed |
|-------|--------|------------|----------------|-----------------|
| GPT-3 | 125M | - | 1.0x | 1.0x |
| Mamba | 130M | 10.56 | 1.2x | 5.0x |
| GPT-3 | 350M | - | 1.0x | 1.0x |
| Mamba | 370M | 8.28 | 1.3x | 5.2x |
| GPT-3 | 1.3B | - | 1.0x | 1.0x |
| Mamba | 1.4B | 7.33 | 1.4x | 5.4x |

Mamba is faster at training and significantly faster at inference, especially for long sequences.

### Scaling Laws

Mamba follows similar scaling laws to Transformers:
- Test loss improves as power law with compute: `L ~ C^(-α)`
- Works well from 125M to 2.8B parameters
- Scaling exponent α similar to GPT

### Selective Copying Task

Mamba excels at tasks requiring selective memory:

Task: Copy specific tokens marked by special indicator:
```
Input:  "The [*] cat [*] sat [*] on [*] the [*] mat [*]"
Markers: [  0     1     0     0     1     0  ]
Output:     "   cat             the        "
```

| Model | Accuracy (len=1K) | Accuracy (len=10K) |
|-------|-------------------|---------------------|
| Transformer | 95% | 23% |
| S4 | 72% | 68% |
| Mamba | 99.9% | 99.7% |

Mamba's selectivity allows it to excel where fixed SSMs fail.

### Efficiency Analysis

Memory usage (batch=16, seq_len):
- 512 tokens: Mamba 2.1 GB, Transformer 2.8 GB (1.3x less)
- 2048 tokens: Mamba 4.2 GB, Transformer 9.6 GB (2.3x less)
- 8192 tokens: Mamba 12.1 GB, Transformer 38.4 GB (3.2x less)

Throughput (tokens/sec, A100 GPU):
- 512 tokens: Mamba 15K, Transformer 14K (1.1x faster)
- 2048 tokens: Mamba 16K, Transformer 11K (1.5x faster)
- 8192 tokens: Mamba 18K, Transformer 5K (3.6x faster)

## Common Pitfalls

### 1. Forgetting the Convolution Layer

**Problem**: Skipping conv1d layer:
```python
# BAD
x = self.in_proj(input)
y = self.ssm(x)
```

**Solution**: Convolution is crucial for local context:
```python
# GOOD
x = self.in_proj(input)
x = self.conv1d(x)  # Local receptive field
x = silu(x)
y = self.ssm(x)
```

### 2. Wrong Δ Initialization

**Problem**: Not initializing dt projection bias:
```python
self.dt_proj = Linear(dt_rank, d_model)  # Random bias
```

**Solution**: Proper inverse initialization:
```python
dt = exp(rand(d_model) * (log(0.1) - log(0.001)) + log(0.001))
inv_dt = dt + log(-expm1(-dt))
self.dt_proj.bias.data = inv_dt
```

This ensures Δ starts in a reasonable range.

### 3. Not Using Selective Parameters

**Problem**: Making B, C fixed (like S4):
```python
self.B = Parameter(randn(d_model, d_state))  # Fixed
self.C = Parameter(randn(d_model, d_state))  # Fixed
```

**Solution**: Project from input:
```python
Δ, B, C = self.x_proj(x).split([dt_rank, d_state, d_state], dim=-1)
```

This is the core of Mamba's selectivity.

### 4. State Shape Confusion

**Problem**: Wrong state shape for recurrence:
```python
state = zeros(batch, d_state)  # Too small!
```

**Solution**: State is per-channel:
```python
state = zeros(batch, d_inner, d_state)
# d_inner = expand * d_model
```

### 5. Causal Convolution Issues

**Problem**: Non-causal convolution sees future:
```python
self.conv = Conv1d(d_inner, d_inner, kernel_size=4)  # Bad padding
```

**Solution**: Proper causal padding:
```python
self.conv = Conv1d(d_inner, d_inner, kernel_size=4, padding=3)
x_conv = self.conv(x)[:, :, :seq_len]  # Truncate to original length
```

### 6. Numerical Instability

**Problem**: Overflow in exp(ΔA) for large Δ.

**Solution**: Clip Δ:
```python
Δ = softplus(self.dt_proj(Δ))
Δ = Δ.clamp(max=1.0)  # Prevent overflow
```

## References

### Primary Papers

1. **Mamba (2023)**
   - Gu & Dao. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
   - https://arxiv.org/abs/2312.00752
   - Main Mamba paper

2. **Hungry Hungry Hippos (H3) (2023)**
   - Fu, Dao, Saab, Thomas, Rudra, Ré. "Hungry Hungry Hippos: Towards Language Modeling with State Space Models"
   - https://arxiv.org/abs/2212.14052
   - Precursor combining SSMs with attention

3. **Hyena (ICML 2023)**
   - Poli, Massaroli, Nguyen, Fu, Dao, Baccus, Bengio, Ermon, Ré. "Hyena Hierarchy: Towards Larger Convolutional Language Models"
   - https://arxiv.org/abs/2302.10866
   - Related long-convolution approach

### Implementation Resources

4. **Mamba Official Implementation**
   - https://github.com/state-spaces/mamba
   - Reference CUDA kernels

5. **Mamba-Minimal**
   - https://github.com/johnma2006/mamba-minimal
   - Simplified PyTorch implementation

### Applications

6. **MambaByte (2024)**
   - Wang et al. "MambaByte: Token-free Selective State Space Model"
   - Byte-level modeling with Mamba

7. **Vision Mamba (2024)**
   - Zhu et al. "Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model"
   - Mamba for computer vision

---

*For implementation reference, see `/Users/kevinyu/Projects/Nexus/nexus/components/ssm/mamba.py`*
