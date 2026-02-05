# Jamba: Hybrid Transformer-Mamba-MoE Architecture

## Overview

Jamba is AI21 Labs' production-scale hybrid architecture that interleaves three types of layers: Transformer attention, Mamba SSM, and Mixture-of-Experts (MoE). It achieves strong performance while maintaining efficiency through strategic layer composition and conditional computation.

**Key Innovation**: Triple-hybrid design combining attention (precision), Mamba (efficiency), and MoE (capacity scaling) in a configurable pattern.

**Paper**: [Jamba: A Hybrid Transformer-Mamba Language Model](https://arxiv.org/abs/2403.19887) (AI21 Labs, 2024)

## Architecture Overview

### Layer Composition

Jamba uses a configurable pattern (example: `AMMAMMAMMAMM`):
- **A**: Attention layer (grouped-query, RoPE)
- **M**: Mamba SSM layer (selective state-space)
- **MoE**: Applied periodically (every 2nd layer)

```
Layer 0:  Attention + Dense FFN
Layer 1:  Mamba + MoE FFN  ←─── MoE
Layer 2:  Mamba + Dense FFN
Layer 3:  Mamba + MoE FFN  ←─── MoE
Layer 4:  Attention + Dense FFN
...
```

### Key Components

1. **Grouped-Query Attention (GQA)**: Efficient attention with reduced KV heads
2. **Mamba SSM**: Selective state-space model for O(N) sequence processing
3. **MoE**: Top-k expert routing for capacity scaling
4. **RMSNorm**: Efficient normalization

## Implementation Details

### Typical Configuration

```python
model = JambaModel(
    d_model=4096,
    num_layers=32,
    num_heads=32,
    num_kv_heads=8,        # GQA: 4x KV cache reduction
    mamba_d_state=16,
    num_experts=8,
    top_k=2,               # Route to top-2 experts
    layer_pattern='AMMAMM', # Repeats to fill layers
    moe_every_n=2          # MoE every 2nd layer
)
```

### Layer Pattern Design

**Attention ratio**: Typically 1:3 to 1:7 (attention:mamba)
```python
# Conservative (more attention)
pattern = 'AMMAMM'  # 1:2 ratio

# Balanced
pattern = 'AMMAMMAMM'  # 1:3 ratio

# Aggressive (more efficient)
pattern = 'AMMAMMAMMAMM'  # 1:4 ratio
```

## Mathematical Formulation

### Mamba Layer

Selective SSM with input-dependent dynamics:

```
Input-dependent parameters:
    Δ[t] = softplus(W_Δ · x[t])      # Step size
    B[t] = W_B · x[t]                # Input matrix
    C[t] = W_C · x[t]                # Output matrix

Discretization:
    A_d[t] = exp(Δ[t] ⊙ A)           # Discrete A
    B_d[t] = Δ[t] ⊙ B[t]             # Discrete B

State update:
    h[t] = A_d[t] ⊙ h[t-1] + B_d[t] ⊙ (W_x · x[t])
    y[t] = C[t]^T h[t] + D ⊙ x[t]
```

### MoE Routing

Top-k gated routing:

```
Router logits:
    g = W_router · x                  # (vocab_size, num_experts)

Top-k selection:
    indices, weights = topk(softmax(g), k=2)

Expert computation:
    y = Σ_{i in topk} weight_i · Expert_i(x)

Load balancing loss:
    L_aux = α · num_experts · Σ_e (f_e · P_e)
    where f_e = fraction of tokens to expert e
          P_e = average routing probability to e
```

## Optimization Tricks

### 1. MoE Expert Balancing

```python
# Add auxiliary loss to prevent expert collapse
total_loss = language_model_loss + 0.01 * aux_loss

# Monitor expert utilization
expert_counts = router_decisions.bincount()
print(f"Expert usage: {expert_counts}")  # Should be roughly uniform
```

### 2. Efficient KV Cache

```python
# Only attention layers need KV cache
kv_caches = {
    layer_idx: (k, v) for layer_idx in attention_layer_indices
}

# Mamba layers use O(d_state) state
mamba_states = [
    torch.zeros(batch, d_inner, d_state) for _ in mamba_layers
]

# Total memory << full transformer
```

### 3. Selective Expert Computation

```python
# Only compute top-k experts per token
mask = (indices == expert_idx)
if mask.any():
    expert_input = x_flat[mask]
    expert_output = experts[expert_idx](expert_input)
    output[mask] += weight[mask] * expert_output
```

## Performance Characteristics

**Efficiency vs Capability:**
- **Training**: 1.5-2x faster than pure transformers
- **Inference**: 3-5x faster (depends on attention ratio)
- **KV Cache**: ~5-8x smaller (GQA + sparse attention)
- **Capacity**: 2-4x increase (via MoE)

**Quality:**
- Matches transformer perplexity
- Strong on retrieval tasks (attention layers)
- Efficient on long-range tasks (Mamba layers)

## Common Pitfalls

### 1. MoE Load Imbalance

```python
# ❌ BAD: Ignoring load balancing
# Experts collapse to using only 1-2 experts

# ✅ GOOD: Monitor and tune aux loss
aux_loss_coeff = 0.01  # Start here
# Increase if experts collapse, decrease if quality suffers
```

### 2. Attention Ratio Too Low

```python
# ❌ BAD: Too few attention layers
pattern = 'MMMMMMMA'  # 7:1, may hurt quality

# ✅ GOOD: At least 1:4 for most tasks
pattern = 'MMMAMMMA'  # 3:1, good balance
```

### 3. State Management

```python
# ❌ BAD: Mixing up Mamba states and attention caches
states = [None] * num_layers  # Confused!

# ✅ GOOD: Separate tracking
layer_states = [
    {'state': None, 'kv_cache': None} for _ in range(num_layers)
]
```

## References

Lieber, O., et al. (2024). **Jamba: A Hybrid Transformer-Mamba Language Model**. AI21 Labs. arXiv:2403.19887.

**Related**: Mamba (Gu & Dao, 2023), Switch Transformer (Fedus et al., 2021)

**Code**: `nexus/models/hybrid/jamba.py`
