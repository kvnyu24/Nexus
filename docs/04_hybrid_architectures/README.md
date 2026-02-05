# Hybrid Architectures: The Attention-SSM Design Space

## Overview

Hybrid architectures combine attention mechanisms with state-space models (SSMs) or recurrent networks to achieve optimal trade-offs between quality, efficiency, and scalability. This document provides a comprehensive overview of the hybrid architecture design space and guides you through the various approaches explored in modern language models.

## Table of Contents

1. [Motivation](#motivation)
2. [The Efficiency-Quality Tradeoff](#the-efficiency-quality-tradeoff)
3. [Architecture Design Dimensions](#architecture-design-dimensions)
4. [Hybrid Architecture Taxonomy](#hybrid-architecture-taxonomy)
5. [Implementation Strategies](#implementation-strategies)
6. [When to Use Each Architecture](#when-to-use-each-architecture)
7. [Getting Started](#getting-started)

## Motivation

### Why Hybrid?

Pure transformer models face fundamental scalability challenges:

- **Quadratic Memory**: O(N²) KV cache for attention grows prohibitively large for long contexts
- **Inference Cost**: Each token generation requires attending to all previous tokens
- **Training Efficiency**: Self-attention is the computational bottleneck at scale

Pure SSM/recurrent models offer efficiency but face quality limitations:

- **Limited Recall**: Difficulty with precise token-level retrieval
- **Associative Memory**: Challenges with tasks requiring exact matches
- **Reasoning**: May struggle with multi-hop reasoning requiring precise information flow

**Hybrid architectures combine the best of both worlds:**
- Use efficient SSMs/recurrence for bulk sequence processing
- Insert strategic attention for tasks requiring precision
- Achieve transformer-level quality with SSM-level efficiency

## The Efficiency-Quality Tradeoff

```
Pure Attention ←————————— Hybrid Spectrum ——————————→ Pure SSM/Recurrence
(Transformers)                                         (Mamba, RWKV)

High Quality          Balanced Tradeoffs           High Efficiency
High Memory           Configurable Design          Low Memory
Quadratic KV Cache    Flexible Patterns           O(1) or No KV Cache
```

### Key Metrics

| Metric | Pure Attention | Hybrid | Pure SSM |
|--------|---------------|--------|----------|
| **Training Speed** | Baseline | 1-2x faster | 2-3x faster |
| **Inference Speed** | Baseline | 2-5x faster | 5-10x faster |
| **KV Cache Size** | O(N²) | O(N) to O(N²)* | O(1) or None |
| **Long Context** | Limited | Strong | Excellent |
| **Precise Recall** | Excellent | Good | Limited |
| **Implementation** | Mature | Emerging | Cutting-edge |

*Depends on attention layer frequency

## Architecture Design Dimensions

When designing a hybrid architecture, you make choices along several key dimensions:

### 1. Layer Composition

**How do you combine attention and SSM layers?**

- **Interleaved**: Alternating pattern (e.g., `AMAMAMAM` where A=attention, M=SSM)
  - Example: Jamba, StripedHyena
  - Pros: Regular pattern, easy to reason about
  - Cons: May not match task requirements

- **Ratio-based**: Fixed ratio (e.g., 1 attention per 4 SSM layers)
  - Example: Griffin (7:1), RecurrentGemma (2:1)
  - Pros: Controllable compute budget
  - Cons: Less flexible positioning

- **Strategic**: Attention at specific critical layers
  - Example: GoldFinch (only at middle and end)
  - Pros: Extreme efficiency
  - Cons: Requires careful layer selection

- **Within-block**: Both mechanisms in each block
  - Example: Griffin (recurrence + local attention per block)
  - Pros: Every layer gets both capabilities
  - Cons: Higher per-layer cost

### 2. Attention Scope

**What kind of attention do you use?**

- **Full Attention**: Standard causal attention
  - Used in: Jamba, Zamba
  - KV cache: O(N) per attention layer

- **Sliding Window**: Local attention within fixed window
  - Used in: Griffin, Based, RecurrentGemma
  - KV cache: O(W) where W is window size

- **Sparse Patterns**: Structured attention patterns
  - Used in: GoldFinch (strategic positions)
  - KV cache: Minimal

### 3. SSM/Recurrence Type

**What efficient mechanism handles long-range dependencies?**

- **Selective SSM (Mamba)**: Input-dependent state transitions
  - Used in: Jamba, Zamba
  - Strengths: Strong context modeling, efficient CUDA kernels

- **Gated Linear Recurrence (RGLRU)**: Diagonal gated RNN
  - Used in: Griffin, Hawk, RecurrentGemma
  - Strengths: Simple, parallelizable via associative scan

- **Long Convolution (Hyena)**: FFT-based implicit convolutions
  - Used in: Hyena, StripedHyena
  - Strengths: Sub-quadratic, data-controlled filters

- **Linear Attention**: Kernelized attention approximation
  - Used in: Based
  - Strengths: Exact linear complexity, recurrent formulation

- **RWKV Time Mixing**: WKV mechanism with matrix-valued states
  - Used in: GoldFinch
  - Strengths: Minimal memory, strong recurrent dynamics

### 4. Parameter Sharing

**Do layers share parameters?**

- **Independent**: Each layer has unique parameters
  - Used in: Most architectures
  - Pros: Maximum expressivity
  - Cons: Higher parameter count

- **Shared Attention**: Reuse attention module across layers
  - Used in: Zamba
  - Pros: Reduced parameters, amortized attention cost
  - Cons: Less layer-specific adaptation

### 5. Additional Mechanisms

**What enhancements are included?**

- **Mixture-of-Experts (MoE)**: Conditional computation
  - Used in: Jamba
  - Effect: 2-4x capacity with modest compute increase

- **Multi-Query/Grouped-Query Attention**: Reduced KV heads
  - Used in: Griffin, Jamba
  - Effect: Smaller KV cache (4-8x reduction)

## Hybrid Architecture Taxonomy

### By Primary Design Philosophy

```
┌─────────────────────────────────────────────────────┐
│           HYBRID ARCHITECTURE TAXONOMY              │
└─────────────────────────────────────────────────────┘

1. SSM-DOMINANT HYBRIDS
   ├── Jamba: Mamba backbone + sparse attention + MoE
   ├── Zamba: Mamba backbone + shared attention
   ├── Hawk: Pure RGLRU (reference point)
   └── GoldFinch: RWKV backbone + strategic attention

2. BALANCED HYBRIDS
   ├── Griffin: RGLRU + local MQA in each block
   ├── RecurrentGemma: RGLRU + local attention (3:1 ratio)
   └── Based: Linear attention + sliding window

3. CONVOLUTION-BASED HYBRIDS
   ├── Hyena: Pure long convolution (reference)
   └── StripedHyena: Hyena + attention stripes

```

### By Efficiency Profile

```
Efficiency Tier 1 (Extreme): 5-10x inference speedup
├── Hawk (pure recurrence)
├── GoldFinch (756-2550x KV cache compression)
└── Hyena (pure convolution)

Efficiency Tier 2 (High): 3-5x inference speedup
├── Zamba (shared attention)
├── RecurrentGemma (mostly recurrence)
└── Based (linear attention)

Efficiency Tier 3 (Moderate): 2-3x inference speedup
├── Jamba (attention + Mamba + MoE)
├── Griffin (recurrence + local attention)
└── StripedHyena (mostly Hyena)
```

## Implementation Strategies

### Pattern 1: Interleaved Layers (Simple)

```python
# Example: Alternating attention and SSM layers
layers = []
for i in range(num_layers):
    if i % 4 == 0:  # Every 4th layer is attention
        layers.append(AttentionLayer(...))
    else:
        layers.append(SSMLayer(...))
```

**Use when:**
- You want a simple, predictable pattern
- You're prototyping or experimenting
- You want to easily adjust the ratio

**Examples:** StripedHyena, Zamba

### Pattern 2: Within-Block Composition (Griffin-style)

```python
# Example: Both mechanisms in each block
class HybridBlock(nn.Module):
    def forward(self, x, state, kv_cache):
        # First: efficient sequence modeling
        x, state = self.ssm_layer(x, state)

        # Second: precise local refinement
        x, kv_cache = self.local_attention(x, kv_cache)

        # Third: feedforward
        x = self.ffn(x)

        return x, state, kv_cache
```

**Use when:**
- Every position needs both global and local context
- You can afford the additional per-block cost
- You want consistent hybrid behavior

**Examples:** Griffin

### Pattern 3: Strategic Placement (GoldFinch-style)

```python
# Example: Attention only at critical positions
attention_layers = {11, 23}  # Middle and end of 24-layer model

layers = []
for i in range(num_layers):
    if i in attention_layers:
        layers.append(AttentionLayer(...))
    else:
        layers.append(RWKVLayer(...))
```

**Use when:**
- Extreme efficiency is paramount
- You've profiled which layers need attention
- You're willing to tune layer positions

**Examples:** GoldFinch

### Pattern 4: Shared Parameters (Zamba-style)

```python
# Example: Reuse attention module across layers
shared_attention = AttentionLayer(...)

layers = []
for i in range(num_layers):
    if i % 6 == 0:  # Every 6th layer
        layers.append(HybridBlock(shared_attention=shared_attention))
    else:
        layers.append(HybridBlock(mamba_only=True))
```

**Use when:**
- Parameter budget is constrained
- Attention is expensive to replicate
- Layers should use consistent attention patterns

**Examples:** Zamba

## When to Use Each Architecture

### Decision Tree

```
START: What's your primary constraint?

├─ Memory/Efficiency Critical?
│  ├─ Yes → Need some attention?
│  │  ├─ Yes → GoldFinch or Hawk+strategic attention
│  │  └─ No → Hawk (pure recurrence)
│  │
│  └─ No → Need long context (>32K)?
│     ├─ Yes → StripedHyena or Griffin
│     └─ No → Based or RecurrentGemma

├─ Quality/Capability Critical?
│  ├─ Production scale?
│  │  ├─ Yes → Jamba (with MoE)
│  │  └─ No → Griffin or RecurrentGemma
│  │
│  └─ Research/Prototyping?
│     └─ Based or Griffin

└─ Balanced Requirements?
   ├─ Need MoE scaling? → Jamba
   ├─ Open source priority? → RecurrentGemma
   └─ Maximum flexibility? → Griffin
```

### Architecture Selection Guide

| Architecture | Best For | Avoid When |
|--------------|----------|------------|
| **Griffin** | General-purpose, balanced needs | Pure efficiency required |
| **Hyena** | Long sequences, convolution-friendly | Need precise recall |
| **Based** | Extreme throughput, research | Production deployment (newer) |
| **Jamba** | Production, need MoE scaling | Memory-constrained |
| **StripedHyena** | 128K+ context, mostly efficient | Need full attention |
| **Zamba** | Parameter budget constrained | Need layer-specific attention |
| **GoldFinch** | Ultra-long context, KV cache critical | Need frequent attention |
| **RecurrentGemma** | Open source, reproducibility | Cutting-edge research |
| **Hawk** | Maximum efficiency benchmark | Need any attention |

## Getting Started

### 1. Understand Your Requirements

Ask yourself:
- What's my context length? (<8K, 8-32K, 32-128K, >128K)
- What's my inference budget? (latency, memory, throughput)
- What's my quality target? (match GPT-4, match GPT-3.5, beat baseline)
- What's my use case? (general text, code, long documents, chat)

### 2. Choose a Starting Point

**For most use cases:** Start with Griffin or RecurrentGemma
- Well-documented
- Good balance of efficiency and quality
- Proven in production/research

**For research:** Try Based or experiment with custom patterns
- Cutting-edge ideas
- Room for innovation
- Easier to modify

**For production:** Consider Jamba or Zamba
- Production-tested
- MoE scaling available
- Strong efficiency

### 3. Experiment with Patterns

```python
# Start simple
model = GriffinModel(
    d_model=512,
    num_layers=12,
    num_heads=8,
    window_size=128,
    hawk_only=False  # Try True to compare
)

# Profile performance
from nexus.utils.profiling import profile_model
metrics = profile_model(model, seq_lens=[1024, 4096, 16384])

# Adjust based on results
```

### 4. Read Architecture-Specific Docs

Each architecture has detailed documentation:

- [Griffin](griffin.md) - Balanced hybrid with RGLRU + local MQA
- [Hyena](hyena.md) - Long convolution with implicit filters
- [Based](based.md) - Linear attention with extreme throughput
- [Jamba](jamba.md) - Production hybrid with Mamba + MoE
- [StripedHyena](striped_hyena.md) - Alternating Hyena-attention for 128K context
- [Zamba](zamba.md) - Mamba with shared attention
- [GoldFinch](goldfinch.md) - RWKV hybrid with extreme KV compression
- [RecurrentGemma](recurrent_gemma.md) - Open Griffin-based LM
- [Hawk](hawk.md) - Pure recurrence (Griffin without attention)

## Research Directions

Current open questions in hybrid architectures:

1. **Optimal Layer Ratios**: How many attention layers do you really need?
2. **Layer Positioning**: Where should attention layers go for maximum impact?
3. **Dynamic Selection**: Can we learn which tokens need attention?
4. **Training Stability**: How to train hybrids as stably as transformers?
5. **Scaling Laws**: Do hybrid scaling laws differ from transformer laws?
6. **Task Transfer**: Which hybrid patterns transfer best across tasks?

## References

1. De et al., "Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models", 2024
2. Poli et al., "Hyena Hierarchy: Towards Larger Convolutional Language Models", 2023
3. Arora et al., "Simple Linear Attention Language Models Balance the Recall-Throughput Tradeoff", 2024
4. Lieber et al., "Jamba: A Hybrid Transformer-Mamba Language Model", 2024
5. Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2023

## Contributing

See individual architecture documents for implementation details, pitfalls, and optimization techniques. The reference implementations are in `nexus/models/hybrid/`.
