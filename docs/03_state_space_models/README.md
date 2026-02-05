# State Space Models (SSMs): A Comprehensive Guide

This directory contains comprehensive documentation for State Space Models (SSMs), a powerful class of sequence models that achieve linear complexity while maintaining strong performance on long-range dependencies.

## Table of Contents

1. [Introduction](#introduction)
2. [SSM Evolution Timeline](#ssm-evolution-timeline)
3. [Model Comparison](#model-comparison)
4. [When to Use Which SSM](#when-to-use-which-ssm)
5. [Documentation Index](#documentation-index)

## Introduction

State Space Models provide an alternative to Transformers for sequence modeling, offering linear-time complexity O(n) instead of quadratic O(n²). SSMs model sequences through continuous-time state space equations:

```
dx/dt = Ax + Bu
y = Cx + Du
```

where:
- `x` is the hidden state (continuous-time)
- `u` is the input
- `y` is the output
- `A`, `B`, `C`, `D` are learnable parameters

The key insight is that these continuous equations can be discretized and computed efficiently in two modes:
1. **Convolution mode** (parallel, for training): O(n log n) via FFT
2. **Recurrence mode** (sequential, for inference): O(1) per step

## SSM Evolution Timeline

### First Generation: Structured SSMs (2022)
- **S4** (Efficiently Modeling Long Sequences)
  - Introduced HiPPO initialization for long-range dependencies
  - DPLR (Diagonal Plus Low-Rank) parameterization
  - Dual computation modes: convolution (training) and recurrence (inference)

- **S4D** (Diagonal State Spaces)
  - Simplified S4 by restricting A to diagonal
  - Easier to implement, similar performance
  - Reduced memory and compute requirements

### Second Generation: Simplified SSMs (2023)
- **S5** (Simplified State Space Layers)
  - MIMO (multi-input multi-output) instead of multiple SISO systems
  - Parallel associative scan replaces frequency-domain computation
  - Complex diagonal parameterization

- **Liquid-S4** (Input-Dependent State Spaces)
  - Makes state transitions input-dependent
  - Liquid time constants adapt to input complexity
  - Enhanced expressivity through dynamic recurrence

### Third Generation: Selective SSMs (2023-2024)
- **Mamba** (Linear-Time Sequence Modeling)
  - Input-dependent parameters (selective SSM)
  - Selective scan algorithm with hardware-aware implementation
  - Combines convolution for local context with SSM for global patterns

- **Mamba-2** (State Space Duality)
  - Introduces SSD (State Space Duality) framework
  - Shows SSMs are equivalent to structured masked attention
  - Multi-head structure, 2-8x faster than Mamba
  - Larger state dimension (128 vs 16)

### Fourth Generation: Linear Attention Variants (2023-2024)
- **RetNet** (Retentive Networks)
  - Retention mechanism with exponential decay
  - Multi-scale retention with different decay rates per head
  - Parallel and recurrent modes with O(1) inference

- **HGRN** (Hierarchically Gated RNN)
  - Hierarchical gating with input/forget/output gates
  - Lower-bound constraint on forget gate
  - Parallel mode via log-space cumsum trick

- **DeltaNet** (Gated DeltaNet)
  - Delta rule for selective memory updates
  - Used in Qwen3-Next, Kimi Linear
  - Learning rate per update step

### Fifth Generation: RWKV Family (2023-2025)
- **RWKV-6 (Finch)**
  - Matrix-valued recurrent states (key-value associations)
  - Data-dependent decay (adaptive forgetting)
  - Token shift mechanism for local context
  - WKV (weighted key-value) recurrence

- **RWKV-7 (Goose)**
  - Generalized delta rule with error correction
  - Vector-valued gating (finer control than scalar decay)
  - Improved stability and expressivity
  - Denominator state for normalized readout

## Model Comparison

| Model | Year | State Complexity | Key Innovation | Best For |
|-------|------|-----------------|----------------|----------|
| S4 | 2022 | O(N²) | HiPPO + DPLR | Long-range dependencies |
| S4D | 2022 | O(N) | Diagonal A matrix | Efficient long-range |
| S5 | 2023 | O(N) | MIMO + parallel scan | Simplified implementation |
| Liquid-S4 | 2023 | O(N) | Input-dependent dynamics | Adaptive sequences |
| Mamba | 2023 | O(N) per channel | Selective SSM | General-purpose LLM |
| Mamba-2 | 2024 | O(N) per head | State space duality | Fast training/inference |
| RetNet | 2023 | O(d²) per head | Multi-scale retention | Language modeling |
| HGRN | 2023 | O(d) | Hierarchical gates | Efficient recurrence |
| DeltaNet | 2024 | O(d²) per head | Delta rule memory | Associative patterns |
| RWKV-6 | 2024 | O(d²) per head | Matrix-valued states | LLM with RNN efficiency |
| RWKV-7 | 2025 | O(d²) per head | Generalized delta rule | Advanced LLM |

## When to Use Which SSM

### Choose S4/S4D when:
- You need strong long-range dependency modeling (>10k tokens)
- Working with continuous signals or time series
- You want theoretical guarantees on memory
- Bidirectional processing is acceptable

### Choose S5 when:
- You want simpler implementation than S4
- You need good long-range modeling with less complexity
- You want to use parallel scan for training

### Choose Mamba when:
- Building general-purpose language models
- You need competitive performance with Transformers
- Memory efficiency is important
- You want selective state space modeling

### Choose Mamba-2 when:
- Training speed is critical
- You have modern GPUs with tensor cores
- You want the best SSM performance
- You're scaling to large models (>1B params)

### Choose RetNet when:
- Building language models with strong recency bias
- You want multi-scale temporal patterns
- Autoregressive generation speed matters
- You need O(1) inference per token

### Choose HGRN when:
- You want simpler gating than full LSTM
- Memory efficiency is critical
- You need fast autoregressive generation
- You're working with moderate sequence lengths

### Choose DeltaNet when:
- You need associative memory patterns
- Working on retrieval or reasoning tasks
- You want selective memory updates
- Following Qwen/Kimi architecture

### Choose RWKV-6 when:
- Building large language models
- You want true O(1) inference complexity
- You need strong performance with RNN efficiency
- You're targeting edge deployment

### Choose RWKV-7 when:
- You need state-of-the-art SSM performance
- Error correction is important for your task
- You want the most advanced RWKV variant
- Building research models

## Documentation Index

### Core SSM Foundations
1. [S4 - Structured State Spaces](./s4.md)
   - HiPPO initialization
   - DPLR parameterization
   - Convolution and recurrence modes

2. [S4D - Diagonal State Spaces](./s4d.md)
   - Diagonal simplification
   - Efficient computation
   - Practical implementation

3. [S5 - Simplified State Space Layers](./s5.md)
   - MIMO formulation
   - Parallel associative scan
   - Complex diagonal parameterization

4. [Liquid-S4 - Input-Dependent SSMs](./liquid_s4.md)
   - Liquid time constants
   - Input-modulated dynamics
   - Adaptive sequence modeling

### Selective State Space Models
5. [Mamba - Linear-Time Sequence Modeling](./mamba.md)
   - Selective state space mechanism
   - Hardware-aware selective scan
   - Architecture and implementation

6. [Mamba-2 - State Space Duality](./mamba2.md)
   - SSD framework
   - Multi-head SSM structure
   - Efficient matrix multiplication

### Linear Attention Variants
7. [RetNet - Retentive Networks](./retnet.md)
   - Multi-scale retention
   - Exponential decay mechanism
   - Parallel and recurrent modes

8. [HGRN - Hierarchically Gated RNN](./hgrn.md)
   - Hierarchical gating structure
   - Forget gate with lower bound
   - Parallel cumsum trick

9. [DeltaNet - Gated Delta Rule](./deltanet.md)
   - Delta rule memory updates
   - Learning rate modulation
   - Associative memory patterns

### RWKV Family
10. [RWKV-6 (Finch) - Matrix-Valued States](./rwkv6.md)
    - Matrix-valued recurrence
    - Data-dependent decay
    - Token shift mechanism
    - WKV algorithm

11. [RWKV-7 (Goose) - Generalized Delta Rule](./rwkv7.md)
    - Generalized delta rule
    - Vector-valued gating
    - Enhanced stability
    - Advanced implementation

### Foundational Concepts
12. [Linear RNN - Base Architecture](./linear_rnn.md)
    - Common infrastructure
    - Short convolution
    - Base recurrence patterns

## Implementation Notes

All SSM implementations in Nexus follow consistent patterns:

1. **Dual Mode Support**: All models support both parallel (training) and recurrent (inference) modes
2. **State Management**: Clean state initialization and propagation for autoregressive generation
3. **Hardware-Aware**: Implementations consider GPU/TPU efficiency
4. **Modular Design**: Easy to swap SSMs in transformer-style architectures

## References

Each individual documentation file contains detailed references to the original papers. Key foundational papers:

- **S4**: Gu et al., "Efficiently Modeling Long Sequences with Structured State Spaces", ICLR 2022
- **Mamba**: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", 2023
- **Mamba-2**: Dao & Gu, "Transformers are SSMs", 2024
- **RWKV-6**: Peng et al., "Eagle and Finch: RWKV with Matrix-Valued States", 2024
- **RWKV-7**: Peng et al., "RWKV-7: Generalized Delta Rule", 2025

## Getting Started

To use SSMs in your models:

```python
from nexus.components.ssm import Mamba2Block, S4Block, RetNet, RWKV6Block

# Example: Using Mamba-2
block = Mamba2Block(
    d_model=512,
    d_state=128,
    num_heads=8,
    expand=2
)

x = torch.randn(2, 100, 512)  # (batch, seq_len, d_model)
output, state = block(x)
```

See individual documentation files for detailed usage examples and best practices.
