# Hybrid Architecture Documentation Index

## Quick Reference

| Architecture | Type | Complexity | Best For | Doc | Code |
|--------------|------|------------|----------|-----|------|
| **Griffin** | RGLRU + Local Attn | Moderate | General purpose | [📖](griffin.md) | [💻](../../nexus/models/hybrid/griffin.py) |
| **Hyena** | Long Convolution | Moderate | Long sequences | [📖](hyena.md) | [💻](../../nexus/models/hybrid/hyena.py) |
| **Based** | Linear Attn + Window | Simple | Extreme throughput | [📖](based.md) | [💻](../../nexus/models/hybrid/based.py) |
| **Jamba** | Mamba + Attn + MoE | Complex | Production scale | [📖](jamba.md) | [💻](../../nexus/models/hybrid/jamba.py) |
| **StripedHyena** | Hyena + Attn stripes | Moderate | 128K+ context | [📖](striped_hyena.md) | [💻](../../nexus/models/hybrid/striped_hyena.py) |
| **Zamba** | Mamba + Shared Attn | Moderate | Parameter efficient | [📖](zamba.md) | [💻](../../nexus/models/hybrid/zamba.py) |
| **GoldFinch** | RWKV + Strategic Attn | Simple | Extreme KV compression | [📖](goldfinch.md) | [💻](../../nexus/models/hybrid/goldfinch.py) |
| **RecurrentGemma** | RGLRU + Local Attn | Moderate | Open source | [📖](recurrent_gemma.md) | [💻](../../nexus/models/hybrid/recurrent_gemma.py) |
| **Hawk** | Pure RGLRU | Simple | Maximum efficiency | [📖](hawk.md) | [💻](../../nexus/models/hybrid/hawk.py) |

## By Use Case

### 🚀 Maximum Efficiency
1. [Hawk](hawk.md) - Pure recurrence, 5x speedup, O(1) memory
2. [GoldFinch](goldfinch.md) - 750x+ KV cache compression
3. [Based](based.md) - 24x inference throughput

### 🎯 Quality-Efficiency Balance
1. [Griffin](griffin.md) - Balanced hybrid, 2.5x speedup
2. [RecurrentGemma](recurrent_gemma.md) - Open source Griffin variant
3. [Zamba](zamba.md) - Parameter-efficient with shared attention

### 📏 Long Context (128K+ tokens)
1. [StripedHyena](striped_hyena.md) - Designed for 128K context
2. [Hyena](hyena.md) - Sub-quadratic convolutions
3. [GoldFinch](goldfinch.md) - RWKV-based, ultra-long support

### 🏭 Production Deployment
1. [Jamba](jamba.md) - Full-featured with MoE
2. [RecurrentGemma](recurrent_gemma.md) - Google's open model
3. [Griffin](griffin.md) - Well-tested architecture

### 🔬 Research / Cutting-Edge
1. [Based](based.md) - Linear attention with Taylor expansion
2. [Hyena](hyena.md) - Implicit convolution filters
3. [GoldFinch](goldfinch.md) - Strategic attention placement

## Documentation Structure

Each architecture document includes:

1. **Overview & Motivation** - Why this architecture? What problem does it solve?
2. **Theoretical Background** - Core concepts and mathematical foundations
3. **Mathematical Formulation** - Detailed equations and operations
4. **High-Level Intuition** - Diagrams and conceptual understanding
5. **Implementation Details** - Hyperparameters, layer patterns, configurations
6. **Code Walkthrough** - Annotated implementation details
7. **Optimization Tricks** - KV cache, throughput, memory optimizations
8. **Experiments & Results** - Performance benchmarks and comparisons
9. **Common Pitfalls** - What to avoid, debugging tips
10. **References** - Papers, code, related work

## Getting Started

**New to hybrid architectures?** Start here:
1. Read the [Design Space Overview](README.md)
2. Try [Griffin](griffin.md) - best balance for learning
3. Experiment with code in `nexus/models/hybrid/`

**Have specific requirements?** Use the decision tree in [README.md](README.md)

**Want to implement your own?** Study [Griffin](griffin.md) and [Hyena](hyena.md) for different paradigms

## Comparison Charts

### Efficiency Spectrum
```
Pure Attention ←─────────── Hybrids ──────────→ Pure SSM/Recurrence
(Transformer)                                   (Mamba, RWKV)

Quality:  ████████░░  ███████░░░  ██████░░░░  ███████░░░  █████░░░░░
Speed:    ██░░░░░░░░  ████░░░░░░  ██████░░░░  ████████░░  ██████████
Memory:   ██░░░░░░░░  █████░░░░░  ███████░░░  █████████░  ██████████
          
Examples: GPT-4       Griffin     Jamba       Based       Hawk
                      RecurGemma  StripedHyena            GoldFinch
```

### Training vs Inference Tradeoff
```
              ┃ Training  │ Inference │ Quality │ Memory
──────────────╋───────────┼───────────┼─────────┼────────
Griffin       ┃    ★★★    │   ★★★★    │  ★★★★   │  ★★★★
Hyena         ┃    ★★★★   │   ★★★     │  ★★★    │  ★★★★
Based         ┃    ★★★    │   ★★★★★   │  ★★★★   │  ★★★★
Jamba         ┃    ★★★★   │   ★★★     │  ★★★★★  │  ★★★
StripedHyena  ┃    ★★★★   │   ★★★     │  ★★★    │  ★★★★
Zamba         ┃    ★★★★   │   ★★★     │  ★★★★   │  ★★★★★
GoldFinch     ┃    ★★★    │   ★★★★★   │  ★★★    │  ★★★★★
RecurGemma    ┃    ★★★    │   ★★★★    │  ★★★★   │  ★★★★
Hawk          ┃    ★★★    │   ★★★★★   │  ★★★    │  ★★★★★
```

## Contributing

Found an error? Want to add an architecture? See the main [Nexus documentation](../../README.md) for contribution guidelines.

## External Resources

- [Mamba paper](https://arxiv.org/abs/2312.00752) - Foundation for Jamba, Zamba
- [S4 paper](https://arxiv.org/abs/2111.00396) - Precursor to modern SSMs
- [RWKV paper](https://arxiv.org/abs/2305.13048) - Foundation for GoldFinch
- [Linear Attention survey](https://arxiv.org/abs/2006.16236) - Background for Based
