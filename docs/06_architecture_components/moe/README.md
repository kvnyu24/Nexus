# Mixture of Experts (MoE)

Mixture of Experts is a conditional computation technique that scales model capacity without proportionally increasing computational cost. By routing each input to a sparse subset of "expert" sub-networks, MoE enables training models with hundreds of billions or even trillions of parameters while maintaining practical inference costs.

## Overview

**Problem**: Dense transformer models scale poorly - doubling parameters doubles compute at every forward pass.

**Solution**: MoE introduces sparsity through expert routing:
- Each token is processed by only a small subset (top-k) of available experts
- Total parameters grow with number of experts
- Compute per token remains roughly constant

**Key Insight**: Different inputs benefit from different transformations. Learned routing enables specialization.

## Available Components

| Component | Description | Best For | Paper |
|-----------|-------------|----------|-------|
| [MoE Layer](./moe_layer.md) | Basic sparse MoE with top-k routing | Foundational understanding | GShard (2020) |
| [Router](./router.md) | Token-to-expert routing mechanisms | Custom MoE implementations | Switch Transformer (2021) |
| [Enhanced MoE](./enhanced_moe.md) | Advanced routing strategies | Research & experimentation | Multiple |
| [DeepSeek MoE](./deepseek_moe.md) | Shared + routed experts, fine-grained | Production LLMs, efficiency | DeepSeek-V3 (2024) |
| [Switch Transformer](./switch_transformer.md) | Simplified top-1 routing | Simplicity, large scale | Google (2021) |
| [SwitchAll](./switch_all.md) | MoE in attention + FFN | Maximum sparsity | SwitchHead (2024) |
| [Mixture-of-Depths](./mixture_of_depths.md) | Dynamic token-level compute | Variable computation | Google (2024) |
| [Mixture-of-Agents](./mixture_of_agents.md) | Multi-LLM collaboration | LLM ensembling | Together AI (2024) |

## When to Use MoE

### Use MoE When:

1. **Scaling to Large Capacity**: Need more than 10B parameters but limited compute budget
2. **Multi-Task Learning**: Different tasks benefit from specialized experts
3. **Specialized Domains**: Model serves diverse input distributions
4. **Research Exploration**: Studying conditional computation

### Avoid MoE When:

1. **Small Models**: Less than 1B parameters - overhead not worth it
2. **Uniform Data**: All inputs need similar processing
3. **Memory Constrained**: Limited VRAM - experts don't fit
4. **Simplicity Required**: Debugging/deployment complexity not acceptable

## Common Pitfalls

### 1. Auxiliary Loss Too High

**Problem**: Model focuses on load balancing, not task performance

**Solution**:
- Reduce aux_loss_coef (try 0.001 - 0.01)
- Use loss-free balancing
- Monitor expert utilization separately

### 2. Expert Collapse

**Problem**: All tokens route to same 1-2 experts

**Solution**:
- Increase jitter noise in router
- Add expert dropout
- Use expert-choice routing
- Verify aux loss is being applied

### 3. Memory OOM

**Problem**: All expert parameters must fit in memory

**Solution**:
- Use expert parallelism (distribute across GPUs)
- Implement expert offloading (CPU/disk)
- Reduce number of experts
- Use smaller expert dimension

## Quick Start Guide

### Basic MoE Layer

```python
from nexus.components.moe import MoELayer

# Replace dense FFN with MoE
moe_ffn = MoELayer(
    dim=2048,
    num_experts=8,
    top_k=2,
    expert_hidden_dim=8192
)

# Forward pass
output, aux_loss = moe_ffn(hidden_states)

# Add aux loss to training objective
total_loss = task_loss + 0.01 * aux_loss
```

### Production-Scale MoE

```python
from nexus.components.moe import DeepSeekMoE

# DeepSeek-V3 style configuration
moe = DeepSeekMoE(
    dim=5120,
    num_shared_experts=2,
    num_routed_experts=160,
    top_k_experts=6,
    expert_dim=1536,
    shared_expert_dim=5120,
    num_segments=4,
    router_aux_loss_coef=0.001
)

# Integrates seamlessly into transformer
output, aux_loss = moe(hidden_states)
```

## References

1. **Shazeer et al. (2017)** - "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"
2. **Fedus et al. (2021)** - "Switch Transformers: Scaling to Trillion Parameter Models"
3. **DeepSeek-V2 (2024)** - "A Strong, Economical, and Efficient MoE Language Model"
4. **DeepSeek-V3 (2024)** - "Technical Report"
