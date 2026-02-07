# Architecture Components Documentation Index

This directory contains comprehensive documentation for fundamental neural network architecture components used in modern deep learning, particularly for large language models.

## Documentation Structure

### Main README Files

1. **[Architecture Components README](./README.md)**
   - Overview of all component categories
   - Component selection guide
   - Design principles and common patterns
   - Performance considerations

2. **[MoE README](./moe/README.md)**
   - Complete guide to Mixture of Experts
   - When to use MoE vs dense models
   - Routing strategies comparison
   - Load balancing techniques
   - Production considerations

3. **[Normalization README](./normalization/README.md)**
   - Overview of normalization techniques
   - Pre-Norm vs Post-Norm comparison
   - Performance characteristics
   - Migration guides

4. **[Activation README](./activation/README.md)**
   - Gated activation functions (GLU variants)
   - Performance comparison (SwiGLU, GeGLU, ReGLU)
   - Hidden dimension sizing
   - Architecture integration

## Detailed Component Documentation

### Mixture of Experts (MoE)

| Component | File | Description | Status |
|-----------|------|-------------|--------|
| MoE Layer | [moe_layer.md](./moe/moe_layer.md) | Foundational MoE with top-k routing | Complete |
| Router | [router.md](./moe/router.md) | Token-to-expert routing mechanisms | Complete |
| DeepSeek MoE | [deepseek_moe.md](./moe/deepseek_moe.md) | Shared + routed experts architecture | Complete |
| Mixture-of-Depths | [mixture_of_depths.md](./moe/mixture_of_depths.md) | Dynamic token-level computation | Complete |
| Enhanced MoE | enhanced_moe.md | Advanced routing strategies | TODO |
| Switch Transformer | switch_transformer.md | Top-1 simplified routing | TODO |
| SwitchAll | switch_all.md | MoE in attention + FFN | TODO |
| Mixture-of-Agents | mixture_of_agents.md | Multi-LLM collaboration | TODO |

### Normalization

| Component | File | Description | Status |
|-----------|------|-------------|--------|
| RMSNorm | [rmsnorm.md](./normalization/rmsnorm.md) | Root Mean Square normalization | Complete |
| QK-Norm | qk_norm.md | Query-Key normalization | TODO |
| DeepNorm | deepnorm.md | Scaled residuals for deep nets | TODO |
| HybridNorm | hybrid_norm.md | Pre-Norm + Post-Norm hybrid | TODO |
| DynamicTanh | dyt.md | Normalization-free alternative | TODO |

### Activation Functions

| Component | File | Description | Status |
|-----------|------|-------------|--------|
| SwiGLU | [swiglu.md](./activation/swiglu.md) | Swish-Gated Linear Unit | Complete |
| GeGLU | geglu.md | GELU-Gated Linear Unit | TODO |
| ReGLU | reglu.md | ReLU-Gated Linear Unit | TODO |

## Documentation Features

Each detailed component documentation includes:

1. **Overview & Motivation**
   - Problem being solved
   - Real-world impact and adoption
   - Key achievements

2. **Theoretical Background**
   - Core concepts and intuition
   - Mathematical foundations
   - Why the approach works

3. **Mathematical Formulation**
   - Complete forward/backward pass equations
   - Parameter counts and complexity analysis
   - Detailed algorithm descriptions

4. **High-Level Intuition**
   - Analogies and visual explanations
   - Comparisons with alternatives
   - When to use each component

5. **Implementation Details**
   - Code location in repository
   - Usage examples (basic to advanced)
   - Configuration parameters

6. **Code Walkthrough**
   - Key design decisions
   - Line-by-line explanation of critical sections
   - Best practices

7. **Optimization Tricks**
   - Performance improvements
   - Memory efficiency techniques
   - Hardware-specific optimizations

8. **Experiments & Results**
   - Benchmark comparisons
   - Ablation studies
   - Production results

9. **Common Pitfalls**
   - Mistakes to avoid
   - Debugging tips
   - Migration guides

10. **References**
    - Original papers
    - Implementation guides
    - Related work

## Quick Reference

### Recommended Configurations

#### Small Models (< 1B parameters)
```python
# Normalization: RMSNorm
norm = RMSNorm(dim=768, eps=1e-6)

# Activation: SwiGLU
ffn = SwiGLU(dim=768, hidden_dim=None)

# MoE: Not recommended (overhead too high)
```

#### Medium Models (1-10B parameters)
```python
# Normalization: RMSNorm
norm = RMSNorm(dim=2048, eps=1e-6)

# Activation: SwiGLU
ffn = SwiGLU(dim=2048, hidden_dim=None, multiple_of=256)

# MoE: Optional, use standard MoE or Switch
moe = MoELayer(dim=2048, num_experts=8, top_k=2)
```

#### Large Models (10-100B parameters)
```python
# Normalization: RMSNorm + QK-Norm
norm = RMSNorm(dim=4096, eps=1e-6)
qk_norm = QKNorm(head_dim=128)

# Activation: SwiGLU
ffn = SwiGLU(dim=4096, hidden_dim=None, bias=False)

# MoE: DeepSeek MoE for efficiency
moe = DeepSeekMoE(
    dim=4096,
    num_shared_experts=2,
    num_routed_experts=64,
    top_k_experts=6
)
```

#### Massive Models (100B+ parameters)
```python
# Normalization: RMSNorm + QK-Norm + DeepNorm
norm = DeepNorm(dim=8192, num_layers=96)
qk_norm = QKNorm(head_dim=128)

# Activation: SwiGLU
ffn = SwiGLU(dim=8192, hidden_dim=None, bias=False)

# MoE: DeepSeek MoE with fine-grained experts
moe = DeepSeekMoE(
    dim=8192,
    num_shared_experts=2,
    num_routed_experts=256,
    top_k_experts=8,
    num_segments=4,
    router_aux_loss_coef=0.0  # Loss-free balancing
)
```

## Code References

All components are implemented in:

```
Nexus/nexus/components/
├── moe/
│   ├── router.py              # Routing mechanisms
│   ├── expert.py              # Expert layers and MoE
│   ├── deepseek_moe.py        # DeepSeek MoE
│   ├── switch_transformer.py  # Switch Transformer
│   ├── switch_all.py          # SwitchAll
│   └── mixture_of_agents.py   # Mixture-of-Agents
├── layers/
│   └── mixture_of_depths.py   # Mixture-of-Depths
├── normalization.py           # All normalization layers
└── activations.py             # All activation functions
```

## Contributing

To add new documentation:

1. Follow the 10-section structure outlined above
2. Include practical code examples
3. Reference actual implementation in `/nexus/components/`
4. Add mathematical formulations with clear notation
5. Provide intuitive analogies
6. List common pitfalls and solutions
7. Include references to original papers
8. Update this index

## Related Documentation

- [Attention Mechanisms](../02_attention_mechanisms/)
- [State Space Models](../03_state_space_models/)
- [Hybrid Architectures](../04_hybrid_architectures/)
- [Positional Encodings](../05_positional_encodings/)
- [Inference Optimizations](../07_inference_optimizations/)
- [Training Infrastructure](../11_training_infrastructure/)

## Documentation Statistics

- **Total Documentation Files**: 10
- **Complete Detailed Docs**: 5
  - MoE Layer
  - DeepSeek MoE
  - Router
  - Mixture-of-Depths
  - RMSNorm
  - SwiGLU
- **README Files**: 4
  - Main README
  - MoE README
  - Normalization README
  - Activation README
- **Pending Detailed Docs**: 8
  - Enhanced MoE
  - Switch Transformer
  - SwitchAll
  - Mixture-of-Agents
  - QK-Norm
  - DeepNorm
  - HybridNorm
  - DynamicTanh
  - GeGLU
  - ReGLU

## Next Steps

To complete the documentation:

1. **Phase 1** (Complete): Core components
   - ✅ MoE Layer
   - ✅ DeepSeek MoE
   - ✅ Router
   - ✅ Mixture-of-Depths
   - ✅ RMSNorm
   - ✅ SwiGLU

2. **Phase 2** (TODO): Additional MoE variants
   - Switch Transformer
   - SwitchAll
   - Enhanced MoE
   - Mixture-of-Agents

3. **Phase 3** (TODO): Additional normalization
   - QK-Norm
   - DeepNorm
   - HybridNorm
   - DynamicTanh

4. **Phase 4** (TODO): Additional activations
   - GeGLU
   - ReGLU

## Feedback

For questions, corrections, or suggestions:
- Open an issue in the Nexus repository
- Reference specific documentation files
- Suggest improvements or clarifications

---

Last Updated: 2026-02-06
