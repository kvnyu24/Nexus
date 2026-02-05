# Architecture Components

This section covers the fundamental building blocks that make up modern neural network architectures, particularly those used in large language models and transformers.

## Overview

Architecture components are the essential layers and modules that define the structure and behavior of neural networks. These components handle critical operations like:

- **Sparse Computation**: Activating only subsets of parameters (MoE)
- **Normalization**: Stabilizing training and improving convergence
- **Activation Functions**: Non-linear transformations that enable learning complex patterns

## Categories

### 1. [Mixture of Experts (MoE)](./moe/)

Sparse computation architectures that scale model capacity without proportionally increasing compute by selectively activating expert sub-networks.

**Key Components:**
- MoE Layer & Router
- Enhanced MoE Variants
- DeepSeek MoE
- Switch Transformer
- SwitchAll
- Mixture-of-Depths
- Mixture-of-Agents

**When to Use:**
- Scaling to massive parameter counts (100B+ parameters)
- Improving model capacity with constant compute budget
- Multi-task learning with specialized experts
- Conditional computation based on input

### 2. [Normalization](./normalization/)

Techniques for stabilizing training, improving convergence speed, and enabling deeper networks.

**Key Components:**
- RMSNorm
- QK-Norm
- DeepNorm
- HybridNorm
- DynamicTanh (DyT)

**When to Use:**
- Training deep transformers (>12 layers)
- Improving training stability
- Accelerating convergence
- Enabling higher learning rates

### 3. [Activation Functions](./activation/)

Non-linear transformations in feed-forward layers, with modern gated variants providing superior performance.

**Key Components:**
- SwiGLU
- GeGLU
- ReGLU

**When to Use:**
- Feed-forward layers in transformers
- Replacing traditional ReLU/GELU activations
- Improving model expressiveness
- Following modern LLM best practices (Llama, Mistral, etc.)

## Component Selection Guide

### For Language Models

| Model Size | MoE Strategy | Normalization | Activation |
|-----------|--------------|---------------|------------|
| Small (<1B) | Dense or None | LayerNorm/RMSNorm | SwiGLU |
| Medium (1-10B) | Optional MoE | RMSNorm | SwiGLU |
| Large (10-100B) | Switch/DeepSeek MoE | RMSNorm + QK-Norm | SwiGLU |
| Massive (100B+) | DeepSeek MoE | RMSNorm + DeepNorm | SwiGLU |

### For Vision Models

| Architecture | MoE Strategy | Normalization | Activation |
|-------------|--------------|---------------|------------|
| CNN-based | Rarely | BatchNorm/GroupNorm | ReLU/GELU |
| ViT-based | Optional | LayerNorm | GELU/GeGLU |
| Hybrid | MoD | LayerNorm + DyT | SwiGLU |

### For Multimodal Models

| Component | Recommendation | Reason |
|-----------|----------------|---------|
| MoE | SwitchAll or DeepSeek | Handle diverse input modalities |
| Normalization | RMSNorm + QK-Norm | Stable cross-modal attention |
| Activation | SwiGLU | Proven performance across modalities |

## Design Principles

### 1. Composability

Components are designed to work together seamlessly:

```python
# Example: Combining components in a transformer layer
from nexus.components.moe import DeepSeekMoE
from nexus.components.normalization import RMSNorm
from nexus.components.activations import SwiGLU

class TransformerLayer(nn.Module):
    def __init__(self, dim):
        self.attn_norm = RMSNorm(dim)
        self.attention = MultiHeadAttention(dim)

        # MoE with SwiGLU experts
        self.ffn = DeepSeekMoE(
            dim=dim,
            activation='swiglu',
            norm_type='rms'
        )
```

### 2. Efficiency

Modern components prioritize computational efficiency:

- **MoE**: Sparse activation reduces FLOPs by 2-8x
- **RMSNorm**: 10-20% faster than LayerNorm
- **SwiGLU**: Better accuracy per FLOP than ReLU/GELU

### 3. Scalability

Components scale from small research projects to production systems:

- **Parameter Scaling**: MoE enables 100B+ parameter models
- **Sequence Scaling**: Normalization techniques enable long contexts
- **Hardware Scaling**: Optimized for modern accelerators (GPUs, TPUs)

## Common Patterns

### Pre-Norm Transformer

```python
# Standard in modern LLMs (Llama, Mistral, GPT)
def forward(self, x):
    # Attention with Pre-Norm
    x = x + self.attn(self.attn_norm(x))
    # FFN with Pre-Norm
    x = x + self.ffn(self.ffn_norm(x))
    return x
```

### Post-Norm Transformer

```python
# Traditional approach (BERT, original Transformer)
def forward(self, x):
    # Attention with Post-Norm
    x = self.attn_norm(x + self.attn(x))
    # FFN with Post-Norm
    x = self.ffn_norm(x + self.ffn(x))
    return x
```

### Hybrid-Norm Transformer

```python
# Best of both worlds
def forward(self, x):
    # Pre-Norm for attention (stability)
    x = x + self.attn(self.attn_norm(x))
    # Post-Norm for FFN (capacity)
    x = self.ffn_norm(x + self.ffn(x))
    return x
```

### Sparse MoE Transformer

```python
# Scaling with conditional computation
def forward(self, x):
    x = x + self.attn(self.attn_norm(x))
    # Sparse MoE FFN
    ffn_out, aux_loss = self.moe_ffn(self.ffn_norm(x))
    x = x + ffn_out
    return x, aux_loss
```

## Performance Considerations

### Memory Usage

| Component | Memory Impact | Mitigation |
|-----------|---------------|------------|
| MoE (64 experts) | 8-16x parameters | Expert parallelism, offloading |
| RMSNorm | Minimal | N/A |
| SwiGLU | 1.5x vs standard FFN | Worth the trade-off |

### Training Stability

1. **Start Conservative**: Use RMSNorm + Pre-Norm + SwiGLU
2. **Add Complexity**: Introduce MoE after baseline converges
3. **Tune Carefully**: Adjust load balancing coefficients
4. **Monitor Metrics**: Track expert utilization, gradient norms

### Inference Optimization

- **MoE**: Cache expert routing decisions
- **Normalization**: Fuse with adjacent operations
- **Activation**: Use kernels optimized for your hardware

## Implementation Notes

All components in this section:

- Inherit from `NexusModule` for consistent interfaces
- Support both training and inference modes
- Include type hints and docstrings
- Provide configuration examples
- Are tested for correctness and performance

## References

See individual component documentation for detailed references and citations.

## Quick Start

```python
# Install Nexus
pip install nexus-ml

# Import components
from nexus.components.moe import DeepSeekMoE
from nexus.components.normalization import RMSNorm
from nexus.components.activations import SwiGLU

# Build a modern transformer layer
layer = TransformerLayer(
    dim=2048,
    moe_config={
        'num_shared_experts': 2,
        'num_routed_experts': 64,
        'top_k': 6
    }
)
```

## Next Steps

1. Explore [MoE documentation](./moe/) for sparse computation
2. Read [Normalization guide](./normalization/) for training stability
3. Check [Activation documentation](./activation/) for modern FFN layers
4. See [Training Infrastructure](../11_training_infrastructure/) for distributed training
5. Review [Optimization Tricks](../07_inference_optimizations/) for deployment
