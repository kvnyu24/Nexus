# SqueezeLLM: Dense-and-Sparse Quantization for Large Language Models

## Overview

SqueezeLLM introduces a novel hybrid quantization approach that combines dense low-bit quantization for most weights with sparse high-precision storage for outlier weights. This method addresses the challenge that certain "outlier" weights in LLMs are critical for model quality and require higher precision.

**Paper**: "SqueezeLLM: Dense-and-Sparse Quantization"
**Authors**: Sehoon Kim, Coleman Hooper, et al.
**Conference**: ICML 2024
**arXiv**: https://arxiv.org/abs/2306.07629

## Key Contributions

1. **Hybrid Quantization**: Combines dense low-bit (3-4 bit) with sparse high-precision (FP16) storage
2. **Outlier Detection**: Identifies and preserves critical weights automatically
3. **Hardware Efficiency**: Optimized sparse+dense representation for fast inference
4. **Superior Quality**: Better accuracy than pure dense quantization at same bit budget

## Theoretical Foundation

### The Outlier Problem

**Observation**: In LLMs, a small fraction (<0.5%) of weights have disproportionate impact on output quality.

**Mathematical Formulation**:
```
Output sensitivity: ∂y/∂w_ij = x_j * g_i

where:
- w_ij: weight at position (i,j)
- x_j: input activation
- g_i: gradient from downstream layers
```

Weights with large |∂y/∂w_ij| are "outliers" - they significantly affect outputs.

### Dense-and-Sparse Representation

**Decomposition**:
```
W = W_dense + W_sparse

where:
- W_dense: Non-outlier weights quantized to k bits
- W_sparse: Outlier weights stored in FP16
```

**Memory Formula**:
```
Memory = (n - m) * k + m * 16 bits

where:
- n: total weights
- m: number of outliers
- k: dense quantization bits (typically 3-4)
```

### Optimal Outlier Selection

**Objective**: Minimize quantization error subject to memory constraint

```
minimize: ||W - W_Q||²_F
subject to: |outliers| ≤ budget
```

**Selection Criterion**:
```
score_ij = w_ij² / H^{-1}_{jj}

Select top-m weights by score
```

where H^{-1} is the inverse Hessian (sensitivity to quantization).

## Implementation Details

### Outlier Detection Algorithm

```python
def identify_outliers(weight, threshold_percentile=99.5):
    """Identify outlier weights based on magnitude"""
    abs_weight = torch.abs(weight)
    threshold = torch.quantile(abs_weight.flatten(), threshold_percentile/100)

    outlier_mask = abs_weight > threshold
    outlier_indices = torch.nonzero(outlier_mask)
    outlier_values = weight[outlier_mask]

    return outlier_mask, outlier_indices, outlier_values
```

### Dense Quantization (Non-Outliers)

```python
def quantize_dense(weight, outlier_mask, bits=4):
    """Quantize non-outlier weights"""
    # Zero out outliers
    weight_dense = weight.clone()
    weight_dense[outlier_mask] = 0.0

    # Group-wise quantization
    max_val = 2**bits - 1
    scale = (weight_dense.max() - weight_dense.min()) / max_val
    zero_point = weight_dense.min()

    quantized = torch.round((weight_dense - zero_point) / scale)
    quantized = torch.clamp(quantized, 0, max_val)

    return quantized, scale, zero_point
```

### Sparse Storage (Outliers)

```python
def store_outliers_sparse(outlier_indices, outlier_values):
    """Store outliers in COO (coordinate) format"""
    # COO format: (row_indices, col_indices, values)
    rows = outlier_indices[:, 0]
    cols = outlier_indices[:, 1]
    values = outlier_values

    return {
        'indices': torch.stack([rows, cols], dim=0),
        'values': values,
        'shape': (weight.shape[0], weight.shape[1])
    }
```

### Efficient Inference

```python
def squeezellm_forward(x, dense_weight, dense_scale, dense_zero,
                       sparse_indices, sparse_values):
    """Forward pass with hybrid weights"""
    # Dequantize dense weights
    W_dense = (dense_weight - dense_zero) * dense_scale

    # Add sparse outliers
    W_sparse = torch.sparse_coo_tensor(
        sparse_indices, sparse_values, dense_weight.shape
    ).to_dense()

    W = W_dense + W_sparse

    # Matrix multiplication
    return torch.matmul(x, W.t())
```

## Code Examples

### Basic Usage

```python
from nexus.models.compression.quantization.squeezellm import (
    SqueezeLLMConfig,
    SqueezeLLMLinear,
    SqueezeLLMQuantizer
)
import torch
import torch.nn as nn

# Create standard linear layer
layer = nn.Linear(512, 512)

# Configure SqueezeLLM
config = SqueezeLLMConfig(
    bits=4,                    # 4-bit dense quantization
    outlier_threshold=99.5,    # Top 0.5% are outliers
    group_size=128,            # Group size for quantization
    use_sparse_format=True     # Use sparse tensor format
)

# Create quantized layer
squeezed_layer = SqueezeLLMLinear(
    in_features=512,
    out_features=512,
    config=config,
    bias=True
)

# Quantize weights
squeezed_layer.quantize_weight(layer.weight.data)

# Copy bias
if layer.bias is not None:
    squeezed_layer.bias.data = layer.bias.data

# Forward pass
x = torch.randn(32, 512)
output = squeezed_layer(x)

# Get compression statistics
stats = squeezed_layer.get_compression_stats()
print(f"Compression ratio: {stats['compression_ratio']:.2f}x")
print(f"Outlier ratio: {stats['outlier_ratio']:.2f}%")
```

### Quantizing a Full Model

```python
from nexus.models.compression.quantization.squeezellm import (
    SqueezeLLMQuantizer,
    SqueezeLLMConfig
)

# Load model
model = load_pretrained_model("gpt2")

# Configure quantization
config = SqueezeLLMConfig(
    bits=3,                    # Aggressive 3-bit quantization
    outlier_threshold=99.0,    # Top 1% preserved
    group_size=128,
    use_sparse_format=True
)

# Quantize
quantizer = SqueezeLLMQuantizer(config)
quantized_model = quantizer.quantize_model(
    model=model,
    calibration_data=None,     # Optional calibration
    verbose=True
)

# Model now uses hybrid quantization
```

### Custom Outlier Detection

```python
from nexus.models.compression.quantization.squeezellm import SqueezeLLMLinear
import torch

layer = SqueezeLLMLinear(512, 512, config=config)

# Custom outlier detection based on activation magnitude
def custom_outlier_detection(weight, activations):
    """Detect outliers based on weight * activation product"""
    importance = torch.abs(weight) * activations.mean(0).unsqueeze(0)
    threshold = torch.quantile(importance.flatten(), 0.995)
    return importance > threshold

# Apply custom detection
activations = collect_activations(model, calibration_data)
outlier_mask = custom_outlier_detection(layer.weight, activations)

# Quantize with custom mask
layer.quantize_weight_with_mask(layer.weight, outlier_mask)
```

### Integration with Large Models

```python
from transformers import AutoModelForCausalLM
from nexus.models.compression.quantization.squeezellm import (
    SqueezeLLMQuantizer,
    SqueezeLLMConfig,
    apply_squeezellm
)

# Load model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Configure for extreme compression
config = SqueezeLLMConfig(
    bits=3,                    # 3-bit dense
    outlier_threshold=99.5,    # 0.5% outliers in FP16
    group_size=128,
    use_sparse_format=True
)

# Quantize (convenience function)
quantized_model = apply_squeezellm(
    model=model,
    config=config,
    calibration_data=calibration_samples,
    verbose=True
)

# Save
torch.save(quantized_model.state_dict(), "llama2-7b-squeezellm-3bit.pt")
```

## Benchmarks

### Perplexity Results (WikiText-2)

| Model | Method | Bits | Outliers | PPL | Memory | Speedup |
|-------|--------|------|----------|-----|--------|---------|
| LLaMA-7B | FP16 | 16 | 0% | 5.68 | 14.0 GB | 1.0x |
| LLaMA-7B | RTN | 4 | 0% | 7.43 | 3.9 GB | 2.8x |
| LLaMA-7B | GPTQ | 4 | 0% | 5.92 | 3.9 GB | 2.5x |
| LLaMA-7B | SqueezeLLM | 4 | 0.5% | 5.76 | 4.1 GB | 2.7x |
| LLaMA-7B | SqueezeLLM | 3 | 1.0% | 6.23 | 3.2 GB | 3.2x |
| | | | | | | |
| LLaMA-13B | FP16 | 16 | 0% | 5.09 | 26.0 GB | 1.0x |
| LLaMA-13B | GPTQ | 4 | 0% | 5.36 | 7.2 GB | 2.6x |
| LLaMA-13B | SqueezeLLM | 4 | 0.5% | 5.21 | 7.5 GB | 2.9x |
| LLaMA-13B | SqueezeLLM | 3 | 1.0% | 5.68 | 5.8 GB | 3.5x |

### Accuracy on Downstream Tasks

**LLaMA-7B SqueezeLLM (3-bit + 1% outliers):**
| Task | FP16 | SqueezeLLM | Δ |
|------|------|------------|---|
| PIQA | 79.8 | 78.9 | -0.9 |
| ARC-e | 76.2 | 75.1 | -1.1 |
| ARC-c | 47.6 | 46.3 | -1.3 |
| HellaSwag | 76.1 | 74.8 | -1.3 |
| WinoGrande | 70.0 | 68.7 | -1.3 |
| MMLU (5-shot) | 35.1 | 33.6 | -1.5 |

### Memory Analysis

**LLaMA-7B with Different Configurations:**
| Dense Bits | Outlier % | Eff. Bits/Weight | Memory (GB) | Compression |
|------------|-----------|------------------|-------------|-------------|
| 4 | 0.0% | 4.00 | 3.90 | 3.59x |
| 4 | 0.5% | 4.06 | 4.12 | 3.40x |
| 4 | 1.0% | 4.12 | 4.35 | 3.22x |
| 3 | 0.5% | 3.065 | 3.18 | 4.40x |
| 3 | 1.0% | 3.13 | 3.46 | 4.05x |
| 3 | 2.0% | 3.26 | 4.02 | 3.48x |

### Outlier Statistics Across Layers

**LLaMA-7B Layer-wise Outlier Distribution:**
| Layer Type | Avg Outliers | Max Outliers | Impact on PPL |
|------------|--------------|--------------|---------------|
| Embedding | 0.02% | 0.05% | Low |
| Attention Q | 0.31% | 0.58% | Medium |
| Attention K | 0.28% | 0.52% | Medium |
| Attention V | 0.25% | 0.47% | Medium |
| Attention O | 0.42% | 0.79% | High |
| MLP Gate | 0.65% | 1.23% | High |
| MLP Up | 0.58% | 1.15% | High |
| MLP Down | 0.71% | 1.38% | Very High |
| LM Head | 0.15% | 0.28% | Medium |

## Comparison with Other Methods

### SqueezeLLM vs Pure Dense Quantization

| Metric | GPTQ-3bit | SqueezeLLM-3bit | Advantage |
|--------|-----------|-----------------|-----------|
| PPL (LLaMA-7B) | 8.47 | 6.23 | 2.24 lower |
| Memory | 2.9 GB | 3.2 GB | 10% more |
| Inference Speed | 3.8x | 3.2x | 16% slower |
| Accuracy Drop | -6.2% | -2.1% | 3x better |

### SqueezeLLM vs Mixed Precision

| Method | Bits Distribution | Memory | PPL | Setup Complexity |
|--------|-------------------|--------|-----|------------------|
| Mixed Precision | 4/8-bit layers | 5.2 GB | 6.05 | High |
| SqueezeLLM | 3-bit + 1% FP16 | 3.2 GB | 6.23 | Low |

**Advantage**: SqueezeLLM is simpler (no layer-wise tuning) and more memory-efficient.

## Advanced Topics

### Optimal Outlier Budget

**Trade-off Analysis**:
```
PPL(outlier_ratio) = PPL_base + α * exp(-β * outlier_ratio)

where:
- α, β: model-specific constants
- Typical optimal: 0.5-1.0% outliers
```

**Finding Optimal Budget**:
```python
def find_optimal_budget(model, test_data, budgets=[0.1, 0.5, 1.0, 2.0]):
    """Find optimal outlier budget via sweep"""
    results = []

    for budget in budgets:
        config = SqueezeLLMConfig(outlier_threshold=100-budget)
        quantized = quantize_model(model, config)
        ppl = evaluate_perplexity(quantized, test_data)
        mem = compute_memory(quantized)

        results.append({
            'budget': budget,
            'perplexity': ppl,
            'memory': mem,
            'score': ppl * mem  # Combined metric
        })

    return min(results, key=lambda x: x['score'])
```

### Sensitivity-Based Outlier Selection

Instead of magnitude, use sensitivity:

```python
def sensitivity_based_outliers(weight, hessian_inv_diag):
    """Select outliers based on OBS sensitivity"""
    sensitivity = weight**2 / hessian_inv_diag
    threshold = torch.quantile(sensitivity.flatten(), 0.995)
    return sensitivity > threshold
```

### Dynamic Outlier Detection

Adapt outliers based on input:

```python
class DynamicSqueezeLLM(nn.Module):
    def forward(self, x):
        # Detect outliers based on current activations
        outlier_mask = detect_dynamic_outliers(self.weight, x)

        # Quantize on-the-fly
        W_quantized = quantize_with_mask(self.weight, outlier_mask)

        return F.linear(x, W_quantized, self.bias)
```

## Best Practices

### 1. Choosing Outlier Threshold

```python
# Conservative (higher accuracy, more memory)
config = SqueezeLLMConfig(outlier_threshold=99.0)  # 1% outliers

# Balanced (recommended)
config = SqueezeLLMConfig(outlier_threshold=99.5)  # 0.5% outliers

# Aggressive (lower memory, less accuracy)
config = SqueezeLLMConfig(outlier_threshold=99.8)  # 0.2% outliers
```

### 2. Dense Bit Width Selection

```python
# 4-bit: Safe choice, minimal accuracy loss
config = SqueezeLLMConfig(bits=4, outlier_threshold=99.5)

# 3-bit: Aggressive compression, use with 1% outliers
config = SqueezeLLMConfig(bits=3, outlier_threshold=99.0)

# 2-bit: Experimental, requires 2%+ outliers
config = SqueezeLLMConfig(bits=2, outlier_threshold=98.0)
```

### 3. Layer-Specific Configuration

```python
# More outliers for critical layers
config_mlp = SqueezeLLMConfig(
    bits=3,
    outlier_threshold=99.0  # 1% outliers
)

# Fewer outliers for less critical layers
config_attention = SqueezeLLMConfig(
    bits=4,
    outlier_threshold=99.5  # 0.5% outliers
)
```

### 4. Validation Strategy

```python
# Multi-metric validation
def validate_squeezellm(model, test_suite):
    metrics = {}

    # Perplexity
    metrics['ppl'] = evaluate_perplexity(model, test_suite['language_modeling'])

    # Downstream tasks
    for task in ['piqa', 'arc', 'hellaswag']:
        metrics[task] = evaluate_task(model, test_suite[task])

    # Memory and speed
    metrics['memory'] = compute_memory_usage(model)
    metrics['speed'] = benchmark_inference_speed(model)

    return metrics
```

## Troubleshooting

### Problem: High Perplexity After Quantization

**Solutions**:
1. Increase outlier budget
2. Use higher dense precision (4-bit instead of 3-bit)
3. Check outlier detection quality

```python
# Diagnostic: Visualize outliers
import matplotlib.pyplot as plt

def visualize_outliers(weight, outlier_mask):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.hist(weight.flatten().numpy(), bins=100)
    plt.title("Weight Distribution")

    plt.subplot(1, 3, 2)
    plt.imshow(outlier_mask.float().cpu(), cmap='hot')
    plt.title("Outlier Locations")

    plt.subplot(1, 3, 3)
    outlier_vals = weight[outlier_mask]
    plt.hist(outlier_vals.flatten().numpy(), bins=50)
    plt.title("Outlier Values")

    plt.tight_layout()
    plt.show()
```

### Problem: Slow Inference

**Causes**:
- Sparse tensor overhead
- Inefficient outlier access pattern

**Solutions**:
```python
# Precompute dense + sparse
class OptimizedSqueezeLLM(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # Precompute full weight matrix
        self.register_buffer('full_weight', None)

    def compute_full_weight(self):
        """Cache dequantized weight"""
        W_dense = self.dequantize_dense()
        W_sparse = self.get_sparse_contribution()
        self.full_weight = W_dense + W_sparse

    def forward(self, x):
        if self.full_weight is None:
            self.compute_full_weight()
        return F.linear(x, self.full_weight, self.bias)
```

### Problem: Memory Not Reduced as Expected

**Cause**: Too many outliers

**Solutions**:
```python
# Analyze outlier distribution
stats = squeezed_layer.get_compression_stats()
print(f"Outlier ratio: {stats['outlier_ratio']:.2f}%")

if stats['outlier_ratio'] > 1.5:
    # Reduce outliers
    config.outlier_threshold = 99.7  # More aggressive
```

## Hardware Considerations

### GPU Optimization

**CUDA Kernels for Sparse Operations**:
```cuda
// Optimized sparse+dense matmul kernel
__global__ void squeezellm_matmul(
    float* output,
    float* input,
    int8_t* dense_weight,
    float* dense_scale,
    int* sparse_indices,
    float* sparse_values,
    int M, int N, int K
) {
    // Fused dense dequantization + sparse addition
    // Optimized for memory coalescing
    ...
}
```

### Memory Layout

**Optimal Storage Format**:
```
+------------------+
| Dense Weights    |  (packed into bits)
|   (4-bit)        |
+------------------+
| Dense Scales     |  (FP32, per-group)
+------------------+
| Dense Zeros      |  (FP32, per-group)
+------------------+
| Sparse Indices   |  (COO format)
+------------------+
| Sparse Values    |  (FP16/FP32)
+------------------+
```

## Conclusion

SqueezeLLM provides an effective hybrid quantization approach that achieves excellent compression ratios while preserving model quality. By identifying and preserving critical outlier weights while aggressively quantizing the rest, SqueezeLLM outperforms pure dense quantization methods at the same bit budget.

**Key Takeaways**:
- ✅ 3-bit quantization with <2% accuracy loss (with 1% outliers)
- ✅ 4x+ memory reduction vs FP16
- ✅ Automatic outlier detection
- ✅ Better accuracy than pure dense quantization
- ✅ Flexible trade-off between memory and accuracy

**When to Use SqueezeLLM**:
- Need aggressive compression (3-bit or lower)
- Willing to trade 10-15% memory for better accuracy
- Want automatic outlier handling without manual tuning

## References

1. **SqueezeLLM Paper**: Kim, S., et al. "SqueezeLLM: Dense-and-Sparse Quantization." ICML 2024.
2. **Implementation**: https://github.com/SqueezeAILab/SqueezeLLM
3. **Related Work**:
   - SpQR: Dettmers, T., et al. "SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression." 2023.
   - Optimal Brain Surgeon: Hassibi, B., & Stork, D. G. "Second order derivatives for network pruning." 1993.
