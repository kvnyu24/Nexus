# Pruning Methods for Large Language Models

Pruning removes unnecessary weights or structures from neural networks, reducing model size and improving inference speed while maintaining accuracy. For LLMs, pruning can achieve 50-70% sparsity with minimal degradation.

## Overview

| Method | Type | Sparsity | Key Innovation | Best For |
|--------|------|----------|----------------|----------|
| [SparseGPT](sparse_gpt.md) | Unstructured | 50-70% | One-shot OBS-based pruning | Post-training compression |
| [Wanda](wanda.md) | Unstructured | 50-60% | Weight × activation magnitude | Fast, training-free |
| [SliceGPT](slice_gpt.md) | Structured | 20-30% | Dimension reduction via PCA | Hardware-friendly |
| [ShortGPT](shortgpt.md) | Structured | 25-40% | Layer/head removal | Efficient deployment |

**Unstructured**: Remove individual weights (requires sparse kernels)
**Structured**: Remove entire rows/columns/layers (standard hardware)

## When to Use Pruning

### Use Cases

**Ideal for Pruning:**
- Reducing inference latency (especially structured pruning)
- Extreme compression combined with quantization
- Edge deployment with sparse hardware support
- Reducing FLOPs for energy efficiency
- Models with significant redundancy

**Consider Other Methods:**
- If you need >70% compression (use quantization + distillation)
- If hardware doesn't support sparse operations well
- For training acceleration (pruning helps inference more)
- When accuracy is paramount (distillation may be better)

### Compression Comparison

For a 7B parameter model:

| Method | Model Size | Sparsity | Inference Speed | Accuracy Loss |
|--------|------------|----------|-----------------|---------------|
| Baseline | 13 GB | 0% | 1.0× | 0% |
| SparseGPT | 7.8 GB (40% saved) | 60% | 1.2× | <1% |
| Wanda | 9.1 GB (30% saved) | 50% | 1.1× | <2% |
| SliceGPT | 10.4 GB (20% saved) | N/A (structured) | 1.4× | <1% |
| ShortGPT | 9.75 GB (25% saved) | N/A (layer removal) | 1.3× | <2% |

## Pruning Fundamentals

### What is Pruning?

Pruning identifies and removes weights that contribute least to model output:

```
Pruned Weight Matrix: W_pruned = W ⊙ M
```

where:
- W: Original weight matrix
- M: Binary mask (1 = keep, 0 = remove)
- ⊙: Element-wise multiplication

**Goal**: Maximize sparsity (zeros) while minimizing accuracy loss.

### Unstructured vs Structured

**Unstructured Pruning**:
```
Original:          Unstructured:
[1.2  0.3  -0.8]   [1.2  0    -0.8]
[0.5  -1.1  0.2]   [0    -1.1  0   ]
[-0.4  0.9  0.7]   [-0.4  0.9  0   ]

- Remove individual weights
- Maximum flexibility
- Requires sparse kernels
- Best accuracy at high sparsity
```

**Structured Pruning**:
```
Original:          Structured (col 2 removed):
[1.2  0.3  -0.8]   [1.2  -0.8]
[0.5  -1.1  0.2]   [0.5   0.2]
[-0.4  0.9  0.7]   [-0.4  0.7]

- Remove rows/columns/blocks
- Hardware-friendly
- No special kernels needed
- Lower maximum sparsity
```

### Magnitude vs Gradient-Based

**Magnitude Pruning**:
```python
# Remove weights with smallest absolute value
mask = torch.abs(W) > threshold
```
- Fast, simple
- Works okay for moderate sparsity
- Ignores weight importance

**Gradient/Hessian-Based** (SparseGPT, OBS):
```python
# Remove weights with lowest saliency
saliency = W² / (2 * [H⁻¹]_diag)
mask = saliency > threshold
```
- Slower, complex
- Better at high sparsity
- Accounts for weight interactions

### One-Shot vs Iterative

**One-Shot Pruning** (SparseGPT, Wanda):
- Prune once to target sparsity
- No retraining
- Fast (hours for 175B model)

**Iterative Pruning**:
- Gradually increase sparsity: 0% → 20% → 40% → 60%
- Retrain between pruning steps
- Better accuracy but requires training

For LLMs, one-shot is preferred due to retraining cost.

## Method Comparison

### SparseGPT vs Wanda vs SliceGPT

**SparseGPT** (OBS-based):
- **Pros**: Best accuracy at high sparsity (60%), theoretically grounded
- **Cons**: Slower pruning, requires calibration data
- **Use when**: You need maximum sparsity with minimal loss

**Wanda** (Weight × Activation):
- **Pros**: Fast (no Hessian), training-free, simple
- **Cons**: Slightly lower accuracy than SparseGPT
- **Use when**: You need quick pruning without calibration

**SliceGPT** (Structured via PCA):
- **Pros**: Hardware-friendly, no sparse kernels needed, composable with quantization
- **Cons**: Lower maximum compression (20-30%)
- **Use when**: Deploying on hardware without sparse support

**ShortGPT** (Layer Removal):
- **Pros**: Simple, effective, good speedup
- **Cons**: Coarse-grained, less flexible
- **Use when**: Model has redundant layers (large models)

### Performance Comparison (LLaMA-7B)

| Method | Sparsity | WikiText2 PPL | C4 PPL | Zero-Shot Avg | Pruning Time |
|--------|----------|---------------|--------|---------------|--------------|
| Dense | 0% | 5.68 | 7.02 | 61.2% | - |
| Magnitude | 50% | 14.32 | 16.78 | 51.3% | 5 min |
| **SparseGPT** | **50%** | **6.21** | **7.55** | **60.1%** | **3 hrs** |
| Wanda | 50% | 6.87 | 8.12 | 58.7% | 20 min |
| SparseGPT | 60% | 7.14 | 8.89 | 57.8% | 3.5 hrs |
| SliceGPT (25% dim) | N/A | 6.12 | 7.43 | 60.5% | 1 hr |

## Quick Start

### SparseGPT (One-Shot Unstructured)

```python
from nexus.models.compression.pruning import SparseGPTPruner, SparseGPTConfig

# Configure pruning
config = SparseGPTConfig(
    sparsity=0.5,  # 50% of weights will be zero
    blocksize=128,
    percdamp=0.01
)

pruner = SparseGPTPruner(pruning_config=config)

# Prune model (requires calibration data)
metrics = pruner.prune_model(
    model,
    calibration_dataloader,
    nsamples=128
)

# Model is now 50% sparse
print(f"Sparsity: {metrics['achieved_sparsity']:.2%}")
```

### Wanda (Fast Training-Free)

```python
from nexus.models.compression.pruning import WandaPruner

pruner = WandaPruner(sparsity=0.5)

# No calibration needed, uses activation statistics
pruner.prune_model(model, calibration_dataloader, nsamples=64)
```

### SliceGPT (Structured Dimension Reduction)

```python
from nexus.models.compression.pruning import SliceGPTPruner

# Remove 25% of embedding dimensions
pruner = SliceGPTPruner(slice_ratio=0.25)
pruner.prune_model(model, calibration_dataloader)

# Model has 75% of original dimensions (hardware-friendly)
```

### ShortGPT (Layer Removal)

```python
from nexus.models.compression.pruning import ShortGPTPruner

# Remove 6 out of 32 layers
pruner = ShortGPTPruner(
    n_layers_to_remove=6,
    removal_strategy="similarity"  # or "attention", "gradient"
)
pruner.prune_model(model)
```

## Hyperparameter Guidelines

### Sparsity Level Selection

| Model Size | Safe Sparsity | Aggressive Sparsity | Extreme Sparsity |
|------------|---------------|---------------------|------------------|
| <1B | 30-40% | 50% | 60% |
| 1-10B | 40-50% | 60% | 70% |
| 10-70B | 50-60% | 70% | 80% |
| >70B | 60-70% | 80% | 90% |

**Observation**: Larger models tolerate higher sparsity (more redundancy).

### Per-Layer Sparsity

Different layers have different redundancy:

```python
# Example: Variable sparsity by layer
sparsity_config = {
    "embed": 0.0,      # Don't prune embeddings
    "layer_0-5": 0.3,  # Low sparsity for early layers
    "layer_6-25": 0.6, # High sparsity for middle layers
    "layer_26-31": 0.4, # Medium sparsity for late layers
    "lm_head": 0.2,    # Low sparsity for output
}
```

**Rule of thumb**:
- Early layers: Lower sparsity (more critical)
- Middle layers: Higher sparsity (more redundancy)
- Final layers: Medium sparsity (task-specific)

### Block Size (for SparseGPT)

| Block Size | Speed | Memory | Accuracy |
|------------|-------|--------|----------|
| 32 | Slow | Low | Best |
| 128 | Medium | Medium | Good (default) |
| 256 | Fast | High | Acceptable |

**Recommendation**: 128 for most cases.

### Calibration Data

| Samples | Pruning Time | Accuracy |
|---------|--------------|----------|
| 32 | Fast | Acceptable |
| 128 | Medium | Good (recommended) |
| 512 | Slow | Best |

Diminishing returns beyond 128 samples.

## Advanced Topics

### Combining Pruning with Quantization

Prune then quantize for extreme compression:

```python
# 1. Prune to 50% sparsity
pruner = SparseGPTPruner(sparsity=0.5)
model = pruner.prune_model(model, calibration_data)

# 2. Quantize to 4-bit
quantizer = GPTQQuantizer(bits=4)
model = quantizer.quantize_model(model, calibration_data)

# Result: 50% sparsity + 4-bit = 8× compression
# 13 GB → 1.625 GB
```

**Order matters**: Prune before quantization (quantization adapts to sparse structure).

### N:M Structured Sparsity

Hardware-accelerated sparsity pattern (e.g., 2:4 = 2 zeros in every 4 elements):

```python
from nexus.models.compression.pruning import NMSparsityPruner

# 2:4 sparsity (50%, hardware accelerated on Ampere GPUs)
pruner = NMSparsityPruner(n=2, m=4)
model = pruner.prune_model(model, calibration_data)

# Inference on A100 gets automatic 2× speedup
```

**Supported patterns**:
- 2:4 (50% sparsity) on NVIDIA Ampere+
- 1:4 (75% sparsity) on specialized hardware

### Gradual Sparsification

For better accuracy, gradually increase sparsity:

```python
# Start at 0%, end at 60%, over 10k steps
scheduler = CubicSparsityScheduler(
    initial_sparsity=0.0,
    final_sparsity=0.6,
    total_steps=10000
)

for step in range(10000):
    current_sparsity = scheduler.get_sparsity(step)
    apply_sparsity(model, current_sparsity)

    # Continue training
    loss.backward()
    optimizer.step()
```

### Dynamic Pruning

Prune during inference based on input:

```python
class DynamicPruner:
    def prune_by_activation(self, model, input, threshold=0.1):
        # Compute activations
        activations = model(input, output_hidden_states=True)

        # Prune weights contributing to low activations
        for layer, act in zip(model.layers, activations):
            mask = act.abs() > threshold
            layer.weight.data *= mask
```

## Common Issues & Solutions

### Issue 1: Accuracy Collapse

**Symptoms**: >10% accuracy drop after pruning.

**Solutions**:
1. Reduce sparsity (60% → 50% → 40%)
2. Use SparseGPT instead of magnitude pruning
3. Increase calibration data (32 → 128)
4. Don't prune critical layers (embeddings, output)
5. Consider layer-wise variable sparsity

### Issue 2: No Speedup Despite Sparsity

**Symptoms**: Pruned model same speed as dense.

**Causes & Solutions**:
- **No sparse kernels**: Use libraries with sparse support (e.g., SparseML, DeepSparse)
- **Unstructured on GPU**: Use N:M sparsity or structured pruning
- **Overhead**: Ensure sparsity >50% (overhead dominates below)

### Issue 3: Pruning Takes Too Long

**Solutions**:
1. Use Wanda instead of SparseGPT (no Hessian)
2. Reduce calibration samples (128 → 32)
3. Increase block size (128 → 256)
4. Prune layer-by-layer and checkpoint

### Issue 4: Memory Overflow During Pruning

**Solutions**:
1. Reduce block size (256 → 128 → 64)
2. Prune layer-by-layer
3. Use CPU for pruning (slower but more memory)
4. Reduce calibration batch size

### Issue 5: Uneven Sparsity Distribution

**Symptoms**: Some layers 90% sparse, others 10% sparse.

**Solution**: Enforce per-layer sparsity constraints:
```python
# Equal sparsity per layer
config = SparseGPTConfig(
    sparsity=0.5,
    uniform=True  # Enforce 50% sparsity in each layer
)
```

### Issue 6: Poor Generalization

**Symptoms**: Pruned model good on calibration data, poor on test.

**Solutions**:
1. Use more diverse calibration data
2. Increase calibration samples
3. Reduce sparsity
4. Fine-tune after pruning (if budget allows)

## Benchmarks

### Sparsity vs Accuracy (LLaMA-7B on MMLU)

| Sparsity | Method | Accuracy | vs Dense |
|----------|--------|----------|----------|
| 0% | Dense | 42.1% | 100% |
| 30% | SparseGPT | 41.6% | 98.8% |
| 50% | SparseGPT | 40.8% | 96.9% |
| 50% | Wanda | 40.2% | 95.5% |
| 60% | SparseGPT | 39.5% | 93.8% |
| 70% | SparseGPT | 36.7% | 87.2% |

### Pruning Time (A100 GPU)

| Model | Method | Sparsity | Time |
|-------|--------|----------|------|
| LLaMA-7B | Magnitude | 50% | 5 min |
| LLaMA-7B | Wanda | 50% | 20 min |
| LLaMA-7B | SparseGPT | 50% | 3 hrs |
| LLaMA-13B | SparseGPT | 50% | 5 hrs |
| LLaMA-70B | SparseGPT | 50% | 24 hrs |

### Inference Speed (A100 with Sparse Kernels)

| Model | Sparsity | Tokens/sec | Speedup | Notes |
|-------|----------|------------|---------|-------|
| LLaMA-7B | 0% | 32 | 1.0× | Dense baseline |
| LLaMA-7B | 50% (unstructured) | 38 | 1.2× | Limited by memory bandwidth |
| LLaMA-7B | 50% (2:4 pattern) | 52 | 1.6× | Hardware accelerated |
| LLaMA-7B | 25% (SliceGPT) | 44 | 1.4× | Structured, no sparse kernels |
| LLaMA-7B | 30% (ShortGPT) | 46 | 1.4× | Layer removal |

**Note**: Speedup varies by hardware. CPUs and edge devices benefit more from structured pruning.

### Combined Compression (Pruning + Quantization)

| Method Combination | Size | Accuracy | Speed |
|-------------------|------|----------|-------|
| Dense FP16 | 13 GB | 42.1% | 1.0× |
| Pruning (50%) only | 6.5 GB | 40.8% | 1.2× |
| Quantization (4-bit) only | 3.5 GB | 41.6% | 3.2× |
| **Pruning (50%) + Quant (4-bit)** | **1.75 GB** | **40.3%** | **3.8×** |

Pruning and quantization are complementary!

## Tools & Libraries

### Production-Ready

- **SparseML**: https://github.com/neuralmagic/sparseml
  - Production-grade sparse inference
  - DeepSparse engine
  - Supports SparseGPT, magnitude, gradual pruning

- **Neural Compressor**: https://github.com/intel/neural-compressor
  - Intel-optimized pruning
  - Good CPU inference with sparsity
  - Multiple pruning algorithms

### Research

- **Nexus**: `/Users/kevinyu/Projects/Nexus/nexus/models/compression/pruning/`
- **SparseGPT**: https://github.com/IST-DASLab/sparsegpt
- **Wanda**: https://github.com/locuslab/wanda

## References

### Papers

1. **SparseGPT**: Frantar & Alistarh. "SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot." ICML 2023.
2. **Wanda**: Sun et al. "A Simple and Effective Pruning Approach for Large Language Models." ICLR 2024.
3. **SliceGPT**: Ashkboos et al. "SliceGPT: Compress Large Language Models by Deleting Rows and Columns." arXiv 2024.
4. **ShortGPT**: Men et al. "ShortGPT: Layers in Large Language Models are More Redundant Than You Expect." arXiv 2024.
5. **Optimal Brain Surgeon**: Hassibi & Stork. "Second Order Derivatives for Network Pruning: Optimal Brain Surgeon." NeurIPS 1992.

### Surveys

- Hoefler et al. "Sparsity in Deep Learning: Pruning and Growth for Efficient Inference and Training." JMLR 2021.
- Blalock et al. "What is the State of Neural Network Pruning?" MLSys 2020.

## See Also

- [Quantization Methods](../quantization/README.md): Reduce precision
- [PEFT Methods](../peft/README.md): Parameter-efficient fine-tuning
- [Distillation](../distillation/README.md): Knowledge transfer
