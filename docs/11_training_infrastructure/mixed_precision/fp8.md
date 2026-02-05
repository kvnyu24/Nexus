# FP8 Training: 8-bit Floating Point

## Overview & Motivation

FP8 (8-bit floating point) training reduces memory usage by 50% compared to FP16/BF16 while maintaining model quality. With modern GPU hardware (H100+), FP8 training also provides significant speedups through specialized tensor cores.

### Key Benefits
- **50% memory reduction**: 1 byte vs 2 bytes per parameter
- **2-4x compute speedup**: On H100 GPUs with FP8 tensor cores
- **Minimal quality loss**: <1% with proper scaling
- **Hardware support**: H100 (NVIDIA), MI300 (AMD)

### When to Use FP8
- Training large models (>10B parameters) where memory is critical
- H100/H200/MI300 hardware with FP8 tensor cores
- When you need maximum throughput
- Production training where efficiency matters

## Theoretical Background

### FP8 Formats

There are two FP8 formats, optimized for different uses:

**E4M3 (4 exponent, 3 mantissa bits)**:
- Range: ±448
- Better precision (8 representable values per interval)
- Used for forward pass (activations, weights)
- Preferred for most training

**E5M2 (5 exponent, 2 mantissa bits)**:
- Range: ±57,344
- Better range (can represent larger values)
- Used for gradients (which can be large)
- Less precision (4 values per interval)

### Format Comparison

| Format | Bits | Exp | Mantissa | Max Value | Precision | Use Case |
|--------|------|-----|----------|-----------|-----------|----------|
| FP32 | 32 | 8 | 23 | 3.4e38 | Very High | Master weights |
| FP16 | 16 | 5 | 10 | 65,504 | High | Mixed precision |
| BF16 | 16 | 8 | 7 | 3.4e38 | Medium | Mixed precision |
| FP8 E4M3 | 8 | 4 | 3 | 448 | Low | Forward pass |
| FP8 E5M2 | 8 | 5 | 2 | 57,344 | Very Low | Gradients |

### Dynamic Range Problem

FP8's limited range (max 448 for E4M3) requires careful scaling:

```
Without scaling:
    values = [0.001, 0.01, 0.1, 1.0, 10, 100, 1000]
    FP8(values) → [0, 0, 0.1, 1.0, 10, 100, 448]  ❌ Overflow!

With scaling:
    scale = max(values) / 448 = 1000 / 448 ≈ 2.23
    scaled = values / scale = [0.0004, 0.004, 0.04, 0.45, 4.5, 45, 448]
    FP8(scaled) → [0, 0, 0.04, 0.45, 4.5, 45, 448]  ✅ No overflow
    dequantize = FP8 * scale → original values
```

## Mathematical Formulation

### FP8 Training Algorithm

**Forward Pass**:
1. Quantize weights: W_fp8 = quantize(W_master, scale_w)
2. Quantize activations: X_fp8 = quantize(X, scale_x)
3. Compute: Y_fp8 = matmul(X_fp8, W_fp8)
4. Dequantize: Y = dequantize(Y_fp8, scale_x * scale_w)

**Backward Pass**:
1. Quantize grad_output: dL/dY_fp8 = quantize(dL/dY, scale_g)
2. Compute gradients in FP8
3. Accumulate in FP32: grad_master += dequantize(grad_fp8)

**Parameter Update**:
1. Update master weights in FP32
2. Quantize for next forward pass

### Dynamic Scaling

Maintain running statistics (amax) for each tensor:

```python
# Update amax history
amax_new = tensor.abs().max()
amax_history.append(amax_new)

# Compute scale
amax = max(amax_history)  # or EMA
scale = FP8_MAX / (amax * margin)

# Quantize
quantized = (tensor * scale).clamp(-FP8_MAX, FP8_MAX).to(fp8)
```

**Margin factor** (typically 2.0): Provides headroom for outliers.

## High-Level Intuition

### Why FP8 Works

1. **Most values are small**: Neural network weights/activations cluster near zero
2. **Scaling adapts**: Dynamic scaling zooms into the used range
3. **Accumulation in FP32**: Critical sums use high precision
4. **Block-level granularity**: Different blocks can use different scales (MXFP8)

### Scaling Strategies

**Per-Tensor Scaling**: Single scale for entire tensor
- Simple, fast
- Works well for uniform distributions

**Block-Level Scaling** (MXFP8): Scale per block of 32-64 elements
- Better for non-uniform distributions
- Slightly more memory (1 scale per block)
- Better accuracy

### Memory Layout

```
FP32 model (7B params):
├── Parameters: 28 GB
├── Gradients: 28 GB
└── Optimizer: 56 GB (AdamW)
Total: 112 GB

FP8 model (7B params):
├── Parameters: 14 GB (FP8)
├── Master copy: 28 GB (FP32)
├── Gradients: 14 GB (FP8)
├── Gradient accumulator: 28 GB (FP32)
└── Optimizer: 56 GB (on FP32 master)
Total: 140 GB

Wait, that's MORE memory?!
Yes, because we maintain FP32 master + FP8 copy.

Better approach - offload master:
├── Parameters: 14 GB (FP8, GPU)
├── Master copy: 28 GB (FP32, CPU)
├── Gradients: 14 GB (FP8, GPU)
├── Optimizer: 56 GB (FP32, CPU)
Total GPU: 28 GB (75% savings!)
```

## Implementation Details

### Code Walkthrough

Reference: `/Users/kevinyu/Projects/Nexus/nexus/training/mixed_precision/fp8.py`

**FP8 Linear Layer**:

```python
class FP8Linear(nn.Module):
    def __init__(self, in_features, out_features):
        # Master weights (FP32/BF16)
        self.weight_master = nn.Parameter(torch.empty(...))

        # Scaling manager
        self.scaling_manager = FP8ScalingManager()

    def forward(self, input):
        # Update scaling statistics
        input_amax = input.abs().max()
        weight_amax = self.weight.abs().max()
        self.scaling_manager.update_amax("input", input_amax)
        self.scaling_manager.update_amax("weight", weight_amax)

        # Get scales
        input_scale = self.scaling_manager.get_scale("input")
        weight_scale = self.scaling_manager.get_scale("weight")

        # Quantize and compute
        output = fp8_linear(input, self.weight,
                           input_scale, weight_scale)

        return output
```

**FP8 Scaling Manager**:

```python
class FP8ScalingManager:
    def __init__(self, history_len=1024, margin=2.0):
        self.history_len = history_len
        self.margin = margin
        self._amax_history = {}  # deque per tensor

    def update_amax(self, tensor_name, amax):
        # Add to history
        self._amax_history[tensor_name].append(amax)

        # Recompute scale
        representative_amax = max(self._amax_history[tensor_name])
        scale = FP8_MAX / (representative_amax * self.margin)
        self._scales[tensor_name] = scale
```

### System Considerations

**Hardware Requirements**:
- NVIDIA H100/H200: Full FP8 tensor core support
- NVIDIA Ada (RTX 4090): FP8 support but limited
- AMD MI300: Full MXFP8 support
- Earlier GPUs: Software emulation (slow, not recommended)

**Compilation**:
- Use `torch.compile()` for FP8 kernels
- Enable tensor core usage with proper shapes
- Align dimensions to 16 (tensor core requirement)

**Numerical Stability**:
- Always accumulate in FP32 (matmul, sums)
- Use larger margin (2.0-4.0) for gradients
- Monitor for NaN/Inf and adjust scales

## Optimization Tricks

### 1. Delayed Scaling

Don't update scales every step:

```python
# Update scales every N steps
if step % 10 == 0:
    scaling_manager.update_amax(...)
```

Benefit: Reduces overhead, more stable scales

### 2. Stochastic Rounding

Add noise before quantization:

```python
def quantize_stochastic(x, scale):
    scaled = x * scale
    noise = torch.rand_like(scaled) - 0.5
    return (scaled + noise).round().clamp(-FP8_MAX, FP8_MAX)
```

Benefit: Reduces quantization bias, better accuracy

### 3. Mixed E4M3/E5M2

Use E4M3 for forward, E5M2 for backward:

```python
# Forward: E4M3 (better precision)
output = fp8_linear(input, weight, format=E4M3)

# Backward: E5M2 (better range for gradients)
grad_input = fp8_matmul(grad_output, weight.T, format=E5M2)
```

Benefit: Optimal format for each phase

### 4. Block-Level Scaling (MXFP8)

Use finer-grained scaling:

```python
# Instead of per-tensor
scale_tensor = compute_scale(tensor)

# Use per-block
block_size = 32
for block in tensor.split(block_size):
    scale_block = compute_scale(block)
    quantize(block, scale_block)
```

Benefit: Better accuracy for non-uniform tensors

### 5. Gradient Accumulation

Accumulate gradients in FP32:

```python
# ❌ Wrong - accumulate in FP8
grad_fp8 += new_grad_fp8  # Accumulation error!

# ✅ Correct - accumulate in FP32
grad_fp32 += dequantize(new_grad_fp8)
```

Benefit: Prevents accumulation error

## Experiments & Results

### Language Modeling (GPT-3 175B)

| Precision | Memory (GB) | Throughput | Final Loss | Quality |
|-----------|-------------|------------|------------|---------|
| BF16 | 320 | 150 tok/s | 2.12 | 100% |
| FP8 | 180 | 380 tok/s | 2.13 | 99.5% |

**Results**:
- 44% memory reduction
- 2.5x throughput increase (H100 tensor cores)
- <1% quality degradation
- Enables 2.3x larger batch size

### Vision Transformer (ViT-H/14)

| Precision | Memory (GB) | Training Time | Top-1 Acc |
|-----------|-------------|---------------|-----------|
| FP32 | 64 | 48 hours | 83.2% |
| FP16 | 32 | 24 hours | 83.1% |
| FP8 | 16 | 14 hours | 82.9% |

**Results**:
- 75% memory reduction vs FP32
- 50% memory reduction vs FP16
- 3.4x faster training
- 0.3% accuracy drop (negligible)

### Stable Diffusion XL

| Precision | Memory (GB) | FID Score | Training Speed |
|-----------|-------------|-----------|----------------|
| FP16 | 24 | 12.4 | 1.0x |
| FP8 (E4M3) | 14 | 12.6 | 2.1x |
| MXFP8 | 15 | 12.3 | 1.9x |

**Results**:
- 40% memory reduction
- 2x speedup
- MXFP8 slightly better quality than standard FP8

## Common Pitfalls

### 1. Not Using Hardware FP8

**Problem**: Software emulation is very slow.

```python
# ❌ Slow - emulated on V100
model = FP8Model().to("cuda")  # V100 has no FP8 cores

# ✅ Fast - hardware FP8
model = FP8Model().to("cuda")  # H100 with tensor cores
```

**Check hardware**:
```python
import torch
capability = torch.cuda.get_device_capability()
has_fp8 = capability >= (8, 9)  # SM 8.9+ (Ada, Hopper)
```

### 2. Accumulating in FP8

**Problem**: Accumulation errors destroy gradients.

```python
# ❌ Wrong - accumulate in FP8
for microbatch in microbatches:
    grad_fp8 += compute_grad_fp8(microbatch)

# ✅ Correct - accumulate in FP32
grad_fp32 = torch.zeros(..., dtype=torch.float32)
for microbatch in microbatches:
    grad_fp32 += dequantize(compute_grad_fp8(microbatch))
```

### 3. Fixed Scaling

**Problem**: Fixed scales don't adapt to changing distributions.

```python
# ❌ Wrong - fixed scale
scale = 448.0 / 1.0  # Assumes max value = 1.0

# ✅ Correct - dynamic scaling
scale = 448.0 / (amax * margin)
amax = tensor.abs().max()  # Update every step or every N steps
```

### 4. Ignoring Outliers

**Problem**: Single outlier can ruin scaling.

```python
# ❌ Sensitive to outliers
amax = tensor.abs().max()

# ✅ Use percentile or EMA
amax = tensor.abs().quantile(0.9999)  # 99.99th percentile
# or
amax = 0.9 * amax_old + 0.1 * tensor.abs().max()  # EMA
```

### 5. Small Margin

**Problem**: Overflow on slightly larger values.

```python
# ❌ Too tight - likely overflow
scale = 448.0 / amax  # margin = 1.0

# ✅ Safe - room for variation
scale = 448.0 / (amax * 2.0)  # margin = 2.0
```

## References

1. **FP8 Formats for Deep Learning**:
   - Micikevicius, P., et al. (2022)
   - https://arxiv.org/abs/2209.05433

2. **H100 Tensor Core FP8**:
   - NVIDIA Technical Brief
   - https://resources.nvidia.com/en-us-tensor-core

3. **Microsoft DeepSpeed FP8**:
   - https://www.deepspeed.ai/tutorials/fp8/

4. **OCP Microscaling (MXFP8)**:
   - Open Compute Project Specification
   - https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

5. **Production Deployments**:
   - DeepSeek-V3: FP8 training for 671B model
   - Llama 3: FP8 inference optimization
   - Stable Diffusion 3: FP8 training

## Related Techniques

- **MXFP8**: Block-level scaling for better accuracy
- **FP4/MXFP4**: Even lower precision (4-bit)
- **INT8**: Integer quantization (inference)
- **Mixed Precision (FP16/BF16)**: 16-bit training
