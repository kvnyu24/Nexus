# MXFP8: Microscaling FP8 for Block-Level Precision Training

## Overview

MXFP8 (OCP Specification, 2024) uses block-level scaling factors instead of tensor-level scaling, providing better dynamic range and numerical stability than standard FP8. Each block of elements (typically 16-32) shares a scaling factor.

## Key Differences from Standard FP8

| Aspect | Standard FP8 | MXFP8 |
|--------|--------------|-------|
| **Scaling** | Per-tensor | Per-block (16-32 elements) |
| **Dynamic Range** | Limited | Better (each block scales independently) |
| **Accuracy** | Lower | Higher |
| **Memory** | Minimal overhead | Small overhead (scales per block) |
| **Hardware** | NVIDIA H100 | AMD MI300, future GPUs |

## Mathematical Formulation

### Block-Level Quantization

For tensor $X$ divided into blocks $B_1, \\ldots, B_k$:

$$\\text{scale}_i = \\frac{\\max|B_i|}{FP8_{max}}$$

$$\\text{quant}(B_i) = \\text{round}\\left(\\frac{B_i}{\\text{scale}_i}\\right)$$

**Dequantization**:
$$B_i = \\text{quant}(B_i) \\times \\text{scale}_i$$

### FP8 Formats

**E4M3** (Forward pass):
- 1 sign bit, 4 exponent bits, 3 mantissa bits
- Range: [-448, 448]
- Better for activations (wider range)

**E5M2** (Backward pass):
- 1 sign bit, 5 exponent bits, 2 mantissa bits  
- Range: [-57344, 57344]
- Better for gradients (higher precision)

## Implementation

### Basic Usage

```python
from nexus.training.mixed_precision import to_mxfp8, MXFP8Linear

# Convert tensor to MXFP8
tensor_fp32 = torch.randn(1024, 1024)
mxfp8_tensor = to_mxfp8(tensor_fp32, block_size=32)

# Dequantize back
tensor_restored = mxfp8_tensor.dequantize()

# Use MXFP8 linear layer
layer = MXFP8Linear(768, 3072, block_size=32)
output = layer(input)
```

### Training with MXFP8

```python
from nexus.training.mixed_precision import MXFP8Config, MXFP8GradientScaler

config = MXFP8Config(
    block_size=32,
    e4m3_forward=True,  # E4M3 for forward
    e5m2_backward=True,  # E5M2 for backward
)

model = MyModel()  # Replace linear layers with MXFP8Linear
optimizer = torch.optim.AdamW(model.parameters())
scaler = MXFP8GradientScaler(config)

for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

## Memory Savings

**Formula**:
$$\\text{Savings} = 1 - \\frac{\\text{params} + \\text{num\_blocks} \\times 2}{\\text{params} \\times 4}$$

**Example** (1B parameters, block_size=32):
- FP32: 4GB
- MXFP8: 1GB + 122MB (scales) = 1.122GB
- **Savings**: 72%

## Performance Characteristics

**Accuracy**: ~0.1-0.5% loss vs FP32 (much better than standard FP8)  
**Memory**: ~75% reduction  
**Compute**: Depends on hardware support (AMD MI300 native)

## Hardware Support

**Native Support**:
- AMD MI300 series
- Future Intel GPUs

**Emulated** (current implementation):
- Works on any GPU
- Saves memory but no speed benefit
- Useful for prototyping

## When to Use

**Best for**:
- Training very large models (>10B params) where memory critical
- AMD MI300 hardware
- When standard FP8 accuracy insufficient

**Not for**:
- Small models (<1B params) - overhead not worth it
- When FP16/BF16 sufficient
- Hardware without native support (use FP16/BF16 instead)

## Configuration

**Block Size**:
- Smaller (16): Better accuracy, more overhead
- Larger (64): Less overhead, slightly lower accuracy
- **Recommended**: 32 (good balance)

## References

**OCP Microscaling Formats (MX) Specification**  
Open Compute Project, 2024  
https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf

**Implementation**: `nexus/training/mixed_precision/mxfp8.py`
