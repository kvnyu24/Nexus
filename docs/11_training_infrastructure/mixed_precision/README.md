# Mixed Precision Training

Low-precision training techniques for memory and compute efficiency.

## Overview

Mixed precision training uses lower-precision number formats (FP16, FP8, FP4) to:
- Reduce memory usage (2-8x)
- Accelerate computation (2-4x on modern GPUs)
- Maintain model quality with careful scaling

## Format Comparison

| Format | Bits | Range | Precision | Memory vs FP32 | Hardware Support |
|--------|------|-------|-----------|----------------|------------------|
| **FP32** | 32 | ±3.4e38 | High | 1x | All GPUs |
| **FP16** | 16 | ±65504 | Medium | 0.5x | V100+ |
| **BF16** | 16 | ±3.4e38 | Medium | 0.5x | A100+ |
| **FP8 E4M3** | 8 | ±448 | Low | 0.25x | H100+ |
| **FP8 E5M2** | 8 | ±57344 | Low | 0.25x | H100+ |
| **MXFP8** | 8 + scales | Block-level | Medium | 0.26x | H100+, MI300 |
| **FP4/MXFP4** | 4 + scales | Block-level | Very Low | 0.13x | Future |

## Quick Start

```python
from nexus.training.mixed_precision import (
    FP8Linear,
    convert_to_fp8,
    FP8Format,
)

# Convert entire model to FP8
model = convert_to_fp8(model, fp8_format=FP8Format.E4M3)

# Or use FP8 layers directly
layer = FP8Linear(
    in_features=768,
    out_features=3072,
    fp8_format=FP8Format.E4M3,
)

# Training with FP8
for batch in dataloader:
    output = model(batch)  # Automatic FP8 computation
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
```

## Detailed Documentation

- [FP8 Training](fp8.md) - 8-bit floating point with dynamic scaling
- [MXFP8 (Microscaling FP8)](mxfp8.md) - Block-level scaled FP8
- [FP4/MXFP4](fp4.md) - 4-bit training for extreme memory reduction

## Memory Savings

For a 7B parameter model:

| Precision | Memory (GB) | Savings vs FP32 |
|-----------|-------------|-----------------|
| FP32 | 28.0 | 0% |
| FP16/BF16 | 14.0 | 50% |
| FP8 | 7.0 | 75% |
| MXFP8 | 7.3 | 74% |
| FP4 | 3.5 | 87.5% |
| MXFP4 | 3.7 | 86.8% |

## References

See individual format documentation for detailed references.
