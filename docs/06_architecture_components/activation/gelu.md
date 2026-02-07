# GELU: Gaussian Error Linear Unit

## Overview & Motivation

GELU (Gaussian Error Linear Unit) is a smooth, non-monotonic activation function that became the standard for transformer-based language models before the rise of SwiGLU. It provides better performance than ReLU while maintaining computational efficiency, making it ideal for deep learning models.

**Key Achievement**: The activation function of choice for major pre-2022 models:
- BERT (Google, 2018)
- GPT-2, GPT-3 (OpenAI, 2019-2020)
- RoBERTa (Facebook, 2019)
- T5 (Google, 2019)
- ELECTRA (Google, 2020)

**Performance**: Consistently outperforms ReLU by 1-3% on language tasks while being smoother and more biologically plausible.

## Theoretical Background

### Motivation for GELU

Traditional activation functions have limitations:

**ReLU**: `f(x) = max(0, x)`
- Hard threshold at 0
- Not differentiable at x=0
- Zero gradient for negative inputs (dead neurons)
- Deterministic gating

**ELU**: `f(x) = x if x > 0 else α(e^x - 1)`
- Smooth for negative values
- Non-zero gradient for x < 0
- Still has hard threshold at 0

**GELU's Innovation**: Smooth, probabilistic gating that combines properties of dropout and ReLU.

### Core Idea

GELU can be interpreted as a smooth approximation to a stochastic regularizer:

**Stochastic perspective**:
```
During training, randomly multiply input by 0 or 1:
x → x * m, where m ~ Bernoulli(Φ(x))
Φ(x) = P(X ≤ x), X ~ N(0, 1)
```

**Deterministic expectation**:
```
GELU(x) = x * Φ(x)
        = x * P(X ≤ x) where X ~ N(0, 1)
```

**Intuition**: Weight the input by its probability under a Gaussian distribution. Inputs far above the mean are weighted close to 1, inputs far below are weighted close to 0, with smooth transitions.

### Mathematical Definition

The exact GELU is defined as:

```
GELU(x) = x · Φ(x)
        = x · (1/2)[1 + erf(x/√2)]
        = x · P(X ≤ x) where X ~ N(0, 1)

where:
Φ(x) = Gaussian CDF (cumulative distribution function)
erf(x) = Error function = (2/√π) ∫₀ˣ e^(-t²) dt
```

**Properties**:
- Smooth everywhere (infinitely differentiable)
- Non-monotonic (has negative region)
- Non-linear (enables deep learning)
- Self-gated (output depends on input magnitude)

## Mathematical Formulation

### Forward Pass

**Exact GELU**:
```
GELU(x) = x · Φ(x)
        = x · (1/2) · [1 + erf(x/√2)]

where:
erf(x) = (2/√π) ∫₀ˣ e^(-t²) dt
```

**Using standard normal CDF**:
```
Φ(x) = (1/2)[1 + erf(x/√2)]

Therefore:
GELU(x) = x · Φ(x)
```

**Component breakdown**:
1. Compute Φ(x) (Gaussian CDF)
2. Multiply input by Φ(x)
3. Result: smooth gating of input

### Approximations

Computing the exact error function is expensive. Two common approximations:

**Tanh Approximation (most common)**:
```
GELU(x) ≈ 0.5 · x · [1 + tanh(√(2/π) · (x + 0.044715 · x³))]

Let:
y = √(2/π) · (x + 0.044715 · x³)

Then:
GELU(x) ≈ 0.5 · x · (1 + tanh(y))
```

**Accuracy**: Very close to exact GELU (error < 0.001 for typical ranges).

**Sigmoid Approximation (faster)**:
```
GELU(x) ≈ x · σ(1.702 · x)

where:
σ(x) = 1 / (1 + e^(-x))  (sigmoid function)
```

**Accuracy**: Slightly less accurate but faster to compute.

### Derivative (Gradient)

**Exact derivative**:
```
d/dx GELU(x) = Φ(x) + x · φ(x)

where:
φ(x) = (1/√(2π)) · e^(-x²/2)  (Gaussian PDF)
```

**Using tanh approximation**:
```
Let:
y = √(2/π) · (x + 0.044715 · x³)

d/dx GELU(x) ≈ 0.5 · [1 + tanh(y)] + 0.5 · x · sech²(y) · dy/dx

where:
dy/dx = √(2/π) · (1 + 0.134145 · x²)
sech²(y) = 1 - tanh²(y)
```

**Gradient properties**:
- Non-zero for all x (no dead neurons)
- Smooth and continuous
- Maximum gradient around x ≈ 1.4
- Approaches 1 for large positive x
- Approaches 0 for large negative x

## Implementation Details

### PyTorch Implementation

**Using built-in GELU**:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Method 1: Functional API
x = torch.randn(10)
y = F.gelu(x)

# Method 2: Module API
gelu = nn.GELU()
y = gelu(x)

# Method 3: Specify approximation
y = F.gelu(x, approximate='none')  # Exact (default)
y = F.gelu(x, approximate='tanh')  # Tanh approximation
```

**Manual implementation (exact)**:
```python
import torch
import math

def gelu_exact(x: torch.Tensor) -> torch.Tensor:
    """Exact GELU using error function."""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GELUExact(nn.Module):
    """Exact GELU activation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
```

**Manual implementation (tanh approximation)**:
```python
import torch
import torch.nn as nn
import math

def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    """GELU with tanh approximation (fast, accurate)."""
    return 0.5 * x * (1.0 + torch.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
    ))


class GELUTanh(nn.Module):
    """GELU with tanh approximation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
        ))
```

**Manual implementation (sigmoid approximation)**:
```python
import torch
import torch.nn as nn

def gelu_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """GELU with sigmoid approximation (fastest)."""
    return x * torch.sigmoid(1.702 * x)


class GELUSigmoid(nn.Module):
    """GELU with sigmoid approximation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)
```

### Complete Implementation with Variants

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Literal

class GELU(nn.Module):
    """Gaussian Error Linear Unit with multiple approximations.

    Args:
        approximate: Approximation method:
            - 'none': Exact GELU using error function
            - 'tanh': Tanh approximation (default, fast and accurate)
            - 'sigmoid': Sigmoid approximation (fastest)
    """

    def __init__(self, approximate: Literal['none', 'tanh', 'sigmoid'] = 'tanh'):
        super().__init__()
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.approximate == 'none':
            # Exact GELU
            return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

        elif self.approximate == 'tanh':
            # Tanh approximation (most common)
            return 0.5 * x * (1.0 + torch.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
            ))

        elif self.approximate == 'sigmoid':
            # Sigmoid approximation (fastest)
            return x * torch.sigmoid(1.702 * x)

        else:
            raise ValueError(f"Unknown approximation: {self.approximate}")

    def extra_repr(self) -> str:
        return f'approximate={self.approximate}'


# Usage examples
gelu_exact = GELU(approximate='none')
gelu_tanh = GELU(approximate='tanh')
gelu_sigmoid = GELU(approximate='sigmoid')

x = torch.randn(100, 512)
y_exact = gelu_exact(x)
y_tanh = gelu_tanh(x)
y_sigmoid = gelu_sigmoid(x)
```

## Usage in Neural Networks

### Feed-Forward Networks

**Standard FFN with GELU**:
```python
class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, dim)

        Returns:
            Output tensor (batch, seq_len, dim)
        """
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Example usage
ffn = FeedForward(dim=768, hidden_dim=3072, dropout=0.1)
x = torch.randn(32, 128, 768)
output = ffn(x)
```

### Transformer Block with GELU

**Complete transformer block**:
```python
class TransformerBlock(nn.Module):
    """Transformer block with GELU activation (BERT-style)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        gelu_approximate: str = 'tanh',
    ):
        super().__init__()

        # Pre-norm for attention
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )

        # Pre-norm for FFN
        self.norm2 = nn.LayerNorm(dim)

        # FFN with GELU
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(approximate=gelu_approximate),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention block
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]

        # FFN block
        x = x + self.mlp(self.norm2(x))

        return x


# Example
block = TransformerBlock(dim=768, num_heads=12, gelu_approximate='tanh')
x = torch.randn(32, 128, 768)
output = block(x)
```

### Vision Models with GELU

**ConvNet with GELU**:
```python
class ConvBlock(nn.Module):
    """Convolutional block with GELU activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.gelu(x)
        return x
```

## GELU Variants and Extensions

### GeGLU (GELU-Gated Linear Unit)

Combines GELU with gated linear units:

```python
class GeGLU(nn.Module):
    """GELU-Gated Linear Unit.

    GeGLU(x) = GELU(xW) ⊙ (xV)

    Used in: GPT-J, Falcon
    Reference: https://arxiv.org/abs/2002.05202
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = None,
        bias: bool = False,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim * 4

        self.w_gate = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_up = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_down = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (..., dim)

        Returns:
            Output tensor (..., dim)
        """
        gate = F.gelu(self.w_gate(x))
        up = self.w_up(x)
        return self.w_down(gate * up)


# Usage
geglu = GeGLU(dim=768, hidden_dim=3072)
x = torch.randn(32, 128, 768)
output = geglu(x)
```

### QuickGELU (OpenAI CLIP)

Faster approximation used in CLIP:

```python
class QuickGELU(nn.Module):
    """QuickGELU approximation used in OpenAI CLIP.

    QuickGELU(x) = x · σ(1.702 · x)

    where σ is the sigmoid function.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


# Usage in CLIP-style MLP
class CLIPMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.activation = QuickGELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
```

## Comparison with Other Activations

### GELU vs ReLU

| Aspect | GELU | ReLU |
|--------|------|------|
| Formula | x · Φ(x) | max(0, x) |
| Smoothness | Smooth everywhere | Non-differentiable at 0 |
| Gradient | Non-zero everywhere | Zero for x < 0 |
| Dead neurons | No | Yes |
| Computation | Moderate (with approx) | Very fast |
| Performance (NLP) | +1-3% better | Baseline |
| Performance (Vision) | Similar | Similar |

**When to use GELU**: Transformers, language models, when quality > speed.

**When to use ReLU**: CNNs, when speed > quality, very deep networks.

### GELU vs SwiGLU

| Aspect | GELU | SwiGLU |
|--------|------|--------|
| Formula | x · Φ(x) | Swish(xW) ⊙ (xV) |
| Parameters | None | 3 weight matrices |
| FLOPs | Low | 1.5x higher |
| Performance | Baseline | +5-10% better |
| Used in | BERT, GPT-2, GPT-3 | Llama, Mistral, PaLM |
| Era | 2018-2021 | 2022+ |

**Trend**: Modern LLMs have largely switched from GELU to SwiGLU.

### GELU vs Swish/SiLU

| Aspect | GELU | Swish/SiLU |
|--------|------|------------|
| Formula | x · Φ(x) | x · σ(x) |
| Distribution | Gaussian CDF | Sigmoid |
| Origin | Probabilistic | Search-based |
| Performance | Very similar | Very similar |
| Usage | NLP (BERT era) | Vision, mobile |

**Note**: GELU and Swish are very similar in practice. Choice is often historical.

### Activation Function Comparison Chart

```
1.0 |                    ___---
    |                ___/
    |            ___/
0.5 |        ___/
    |    ___/
    | __/
0.0 |/________________
    |          GELU (smooth)
   -3  -2  -1  0   1   2   3

1.0 |              ___---
    |          ___/
    |      ___/
0.5 |  ___/
    |_/
    |
0.0 |________________
    |          Swish/SiLU
   -3  -2  -1  0   1   2   3

1.0 |              ___---
    |          ___/
    |      ___/
0.5 |  ___/
    |_|
    |
0.0 |________________
    |          ReLU (hard threshold)
   -3  -2  -1  0   1   2   3
```

## Practical Considerations

### When to Use GELU

**Best for**:
- Transformer language models (BERT-style)
- Pre-training large models
- Tasks requiring smooth gradients
- When using Pre-Norm transformers
- Models with many layers (50+)

**Not recommended for**:
- Modern LLMs → Use SwiGLU instead
- Inference-critical applications → Consider QuickGELU
- Very simple models → ReLU is sufficient
- CNNs for image classification → ReLU often sufficient

### Approximation Choice

**Exact GELU (`approximate='none'`)**:
- Most accurate
- ~2x slower than approximations
- Use for: Research, benchmarking

**Tanh approximation (`approximate='tanh'`)**:
- Very accurate (error < 0.001)
- Standard choice in practice
- Use for: Production models, default choice

**Sigmoid approximation (QuickGELU)**:
- Slightly less accurate
- Fastest computation
- Use for: Inference optimization, CLIP-style models

### Performance Optimization

**Speed comparison** (relative to ReLU = 1.0):
```
ReLU:          1.0x (baseline)
QuickGELU:     1.2x
GELU (tanh):   1.5x
GELU (exact):  3.0x
SwiGLU:        2.0x
```

**Recommendations**:
```python
# Training large models
activation = nn.GELU(approximate='tanh')  # Good balance

# Inference optimization
activation = QuickGELU()  # Fastest approximation

# Research/benchmarking
activation = nn.GELU(approximate='none')  # Most accurate

# Modern LLMs
from nexus.components.activations import SwiGLU
activation = SwiGLU(dim=768)  # Best quality
```

### Memory Considerations

GELU is not a memory bottleneck:
- No learnable parameters
- Minimal activation memory
- Can be recomputed during backward pass

**Activation checkpointing**:
```python
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    # Checkpoint GELU computation to save memory
    x = self.fc1(x)
    x = checkpoint(self.gelu, x)  # Recompute in backward
    x = self.fc2(x)
    return x
```

## Advanced Topics

### Theoretical Properties

**Non-monotonicity**:
GELU has a slight negative region (x ≈ -0.2):
```
GELU(-0.2) ≈ -0.0084
```

**Bounded below**:
```
min GELU(x) ≈ -0.17  (at x ≈ -0.76)
```

**Asymptotic behavior**:
```
x → +∞: GELU(x) → x  (linear)
x → -∞: GELU(x) → 0  (bounded)
```

### Connection to Dropout

GELU can be viewed as an adaptive, deterministic dropout:

**Stochastic interpretation**:
```
Training: x' = x · m, m ~ Bernoulli(Φ(x))
Inference: x' = E[x · m] = x · Φ(x) = GELU(x)
```

**Adaptive dropout rate**:
```
p_drop(x) = 1 - Φ(x)

For x = -1: p_drop ≈ 0.84 (high dropout)
For x =  0: p_drop = 0.5  (moderate)
For x = +1: p_drop ≈ 0.16 (low dropout)
```

### Gradient Flow Analysis

GELU provides better gradient flow than ReLU:

**ReLU gradient**:
```
∂ReLU/∂x = 1 if x > 0 else 0
```
Problem: Zero gradient for x < 0 (dead neurons).

**GELU gradient**:
```
∂GELU/∂x = Φ(x) + x · φ(x)

Always > 0 for x > -1
Approaches 0 smoothly for x < -2
```

**Empirical observation**: Networks with GELU have fewer dead neurons.

### Initialization Considerations

GELU works well with standard initialization:

```python
# Xavier/Glorot initialization (default)
nn.init.xavier_uniform_(layer.weight)

# He initialization (for ReLU-like activations)
nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

# For GELU, Xavier typically works better
nn.init.xavier_normal_(layer.weight, gain=1.0)
```

## Code Examples from Nexus

### Using Nexus Activation Components

```python
import torch
from nexus.components.activations import GeGLU, GLUFeedForward

# GeGLU activation
geglu = GeGLU(dim=768, hidden_dim=3072)
x = torch.randn(32, 128, 768)
output = geglu(x)

# GLU-based FFN with GELU
ffn = GLUFeedForward(
    dim=768,
    activation='gelu',  # Use GELU variant
    dropout=0.1,
    norm_type='layer',
)
output = ffn(x)
```

### Complete BERT-Style Model

```python
import torch
import torch.nn as nn

class BERTBlock(nn.Module):
    """BERT-style transformer block with GELU."""

    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Layer norms
        self.norm1 = nn.LayerNorm(dim, eps=1e-12)  # BERT uses 1e-12
        self.norm2 = nn.LayerNorm(dim, eps=1e-12)

        # Multi-head attention
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)

        # Feed-forward with GELU
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(approximate='tanh'),  # BERT uses tanh approximation
            nn.Linear(hidden_dim, dim),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        # Post-Norm attention (BERT style)
        attn_out = self.attn(x, x, x, attn_mask=mask)[0]
        x = self.norm1(x + self.dropout1(attn_out))

        # Post-Norm FFN
        mlp_out = self.mlp(x)
        x = self.norm2(x + self.dropout2(mlp_out))

        return x


class BERTEncoder(nn.Module):
    """Stack of BERT blocks."""

    def __init__(
        self,
        num_layers: int = 12,
        dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            BERTBlock(dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x


# Example usage
encoder = BERTEncoder(num_layers=12, dim=768, num_heads=12)
x = torch.randn(32, 128, 768)  # (batch, seq_len, dim)
output = encoder(x)
print(output.shape)  # torch.Size([32, 128, 768])
```

### Benchmarking GELU Approximations

```python
import torch
import time

def benchmark_gelu_variants():
    """Benchmark different GELU implementations."""
    x = torch.randn(1000, 10000, device='cuda')

    # Warm up
    _ = F.gelu(x)

    # Exact GELU
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        y = F.gelu(x, approximate='none')
    torch.cuda.synchronize()
    exact_time = time.time() - start

    # Tanh approximation
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        y = F.gelu(x, approximate='tanh')
    torch.cuda.synchronize()
    tanh_time = time.time() - start

    # Sigmoid approximation (QuickGELU)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        y = x * torch.sigmoid(1.702 * x)
    torch.cuda.synchronize()
    sigmoid_time = time.time() - start

    print(f"Exact GELU:   {exact_time:.4f}s (1.00x)")
    print(f"Tanh approx:  {tanh_time:.4f}s ({exact_time/tanh_time:.2f}x)")
    print(f"Sigmoid approx: {sigmoid_time:.4f}s ({exact_time/sigmoid_time:.2f}x)")

# Run benchmark
benchmark_gelu_variants()
```

## Common Pitfalls and Solutions

### Issue 1: Wrong Approximation in Production

**Problem**: Using exact GELU slows down inference.

**Solution**:
```python
# Don't use exact GELU in production
gelu = nn.GELU(approximate='none')  # Slow!

# Use tanh approximation instead
gelu = nn.GELU(approximate='tanh')  # Fast and accurate

# Or QuickGELU for maximum speed
class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)
```

### Issue 2: Inconsistent Approximation with Pretrained Models

**Problem**: Loading pretrained weights trained with different approximation.

**Solution**:
```python
# Check model card for approximation used
# BERT models typically use tanh approximation
gelu = nn.GELU(approximate='tanh')

# GPT-2 PyTorch implementation uses exact
# But converting to tanh usually works fine
gelu = nn.GELU(approximate='tanh')  # Close enough
```

### Issue 3: Forgetting GELU in Custom Modules

**Problem**: Implementing FFN but using ReLU out of habit.

**Solution**:
```python
# Wrong (using ReLU when model expects GELU)
self.mlp = nn.Sequential(
    nn.Linear(dim, hidden_dim),
    nn.ReLU(),  # Wrong!
    nn.Linear(hidden_dim, dim),
)

# Correct (use GELU for transformer models)
self.mlp = nn.Sequential(
    nn.Linear(dim, hidden_dim),
    nn.GELU(approximate='tanh'),  # Correct
    nn.Linear(hidden_dim, dim),
)
```

### Issue 4: Performance Regression When Switching from ReLU

**Problem**: Replacing ReLU with GELU in CNN hurts performance.

**Solution**:
```python
# CNNs often work better with ReLU
# Don't blindly replace all activations

# For transformers: Use GELU
transformer_mlp = nn.Sequential(
    nn.Linear(768, 3072),
    nn.GELU(),
    nn.Linear(3072, 768),
)

# For CNNs: Usually stick with ReLU
cnn_block = nn.Sequential(
    nn.Conv2d(64, 128, 3, padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(inplace=True),  # ReLU is fine for CNNs
)
```

## Migration Guide

### From ReLU to GELU

```python
# Before (ReLU-based transformer)
class OldFFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# After (GELU-based transformer)
class NewFFN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.gelu = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))
```

**Expected improvements**: 1-3% better perplexity on language tasks.

### From GELU to SwiGLU (Modern LLMs)

```python
# Before (GELU FFN)
class GELUMlp(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))

# After (SwiGLU FFN)
from nexus.components.activations import SwiGLU

class SwiGLUMlp(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.swiglu = SwiGLU(dim, hidden_dim)

    def forward(self, x):
        return self.swiglu(x)
```

**Expected improvements**: 5-10% better perplexity on language tasks.

## References and Further Reading

### Original Papers

1. **Gaussian Error Linear Units (GELUs)** (Hendrycks & Gimpel, 2016)
   - https://arxiv.org/abs/1606.08415
   - Introduces GELU activation function

2. **BERT: Pre-training of Deep Bidirectional Transformers** (Devlin et al., 2018)
   - https://arxiv.org/abs/1810.04805
   - Popularizes GELU in transformers

3. **Language Models are Unsupervised Multitask Learners** (Radford et al., 2019)
   - GPT-2 paper, uses GELU in all FFN layers

### Comparison Studies

4. **GLU Variants Improve Transformer** (Shazeer, 2020)
   - https://arxiv.org/abs/2002.05202
   - Introduces GeGLU and other GLU variants

5. **Swish: A Self-Gated Activation Function** (Ramachandran et al., 2017)
   - https://arxiv.org/abs/1710.05941
   - Introduces Swish (SiLU), very similar to GELU

### Implementation References

6. **PyTorch Documentation**
   - https://pytorch.org/docs/stable/generated/torch.nn.GELU.html

7. **Hugging Face Transformers**
   - https://github.com/huggingface/transformers
   - Production BERT, GPT-2 implementations with GELU

## Summary

GELU is a smooth, probabilistic activation function that:

- **Dominated transformers 2018-2021** (BERT, GPT-2, GPT-3 era)
- **Outperforms ReLU** by 1-3% on language tasks
- **Has efficient approximations** (tanh, sigmoid)
- **Provides smooth gradients** (no dead neurons)
- **Being replaced by SwiGLU** in newest LLMs (2022+)

**Key Takeaways**:
- Use tanh approximation for best speed/accuracy trade-off
- Standard for BERT-style transformers
- Consider SwiGLU for new LLM architectures
- Works well with Pre-Norm and Post-Norm transformers

**When to use**:
- BERT-style models
- Pre-training transformers
- When you need smooth, non-monotonic activation
- As drop-in replacement for ReLU in transformers
