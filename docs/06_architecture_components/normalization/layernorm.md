# LayerNorm: Layer Normalization

## Overview & Motivation

Layer Normalization (LayerNorm) is a fundamental normalization technique that normalizes activations across the feature dimension for each example independently. Unlike Batch Normalization which normalizes across the batch dimension, LayerNorm normalizes across features, making it ideal for sequence models and transformers where batch statistics are unreliable.

**Key Achievement**: The foundation of transformer architectures:
- Original Transformer (Vaswani et al., 2017)
- BERT (Google, 2018)
- GPT-1, GPT-2 (OpenAI, 2018-2019)
- T5 (Google, 2019)
- Still used in vision models (ViT, CLIP)

**Performance**: Enables stable training of deep transformers, though newer models (Llama, Mistral) have largely replaced it with RMSNorm for efficiency.

## Theoretical Background

### The Normalization Problem

Deep neural networks suffer from internal covariate shift - the distribution of layer inputs changes during training as parameters update. This causes:
- Vanishing/exploding gradients
- Slower convergence
- Sensitivity to initialization
- Difficulty training deep networks

### Batch Normalization vs Layer Normalization

**Batch Normalization (BatchNorm)**:
```
Normalizes across batch dimension:
For feature i: normalize over all examples in batch
```

**Problems with BatchNorm**:
- Requires large batches for stable statistics
- Batch statistics differ between training and inference
- Doesn't work well with sequence models (variable lengths)
- Breaks independence between examples

**Layer Normalization (LayerNorm)**:
```
Normalizes across feature dimension:
For each example: normalize over all features
```

**Advantages of LayerNorm**:
- No dependence on batch size (works with batch_size=1)
- Same computation in training and inference
- Each example normalized independently
- Perfect for RNNs, transformers, sequence models

### Core Idea

For each training example, compute the mean and variance of all activations in a layer, then normalize:

```
LayerNorm(x) = γ · (x - μ) / √(σ² + ε) + β

where:
μ = (1/d) Σᵢ xᵢ           (mean across features)
σ² = (1/d) Σᵢ (xᵢ - μ)²   (variance across features)
γ, β = learnable affine parameters
ε = small constant for stability
```

**Intuition**:
- Standardize each layer's activations to have mean 0 and variance 1
- Learn optimal scale (γ) and shift (β) for each feature
- Reduces internal covariate shift

## Mathematical Formulation

### Forward Pass

Given input `x ∈ ℝ^d` (single example) or `x ∈ ℝ^(B×d)` (batch):

**Step 1: Compute Statistics (per example)**
```
For each example in the batch:

μ = (1/d) Σᵢ₌₁ᵈ xᵢ

σ² = (1/d) Σᵢ₌₁ᵈ (xᵢ - μ)²
   = (1/d) Σᵢ₌₁ᵈ xᵢ² - μ²

where:
d = feature dimension
μ = mean of features for this example
σ² = variance of features for this example
```

**Step 2: Normalize**
```
x_norm = (x - μ) / √(σ² + ε)

where:
ε = small constant (typically 1e-5)
```

After normalization:
- mean(x_norm) ≈ 0
- var(x_norm) ≈ 1

**Step 3: Affine Transformation**
```
output = γ ⊙ x_norm + β

where:
γ ∈ ℝ^d = learnable scale parameter
β ∈ ℝ^d = learnable shift parameter
⊙ = element-wise multiplication
```

**Purpose of γ and β**: Allow the network to undo normalization if needed. The network can learn γ=√(σ²) and β=μ to recover the original distribution.

### Backward Pass

Gradients through LayerNorm:

**Gradient w.r.t. scale and shift (simple)**:
```
∂L/∂γ = Σₑₓₐₘₚₗₑₛ (∂L/∂output) ⊙ x_norm

∂L/∂β = Σₑₓₐₘₚₗₑₛ (∂L/∂output)
```

**Gradient w.r.t. input (complex)**:
```
Let:
g = ∂L/∂output
x̂ = x_norm = (x - μ) / σ

Then:
∂L/∂x = (γ/σ) ⊙ [g - mean(g) - x̂ ⊙ mean(g ⊙ x̂)]

where:
σ = √(σ² + ε)
mean(·) is taken over feature dimension d
```

**Intuition**: The gradient must account for:
1. Direct effect through normalization
2. Indirect effect through changing μ
3. Indirect effect through changing σ²

### Numerical Stability

**Computing Variance**:
```
# Numerically unstable (cancellation):
σ² = mean(x²) - mean(x)²

# Numerically stable (Welford's method):
μ = mean(x)
σ² = mean((x - μ)²)
```

**Using rsqrt**:
```
# Standard approach:
x_norm = (x - μ) / sqrt(σ² + ε)

# Optimized (single reciprocal):
rstd = rsqrt(σ² + ε)  # 1/sqrt(σ² + ε)
x_norm = (x - μ) * rstd
```

## Implementation Details

### PyTorch Implementation

```python
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    """Layer Normalization.

    Args:
        normalized_shape: Input shape (int or list/tuple)
        eps: Small constant for numerical stability
        elementwise_affine: Whether to include learnable γ and β
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True
    ):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (..., normalized_shape)

        Returns:
            Normalized tensor of same shape
        """
        # Compute mean and variance over last dimension(s)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Apply affine transformation
        if self.elementwise_affine:
            x_norm = x_norm * self.weight + self.bias

        return x_norm
```

### Using PyTorch's Built-in LayerNorm

```python
import torch.nn as nn

# Standard usage
layer_norm = nn.LayerNorm(normalized_shape=768)

# Without affine parameters
layer_norm = nn.LayerNorm(768, elementwise_affine=False)

# Custom epsilon
layer_norm = nn.LayerNorm(768, eps=1e-6)

# Example
x = torch.randn(32, 128, 768)  # (batch, seq_len, hidden_dim)
output = layer_norm(x)  # Normalize over hidden_dim for each token
```

### 2D Layer Normalization (Vision)

For 2D images (B, C, H, W), normalize over spatial dimensions:

```python
class LayerNorm2d(nn.Module):
    """Layer Normalization for 2D inputs (images).

    Normalizes over channel, height, and width dimensions.
    Used in vision transformers and ConvNeXt.
    """

    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Normalized tensor of shape (B, C, H, W)
        """
        # Compute mean and variance over C, H, W dimensions
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        var = x.var(dim=(1, 2, 3), keepdim=True, unbiased=False)

        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Apply affine (broadcast over H, W)
        x_norm = x_norm * self.weight.view(1, -1, 1, 1)
        x_norm = x_norm + self.bias.view(1, -1, 1, 1)

        return x_norm


# Alternative: Normalize over H, W only (channel-wise)
class ChannelLayerNorm2d(nn.Module):
    """Channel-wise Layer Normalization for images."""

    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize over H, W dimensions only
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)

        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        x_norm = x_norm * self.weight.view(1, -1, 1, 1)
        x_norm = x_norm + self.bias.view(1, -1, 1, 1)

        return x_norm
```

## Usage in Transformers

### Pre-Norm vs Post-Norm

**Post-Norm (Original Transformer)**:
```python
# Post-LayerNorm (used in original Transformer)
def transformer_block_post_norm(x):
    # Attention
    x = x + attention(x)
    x = layer_norm_1(x)

    # FFN
    x = x + ffn(x)
    x = layer_norm_2(x)

    return x
```

**Pre-Norm (Modern Standard)**:
```python
# Pre-LayerNorm (used in GPT-2, GPT-3, most modern models)
def transformer_block_pre_norm(x):
    # Attention
    x = x + attention(layer_norm_1(x))

    # FFN
    x = x + ffn(layer_norm_2(x))

    return x
```

**Key Differences**:
- **Post-Norm**: Normalize after residual addition
  - Better representation learning
  - Less stable training (requires warmup)
  - Used in original Transformer, BERT

- **Pre-Norm**: Normalize before sublayer
  - More stable training
  - Can train without warmup
  - Used in GPT-2, GPT-3, T5
  - Slightly worse performance at same depth
  - Enables training much deeper models

### Complete Transformer Block

```python
class TransformerBlock(nn.Module):
    """Standard transformer block with Pre-LayerNorm."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Pre-norm for attention
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)

        # Pre-norm for FFN
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention block with Pre-Norm
        x = x + self.attn(self.norm1(x))

        # FFN block with Pre-Norm
        x = x + self.mlp(self.norm2(x))

        return x
```

## LayerNorm Variants

### RMSNorm (Simplified LayerNorm)

RMSNorm removes mean-centering for efficiency:

```python
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # No mean subtraction, only RMS normalization
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return x_norm * self.weight
```

**When to use**: Modern LLMs (Llama, Mistral) - 10-20% faster than LayerNorm.

### DeepNorm (For Very Deep Networks)

DeepNorm scales residuals for training 1000+ layer transformers:

```python
class DeepNorm(nn.Module):
    """DeepNorm for ultra-deep transformers."""

    def __init__(self, dim: int, num_layers: int = 1):
        super().__init__()
        self.alpha = (2.0 * num_layers) ** 0.25  # Residual scaling
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, sublayer_out: torch.Tensor) -> torch.Tensor:
        # Post-norm with scaled residual
        return self.norm(x * self.alpha + sublayer_out)
```

**When to use**: Transformers with 100+ layers.

### QKNorm (For Large Head Dimensions)

Normalizes queries and keys before attention:

```python
class QKNorm(nn.Module):
    """Query-Key Normalization (used in Gemma 2)."""

    def __init__(self, head_dim: int):
        super().__init__()
        self.q_norm = nn.LayerNorm(head_dim)
        self.k_norm = nn.LayerNorm(head_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        return self.q_norm(q), self.k_norm(k)
```

**When to use**: Large head dimensions (256+) or low-precision training.

## Practical Considerations

### When to Use LayerNorm

**Best for**:
- Transformer architectures
- Sequence models (RNNs, LSTMs)
- Models with variable batch sizes
- Vision transformers
- Any model where batch statistics are unreliable

**Not recommended for**:
- CNNs for image classification → Use BatchNorm
- When you need faster inference → Consider RMSNorm
- Very large models → Consider RMSNorm (10-20% faster)

### Hyperparameter Tuning

**Epsilon (ε)**:
- Default: `1e-5` (PyTorch default)
- Lower precision: `1e-6` (used in some LLMs)
- Mixed precision: `1e-5` to `1e-6`
- Too small: Numerical instability
- Too large: Reduced normalization effect

**Affine Parameters**:
- Usually keep `elementwise_affine=True`
- Can disable for intermediate layers in very deep networks
- Disabling saves parameters but may hurt performance

### Placement in Architecture

**Standard Transformer**:
```python
# Pre-Norm (recommended)
x = x + attention(norm(x))
x = x + ffn(norm(x))

# Post-Norm (less stable)
x = norm(x + attention(x))
x = norm(x + ffn(x))
```

**Vision Transformer**:
```python
# After patch embedding
x = patch_embed(images)
x = x + pos_embed
x = layer_norm(x)  # Often added here

# In transformer blocks
for block in blocks:
    x = block(x)  # Each block has 2 LayerNorms

# Before head
x = layer_norm(x)  # Final norm before classification
```

### Performance Optimization

**Memory Efficiency**:
```python
# Standard LayerNorm (saves stats for backward)
x = layer_norm(x)

# In-place operations (when possible)
x = layer_norm(x)  # PyTorch handles this automatically

# Fused LayerNorm (CUDA kernel)
# apex.normalization.FusedLayerNorm  # 20-30% faster
```

**Computation Cost**:
- LayerNorm: ~0.1% of total transformer FLOPs
- Not a bottleneck for most models
- Optimization usually not critical unless very large hidden_dim

## Comparison with Other Normalizations

### LayerNorm vs BatchNorm

| Aspect | LayerNorm | BatchNorm |
|--------|-----------|-----------|
| Normalization dimension | Features | Batch |
| Batch size dependency | None | Requires large batches |
| Train/Inference | Same | Different (running stats) |
| Best for | Transformers, RNNs | CNNs |
| Speed | Slightly slower | Slightly faster |

### LayerNorm vs RMSNorm

| Aspect | LayerNorm | RMSNorm |
|--------|-----------|---------|
| Operations | Mean + Variance | RMS only |
| Parameters | γ and β | γ only |
| Speed | Baseline | 10-20% faster |
| Quality | Baseline | Equivalent |
| Used in | BERT, GPT-2, ViT | Llama, Mistral, Qwen |

### LayerNorm vs GroupNorm

| Aspect | LayerNorm | GroupNorm |
|--------|-----------|-----------|
| Normalization | All features | Feature groups |
| Use case | Transformers | CNNs (small batch) |
| Batch dependency | None | None |
| Flexibility | Fixed | Tunable (num_groups) |

## Advanced Topics

### LayerNorm Gradient Flow

LayerNorm improves gradient flow by:

1. **Preventing gradient explosion**: Normalization bounds gradient magnitudes
2. **Preventing gradient vanishing**: Maintains gradient scale across layers
3. **Reducing sensitivity to initialization**: Normalizes regardless of initial weights

**Empirical observation**: Networks with LayerNorm can use higher learning rates.

### Effect on Loss Landscape

LayerNorm smooths the loss landscape:

```
Without LayerNorm: Rugged landscape, slow optimization
With LayerNorm: Smoother landscape, faster convergence
```

**Research findings**:
- Reduces Lipschitz constant of loss function
- Makes optimization landscape more convex-like
- Enables larger learning rates

### Zero Initialization Trick

In very deep networks, initialize last layer before LayerNorm to zero:

```python
def init_transformer_block(block):
    # Initialize output projections to zero
    nn.init.zeros_(block.attn.out_proj.weight)
    nn.init.zeros_(block.mlp.fc2.weight)
```

**Effect**:
- Network starts as identity mapping
- Gradual learning from simple to complex
- Stabler training for 100+ layers

### LayerNorm and Attention

LayerNorm before attention prevents extreme attention scores:

```python
# Without LayerNorm
Q = X @ W_q  # Unbounded
K = X @ W_k
attn = softmax(Q @ K.T / sqrt(d))  # Can become peaked

# With LayerNorm
X_norm = layer_norm(X)  # Bounded variance
Q = X_norm @ W_q
K = X_norm @ W_k
attn = softmax(Q @ K.T / sqrt(d))  # More diffuse attention
```

## Code Examples from Nexus

### Using Nexus LayerNorm Components

```python
import torch
from nexus.components.normalization import RMSNorm, QKNorm, DeepNorm

# RMSNorm (faster alternative to LayerNorm)
norm = RMSNorm(dim=768, eps=1e-6)
x = torch.randn(32, 128, 768)
x_norm = norm(x)

# QKNorm for attention
qk_norm = QKNorm(head_dim=64)
q = torch.randn(32, 8, 128, 64)  # (batch, heads, seq_len, head_dim)
k = torch.randn(32, 8, 128, 64)
q_norm, k_norm = qk_norm(q, k)

# DeepNorm for very deep networks
deep_norm = DeepNorm(dim=768, num_layers=100)
x = torch.randn(32, 128, 768)
sublayer_out = torch.randn(32, 128, 768)
x_norm = deep_norm(x, sublayer_out)
```

### Building a Transformer with LayerNorm

```python
import torch
import torch.nn as nn
from typing import Optional

class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with Pre-LayerNorm."""

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = True,  # Pre-Norm if True
    ):
        super().__init__()
        self.norm_first = norm_first

        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # LayerNorms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Activation
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            src: Input tensor (batch, seq_len, d_model)
            src_mask: Attention mask

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        if self.norm_first:
            # Pre-Norm
            src = src + self._sa_block(self.norm1(src), src_mask)
            src = src + self._ff_block(self.norm2(src))
        else:
            # Post-Norm
            src = self.norm1(src + self._sa_block(src, src_mask))
            src = self.norm2(src + self._ff_block(src))

        return src

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Self-attention block."""
        x = self.self_attn(x, x, x, attn_mask=attn_mask)[0]
        return self.dropout1(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        """Feed-forward block."""
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


# Example usage
layer = TransformerEncoderLayer(
    d_model=512,
    nhead=8,
    dim_feedforward=2048,
    dropout=0.1,
    norm_first=True,  # Pre-Norm (recommended)
)

x = torch.randn(32, 100, 512)  # (batch, seq_len, d_model)
output = layer(x)
print(output.shape)  # torch.Size([32, 100, 512])
```

## Common Pitfalls and Solutions

### Issue 1: Training Instability with Post-Norm

**Problem**: Post-Norm transformers diverge during training.

**Solution**:
```python
# Use Pre-Norm instead
x = x + attention(layer_norm(x))  # Pre-Norm

# Or use learning rate warmup for Post-Norm
from torch.optim.lr_scheduler import LambdaLR

def lr_lambda(step):
    warmup_steps = 4000
    if step < warmup_steps:
        return step / warmup_steps
    return (warmup_steps / step) ** 0.5

scheduler = LambdaLR(optimizer, lr_lambda)
```

### Issue 2: Gradient Explosion in Deep Networks

**Problem**: Gradients explode in 50+ layer transformers.

**Solution**:
```python
# Use gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Or use DeepNorm
from nexus.components.normalization import DeepNorm

norm = DeepNorm(dim=768, num_layers=num_layers)
```

### Issue 3: Slow Inference

**Problem**: LayerNorm slows down inference.

**Solution**:
```python
# Switch to RMSNorm (10-20% faster)
from nexus.components.normalization import RMSNorm

norm = RMSNorm(dim=768, eps=1e-6)

# Or use fused LayerNorm
try:
    from apex.normalization import FusedLayerNorm
    norm = FusedLayerNorm(768)
except ImportError:
    norm = nn.LayerNorm(768)
```

### Issue 4: Numerical Instability in FP16

**Problem**: NaN values in mixed precision training.

**Solution**:
```python
# Increase epsilon
norm = nn.LayerNorm(768, eps=1e-6)  # Instead of 1e-5

# Or ensure LayerNorm runs in FP32
class LayerNormFP32(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.float()).type_as(x)
```

## References and Further Reading

### Original Papers

1. **Layer Normalization** (Ba et al., 2016)
   - https://arxiv.org/abs/1607.06450
   - Introduces LayerNorm, shows benefits for RNNs

2. **Attention Is All You Need** (Vaswani et al., 2017)
   - https://arxiv.org/abs/1706.03762
   - Uses LayerNorm in transformer architecture

3. **Understanding the Difficulty of Training Deep Feedforward Neural Networks** (Glorot & Bengio, 2010)
   - Early work on normalization importance

### Advanced Techniques

4. **RMSNorm** (Zhang & Sennrich, 2019)
   - https://arxiv.org/abs/1910.07467
   - Simplified LayerNorm used in modern LLMs

5. **DeepNet: Scaling Transformers to 1,000 Layers** (Wang et al., 2022)
   - https://arxiv.org/abs/2203.00555
   - Post-Norm with residual scaling

6. **On Layer Normalization in the Transformer Architecture** (Xiong et al., 2020)
   - https://arxiv.org/abs/2002.04745
   - Analysis of Pre-Norm vs Post-Norm

### Implementation References

7. **PyTorch Documentation**
   - https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html

8. **Hugging Face Transformers**
   - https://github.com/huggingface/transformers
   - Production-grade implementations

## Summary

LayerNorm is a fundamental normalization technique that:

- **Normalizes across features** instead of batch dimension
- **Enables stable transformer training** through better gradient flow
- **Works with any batch size** including batch_size=1
- **Used in Pre-Norm** (modern) or Post-Norm (original) configurations
- **Being replaced by RMSNorm** in newest LLMs for efficiency

**Key Takeaways**:
- Use Pre-Norm for better training stability
- Consider RMSNorm for production LLMs (10-20% faster)
- Essential for transformers, RNNs, and sequence models
- Not a computational bottleneck in most architectures

**When to use**:
- Default choice for transformers and sequence models
- Vision transformers (ViT, CLIP)
- Any architecture with variable batch sizes
- Models requiring stable training of deep networks
