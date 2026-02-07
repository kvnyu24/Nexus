# Diffusion Transformer (DiT)

## 1. Overview and Motivation

### The Problem with U-Nets

Traditional diffusion models rely on U-Net architectures inherited from segmentation tasks. While U-Nets work well, they have several limitations:

- **Limited Scalability**: U-Nets don't scale as cleanly as transformers when increasing model capacity
- **Inductive Biases**: Strong spatial biases may limit flexibility for diverse modalities
- **Architectural Complexity**: Skip connections and multi-scale processing add complexity
- **Training Efficiency**: Harder to leverage modern distributed training infrastructure

### DiT's Solution

**Diffusion Transformer (DiT)** replaces the U-Net backbone with a Vision Transformer (ViT) architecture, bringing the benefits of transformers to diffusion models:

- **Clean Scaling**: Standard transformer depth/width scaling laws apply
- **Architectural Simplicity**: Uniform blocks without skip connections
- **Better Training**: Leverages transformer optimization techniques
- **State-of-the-Art Results**: Achieves best FID scores on ImageNet generation

**Key Innovation**: **adaLN-Zero** conditioning mechanism that modulates transformer activations based on timestep and class labels, initialized so each block starts as an identity function for stable deep network training.

### Architecture at a Glance

```
Input Latent (B, 4, 32, 32)
         ↓
    Patch Embed (2×2 patches)
         ↓
    Token Sequence (B, 256, 1152) + Positional Embedding
         ↓
    ┌─────────────────────┐
    │  DiT Block × 28     │
    │  (adaLN-Zero)       │ ← Timestep + Class Conditioning
    └─────────────────────┘
         ↓
    Final Layer (adaLN + Linear)
         ↓
    Unpatchify
         ↓
    Output Prediction (B, 4, 32, 32)
```

### Why It Matters

DiT demonstrates that:
1. **Transformers outperform U-Nets** when scaled properly for diffusion
2. **Simple architectures** can achieve state-of-the-art results
3. **Conditioning through normalization** (adaLN-Zero) is highly effective
4. **Latent diffusion** + transformers = powerful combination

## 2. Theoretical Background

### From U-Net to Transformer

**Traditional Approach (DDPM with U-Net):**
- Processes images at multiple resolutions
- Uses skip connections for spatial details
- Injects time via projection + addition
- Convolutional inductive biases

**DiT Approach (Transformer):**
- Treats image as sequence of patches
- Uniform transformer blocks without skips
- Injects time + class via adaptive normalization
- Self-attention for global context

### The ViT Foundation

DiT builds on Vision Transformer (ViT):

1. **Patch Embedding**: Split image into non-overlapping patches
2. **Position Encoding**: Add learnable position embeddings
3. **Transformer Encoder**: Process with self-attention
4. **Linear Head**: Project to output space

**Key Difference**: DiT adds conditioning on diffusion timestep and class labels through adaLN-Zero.

### Adaptive Layer Normalization (adaLN)

Standard LayerNorm:
```
LN(x) = γ * (x - μ) / σ + β
```

Adaptive LayerNorm (adaLN):
```
adaLN(x, c) = scale(c) * LN(x) + shift(c)
```

Where scale and shift are predicted from conditioning c (timestep + class).

### adaLN-Zero: Identity Initialization

**Problem**: Deep networks are hard to train

**Solution**: Initialize so each block starts as identity function

**adaLN-Zero**:
```
x_out = x + gate(c) * Attention(adaLN(x, c))
x_out = x + gate(c) * MLP(adaLN(x, c))
```

Where gate(c) is initialized to zero, making each block initially:
```
x_out = x + 0 * Attention(...) = x  (identity!)
```

This enables stable training of very deep networks (28+ blocks).

### Diffusion Process Recap

**Forward (noising)**:
```
x_t = √(α̅_t) x_0 + √(1 - α̅_t) ε
```

**Training Objective**:
```
L = E_t,ε [||ε - ε_θ(x_t, t, y)||²]
```

**Classifier-Free Guidance**:
```
ε̃ = ε_θ(x_t, t, ∅) + w * (ε_θ(x_t, t, y) - ε_θ(x_t, t, ∅))
```

DiT predicts the noise ε added at timestep t, conditioned on class label y.

## 3. Mathematical Formulation

### Model Architecture

**Input**: Noisy latent x_t ∈ ℝ^(C×H×W), timestep t ∈ [0, T-1], class y ∈ [0, K-1]

**Patch Embedding**:
```
x_patches = Patchify(x_t)  ∈ ℝ^(N×D)
where N = (H/P)² is number of patches, P is patch size
```

**Positional Encoding**:
```
x_0 = x_patches + E_pos  ∈ ℝ^(N×D)
```

**Conditioning Embedding**:
```
t_emb = MLP(SinusoidalEmbed(t))  ∈ ℝ^D
y_emb = Embed(y)  ∈ ℝ^D
c = t_emb + y_emb  ∈ ℝ^D
```

**Transformer Blocks**:
For each block l = 1, ..., L:

```
# Predict modulation parameters
(shift₁, scale₁, gate₁, shift₂, scale₂, gate₂) = MLP(SiLU(c))

# Attention with adaLN-Zero
x_norm = (scale₁ + 1) * LN(x_{l-1}) + shift₁
x_attn = MultiHeadAttention(x_norm)
x_l = x_{l-1} + gate₁ * x_attn

# MLP with adaLN-Zero
x_norm = (scale₂ + 1) * LN(x_l) + shift₂
x_mlp = MLP(x_norm)
x_l = x_l + gate₂ * x_mlp
```

**Final Layer**:
```
(shift_f, scale_f) = MLP(SiLU(c))
x_norm = (scale_f + 1) * LN(x_L) + shift_f
x_pred = Linear(x_norm)  ∈ ℝ^(N×P²C)
```

**Unpatchify**:
```
ε_pred = Unpatchify(x_pred)  ∈ ℝ^(C×H×W)
```

### Training Loss

**Simple MSE Loss**:
```
L_simple = ||ε - ε_θ(x_t, t, y)||²
```

**With Variance Prediction** (learn_sigma=True):
```
[ε_pred, Σ_pred] = ε_θ(x_t, t, y)
L = L_simple + λ * L_vlb(Σ_pred)
```
In practice, DiT often uses only L_simple.

### Classifier-Free Guidance

**Training**: Randomly drop class labels with probability p_uncond ≈ 0.1
```
y_dropped = {
    y      with probability 1 - p_uncond
    y_∅    with probability p_uncond
}
```

**Sampling**: Mix conditional and unconditional predictions
```
ε̃ = (1 - w) * ε_θ(x_t, t, ∅) + w * ε_θ(x_t, t, y)
  = ε_θ(x_t, t, ∅) + w * (ε_θ(x_t, t, y) - ε_θ(x_t, t, ∅))
```

Typical guidance scale: w = 4.0 for DiT-XL/2

### Model Variants

DiT comes in different sizes (similar to ViT-S/B/L/XL):

| Model | Depth | Hidden Dim | Heads | Params | Gflops |
|-------|-------|------------|-------|--------|--------|
| DiT-S/2 | 12 | 384 | 6 | 33M | 4 |
| DiT-B/2 | 12 | 768 | 12 | 130M | 15 |
| DiT-L/2 | 24 | 1024 | 16 | 458M | 56 |
| DiT-XL/2 | 28 | 1152 | 16 | 675M | 83 |

The "/2" indicates patch size P=2.

## 4. High-Level Intuition

### The Core Idea

Think of DiT as:
1. **Breaking the image into patches** (like cutting a photo into tiles)
2. **Treating patches as words** in a sentence
3. **Using transformer to understand relationships** between patches
4. **Conditioning on "when" (timestep) and "what" (class)** through normalization

### Why Patches?

**Problem**: Images are 2D, transformers expect 1D sequences

**Solution**: Treat image as grid of patches
- Original: (B, 4, 32, 32) = 4096 values
- Patchified: (B, 256, 32) = 256 patches of 32 features each
- Much more manageable for self-attention!

### Why adaLN-Zero?

**Intuition**: The model needs to know "when" (timestep) and "what" (class)

**Bad approach**: Concatenate conditioning → wastes parameters
**Good approach**: Modulate activations based on conditioning → efficient

**adaLN-Zero ensures**:
- At initialization, model outputs zero everywhere (safe!)
- During training, each block gradually learns useful transformations
- Deep networks remain stable

### The Denoising Process

**Timestep t=999** (pure noise):
- Model sees random patches
- Conditioning says "this should be an ImageNet cat"
- Predicts noise to remove for first denoising step

**Timestep t=500** (half-noised):
- Model sees partially formed shapes
- Uses self-attention to understand spatial relationships
- Predicts noise, gradually revealing structure

**Timestep t=0** (clean):
- Model sees nearly clean image
- Removes final artifacts
- Outputs sharp, class-consistent sample

### Classifier-Free Guidance Intuition

**Without guidance** (w=1.0):
- Model generates diverse samples
- May not strongly follow class label
- High variety, lower class consistency

**With guidance** (w=4.0):
- Model "pushes away" from unconditional
- Generates samples that strongly match class
- Lower variety, higher class consistency

Think of guidance as a "class adherence knob":
- w=1.0: Generate anything remotely cat-like
- w=4.0: Generate the most cat-like cat possible
- w=7.5: Generate an exaggerated, hyper-cat (may sacrifice realism)

## 5. Implementation Details

### Model Configuration

**Default DiT-XL/2 Config**:
```python
config = {
    # Architecture
    "input_size": 32,           # Latent spatial resolution
    "patch_size": 2,            # Patch size (32/2 = 16 patches per side)
    "in_channels": 4,           # VAE latent channels
    "hidden_dim": 1152,         # Transformer dimension
    "depth": 28,                # Number of transformer blocks
    "num_heads": 16,            # Attention heads
    "mlp_ratio": 4.0,           # MLP expansion ratio
    "dropout": 0.0,             # Dropout (usually 0 for generation)

    # Conditioning
    "num_classes": 1000,        # ImageNet classes
    "class_dropout_prob": 0.1,  # CFG training dropout

    # Diffusion
    "num_timesteps": 1000,      # Diffusion steps
    "beta_start": 0.0001,       # Linear schedule start
    "beta_end": 0.02,           # Linear schedule end
    "learn_sigma": True,        # Predict variance
}
```

### Key Components

#### 1. Patch Embedding

Converts latent image to patch sequence:

```python
class PatchEmbed(nn.Module):
    def __init__(self, input_size=32, patch_size=2,
                 in_channels=4, hidden_dim=1152):
        super().__init__()
        self.num_patches = (input_size // patch_size) ** 2
        # Use conv2d with kernel=stride=patch_size for non-overlapping patches
        self.proj = nn.Conv2d(in_channels, hidden_dim,
                             kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # (B, C, H, W) -> (B, D, H/P, W/P) -> (B, num_patches, D)
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
```

#### 2. Timestep Embedding

Embeds scalar timesteps to vectors:

```python
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_dim=1152):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def sinusoidal_embedding(self, t, dim=256):
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half_dim) / half_dim
        ).to(t.device)
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t):
        t_freq = self.sinusoidal_embedding(t)
        return self.mlp(t_freq)
```

#### 3. Label Embedding with CFG Dropout

```python
class LabelEmbedder(nn.Module):
    def __init__(self, num_classes=1000, hidden_dim=1152, dropout_prob=0.1):
        super().__init__()
        # +1 for null class
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_dim)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def forward(self, labels, train=True):
        # During training, randomly replace labels with null class
        if train and self.dropout_prob > 0:
            drop_mask = torch.rand(labels.shape[0]) < self.dropout_prob
            labels = torch.where(drop_mask.to(labels.device),
                               self.num_classes, labels)
        return self.embedding_table(labels)
```

#### 4. DiT Block with adaLN-Zero

```python
class DiTBlock(nn.Module):
    def __init__(self, hidden_dim=1152, num_heads=16,
                 mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        # Normalization without learnable parameters
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads,
                                         dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)

        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(dropout),
        )

        # Predict 6 modulation params from conditioning
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim),
        )
        # Zero initialization for identity at init
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x, c):
        # Split modulation parameters
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)

        # Attention with adaLN-Zero
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # MLP with adaLN-Zero
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)

        return x

def modulate(x, shift, scale):
    """Apply affine modulation: x * (1 + scale) + shift"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
```

#### 5. Final Layer

```python
class FinalLayer(nn.Module):
    def __init__(self, hidden_dim=1152, patch_size=2, out_channels=4):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.linear = nn.Linear(hidden_dim, patch_size**2 * out_channels)

        # AdaLN modulation
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )

        # Zero initialization
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)
```

### Unpatchify Operation

```python
def unpatchify(self, x):
    """
    x: (B, N, patch_size^2 * C)
    return: (B, C, H, W)
    """
    c = self.out_channels
    p = self.patch_size
    h = w = self.input_size // p

    # (B, N, P²C) -> (B, h, w, P, P, C)
    x = x.reshape(-1, h, w, p, p, c)
    # Rearrange to (B, C, h, P, w, P) -> (B, C, h*P, w*P)
    x = torch.einsum("nhwpqc->nchpwq", x)
    x = x.reshape(-1, c, h * p, w * p)
    return x
```

## 6. Code Walkthrough

### Full Training Loop

```python
import torch
import torch.nn.functional as F
from nexus.models.diffusion import DiT

# Initialize model
config = {
    "input_size": 32,
    "patch_size": 2,
    "in_channels": 4,
    "hidden_dim": 1152,
    "depth": 28,
    "num_heads": 16,
    "num_classes": 1000,
    "class_dropout_prob": 0.1,
    "num_timesteps": 1000,
}
model = DiT(config).cuda()

# EMA model for sampling
ema_model = torch.optim.swa_utils.AveragedModel(
    model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.9999)
)

# Optimizer (AdamW with weight decay)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)

# Training step
def train_step(batch):
    x_0, labels = batch  # x_0: (B, 4, 32, 32), labels: (B,)
    x_0 = x_0.cuda()
    labels = labels.cuda()

    # Forward pass
    loss_dict = model.compute_loss(x_0, labels)
    loss = loss_dict["loss"]

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # Update EMA
    ema_model.update_parameters(model)

    return loss.item()

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        loss = train_step(batch)
        print(f"Loss: {loss:.4f}")
```

### Sampling with Classifier-Free Guidance

```python
@torch.no_grad()
def sample_images(model, labels, cfg_scale=4.0, num_steps=250):
    """
    Generate images using DDPM sampling with CFG.

    Args:
        model: DiT model
        labels: Class labels (B,)
        cfg_scale: Guidance scale (1.0 = no guidance, 4.0 = strong)
        num_steps: Number of denoising steps

    Returns:
        Generated latents (B, 4, 32, 32)
    """
    batch_size = labels.shape[0]
    device = labels.device

    # Start from pure noise
    x = torch.randn(batch_size, 4, 32, 32, device=device)

    # Prepare unconditional labels
    y_null = torch.full_like(labels, 1000)  # null class

    # Timestep schedule (uniform)
    timesteps = torch.linspace(999, 0, num_steps, device=device).long()

    for i, t in enumerate(timesteps):
        t_batch = t.expand(batch_size)

        # Batched conditional + unconditional forward pass
        x_input = torch.cat([x, x], dim=0)
        t_input = torch.cat([t_batch, t_batch], dim=0)
        y_input = torch.cat([labels, y_null], dim=0)

        # Predict noise
        output = model(x_input, t_input, y_input)
        pred = output["prediction"]

        if model.learn_sigma:
            pred, _ = pred.chunk(2, dim=1)  # Ignore variance

        pred_cond, pred_uncond = pred.chunk(2, dim=0)

        # Apply classifier-free guidance
        pred_guided = pred_uncond + cfg_scale * (pred_cond - pred_uncond)

        # DDPM update step
        alpha_bar = model.alphas_cumprod[t]
        alpha_bar_prev = model.alphas_cumprod[t - 1] if t > 0 else 1.0
        beta_t = model.betas[t]

        # Predict x_0 from noise prediction
        x0_pred = (x - torch.sqrt(1 - alpha_bar) * pred_guided) / torch.sqrt(alpha_bar)
        x0_pred = torch.clamp(x0_pred, -1, 1)  # Clip for stability

        # Compute posterior mean
        coeff1 = torch.sqrt(alpha_bar_prev) * beta_t / (1 - alpha_bar)
        coeff2 = torch.sqrt(1 - beta_t) * (1 - alpha_bar_prev) / (1 - alpha_bar)
        mean = coeff1 * x0_pred + coeff2 * x

        # Add noise (except at last step)
        if t > 0:
            noise = torch.randn_like(x)
            variance = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar)
            x = mean + torch.sqrt(variance) * noise
        else:
            x = mean

    return x

# Usage
labels = torch.tensor([207, 360, 387, 974]).cuda()  # ImageNet classes
latents = sample_images(ema_model.module, labels, cfg_scale=4.0)

# Decode latents to images using VAE decoder
# images = vae.decode(latents)
```

### DDIM Sampling (Faster)

```python
@torch.no_grad()
def sample_ddim(model, labels, cfg_scale=4.0, num_steps=50):
    """DDIM sampling for faster generation."""
    batch_size = labels.shape[0]
    device = labels.device

    x = torch.randn(batch_size, 4, 32, 32, device=device)
    y_null = torch.full_like(labels, 1000)

    # Evenly spaced timesteps
    timesteps = torch.linspace(999, 0, num_steps, device=device).long()

    for i, t in enumerate(timesteps):
        t_batch = t.expand(batch_size)

        # CFG forward pass
        x_input = torch.cat([x, x], dim=0)
        t_input = torch.cat([t_batch, t_batch], dim=0)
        y_input = torch.cat([labels, y_null], dim=0)

        output = model(x_input, t_input, y_input)
        pred = output["prediction"]
        if model.learn_sigma:
            pred, _ = pred.chunk(2, dim=1)

        pred_cond, pred_uncond = pred.chunk(2, dim=0)
        pred_guided = pred_uncond + cfg_scale * (pred_cond - pred_uncond)

        # DDIM update (deterministic)
        alpha_bar = model.alphas_cumprod[t]
        alpha_bar_prev = model.alphas_cumprod[t - 1] if t > 0 else 1.0

        # Predict x_0
        x0_pred = (x - torch.sqrt(1 - alpha_bar) * pred_guided) / torch.sqrt(alpha_bar)
        x0_pred = torch.clamp(x0_pred, -1, 1)

        # Deterministic direction pointing to x_t
        dir_xt = torch.sqrt(1 - alpha_bar_prev) * pred_guided

        # DDIM step
        x = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt

    return x
```

## 7. Optimization Tricks

### 1. EMA (Exponential Moving Average)

**Critical for stable sampling**:

```python
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

# Create EMA model
ema_model = AveragedModel(
    model,
    multi_avg_fn=get_ema_multi_avg_fn(0.9999)  # decay rate
)

# Update after each training step
for batch in dataloader:
    loss = train_step(batch)
    ema_model.update_parameters(model)

# Use EMA for sampling
latents = sample_images(ema_model.module, labels)
```

**Why it works**:
- Smooths out training noise
- More stable weight trajectory
- Better sample quality

### 2. Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

def train_step(batch):
    x_0, labels = batch

    with autocast():  # fp16 forward pass
        loss_dict = model.compute_loss(x_0.cuda(), labels.cuda())
        loss = loss_dict["loss"]

    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    ema_model.update_parameters(model)

    return loss.item()
```

**Benefits**:
- 2x faster training
- 2x less memory
- Minimal quality loss with bfloat16

### 3. Gradient Checkpointing

For deeper models (depth > 28):

```python
from torch.utils.checkpoint import checkpoint

class DiTBlock(nn.Module):
    def forward(self, x, c):
        if self.training and self.use_checkpoint:
            return checkpoint(self._forward, x, c)
        return self._forward(x, c)

    def _forward(self, x, c):
        # ... actual forward pass
        pass
```

**Trade-off**:
- 30% slower training
- 40% less memory
- Enables training larger models

### 4. Efficient Attention

Use Flash Attention for 2-3x speedup:

```python
from flash_attn import flash_attn_func

class DiTBlock(nn.Module):
    def forward(self, x, c):
        # ... adaLN modulation ...

        # Replace standard attention with Flash Attention
        B, N, D = x_norm.shape
        qkv = self.qkv(x_norm).reshape(B, N, 3, self.num_heads, D // self.num_heads)
        q, k, v = qkv.unbind(2)

        attn_out = flash_attn_func(q, k, v)
        attn_out = attn_out.reshape(B, N, D)

        x = x + gate_msa.unsqueeze(1) * self.proj(attn_out)
        # ...
```

### 5. Optimized Sampling

**Batched CFG** (2x faster than sequential):
```python
# Instead of:
# pred_cond = model(x, t, y)
# pred_uncond = model(x, t, y_null)

# Do:
x_input = torch.cat([x, x], dim=0)
y_input = torch.cat([y, y_null], dim=0)
pred = model(x_input, t, y_input)
pred_cond, pred_uncond = pred.chunk(2, dim=0)
```

**Fewer steps with DDIM**:
- DDPM: 1000 steps, stochastic
- DDIM: 50-250 steps, deterministic
- Quality trade-off minimal with 100+ steps

### 6. Data Augmentation

For training stability:

```python
def augment_latent(x):
    # Random horizontal flip
    if torch.rand(1) > 0.5:
        x = torch.flip(x, [-1])
    # No other augmentations needed for latents
    return x
```

### 7. Learning Rate Schedule

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

for epoch in range(num_epochs):
    for batch in dataloader:
        train_step(batch)
    scheduler.step()
```

## 8. Experiments and Results

### ImageNet 256×256 Results

**FID-50K Scores** (lower is better):

| Model | Params | FID | Sampling Steps |
|-------|--------|-----|----------------|
| DiT-XL/2 | 675M | **2.27** | 250 (DDPM) |
| DiT-L/2 | 458M | 3.04 | 250 |
| DiT-B/2 | 130M | 5.02 | 250 |
| Latent Diffusion | 400M | 3.60 | 250 |
| ADM | 554M | 3.94 | 250 |
| StyleGAN-XL | 166M | 2.30 | 1 (GAN) |

**DiT-XL/2 achieves SOTA FID among diffusion models on ImageNet 256×256 class-conditional generation.**

### Scaling Analysis

**Model Size vs FID**:

```
FID vs Params (log scale):
6.0 |        DiT-S/2
5.0 |           |
4.0 |           DiT-B/2
3.0 |               |
2.5 |                DiT-L/2
2.3 |                    DiT-XL/2
    +----+----+----+----+----
        50M 100M 300M 600M
```

**Key Findings**:
1. DiT scales smoothly like transformers (power law)
2. Larger models consistently outperform smaller ones
3. Diminishing returns after ~500M params

### Guidance Scale Impact

**CFG Scale vs Quality** (DiT-XL/2):

| Guidance Scale | FID | Precision | Recall |
|----------------|-----|-----------|--------|
| 1.0 (no CFG) | 12.03 | 0.62 | 0.59 |
| 1.5 | 5.02 | 0.74 | 0.52 |
| 2.0 | 3.14 | 0.79 | 0.48 |
| 3.0 | 2.57 | 0.83 | 0.43 |
| 4.0 | **2.27** | 0.85 | 0.40 |
| 6.0 | 2.89 | 0.88 | 0.33 |

**Observations**:
- w=4.0 gives best FID
- Higher guidance → better precision, lower recall
- w=1.5-3.0 for more diverse generations
- w>6.0 causes artifacts

### Sampling Steps vs Quality

**DDIM Steps vs FID** (DiT-XL/2, CFG=4.0):

| Steps | FID | Time (A100) |
|-------|-----|-------------|
| 25 | 6.21 | 0.8s |
| 50 | 3.45 | 1.5s |
| 100 | 2.52 | 3.0s |
| 250 | **2.27** | 7.5s |
| 1000 (DDPM) | 2.18 | 30s |

**Sweet spot**: 100-250 steps for good quality/speed trade-off

### Architecture Ablations

**Conditioning Methods** (DiT-B/2):

| Method | FID | Params |
|--------|-----|--------|
| Cross-attention (U-Net style) | 6.12 | 135M |
| Concatenation | 5.89 | 132M |
| AdaLN | 5.31 | 130M |
| **AdaLN-Zero** | **5.02** | 130M |

**adaLN-Zero wins**: Better FID with same param count

### Latent Space Analysis

Working in VAE latent space (4×32×32 instead of 3×256×256):

**Benefits**:
- 64x fewer pixels to model
- 8x faster training
- 8x faster sampling
- Comparable quality to pixel-space diffusion

**Latent Space Properties**:
- Compressed representation
- Semantic structure preserved
- Smooth interpolation
- Enables high-resolution generation

## 9. Common Pitfalls

### 1. Forgetting EMA

**Problem**: Training model generates blurry samples

```python
# BAD: Using training weights for sampling
samples = sample_images(model, labels)  # Noisy!

# GOOD: Use EMA weights
samples = sample_images(ema_model.module, labels)  # Sharp!
```

**Solution**: Always use EMA model for inference

### 2. Wrong Guidance Scale

**Problem**: Samples look bad or lack diversity

**Too low (w=1.0)**:
- Samples don't match class
- Too diverse, low quality

**Too high (w>10)**:
- Oversaturated colors
- Artifacts and distortions
- Mode collapse

**Solution**: Use w=3.0-5.0 for most cases

### 3. Insufficient Training Steps

**Problem**: FID hasn't converged

DiT requires substantial training:
- DiT-XL/2: ~400K iterations (7M images)
- Batch size 256
- ~1-2 weeks on 8×A100

**Symptom**: FID > 10 after many epochs

**Solution**: Train longer, larger batches, or use pre-trained checkpoints

### 4. Incorrect Normalization

**Problem**: Training unstable, NaN losses

```python
# BAD: Using standard LayerNorm with affine
self.norm = nn.LayerNorm(hidden_dim)  # Has learnable γ, β

# GOOD: Disable affine (adaLN provides modulation)
self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
```

**Solution**: Use `elementwise_affine=False` in all LayerNorm

### 5. Not Clipping x0 Predictions

**Problem**: Samples explode during generation

```python
# BAD: No clipping
x0_pred = (x - sqrt_one_minus_alpha * noise_pred) / sqrt_alpha

# GOOD: Clip to valid range
x0_pred = (x - sqrt_one_minus_alpha * noise_pred) / sqrt_alpha
x0_pred = torch.clamp(x0_pred, -1, 1)  # Latents in [-1, 1]
```

**Solution**: Always clip x0 predictions during sampling

### 6. Incorrect Positional Embeddings

**Problem**: Model can't distinguish patch positions

```python
# BAD: Not adding positional embeddings
x = self.patch_embed(x)  # Missing position info!

# GOOD: Add learned positional embeddings
x = self.patch_embed(x) + self.pos_embed
```

**Solution**: Always add positional embeddings after patch embedding

### 7. Wrong Label Dropout Implementation

**Problem**: CFG doesn't work properly

```python
# BAD: Dropping labels incorrectly
if random.random() < 0.1:
    y = None  # Wrong! Need to use null class index

# GOOD: Replace with null class
if random.random() < 0.1:
    y = torch.full_like(y, num_classes)  # Last index = null class
```

**Solution**: Use a special null class index, not None

### 8. Not Initializing Gates to Zero

**Problem**: Deep networks don't train well

```python
# BAD: Standard initialization
self.adaLN_modulation = nn.Linear(hidden_dim, 6 * hidden_dim)

# GOOD: Zero initialization for gates
self.adaLN_modulation = nn.Linear(hidden_dim, 6 * hidden_dim)
nn.init.zeros_(self.adaLN_modulation.weight)
nn.init.zeros_(self.adaLN_modulation.bias)
```

**Solution**: Initialize final layers to zero (identity initialization)

### 9. Memory Issues

**Problem**: OOM errors during training

**Solutions**:
- Enable gradient checkpointing
- Reduce batch size
- Use mixed precision (bf16)
- Use Flash Attention
- Reduce model size (use DiT-L/2 instead of XL/2)

### 10. Incorrect Unpatchify

**Problem**: Generated images have wrong spatial structure

```python
# Make sure patch reconstruction is correct
def unpatchify(self, x):
    # x: (B, N, P²C) where N = (H/P)²
    c = self.out_channels
    p = self.patch_size
    h = w = self.input_size // p

    # Reshape: (B, N, P²C) -> (B, h, w, P, P, C)
    x = x.reshape(-1, h, w, p, p, c)
    # Rearrange: (B, h, w, P, P, C) -> (B, C, h*P, w*P)
    x = torch.einsum("nhwpqc->nchpwq", x)
    return x.reshape(-1, c, h * p, w * p)
```

## 10. References

### Original Paper

**"Scalable Diffusion Models with Transformers"**
- Authors: William Peebles, Saining Xie
- Conference: ICCV 2023 (Oral)
- Paper: https://arxiv.org/abs/2212.09748
- Project: https://www.wpeebles.com/DiT

### Key Contributions

1. First to replace U-Net with pure transformer for diffusion
2. Introduced adaLN-Zero conditioning mechanism
3. Demonstrated clean scaling properties
4. Achieved SOTA FID on ImageNet 256×256

### Related Papers

#### Vision Transformers
**"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"**
- Dosovitskiy et al., ICLR 2021
- https://arxiv.org/abs/2010.11929

#### Latent Diffusion
**"High-Resolution Image Synthesis with Latent Diffusion Models"**
- Rombach et al., CVPR 2022
- https://arxiv.org/abs/2112.10752

#### Diffusion Models
**"Denoising Diffusion Probabilistic Models"**
- Ho et al., NeurIPS 2020
- https://arxiv.org/abs/2006.11239

**"Classifier-Free Diffusion Guidance"**
- Ho & Salimans, NeurIPS 2021 Workshop
- https://arxiv.org/abs/2207.12598

#### Architecture Inspiration
**"Layer Normalization"**
- Ba et al., 2016
- https://arxiv.org/abs/1607.06450

**"Adaptive Instance Normalization"**
- Huang & Belongie, ICCV 2017
- Used in style transfer, inspired adaLN

### Code Repositories

**Official Implementation**:
- https://github.com/facebookresearch/DiT
- PyTorch, clean codebase
- Pre-trained checkpoints available

**Nexus Implementation**:
```
/Users/kevinyu/Projects/Nexus/nexus/models/diffusion/dit.py
```

### Follow-up Work

**PixArt-α** (2023): More efficient training of DiT-style models
**Stable Diffusion 3** (2024): Uses MMDiT (multimodal DiT) architecture
**FLUX** (2024): Rectified Flow + DiT for text-to-image

### Benchmarks

**ImageNet 256×256 Class-Conditional Generation**:
- Dataset: ImageNet-1K (1.28M training images)
- Metric: FID-50K (Fréchet Inception Distance on 50K samples)
- Current SOTA: DiT-XL/2 (FID 2.27)

**Comparison Sites**:
- Papers with Code: https://paperswithcode.com/sota/image-generation-on-imagenet-256x256
- StyleGAN benchmarks for reference

### Training Resources

**Compute Requirements**:
- DiT-XL/2: ~8×A100 GPUs, 1-2 weeks
- DiT-L/2: ~4×A100 GPUs, 1 week
- Estimated cost: $5K-20K depending on cloud provider

**Dataset**:
- ImageNet: https://image-net.org/
- Requires access request and download
- ~140GB for full dataset

### Additional Reading

**Excellent Blog Posts**:
1. Lil'Log - What are Diffusion Models?
   - https://lilianweng.github.io/posts/2021-07-11-diffusion-models/

2. Hugging Face - Annotated Diffusion
   - https://huggingface.co/blog/annotated-diffusion

3. Sander Dieleman - Diffusion Language
   - https://benanne.github.io/2022/01/31/diffusion.html

**Video Lectures**:
- Pieter Abbeel's Diffusion Models Course
- Stefano Ermon's CS236 Generative Models

---

**Implementation Status**: ✅ Complete
**Documentation**: ✅ Complete
**File**: `/Users/kevinyu/Projects/Nexus/nexus/models/diffusion/dit.py`
**Tests**: Available in `/Users/kevinyu/Projects/Nexus/tests/`

For questions or contributions, see the main generative models README.
