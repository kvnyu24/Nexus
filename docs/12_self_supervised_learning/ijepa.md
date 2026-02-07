# I-JEPA: Image Joint-Embedding Predictive Architecture

## Overview & Motivation

I-JEPA (Image Joint-Embedding Predictive Architecture) is a self-supervised learning method that learns visual representations by predicting the representations of masked image regions from visible context regions in a shared embedding space. Unlike MAE (Masked Autoencoder) which reconstructs pixels, I-JEPA predicts abstract feature representations, enabling it to learn more semantic, high-level features while avoiding the computational cost and potential shortcuts associated with pixel-level prediction.

### Key Innovation

**Predicting in representation space is fundamentally better than predicting in pixel space**:

- **Semantic Learning**: Focuses on semantic content rather than texture details and low-level statistics
- **Avoids Shortcuts**: Cannot exploit trivial pixel correlations or high-frequency patterns
- **No Augmentation Required**: Unlike contrastive methods, doesn't need hand-crafted augmentations
- **Better Downstream Transfer**: Learned features transfer better to various downstream tasks
- **Computational Efficiency**: Predicting compact representations is faster than reconstructing high-resolution pixels
- **Flexible Masking**: Multi-block masking strategy encourages understanding of spatial relationships

### Why I-JEPA Matters

Traditional self-supervised methods have limitations:

1. **Pixel reconstruction (MAE)**: Can focus on low-level details, missing semantic structure
2. **Contrastive methods (SimCLR, MoCo)**: Require carefully designed augmentations and large batch sizes
3. **Knowledge distillation (DINO)**: Needs specific mechanisms to avoid collapse

I-JEPA addresses these by:
- Learning in representation space (semantic by design)
- Using EMA targets (stable, no collapse)
- No augmentation dependence (learns invariances naturally)
- Efficient training (smaller predictor, compact targets)

## Theoretical Background

### Energy-Based Self-Supervised Learning

I-JEPA follows Yann LeCun's framework for energy-based models. The goal is to learn an energy function that assigns low energy to compatible pairs of image regions:

$$E(x_{\text{visible}}, x_{\text{target}}) = \|f_{\text{context}}(x_{\text{visible}}) - f_{\text{target}}(x_{\text{target}})\|^2$$

Where:
- $x_{\text{visible}}$: Visible (context) patches
- $x_{\text{target}}$: Masked (target) patches
- $f_{\text{context}}$: Context encoder (trained via backprop)
- $f_{\text{target}}$: Target encoder (updated via EMA)
- Energy is low when context correctly predicts target representations

### Joint Embedding Architecture

The "Joint-Embedding" aspect means both context and target are embedded in the same representation space. The predictor maps from context space to target space:

```
Context Image → Context Encoder → Context Embeddings
                                         ↓
                                    Predictor
                                         ↓
Target Image → Target Encoder → Target Embeddings ← Compare (MSE Loss)
```

Key insight: By predicting **representations** rather than raw inputs, the model learns **abstract, semantic features**.

### Why EMA Target Encoder?

Exponential Moving Average prevents representation collapse:

1. **Provides stable learning targets**: Target encoder changes slowly, giving consistent supervision
2. **Prevents trivial solutions**: Cannot output constant values (would conflict with EMA update)
3. **No negative pairs needed**: Unlike contrastive learning, don't need to explicitly separate embeddings
4. **Self-supervised bootstrap**: Teacher provides targets for student to learn from

Update rule:

$$\theta_{\text{target}} \leftarrow \tau \cdot \theta_{\text{target}} + (1-\tau) \cdot \theta_{\text{context}}$$

Where $\tau \approx 0.996-0.999$ (very high momentum = slow updates).

### Multi-Block Masking Strategy

Unlike random patch masking (MAE), I-JEPA uses **multi-block masking**:

**Context blocks**:
- 4-6 scattered rectangular regions
- Cover ~25% of image
- Provide spatial context from different locations

**Target blocks**:
- 2-4 separate rectangular regions
- Cover ~15% of image
- Do not overlap with context blocks

**Why this works**:
- Forces **spatial reasoning** (cannot rely on local correlations)
- Encourages **semantic understanding** (must understand scene structure)
- More **challenging** than random masking (harder to cheat with low-level statistics)

### Predictor Architecture

The predictor is a **smaller transformer** that:
- Takes context embeddings + target position encodings
- Predicts what should appear at target positions
- Is intentionally smaller than encoder (prevents shortcuts)

Design choices:
- Depth: 6 layers (vs 12 for encoder)
- Width: 384 dim (vs 768 for encoder)
- Prevents memorization of training data
- Forces learning of general patterns

## Mathematical Formulation

### Complete Loss Function

The full I-JEPA loss is:

$$\mathcal{L} = \frac{1}{N_{\text{target}}} \sum_{i=1}^{N_{\text{target}}} \|\text{predictor}(z_{\text{ctx}}, \text{pos}_{\text{tgt},i}) - \text{encoder}_{\text{tgt}}(x_{\text{tgt},i})\|^2$$

Where:
- $z_{\text{ctx}}$: Context encoder output
- $\text{pos}_{\text{tgt},i}$: Positional embedding for $i$-th target patch
- $\text{encoder}_{\text{tgt}}$: Target encoder (no gradient, EMA updated)
- $N_{\text{target}}$: Number of target patches

### Detailed Architecture Components

**Context Encoder** (Vision Transformer):

```python
# Patch embedding
x = patch_embed(images)  # (B, N, D)

# Add positional embeddings
x = x + pos_embed

# Keep only visible patches
x_visible = x[:, visible_mask]  # (B, N_visible, D)

# Transformer encoding
z_ctx = transformer(x_visible)  # (B, N_visible, D)
```

**Target Encoder** (EMA, no gradients):

```python
with torch.no_grad():
    # Full patch embedding
    x = patch_embed(images)
    x = x + pos_embed

    # Extract target patches
    x_target = x[:, target_mask]  # (B, N_target, D)

    # Encode targets
    z_tgt = transformer(x_target)  # (B, N_target, D)
```

**Predictor** (Smaller Transformer):

```python
# Project context to predictor dimension
ctx = linear_proj(z_ctx) + pos_embed[ctx_positions]  # (B, N_ctx, D_pred)

# Create mask tokens with target positions
masks = learnable_mask_token + pos_embed[tgt_positions]  # (B, N_tgt, D_pred)

# Concatenate and predict
combined = torch.cat([ctx, masks], dim=1)  # (B, N_ctx + N_tgt, D_pred)
output = transformer(combined)  # (B, N_ctx + N_tgt, D_pred)

# Extract predictions for target positions
predictions = output[:, -N_tgt:]  # (B, N_tgt, D_pred)

# Project back to encoder dimension
predictions = output_proj(predictions)  # (B, N_tgt, D)
```

### Masking Strategy Details

**Multi-block mask generation**:

$$\text{context\_blocks} = \text{sample\_rectangles}(n=4, \text{scale}=(0.15, 0.2), \text{aspect}=(0.75, 1.5))$$

$$\text{target\_blocks} = \text{sample\_rectangles}(n=2, \text{scale}=(0.10, 0.15), \text{aspect}=(0.75, 1.5))$$

Properties:
- Blocks are **rectangular** (not single patches)
- **Random positions** each iteration
- **Non-overlapping** (context and targets don't intersect)
- **Varied sizes** (scale and aspect ratio randomized)

### EMA Update Schedule

The EMA momentum follows a cosine schedule:

$$\tau(t) = \tau_{\text{end}} - (\tau_{\text{end}} - \tau_{\text{start}}) \cdot \frac{1 + \cos(\pi t / T)}{2}$$

Where:
- $\tau_{\text{start}} = 0.996$
- $\tau_{\text{end}} = 1.0$
- $t$: current iteration
- $T$: total iterations

**Why increasing momentum**:
- Early training: Teacher tracks student more closely (τ=0.996)
- Late training: Teacher stabilizes (τ→1.0)
- Smooth transition prevents disruption

## High-Level Intuition

I-JEPA is like describing a partially hidden photo to someone:

1. **Context Encoder**: You see visible parts and understand their meaning
2. **Predictor**: Based on what you see, you imagine what should be in hidden regions
3. **Target Encoder**: Ground truth encoding of what's actually hidden
4. **Learning**: Get better at imagination by minimizing prediction error

**Crucial insight**: You predict **semantic descriptions** ("probably a tree with green leaves"), not **pixel values** ("RGB channels [145, 203, 87] at position (i,j)").

This forces learning of:
- Object shapes and structures
- Spatial relationships
- Scene composition
- Semantic categories

Rather than:
- Texture patterns
- Color statistics
- Edge detection
- High-frequency details

## Implementation Details

### Network Architecture

**Context Encoder** (ViT-Base/16):
- Patch size: 16×16
- Embedding dim: 768
- Layers: 12
- Attention heads: 12
- MLP ratio: 4.0
- Parameters: ~86M

**Target Encoder** (same as context):
- Identical architecture
- EMA updated, no gradients
- Initialized from context encoder

**Predictor** (Smaller Transformer):
- Embedding dim: 384 (half of encoder)
- Layers: 6 (half of encoder)
- Attention heads: 12
- MLP ratio: 4.0
- Parameters: ~12M

**Design rationale**:
- Predictor is **deliberately smaller** to prevent shortcuts
- If predictor is too powerful, it might memorize rather than learn patterns
- Asymmetry encourages learning generalizable features

### Complete Nexus Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from nexus.models.vision import VisionTransformer
from nexus.models.ssl.base import BaseSSL

class MultiBlockMaskGenerator:
    """Generates multi-block masks for I-JEPA."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        num_context_blocks=4,
        num_target_blocks=2,
        context_scale=(0.15, 0.20),
        target_scale=(0.10, 0.15),
        aspect_ratio=(0.75, 1.5),
    ):
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.grid_size = img_size // patch_size

        self.num_context_blocks = num_context_blocks
        self.num_target_blocks = num_target_blocks
        self.context_scale = context_scale
        self.target_scale = target_scale
        self.aspect_ratio = aspect_ratio

    def sample_block(self, scale, aspect_ratio):
        """Sample a rectangular block."""
        # Sample scale and aspect ratio
        scale_factor = torch.FloatTensor(1).uniform_(*scale).item()
        aspect = torch.FloatTensor(1).uniform_(*aspect_ratio).item()

        # Calculate block size
        area = self.num_patches * scale_factor
        height = int(round((area / aspect) ** 0.5))
        width = int(round(height * aspect))

        # Clip to valid range
        height = max(1, min(height, self.grid_size))
        width = max(1, min(width, self.grid_size))

        # Sample position
        top = torch.randint(0, self.grid_size - height + 1, (1,)).item()
        left = torch.randint(0, self.grid_size - width + 1, (1,)).item()

        return top, left, height, width

    def generate_masks(self, batch_size):
        """Generate context and target masks for a batch."""
        context_masks = []
        target_masks = []

        for _ in range(batch_size):
            # Initialize mask grid
            mask = torch.zeros(self.grid_size, self.grid_size, dtype=torch.bool)

            # Sample context blocks
            context_blocks = []
            for _ in range(self.num_context_blocks):
                top, left, h, w = self.sample_block(
                    self.context_scale, self.aspect_ratio
                )
                mask[top:top+h, left:left+w] = True
                context_blocks.append((top, left, h, w))

            context_mask = mask.flatten()

            # Reset mask for targets (non-overlapping)
            mask_target = torch.zeros_like(mask)

            # Sample target blocks (avoid context)
            attempts = 0
            targets_added = 0
            while targets_added < self.num_target_blocks and attempts < 100:
                top, left, h, w = self.sample_block(
                    self.target_scale, self.aspect_ratio
                )

                # Check if overlaps with context
                if not mask[top:top+h, left:left+w].any():
                    mask_target[top:top+h, left:left+w] = True
                    targets_added += 1

                attempts += 1

            target_mask = mask_target.flatten()

            context_masks.append(context_mask)
            target_masks.append(target_mask)

        return torch.stack(context_masks), torch.stack(target_masks)


class IJEPAPredictor(nn.Module):
    """Predictor module for I-JEPA."""

    def __init__(
        self,
        encoder_dim=768,
        predictor_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
    ):
        super().__init__()

        # Project encoder dim to predictor dim
        self.input_proj = nn.Linear(encoder_dim, predictor_dim)

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Predictor transformer
        from nexus.models.vision.vit import TransformerBlock
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=predictor_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(predictor_dim)

        # Project back to encoder dim
        self.output_proj = nn.Linear(predictor_dim, encoder_dim)

    def forward(self, context_tokens, context_pos, target_pos):
        """
        Args:
            context_tokens: (B, N_ctx, D_enc)
            context_pos: (B, N_ctx, D_enc) positional embeddings
            target_pos: (B, N_tgt, D_enc) positional embeddings

        Returns:
            predictions: (B, N_tgt, D_enc)
        """
        B, N_ctx, _ = context_tokens.shape
        N_tgt = target_pos.shape[1]

        # Project context to predictor dimension
        ctx = self.input_proj(context_tokens + context_pos)  # (B, N_ctx, D_pred)

        # Create mask tokens with target positions
        mask_tokens = self.mask_token.expand(B, N_tgt, -1)  # (B, N_tgt, D_pred)
        tgt_pos = self.input_proj(target_pos)  # (B, N_tgt, D_pred)
        masks = mask_tokens + tgt_pos

        # Concatenate context and mask tokens
        x = torch.cat([ctx, masks], dim=1)  # (B, N_ctx + N_tgt, D_pred)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Extract predictions for target positions
        predictions = x[:, -N_tgt:]  # (B, N_tgt, D_pred)

        # Project back to encoder dimension
        predictions = self.output_proj(predictions)  # (B, N_tgt, D_enc)

        return predictions


class IJEPA(BaseSSL):
    """I-JEPA: Image Joint-Embedding Predictive Architecture."""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        encoder_dim=768,
        encoder_depth=12,
        encoder_heads=12,
        predictor_dim=384,
        predictor_depth=6,
        predictor_heads=12,
        ema_momentum=0.996,
        ema_end=1.0,
        num_context_blocks=4,
        num_target_blocks=2,
    ):
        super().__init__()

        # Context encoder (trained with backprop)
        self.context_encoder = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=encoder_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
        )

        # Target encoder (EMA updated, no gradients)
        self.target_encoder = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=encoder_dim,
            depth=encoder_depth,
            num_heads=encoder_heads,
        )

        # Initialize target encoder with context encoder weights
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())

        # Target encoder has no gradients
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # Predictor
        self.predictor = IJEPAPredictor(
            encoder_dim=encoder_dim,
            predictor_dim=predictor_dim,
            depth=predictor_depth,
            num_heads=predictor_heads,
        )

        # Mask generator
        self.mask_generator = MultiBlockMaskGenerator(
            img_size=img_size,
            patch_size=patch_size,
            num_context_blocks=num_context_blocks,
            num_target_blocks=num_target_blocks,
        )

        # EMA parameters
        self.ema_momentum = ema_momentum
        self.ema_end = ema_end
        self.current_step = 0
        self.total_steps = 100000  # Set based on training schedule

    def get_ema_momentum(self):
        """Get EMA momentum with cosine schedule."""
        progress = self.current_step / self.total_steps
        momentum = self.ema_end - (self.ema_end - self.ema_momentum) * \
                   0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
        return momentum.item()

    @torch.no_grad()
    def update_target_encoder(self):
        """Update target encoder with EMA of context encoder."""
        momentum = self.get_ema_momentum()

        for param_ctx, param_tgt in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            param_tgt.data.mul_(momentum).add_(
                param_ctx.detach().data,
                alpha=1 - momentum
            )

    def forward(self, images):
        """
        Args:
            images: (B, 3, H, W)

        Returns:
            loss: Scalar loss
            metrics: Dictionary of metrics
        """
        B = images.shape[0]

        # Generate masks
        context_mask, target_mask = self.mask_generator.generate_masks(B)
        context_mask = context_mask.to(images.device)
        target_mask = target_mask.to(images.device)

        # Extract patches and add positional embeddings
        patches = self.context_encoder.patch_embed(images)  # (B, N, D)
        pos_embed = self.context_encoder.pos_embed[:, 1:]  # Remove CLS token

        # Context encoder forward (only on visible patches)
        context_patches = []
        context_pos = []
        for i in range(B):
            ctx_idx = context_mask[i].nonzero(as_tuple=True)[0]
            context_patches.append(patches[i, ctx_idx])
            context_pos.append(pos_embed[0, ctx_idx])

        context_patches = torch.stack([
            F.pad(cp, (0, 0, 0, context_mask.sum(1).max() - cp.shape[0]))
            for cp in context_patches
        ])
        context_pos = torch.stack([
            F.pad(cp, (0, 0, 0, context_mask.sum(1).max() - cp.shape[0]))
            for cp in context_pos
        ])

        context_features = self.context_encoder.forward_features(
            context_patches, add_pos=False
        )

        # Target encoder forward (only on target patches, no gradients)
        with torch.no_grad():
            target_patches_list = []
            target_pos_list = []
            for i in range(B):
                tgt_idx = target_mask[i].nonzero(as_tuple=True)[0]
                target_patches_list.append(patches[i, tgt_idx])
                target_pos_list.append(pos_embed[0, tgt_idx])

            target_patches = torch.stack([
                F.pad(tp, (0, 0, 0, target_mask.sum(1).max() - tp.shape[0]))
                for tp in target_patches_list
            ])
            target_pos = torch.stack([
                F.pad(tp, (0, 0, 0, target_mask.sum(1).max() - tp.shape[0]))
                for tp in target_pos_list
            ])

            target_features = self.target_encoder.forward_features(
                target_patches, add_pos=False
            )

        # Predictor forward
        predictions = self.predictor(context_features, context_pos, target_pos)

        # Compute loss (MSE between predictions and targets)
        loss = F.mse_loss(predictions, target_features.detach())

        # Metrics
        metrics = {
            "loss": loss.item(),
            "ema_momentum": self.get_ema_momentum(),
            "num_context": context_mask.sum(1).float().mean().item(),
            "num_target": target_mask.sum(1).float().mean().item(),
        }

        return loss, metrics


# Training example
def train_ijepa():
    """Example training loop for I-JEPA."""
    from torch.utils.data import DataLoader
    from torchvision.datasets import ImageFolder
    from torchvision import transforms

    # Configuration
    config = {
        "img_size": 224,
        "patch_size": 16,
        "encoder_dim": 768,
        "encoder_depth": 12,
        "predictor_dim": 384,
        "predictor_depth": 6,
        "batch_size": 256,
        "lr": 1.5e-4,
        "weight_decay": 0.05,
        "epochs": 300,
        "warmup_epochs": 40,
    }

    # Model
    model = IJEPA(
        img_size=config["img_size"],
        patch_size=config["patch_size"],
        encoder_dim=config["encoder_dim"],
        encoder_depth=config["encoder_depth"],
        predictor_dim=config["predictor_dim"],
        predictor_depth=config["predictor_depth"],
    )
    model = model.cuda()

    # Data (minimal augmentation for I-JEPA)
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.3, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225]),
    ])

    dataset = ImageFolder("/path/to/imagenet", transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=16,
        pin_memory=True,
    )

    # Optimizer with layer-wise LR decay
    param_groups = []
    for name, param in model.context_encoder.named_parameters():
        if "blocks" in name:
            layer_id = int(name.split(".")[1])
            lr_scale = 0.75 ** (config["encoder_depth"] - layer_id - 1)
        else:
            lr_scale = 1.0

        param_groups.append({
            "params": param,
            "lr": config["lr"] * lr_scale,
            "weight_decay": config["weight_decay"],
        })

    # Add predictor params
    param_groups.append({
        "params": model.predictor.parameters(),
        "lr": config["lr"],
        "weight_decay": config["weight_decay"],
    })

    optimizer = torch.optim.AdamW(param_groups)

    # Training loop
    total_steps = len(dataloader) * config["epochs"]
    model.total_steps = total_steps

    for epoch in range(config["epochs"]):
        model.train()

        for step, (images, _) in enumerate(dataloader):
            images = images.cuda()

            # Forward pass
            loss, metrics = model(images)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            # Update target encoder
            model.update_target_encoder()
            model.current_step += 1

            if step % 100 == 0:
                print(f"Epoch {epoch}, Step {step}, "
                      f"Loss: {metrics['loss']:.4f}, "
                      f"EMA: {metrics['ema_momentum']:.4f}")

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch,
                "context_encoder": model.context_encoder.state_dict(),
                "target_encoder": model.target_encoder.state_dict(),
                "predictor": model.predictor.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, f"ijepa_checkpoint_epoch_{epoch}.pth")
```

### Minimal Augmentation Strategy

Unlike contrastive methods, I-JEPA requires **minimal augmentation**:

```python
# I-JEPA augmentation (minimal)
transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.3, 1.0)),  # Reasonable crops
    transforms.RandomHorizontalFlip(),  # Simple flip
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])

# No need for:
# - Strong color jitter
# - Gaussian blur
# - Solarization
# - Multi-crop
```

**Why minimal augmentation works**:
- Representation prediction is augmentation by design
- Masking provides the core learning signal
- Over-augmentation can hurt (model learns augmentation artifacts)

## Optimization Tricks

### 1. EMA Momentum Schedule

```python
def cosine_ema_schedule(step, total_steps, start=0.996, end=1.0):
    """Cosine schedule for EMA momentum."""
    progress = step / total_steps
    return end - (end - start) * 0.5 * (1 + math.cos(math.pi * progress))

# Usage
momentum = cosine_ema_schedule(current_step, total_steps)
```

**Why this schedule**:
- Start lower (0.996): Teacher tracks student, provides dynamic targets
- End higher (1.0): Teacher stabilizes, provides consistent supervision
- Smooth transition: No sudden jumps that could destabilize training

### 2. Learning Rate Warmup

```python
def lr_schedule(epoch, total_epochs, warmup_epochs=40, base_lr=1.5e-4):
    """Learning rate schedule with warmup."""
    if epoch < warmup_epochs:
        return base_lr * epoch / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return base_lr * 0.5 * (1 + math.cos(math.pi * progress))
```

**Warmup is critical**:
- Prevents early training instability
- Allows EMA teacher to initialize properly
- 40 epochs is optimal for ImageNet-scale training

### 3. Layer-wise Learning Rate Decay

```python
def get_layer_wise_lr(name, base_lr, num_layers=12, decay_rate=0.75):
    """Earlier layers get smaller learning rates."""
    if "blocks" not in name:
        return base_lr

    layer_id = int(name.split(".blocks.")[1].split(".")[0])
    lr_scale = decay_rate ** (num_layers - layer_id - 1)
    return base_lr * lr_scale

# Example
for layer in range(12):
    lr = get_layer_wise_lr(f"blocks.{layer}", 1.5e-4, 12, 0.75)
    print(f"Layer {layer}: {lr:.6f}")
```

**Rationale**:
- Early layers: General features (edges, colors) - change slowly
- Late layers: Task-specific features - can adapt faster
- Improves convergence and final performance

### 4. Gradient Clipping

```python
# Clip to prevent exploding gradients
max_norm = 1.0
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=max_norm
)
```

**Important for I-JEPA**:
- Predictor can have large gradients early in training
- Clipping prevents instability
- More aggressive than DINO (1.0 vs 3.0)

### 5. Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for images, _ in dataloader:
    with autocast():
        loss, metrics = model(images)

    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)

    # Gradient clipping after unscaling
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    scaler.step(optimizer)
    scaler.update()
```

**Benefits**:
- 2× speedup typical
- Reduces memory usage
- Critical for scaling to ViT-Large/Huge

### 6. Masking Strategy Tuning

```python
# Good default parameters
mask_config = {
    "num_context_blocks": 4,  # More context = easier task
    "num_target_blocks": 2,   # Fewer targets = harder task
    "context_scale": (0.15, 0.20),  # 15-20% per block
    "target_scale": (0.10, 0.15),   # 10-15% per block
    "aspect_ratio": (0.75, 1.5),    # Rectangular blocks
}

# Total coverage: ~25% context, ~15% target, ~60% unseen
```

**Tuning guidelines**:
- More context blocks → easier, faster convergence
- Larger target blocks → harder, better representations
- Balance: want challenging but learnable task

### 7. Predictor Size

```python
# Rule of thumb: predictor = 0.5× encoder size
encoder_dim = 768
predictor_dim = 384  # Half

encoder_depth = 12
predictor_depth = 6  # Half
```

**Why smaller predictor**:
- Too large: Can memorize, doesn't learn patterns
- Too small: Underfits, poor predictions
- 0.5× is empirically optimal

### 8. Weight Decay

```python
# Separate weight decay for different components
param_groups = [
    # Encoder: standard weight decay
    {"params": encoder_params, "weight_decay": 0.05},

    # Predictor: less weight decay (smaller model)
    {"params": predictor_params, "weight_decay": 0.04},

    # Biases and norms: no weight decay
    {"params": bias_params, "weight_decay": 0.0},
]
```

## Experiments & Results

### ImageNet-1K Linear Probing

Training a linear classifier on frozen features:

| Model | Architecture | Top-1 Acc | Params |
|-------|--------------|-----------|--------|
| MAE | ViT-H/14 | 76.6% | 632M |
| data2vec | ViT-L/16 | 78.5% | 304M |
| iBOT | ViT-L/16 | 79.5% | 304M |
| **I-JEPA** | **ViT-H/14** | **80.3%** | **632M** |

**Key insight**: Representation prediction outperforms pixel reconstruction by 3.7%!

### Fine-tuning Performance

Full model fine-tuning on ImageNet-1K:

| Method | Pre-training | Fine-tuning | Top-1 Acc |
|--------|--------------|-------------|-----------|
| Supervised | - | ImageNet-1K | 82.3% |
| MAE | ImageNet-1K | ImageNet-1K | 83.1% |
| **I-JEPA** | **ImageNet-1K** | **ImageNet-1K** | **83.6%** |

### Ablation Studies

**Effect of Masking Strategy**:

| Type | Context | Target | Top-1 Acc |
|------|---------|--------|-----------|
| Random patches | 25% | 15% | 77.2% |
| Single large block | 25% | 15% | 78.5% |
| **Multi-block** | **25%** | **15%** | **80.3%** |

**Multi-block masking is crucial** for spatial reasoning!

**Effect of Prediction Target**:

| Target | Top-1 Acc | Training Speed |
|--------|-----------|----------------|
| Pixels (MAE) | 76.6% | 1.0× |
| Normalized pixels | 77.2% | 1.0× |
| PCA features | 78.1% | 0.9× |
| Average pool features | 78.9% | 1.2× |
| **Patch features** | **80.3%** | **1.5×** |

**Patch-level representations** give best accuracy and speed!

**Effect of EMA Momentum**:

| Momentum | Schedule | Top-1 Acc |
|----------|----------|-----------|
| 0.99 | Fixed | 78.5% |
| 0.996 | Fixed | 79.8% |
| 0.999 | Fixed | 79.2% |
| **0.996 → 1.0** | **Cosine** | **80.3%** |

**Scheduling momentum** is important!

**Effect of Predictor Size**:

| Predictor Dim | Depth | Top-1 Acc | Note |
|---------------|-------|-----------|------|
| 192 | 3 | 78.9% | Too small |
| **384** | **6** | **80.3%** | **Optimal** |
| 768 | 12 | 79.7% | Too large (overfits) |
| 1024 | 12 | 78.8% | Memorizes |

**Half-size predictor** prevents shortcuts!

**Effect of Augmentation Strength**:

| Augmentation | Top-1 Acc |
|--------------|-----------|
| None (crop + flip only) | 79.8% |
| **Minimal** (crop + flip) | **80.3%** |
| Moderate (+ color jitter) | 79.9% |
| Strong (+ blur + solarize) | 78.7% |

**Less is more** for I-JEPA augmentation!

### Downstream Tasks

**Object Detection (COCO)**:

| Method | Backbone | AP | AP50 | AP75 |
|--------|----------|-----|------|------|
| Supervised | ViT-B/16 | 46.2 | 66.3 | 50.1 |
| MAE | ViT-B/16 | 47.3 | 67.1 | 51.2 |
| **I-JEPA** | **ViT-B/16** | **48.1** | **68.2** | **52.3** |

**Semantic Segmentation (ADE20K)**:

| Method | Backbone | mIoU |
|--------|----------|------|
| Supervised | ViT-B/16 | 47.3 |
| MAE | ViT-B/16 | 48.1 |
| **I-JEPA** | **ViT-B/16** | **49.2** |

**Instance Segmentation (COCO)**:

| Method | Backbone | AP (mask) |
|--------|----------|-----------|
| Supervised | ViT-B/16 | 41.2 |
| MAE | ViT-B/16 | 42.4 |
| **I-JEPA** | **ViT-B/16** | **43.7** |

### Low-Shot Transfer

Performance when fine-tuning with limited labels:

| Labels per class | Supervised | MAE | I-JEPA |
|------------------|-----------|-----|--------|
| 1 | 25.3% | 42.1% | **48.7%** |
| 5 | 48.2% | 61.3% | **66.2%** |
| 10 | 59.7% | 69.8% | **73.1%** |
| 100 | 76.8% | 79.2% | **80.9%** |

**I-JEPA excels in low-data regimes**!

## Common Pitfalls

### 1. Representation Collapse

**Symptoms**:
- Loss drops to near zero quickly
- All embeddings become identical
- Poor downstream performance

**Root causes**:
- EMA momentum too high initially
- Predictor too powerful
- Insufficient diversity in masks

**Solutions**:

```python
# 1. Appropriate EMA momentum
ema_momentum_start = 0.996  # Not too high!
ema_momentum_end = 1.0

# 2. Smaller predictor (half size of encoder)
predictor_dim = encoder_dim // 2
predictor_depth = encoder_depth // 2

# 3. Proper initialization
target_encoder.load_state_dict(context_encoder.state_dict())

# 4. Diverse masking
num_context_blocks = 4
num_target_blocks = 2
# Randomize positions each iteration

# 5. Monitor embedding variance
variance = torch.var(embeddings, dim=0).mean()
if variance < 0.01:
    print("Warning: Low variance, possible collapse!")
```

### 2. Training Instability

**Symptoms**:
- Loss spikes or oscillates
- NaN values appear
- Gradients explode

**Solutions**:

```python
# 1. Lower learning rate
lr = 1e-4  # Instead of 1.5e-4

# 2. Longer warmup
warmup_epochs = 40  # Critical for I-JEPA

# 3. Gradient clipping (aggressive)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

# 4. Check for NaN
if torch.isnan(loss):
    print("NaN detected! Skipping batch...")
    continue

# 5. Monitor gradient norms
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
if grad_norm > 10.0:
    print(f"Warning: Large gradient norm: {grad_norm:.2f}")
```

### 3. Poor Masking Strategy

**Symptoms**:
- Model learns to predict textures, not semantics
- Poor spatial understanding
- Weak downstream transfer

**Solutions**:

```python
# 1. Use multi-block masking (not random patches)
mask_generator = MultiBlockMaskGenerator(
    num_context_blocks=4,
    num_target_blocks=2,
)

# 2. Appropriate mask ratio
context_coverage = 0.25  # 25% of image
target_coverage = 0.15   # 15% of image

# 3. Non-overlapping masks
# Ensure context and target don't overlap

# 4. Varied block sizes
scale_range = (0.10, 0.20)
aspect_ratio_range = (0.75, 1.5)

# 5. Test masking visually
import matplotlib.pyplot as plt
# Visualize masks to ensure they look reasonable
```

### 4. Wrong Encoder at Inference

**Symptoms**:
- Training metrics look good
- Evaluation/downstream performance is poor
- Features don't match expected quality

**Solution**:

```python
# ALWAYS use context encoder for inference (not target encoder!)
@torch.no_grad()
def extract_features(model, images):
    model.eval()
    features = model.context_encoder(images)  # Context encoder!
    return features

# Common mistake: using target encoder
# features = model.target_encoder(images)  # WRONG!

# The target encoder is only for providing training targets
```

### 5. Excessive Augmentation

**Symptoms**:
- Model performance degrades with stronger augmentation
- Training becomes slower
- Features become augmentation-specific

**Solution**:

```python
# I-JEPA needs MINIMAL augmentation
transform = transforms.Compose([
    # Only basic augmentation
    transforms.RandomResizedCrop(224, scale=(0.3, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])

# DO NOT ADD:
# - Strong color jitter
# - Gaussian blur
# - Solarization
# - Random grayscale
# These hurt I-JEPA performance!
```

### 6. Predictor Too Large

**Symptoms**:
- Training loss very low but poor transfer
- Model memorizes training data
- Validation performance plateaus early

**Solution**:

```python
# Predictor should be SMALLER than encoder
encoder_config = {
    "dim": 768,
    "depth": 12,
}

predictor_config = {
    "dim": 384,   # Half of encoder
    "depth": 6,    # Half of encoder
}

# If predictor is too large, it can take shortcuts:
# - Memorize training samples
# - Exploit low-level correlations
# - Not learn transferable features
```

### 7. Incorrect EMA Update

**Symptoms**:
- Target and context encoders diverge
- Training becomes unstable
- Loss doesn't decrease properly

**Solution**:

```python
# Correct EMA update
@torch.no_grad()
def update_target_encoder(context_encoder, target_encoder, momentum):
    for param_ctx, param_tgt in zip(
        context_encoder.parameters(),
        target_encoder.parameters()
    ):
        # Correct formula
        param_tgt.data = momentum * param_tgt.data + \
                        (1 - momentum) * param_ctx.data

        # Ensure detached (no gradient graph)
        param_tgt.requires_grad = False

# Common mistakes:
# 1. Forgetting to detach context encoder
# 2. Wrong momentum formula
# 3. Not updating every step
```

### 8. Memory Issues with Large Models

**Symptoms**:
- Out of memory errors
- Training very slow
- Cannot use desired batch size

**Solutions**:

```python
# 1. Gradient checkpointing
from torch.utils.checkpoint import checkpoint

for block in model.context_encoder.blocks:
    x = checkpoint(block, x)

# 2. Mixed precision
from torch.cuda.amp import autocast
with autocast():
    loss, metrics = model(images)

# 3. Reduce batch size, accumulate gradients
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 4. Smaller predictor
predictor_dim = 256  # Instead of 384

# 5. Fewer attention heads
num_heads = 8  # Instead of 12
```

## References

```bibtex
@article{assran2023self,
  title={Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture},
  author={Assran, Mahmoud and Duval, Quentin and Misra, Ishan and Bojanowski, Piotr and Vincent, Pascal and Rabbat, Michael and LeCun, Yann and Ballas, Nicolas},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={15619--15629},
  year={2023}
}

@article{lecun2022path,
  title={A Path Towards Autonomous Machine Intelligence},
  author={LeCun, Yann},
  journal={OpenReview},
  year={2022}
}

@article{he2022masked,
  title={Masked Autoencoders Are Scalable Vision Learners},
  author={He, Kaiming and Chen, Xinlei and Xie, Saining and Li, Yanghao and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={16000--16009},
  year={2022}
}

@inproceedings{caron2021emerging,
  title={Emerging Properties in Self-Supervised Vision Transformers},
  author={Caron, Mathilde and Touvron, Hugo and Misra, Ishan and J{\'e}gou, Herv{\'e} and Mairal, Julien and Bojanowski, Piotr and Joulin, Armand},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={9650--9660},
  year={2021}
}

@inproceedings{chen2021empirical,
  title={An Empirical Study of Training Self-Supervised Vision Transformers},
  author={Chen, Xinlei and Xie, Saining and He, Kaiming},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={9640--9649},
  year={2021}
}

@article{zhou2021ibot,
  title={iBOT: Image BERT Pre-Training with Online Tokenizer},
  author={Zhou, Jinghao and Wei, Chen and Wang, Huiyu and Shen, Wei and Xie, Cihang and Yuille, Alan and Kong, Tao},
  journal={arXiv preprint arXiv:2111.07832},
  year={2021}
}

@inproceedings{baevski2022data2vec,
  title={data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language},
  author={Baevski, Alexei and Hsu, Wei-Ning and Xu, Qiantong and Babu, Arun and Gu, Jiatao and Auli, Michael},
  booktitle={International Conference on Machine Learning},
  pages={1298--1312},
  year={2022}
}

@article{grill2020bootstrap,
  title={Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning},
  author={Grill, Jean-Bastien and Strub, Florian and Altch{\'e}, Florent and Tallec, Corentin and Richemond, Pierre H and Buchatskaya, Elena and Doersch, Carl and Pires, Bernardo Avila and Guo, Zhaohan Daniel and Azar, Mohammad Gheshlaghi and others},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={21271--21284},
  year={2020}
}
```

## Summary

I-JEPA key advantages:

**Strengths**:
- Predicts representations, not pixels (semantic learning)
- Multi-block masking for spatial reasoning
- Minimal augmentation required
- State-of-the-art ImageNet linear probing
- Strong downstream transfer
- Computationally efficient

**Key Hyperparameters**:
- EMA momentum: 0.996 → 1.0 (cosine schedule)
- Predictor size: 0.5× encoder size
- Mask ratio: ~25% context, ~15% target
- Learning rate: 1.5e-4 with 40-epoch warmup
- Gradient clipping: 1.0
- Weight decay: 0.05
- Minimal augmentation: crop + flip only

**When to use I-JEPA**:
- Want strong visual representations without labels
- Don't want complex augmentation pipelines
- Need good transfer learning performance
- Have limited computational resources
- Working on dense prediction tasks

**Comparison to alternatives**:
- vs MAE: Better transfer (+3.7% linear probing), similar speed
- vs DINO: No collapse issues, simpler training
- vs Contrastive: No negative pairs, smaller batches OK
- vs Supervised: Approaches supervised with no labels

**Official Resources**:
- Paper: https://arxiv.org/abs/2301.08243
- Code: https://github.com/facebookresearch/ijepa
- Nexus Implementation: `nexus/models/ssl/jepa.py`

I-JEPA demonstrates that predicting abstract representations rather than pixels is a more effective approach to self-supervised learning, achieving excellent performance with minimal augmentation and stable training dynamics.
