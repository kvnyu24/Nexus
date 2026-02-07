# InternVL (Intern Vision-Language)

## Overview & Motivation

InternVL is a large-scale vision-language foundation model that bridges powerful vision encoders with language models through progressive alignment and contrastive learning. It scales vision encoders to 6B parameters and achieves state-of-the-art performance on multimodal understanding tasks by learning robust cross-modal representations.

**Key Innovation**: Progressive scaling strategy from 300M to 6B vision parameters with efficient cross-modal alignment, enabling strong zero-shot transfer and multimodal understanding without task-specific architectures.

**Why It Matters**:
- Largest open-source vision encoder (6B parameters)
- Unified framework for image-text and multimodal tasks
- Strong zero-shot capabilities on diverse benchmarks
- Efficient cross-modal fusion with QLLaMA integration
- Foundation for visual question answering, captioning, retrieval

## Theoretical Background

### Problem Setting
Build a unified vision-language model that (1) learns powerful visual representations that transfer across tasks, (2) aligns vision and language in a shared embedding space, and (3) enables multimodal reasoning through efficient fusion.

### Core Approach
1. Progressive vision encoder scaling: 300M → 1B → 6B parameters
2. Contrastive pre-training on large-scale image-text pairs
3. Cross-modal projection with multi-layer MLP
4. Integration with quantized language models (QLLaMA)
5. Multi-stage training: contrastive → alignment → instruction tuning

### Key Insight
Larger vision encoders learn richer visual representations that generalize better to downstream tasks. Progressive scaling with intermediate checkpoints prevents training instability. Contrastive learning followed by generative alignment combines the benefits of both paradigms.

## Mathematical Formulation

### 1. Vision Encoder Architecture
Standard ViT with progressive scaling:
```
InternViT-300M: D=768,  L=12,  H=12
InternViT-1B:   D=1024, L=24,  H=16
InternViT-6B:   D=3200, L=48,  H=25

Input: x ∈ R^(H×W×3)
Patch embedding: P ∈ R^(N×D) where N=(H×W)/P²

Standard ViT blocks:
For l=1 to L:
    P_l = MSA(LN(P_{l-1})) + P_{l-1}
    P_l = MLP(LN(P_l)) + P_l

Output: CLS token embedding ∈ R^D
```

### 2. Contrastive Pre-training
Similar to CLIP, learn aligned image-text embeddings:
```
Image encoder: v = f_vision(img) ∈ R^D
Text encoder: t = f_text(text) ∈ R^D

L2 normalization:
v̄ = v / ||v||₂
t̄ = t / ||t||₂

Similarity matrix:
S = v̄ · t̄^T ∈ R^(B×B)

Contrastive loss (InfoNCE):
L_i2t = -log(exp(S_ii/τ) / Σ_j exp(S_ij/τ))
L_t2i = -log(exp(S_ii/τ) / Σ_i exp(S_ij/τ))

L_contrast = (L_i2t + L_t2i) / 2

where τ is temperature parameter
```

### 3. Cross-Modal Projection
Project vision features to language model dimension:
```
Vision features: v ∈ R^D_v (from InternViT)
Language model dimension: D_l (e.g., 4096 for LLaMA)

Multi-layer projection:
h₁ = GELU(Linear(v, D_v))
h₂ = GELU(Linear(h₁, D_v))
v_proj = Linear(h₂, D_l) ∈ R^{D_l}

L2 normalization for alignment:
v̄_proj = v_proj / ||v_proj||₂
```

### 4. Multimodal Fusion with QLLaMA
Integrate vision and text for generation:
```
Image: I → InternViT → v_img ∈ R^{N_v × D_l}
Text: T → Tokenize → t_tokens ∈ R^{N_t}

# Interleave vision and text tokens
multimodal_seq = [v_img[1], ..., v_img[N_v], t_tokens[1], ..., t_tokens[N_t]]

# QLLaMA decoder (quantized LLaMA)
output = QLLaMA(multimodal_seq)

# Generative loss
L_gen = -Σ log P(t_{i+1} | v_img, t_{1:i})
```

### 5. Progressive Training Strategy
Three-stage training:
```
Stage 1: Vision-only contrastive pre-training
    Freeze: None
    Train: InternViT on image-text pairs
    Loss: L_contrast
    Data: LAION-2B, etc.

Stage 2: Vision-language alignment
    Freeze: None
    Train: InternViT + projection + text encoder
    Loss: L_contrast + λ₁·L_gen
    Data: High-quality image-text pairs

Stage 3: Instruction tuning
    Freeze: InternViT (optional)
    Train: Projection + QLLaMA
    Loss: L_gen
    Data: Multimodal instruction datasets
```

### 6. Position Interpolation for Dynamic Resolution
Support variable input resolutions:
```
Training: 224×224 → 16×16 patches
Testing: 448×448 → 32×32 patches

Interpolate positional embeddings:
pos_embed_train ∈ R^{(16×16+1)×D}
pos_embed_test = interpolate(pos_embed_train, size=32×32)

Uses bicubic interpolation for smooth adaptation
```

### 7. Complete Forward Pass
```
Input: Image I, Text query Q

# Vision encoding
patches = PatchEmbed(I) ∈ R^{N×D_v}
patches = patches + pos_embed
cls_token = learnable ∈ R^{1×D_v}
x = [cls_token; patches] ∈ R^{(N+1)×D_v}

For each ViT block:
    x = Block(x)

v_global = x[0]  # CLS token
v_patches = x[1:]  # Patch tokens

# Cross-modal projection
v_aligned = Projection(v_global) ∈ R^{D_l}

# Optional: project all patches for dense tasks
v_patches_aligned = Projection(v_patches) ∈ R^{N×D_l}

# Text encoding and fusion
t_embed = TextEncoder(Q) ∈ R^{L_q×D_l}

# Multimodal sequence
seq = [v_aligned; t_embed] or [v_patches_aligned; t_embed]

# Generation
output = QLLaMA(seq)
```

## High-Level Intuition

```
┌─────────────────────────────────────────────────┐
│           InternVL Architecture                 │
└─────────────────────────────────────────────────┘

Input Image (224×224)
        ↓
┌──────────────────────┐
│   InternViT-6B       │  ← Massive vision encoder
│   (6 billion params) │     (48 layers, 3200 dim)
│                      │
│  • Patch Embed       │
│  • 48 ViT Blocks     │
│  • Layer Norm        │
└──────────────────────┘
        ↓
   [CLS Token]  [Patch Tokens]
        │              │
        v              v
   Global (1×3200)  Dense (256×3200)
        │              │
        ↓              ↓
┌──────────────────────────────┐
│   Cross-Modal Projection     │
│                              │
│   3200 → 3200 (GELU)        │
│   3200 → 3200 (GELU)        │
│   3200 → 4096 (Linear)      │
│                              │
│   L2 Normalize               │
└──────────────────────────────┘
        ↓              ↓
  Aligned (1×4096) Dense (256×4096)
        │              │
        │              │
Input Text Query       │
        ↓              │
┌──────────────────────┘
│ [Concatenate]
│  Vision + Text Tokens
│
↓
┌──────────────────────────────┐
│      QLLaMA Decoder          │
│  (Quantized LLaMA-7B/13B)    │
│                              │
│  • Causal Attention          │
│  • Cross-modal understanding │
│  • Generative outputs        │
└──────────────────────────────┘
        ↓
  Generated Response
  (Caption / Answer / Description)


Training Pipeline:
┌─────────────────────────────────────────────────┐
│  Stage 1: Contrastive Pre-training             │
│  ┌───────────┐        ┌──────────┐             │
│  │ Image     │───────→│ViT-300M  │             │
│  └───────────┘        └──────────┘             │
│                            ↓                    │
│  ┌───────────┐        ┌──────────┐             │
│  │ Text      │───────→│ BERT     │             │
│  └───────────┘        └──────────┘             │
│                            ↓                    │
│                    [Contrastive Loss]           │
│                                                 │
│  Data: 2B image-text pairs                     │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  Stage 2: Scale to 1B → 6B                     │
│  Progressive scaling with checkpoints           │
│  Continue contrastive + add generative          │
└─────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────┐
│  Stage 3: Multimodal Instruction Tuning        │
│  ┌─────────┐         ┌──────────┐             │
│  │ Image   │────────→│ViT-6B    │             │
│  │+ Text   │  Frozen └──────────┘             │
│  └─────────┘              ↓                    │
│                     [Projection]                │
│                           ↓                    │
│                    ┌──────────┐               │
│                    │ QLLaMA   │ ← Trainable   │
│                    └──────────┘               │
│                           ↓                    │
│                  [Generative Loss]             │
│                                                │
│  Data: VQA, Captioning, Instruction datasets   │
└─────────────────────────────────────────────────┘
```

**Scaling Benefits**:
- 300M: Basic visual recognition
- 1B: Richer semantic understanding
- 6B: Fine-grained details, compositional reasoning, zero-shot transfer

## Implementation Details

### Model Variants

| Variant | Vision Params | Vision Dim | Vision Layers | Lang Model | Total Params |
|---------|--------------|------------|---------------|------------|--------------|
| InternVL-300M | 300M | 768 | 12 | BERT-B | 410M |
| InternVL-1B | 1B | 1024 | 24 | LLaMA-7B | 8B |
| InternVL-6B | 6B | 3200 | 48 | LLaMA-13B | 19B |

### Configuration Example

```python
config = {
    # Vision encoder (InternViT-6B)
    "img_size": 224,
    "patch_size": 14,
    "in_channels": 3,
    "vision_embed_dim": 3200,
    "vision_depth": 48,
    "vision_num_heads": 25,
    "vision_mlp_ratio": 4.0,

    # Cross-modal projection
    "language_dim": 4096,       # LLaMA hidden size
    "num_proj_layers": 2,       # Depth of projection MLP
    "proj_dropout": 0.0,

    # Training
    "dropout": 0.0,
    "drop_path_rate": 0.1,
    "temperature": 0.07,        # Contrastive temperature

    # Multimodal
    "use_cls_token": True,
    "qkv_bias": True,
}
```

## Code Walkthrough

Reference: `Nexus/nexus/models/cv/intern_vl.py`

### 1. InternVision Encoder

```python
class InternVisionModel(NexusModule):
    """Large-scale ViT for InternVL."""

    def __init__(self, config):
        super().__init__(config)

        self.img_size = config.get("img_size", 224)
        self.patch_size = config.get("patch_size", 14)
        self.embed_dim = config.get("embed_dim", 1024)
        self.depth = config.get("depth", 24)
        self.num_heads = config.get("num_heads", 16)

        self.grid_size = self.img_size // self.patch_size
        self.num_patches = self.grid_size ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            config.get("in_channels", 3),
            self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

        # CLS token and positional embeddings
        if config.get("use_cls_token", True):
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            num_tokens = self.num_patches + 1
        else:
            num_tokens = self.num_patches

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_tokens, self.embed_dim)
        )

        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(
            0, config.get("drop_path_rate", 0.0), self.depth
        )]

        # Transformer blocks
        self.blocks = nn.ModuleList([
            InternVisionBlock(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=config.get("mlp_ratio", 4.0),
                dropout=config.get("dropout", 0.0),
                drop_path=dpr[i],
                qkv_bias=config.get("qkv_bias", True)
            ) for i in range(self.depth)
        ])

        self.norm = nn.LayerNorm(self.embed_dim)

        # Initialize
        if hasattr(self, 'cls_token'):
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.init_weights_vit()

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            dict with embeddings, patch_tokens, all_tokens
        """
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)

        # Add CLS token if used
        if hasattr(self, 'cls_token'):
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embeddings
        x = x + self.pos_embed

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Extract features
        if hasattr(self, 'cls_token'):
            embeddings = x[:, 0]
            patch_tokens = x[:, 1:]
        else:
            embeddings = x.mean(dim=1)
            patch_tokens = x

        return {
            "embeddings": embeddings,
            "patch_tokens": patch_tokens,
            "all_tokens": x
        }
```

### 2. Cross-Modal Projection

```python
class InternVLEmbedding(NexusModule):
    """Project vision features to language dimension."""

    def __init__(self, config):
        super().__init__(config)

        self.vision_dim = config.get("vision_dim", 1024)
        self.language_dim = config.get("language_dim", 4096)
        num_layers = config.get("num_proj_layers", 2)
        dropout = config.get("dropout", 0.0)

        # Multi-layer projection
        layers = []
        in_dim = self.vision_dim

        for i in range(num_layers):
            out_dim = self.language_dim if i == num_layers - 1 else self.vision_dim

            layers.append(nn.Linear(in_dim, out_dim))

            # No activation on last layer
            if i < num_layers - 1:
                layers.append(nn.GELU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

            in_dim = out_dim

        self.projection = nn.Sequential(*layers)

    def forward(self, vision_features):
        """
        Args:
            vision_features: (B, N, vision_dim) or (B, vision_dim)

        Returns:
            (B, N, language_dim) or (B, language_dim)
        """
        return self.projection(vision_features)
```

### 3. Complete InternVL Model

```python
class InternVL(NexusModule):
    """InternVL: Vision-Language foundation model."""

    def __init__(self, config):
        super().__init__(config)

        # Vision encoder
        vision_config = {
            "img_size": config.get("img_size", 224),
            "patch_size": config.get("patch_size", 14),
            "in_channels": config.get("in_channels", 3),
            "embed_dim": config.get("vision_embed_dim", 1024),
            "depth": config.get("vision_depth", 24),
            "num_heads": config.get("vision_num_heads", 16),
            "mlp_ratio": config.get("mlp_ratio", 4.0),
            "dropout": config.get("dropout", 0.0),
            "drop_path_rate": config.get("drop_path_rate", 0.0),
            "use_cls_token": config.get("use_cls_token", True),
            "qkv_bias": config.get("qkv_bias", True),
        }
        self.vision_encoder = InternVisionModel(vision_config)

        # Cross-modal projection
        proj_config = {
            "vision_dim": config.get("vision_embed_dim", 1024),
            "language_dim": config.get("language_dim", 4096),
            "num_proj_layers": config.get("num_proj_layers", 2),
            "dropout": config.get("dropout", 0.0),
        }
        self.projection = InternVLEmbedding(proj_config)

        # Temperature for contrastive learning
        self.temperature = nn.Parameter(
            torch.tensor(config.get("temperature", 0.07))
        )

    def encode_image(self, images):
        """Encode images to language-aligned features.

        Args:
            images: (B, C, H, W)

        Returns:
            (B, language_dim) normalized features
        """
        vision_out = self.vision_encoder(images)
        embeddings = vision_out["embeddings"]

        # Project to language space
        projected = self.projection(embeddings.unsqueeze(1)).squeeze(1)

        # L2 normalize for contrastive learning
        return F.normalize(projected, dim=-1)

    def forward(self, images, return_patch_tokens=False):
        """Forward pass.

        Args:
            images: (B, C, H, W)
            return_patch_tokens: Whether to return patch-level features

        Returns:
            dict with embeddings, patch_embeddings (optional), vision_embeddings
        """
        # Vision encoding
        vision_out = self.vision_encoder(images)

        # Project global features
        global_features = self.projection(
            vision_out["embeddings"].unsqueeze(1)
        ).squeeze(1)
        global_features = F.normalize(global_features, dim=-1)

        output = {
            "embeddings": global_features,
            "vision_embeddings": vision_out["embeddings"]
        }

        # Optionally project patch tokens
        if return_patch_tokens:
            patch_features = self.projection(vision_out["patch_tokens"])
            output["patch_embeddings"] = patch_features

        return output
```

### 4. Contrastive Training

```python
def train_internvl_contrastive(model, dataloader, optimizer, config):
    """Contrastive pre-training for InternVL."""

    for epoch in range(config["epochs"]):
        for images, texts in dataloader:
            images = images.to(device)
            texts = texts.to(device)

            # Encode images
            image_features = model.encode_image(images)

            # Encode texts (assume text encoder exists)
            text_features = text_encoder(texts)
            text_features = F.normalize(text_features, dim=-1)

            # Compute similarity matrix
            logits = image_features @ text_features.T
            logits = logits / model.temperature

            # Labels: identity matrix
            B = images.shape[0]
            labels = torch.arange(B, device=device)

            # Contrastive loss (both directions)
            loss_i2t = F.cross_entropy(logits, labels)
            loss_t2i = F.cross_entropy(logits.T, labels)
            loss = (loss_i2t + loss_t2i) / 2

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}, "
                      f"Temp: {model.temperature.item():.4f}")
```

### 5. Multimodal Generation (with QLLaMA)

```python
class InternVLChat(NexusModule):
    """InternVL for visual question answering."""

    def __init__(self, config):
        super().__init__(config)

        # Vision encoder (frozen after stage 2)
        self.vision_model = InternVL(config)
        if config.get("freeze_vision", False):
            for param in self.vision_model.parameters():
                param.requires_grad = False

        # Language model (QLLaMA)
        self.language_model = QLLaMA(config["language_config"])

    def forward(self, images, input_ids, attention_mask=None):
        """
        Args:
            images: (B, C, H, W)
            input_ids: (B, L) text token IDs
            attention_mask: (B, L)

        Returns:
            dict with logits, loss
        """
        # Encode image
        vision_out = self.vision_model(images, return_patch_tokens=True)

        # Option 1: Use global feature
        vision_tokens = vision_out["embeddings"].unsqueeze(1)

        # Option 2: Use patch features for dense understanding
        # vision_tokens = vision_out["patch_embeddings"]

        # Get text embeddings
        text_embeds = self.language_model.embed_tokens(input_ids)

        # Concatenate vision and text
        combined = torch.cat([vision_tokens, text_embeds], dim=1)

        # Update attention mask
        if attention_mask is not None:
            vision_mask = torch.ones(
                (images.shape[0], vision_tokens.shape[1]),
                device=attention_mask.device
            )
            attention_mask = torch.cat([vision_mask, attention_mask], dim=1)

        # Generate
        outputs = self.language_model(
            inputs_embeds=combined,
            attention_mask=attention_mask
        )

        return {
            "logits": outputs.logits,
            "loss": outputs.loss if hasattr(outputs, 'loss') else None
        }
```

## Optimization Tricks

### 1. Progressive Scaling

```python
# Train 300M → 1B → 6B with warm-starting
def progressive_scaling(configs):
    """Train progressively larger models."""

    model_300m = InternVL(configs["300m"])
    train(model_300m)

    # Initialize 1B from 300M
    model_1b = InternVL(configs["1b"])
    load_partial_weights(model_1b, model_300m)
    train(model_1b)

    # Initialize 6B from 1B
    model_6b = InternVL(configs["6b"])
    load_partial_weights(model_6b, model_1b)
    train(model_6b)

    return model_6b
```

### 2. Efficient Large Batch Training

```python
# Gradient accumulation for large batch sizes
accumulation_steps = 8
effective_batch_size = batch_size * accumulation_steps

for i, (images, texts) in enumerate(dataloader):
    loss = model(images, texts)["loss"]
    loss = loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 3. Mixed Precision for 6B Model

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(images, texts)
    loss = outputs["loss"]

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 4. Checkpoint Activation

```python
# Save memory on 6B model
from torch.utils.checkpoint import checkpoint

class InternVisionBlock(nn.Module):
    def forward(self, x):
        if self.training and self.use_checkpoint:
            return checkpoint(self._forward, x)
        return self._forward(x)

    def _forward(self, x):
        # Actual block computation
        ...
```

### 5. Flash Attention for Long Sequences

```python
# For processing many patch tokens
from flash_attn import flash_attn_func

class InternVisionBlock(nn.Module):
    def forward(self, x):
        if self.use_flash_attn:
            qkv = self.qkv(x)
            q, k, v = qkv.chunk(3, dim=-1)
            out = flash_attn_func(q, k, v)
        else:
            out = standard_attention(x)
        return out
```

## Experiments & Results

### Image-Text Retrieval (Flickr30K)

**Zero-shot Recall@1**:

| Model | Params | Image→Text | Text→Image |
|-------|--------|-----------|------------|
| CLIP-L | 428M | 88.0% | 68.7% |
| ALIGN | 1.8B | 88.6% | 75.7% |
| **InternVL-1B** | **1B** | **91.2%** | **77.3%** |
| **InternVL-6B** | **6B** | **94.7%** | **83.2%** |

### Visual Question Answering (VQAv2)

| Model | Vision Params | Test-dev |
|-------|--------------|----------|
| BLIP-2 | 224M | 65.0% |
| Flamingo | 80B | 67.6% |
| **InternVL-1B** | **1B** | **70.8%** |
| **InternVL-6B** | **6B** | **75.4%** |

### Image Captioning (COCO)

**CIDEr Score**:

| Model | Karpathy Test |
|-------|--------------|
| BLIP | 136.7 |
| CoCa | 143.6 |
| **InternVL-6B** | **151.2** |

### Zero-Shot Classification

**ImageNet-1K**:

| Model | Vision Params | Top-1 |
|-------|--------------|-------|
| CLIP-L | 304M | 75.4% |
| OpenCLIP-G | 1.8B | 80.1% |
| **InternVL-6B** | **6B** | **83.2%** |

### Scaling Trends

**Effect of Vision Encoder Size**:

| Vision Params | Flickr30K R@1 | VQAv2 |
|--------------|---------------|-------|
| 300M | 88.5% | 67.2% |
| 1B | 91.2% | 70.8% |
| 6B | 94.7% | 75.4% |

Consistent improvements with scaling!

### Ablation Studies

**Projection Layers**:

| Num Layers | VQAv2 |
|-----------|-------|
| 1 | 73.8% |
| 2 | **75.4%** |
| 3 | 75.2% |

**Training Stages**:

| Stage | Flickr R@1 |
|-------|-----------|
| Contrastive only | 91.2% |
| + Alignment | 93.5% |
| + Instruction | **94.7%** |

## Common Pitfalls

### 1. Not Freezing Vision in Stage 3
Wrong: Training entire model in instruction tuning
```python
# Stage 3: All parameters trainable (slow, unstable)
model = InternVLChat(config)
optimizer = AdamW(model.parameters())
```

Correct: Freeze vision encoder
```python
for param in model.vision_model.parameters():
    param.requires_grad = False
optimizer = AdamW(model.language_model.parameters())
```

### 2. Mismatched Projection Dimensions
Wrong: Vision and language dims don't match
```python
vision_dim = 3200
language_dim = 4096
projection = nn.Linear(vision_dim, 2048)  # BUG: Wrong target
```

Correct: Project to language model dimension
```python
projection = nn.Linear(vision_dim, language_dim)
```

### 3. Forgetting to Normalize
Wrong: Skip normalization in contrastive learning
```python
img_feat = vision_encoder(img)
txt_feat = text_encoder(txt)
sim = img_feat @ txt_feat.T  # BUG: Not normalized
```

Correct: L2 normalize
```python
img_feat = F.normalize(vision_encoder(img), dim=-1)
txt_feat = F.normalize(text_encoder(txt), dim=-1)
sim = img_feat @ txt_feat.T
```

### 4. Not Scaling Learning Rate
Wrong: Same LR for all stages
```python
optimizer = AdamW(model.parameters(), lr=1e-3)  # Too high for stage 3
```

Correct: Adjust per stage
```python
# Stage 1 & 2: Higher LR
lr_pretrain = 1e-3
# Stage 3: Lower LR for fine-tuning
lr_finetune = 1e-5
```

### 5. OOM with Large Vision Encoder
Wrong: Not using memory optimizations
```python
model_6b = InternVL({"vision_depth": 48})  # OOM!
```

Correct: Enable optimizations
```python
config = {
    "vision_depth": 48,
    "use_checkpoint": True,      # Activation checkpointing
    "use_flash_attn": True,      # Flash attention
}
model_6b = InternVL(config)
```

## References

### Original Papers
1. **InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks**
   Chen, Z., Wu, J., Wang, W., et al., CVPR 2024
   https://arxiv.org/abs/2312.14238

2. **InternVL 1.5: Bridging Vision and Language for Enhanced Multimodal Understanding**
   Chen, Z., et al., 2024
   https://arxiv.org/abs/2404.16821

### Related Vision-Language Models
3. **CLIP: Learning Transferable Visual Models from Natural Language Supervision**
   Radford, A., et al., ICML 2021
   https://arxiv.org/abs/2103.00020

4. **BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders**
   Li, J., et al., ICML 2023
   https://arxiv.org/abs/2301.12597

5. **Flamingo: a Visual Language Model for Few-Shot Learning**
   Alayrac, J., et al., NeurIPS 2022
   https://arxiv.org/abs/2204.14198

### Language Models
6. **LLaMA: Open and Efficient Foundation Language Models**
   Touvron, H., et al., 2023
   https://arxiv.org/abs/2302.13971

### Implementation
- Official: https://github.com/OpenGVLab/InternVL
- Nexus: `Nexus/nexus/models/cv/intern_vl.py`
- Hugging Face: https://huggingface.co/OpenGVLab/InternVL

### Applications
7. **Mini-InternVL: A Flexible-Transfer Pocket Multimodal Model**
   Compact variant for edge deployment

### Datasets
8. **Large-scale Vision-Language Datasets**
   LAION-5B, COYO-700M, etc.
