# SigLIP (Sigmoid Loss for Language-Image Pre-training)

## Overview & Motivation

SigLIP is an efficient vision-language pre-training method that replaces CLIP's softmax-based contrastive loss with a simple sigmoid loss. This change eliminates the need for large batch sizes and global normalization, making training more memory-efficient while achieving comparable or better performance.

**Key Innovation**: Sigmoid loss treats each image-text pair as an independent binary classification problem, removing the dependence on batch statistics and enabling better scalability.

**Why It Matters**:
- More memory-efficient than CLIP (no need for large batches)
- Better performance with smaller batch sizes
- Simpler loss function (binary cross-entropy vs InfoNCE)
- Easier to implement and train
- Scales better to distributed settings

## Theoretical Background

### Problem Setting
Learn joint embeddings of images and text such that matching pairs have high similarity and non-matching pairs have low similarity. Traditional CLIP uses softmax over the batch, requiring large batches (32K+) for good performance.

### Core Approach
1. Encode images and text into a shared embedding space
2. Compute pairwise similarities (dot products)
3. Treat each pair as binary classification: match (1) or no match (0)
4. Use sigmoid loss instead of softmax contrastive loss
5. Apply learnable temperature and bias parameters

### Key Insight
Softmax normalization couples all examples in a batch through the partition function, requiring large batches to estimate the denominator well. Sigmoid loss treats each pair independently, removing this constraint and making training more stable with smaller batches.

## Mathematical Formulation

### 1. Vision and Text Encoders
```
Image encoder: f_v: R^(H×W×3) → R^D
Text encoder: f_t: Z^L → R^D

where:
- f_v is a Vision Transformer (ViT)
- f_t is a Text Transformer
- Both outputs are L2-normalized
```

### 2. Similarity Computation
For batch of size B with images {I₁, ..., I_B} and texts {T₁, ..., T_B}:
```
Image embeddings: v_i = L2Normalize(f_v(I_i)) ∈ R^D
Text embeddings: t_j = L2Normalize(f_t(T_j)) ∈ R^D

Similarity matrix:
S_ij = ⟨v_i, t_j⟩ = v_i^T · t_j ∈ [-1, 1]

for i,j ∈ {1, ..., B}
```

### 3. CLIP's Softmax Loss (for comparison)
```
# Softmax over rows (image-to-text)
L_i2t = -log(exp(S_ii / τ) / Σ_j exp(S_ij / τ))

# Softmax over columns (text-to-image)
L_t2i = -log(exp(S_ii / τ) / Σ_i exp(S_ij / τ))

# Total
L_CLIP = (L_i2t + L_t2i) / 2

Problem: Denominator sums over entire batch → needs large batches
```

### 4. SigLIP's Sigmoid Loss
```
# Apply learnable temperature and bias
z_ij = t · S_ij + b

where:
- t is learnable temperature (init: 10.0)
- b is learnable bias (init: -10.0)

# Binary labels: y_ij = 1 if i==j, else 0
Labels = I_B  (identity matrix)

# Binary cross-entropy for each pair independently
L_ij = -y_ij·log(σ(z_ij)) - (1-y_ij)·log(1-σ(z_ij))

Equivalent to:
L_ij = log(1 + exp(-z_ij))  if i==j  (positive pair)
L_ij = log(1 + exp(z_ij))   if i≠j  (negative pair)

# Average over all B² pairs
L_SigLIP = (1/B²) Σ_i Σ_j L_ij
```

### 5. Simplified Form
Using PyTorch's binary_cross_entropy_with_logits:
```
logits = t · S + b  (element-wise)
labels = eye(B)  (identity matrix)

loss = BCEWithLogitsLoss(logits, labels)
```

### 6. Why Sigmoid Works Better
```
Softmax (CLIP):
- Couples all negatives in denominator: exp(s₁)+exp(s₂)+...+exp(s_B)
- Requires large B to accurately estimate partition function
- Gradient dominated by hardest negatives

Sigmoid (SigLIP):
- Independent binary decisions: Is (i,j) a match?
- No coupling between examples
- Works well with small batches
- All negatives contribute equally
```

### 7. Learnable Temperature and Bias
```
Initial values chosen to match softmax behavior:
t = 10.0  → makes sigmoid steeper (sharpens decisions)
b = -10.0 → shifts decision boundary

During training, these adapt to the data:
- t increases → sharper boundaries
- b adjusts to balance positive/negative pairs
```

## High-Level Intuition

```
Input Image              Input Text
    ↓                        ↓
┌─────────────────┐    ┌─────────────────┐
│ Vision Encoder  │    │ Text Encoder    │
│   (ViT-B/16)    │    │ (Transformer)   │
│                 │    │                 │
│ • Patch Embed   │    │ • Token Embed   │
│ • 12 Layers     │    │ • 12 Layers     │
│ • Hidden: 768   │    │ • Hidden: 512   │
└─────────────────┘    └─────────────────┘
        ↓                      ↓
    [CLS Token]           [EOS Token]
        ↓                      ↓
┌─────────────────┐    ┌─────────────────┐
│ Projection      │    │ Projection      │
│ 768 → 512       │    │ 512 → 512       │
└─────────────────┘    └─────────────────┘
        ↓                      ↓
   [L2 Normalize]        [L2 Normalize]
        ↓                      ↓
    v ∈ R^512             t ∈ R^512
        │                      │
        └──────────┬───────────┘
                   ↓
           [Dot Product Matrix]
               S = v·t^T
         ┌─────────────────┐
         │ s₁₁  s₁₂  s₁₃  │  Each entry: similarity
         │ s₂₁  s₂₂  s₂₃  │
         │ s₃₁  s₃₂  s₃₃  │
         └─────────────────┘
                   ↓
      [Temperature & Bias]
         z = 10·S + (-10)
                   ↓
         [Sigmoid Loss]
      For each entry (i,j):
        If i==j: -log(σ(z_ij))     Match
        If i≠j: -log(1-σ(z_ij))    No match
                   ↓
           [Average All]
              Loss

CLIP Alternative:         SigLIP (Ours):
┌──────────────┐         ┌──────────────┐
│  Softmax     │         │   Sigmoid    │
│  (coupled)   │         │ (independent)│
│              │         │              │
│ Needs large  │  vs     │ Works with   │
│ batches      │         │ small batches│
│ (32K+)       │         │ (512)        │
└──────────────┘         └──────────────┘
```

**Why Independent Classification Works**:
- Each pair asks: "Do these match?" → Binary question
- No need to compare against all other pairs simultaneously
- More stable gradients with small batches
- Can use standard BCE loss (well-understood)

## Implementation Details

### Model Variants

| Variant | Vision Model | Text Model | Embed Dim | Params |
|---------|--------------|------------|-----------|--------|
| SigLIP-B/16 | ViT-B/16 | BERT-Base | 512 | 149M |
| SigLIP-L/16 | ViT-L/16 | BERT-Large | 768 | 428M |
| SigLIP-So400m | ViT-So400m | BERT-Large | 1152 | 878M |

### Configuration Example

```python
config = {
    # Vision encoder
    "img_size": 224,
    "patch_size": 16,
    "vision_embed_dim": 768,
    "vision_depth": 12,
    "vision_num_heads": 12,
    "vision_mlp_ratio": 4.0,

    # Text encoder
    "vocab_size": 49408,
    "max_seq_len": 77,
    "text_embed_dim": 512,
    "text_depth": 12,
    "text_num_heads": 8,
    "text_causal": False,  # Bidirectional

    # Joint embedding
    "output_dim": 512,
    "dropout": 0.0,

    # Loss
    "init_temperature": 10.0,
    "init_bias": -10.0,
    "learnable_temperature": True,
}
```

## Code Walkthrough

Reference: `Nexus/nexus/models/cv/siglip.py`

### 1. Vision Encoder

```python
class SigLIPVisionEncoder(NexusModule):
    """ViT-based image encoder."""

    def __init__(self, config):
        super().__init__(config)

        # Standard ViT architecture
        self.patch_embed = nn.Conv2d(
            config["in_channels"],
            config["embed_dim"],
            kernel_size=config["patch_size"],
            stride=config["patch_size"]
        )

        # CLS token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config["embed_dim"]))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, config["embed_dim"])
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            VisionTransformerBlock(
                dim=config["embed_dim"],
                num_heads=config["num_heads"],
                mlp_ratio=config["mlp_ratio"]
            ) for _ in range(config["depth"])
        ])

        self.norm = nn.LayerNorm(config["embed_dim"])

        # Project to output dimension
        self.head = nn.Linear(
            config["embed_dim"],
            config["output_dim"],
            bias=False
        )

    def forward(self, images):
        """
        Args:
            images: (B, 3, H, W)
        Returns:
            features: (B, output_dim) normalized
        """
        # Patch embedding
        x = self.patch_embed(images)
        x = x.flatten(2).transpose(1, 2)

        # Add CLS token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional encoding
        x = x + self.pos_embed

        # Transformer
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Extract CLS token and project
        cls_output = x[:, 0]
        features = self.head(cls_output)

        # L2 normalization (crucial!)
        features = F.normalize(features, dim=-1)

        return features
```

### 2. Text Encoder

```python
class SigLIPTextEncoder(NexusModule):
    """Transformer-based text encoder."""

    def __init__(self, config):
        super().__init__(config)

        # Token and position embeddings
        self.token_embed = nn.Embedding(
            config["vocab_size"],
            config["embed_dim"]
        )
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config["max_seq_len"], config["embed_dim"])
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TextTransformerBlock(
                dim=config["embed_dim"],
                num_heads=config["num_heads"],
                causal=config.get("causal", False)
            ) for _ in range(config["depth"])
        ])

        self.norm = nn.LayerNorm(config["embed_dim"])

        # Project to output dimension
        self.head = nn.Linear(
            config["embed_dim"],
            config["output_dim"],
            bias=False
        )

    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: (B, L) token indices
            attention_mask: (B, L) attention mask
        Returns:
            features: (B, output_dim) normalized
        """
        # Token + position embeddings
        x = self.token_embed(input_ids)
        x = x + self.pos_embed[:, :input_ids.shape[1]]

        # Transformer
        for block in self.blocks:
            x = block(x, attention_mask)

        x = self.norm(x)

        # Pool: use last valid token (EOS)
        if attention_mask is not None:
            seq_lengths = attention_mask.sum(dim=1) - 1
            pooled = x[torch.arange(x.shape[0]), seq_lengths]
        else:
            pooled = x[:, -1]  # Last token

        # Project and normalize
        features = self.head(pooled)
        features = F.normalize(features, dim=-1)

        return features
```

### 3. Sigmoid Loss

```python
class SigLIPLoss(NexusModule):
    """Sigmoid contrastive loss."""

    def __init__(self, init_temperature=10.0, init_bias=-10.0,
                 learnable=True):
        super().__init__()

        if learnable:
            self.temperature = nn.Parameter(torch.tensor(init_temperature))
            self.bias = nn.Parameter(torch.tensor(init_bias))
        else:
            self.register_buffer("temperature", torch.tensor(init_temperature))
            self.register_buffer("bias", torch.tensor(init_bias))

    def forward(self, image_features, text_features):
        """
        Args:
            image_features: (B, D) normalized
            text_features: (B, D) normalized
        Returns:
            loss: scalar
        """
        B = image_features.shape[0]

        # Compute similarity matrix
        logits = image_features @ text_features.T  # (B, B)

        # Apply temperature and bias
        logits = self.temperature * logits + self.bias

        # Create labels (identity matrix)
        labels = torch.eye(B, device=logits.device, dtype=torch.float32)

        # Binary cross-entropy with logits
        # This is equivalent to:
        # -labels * log(sigmoid(logits)) - (1-labels) * log(1-sigmoid(logits))
        loss = F.binary_cross_entropy_with_logits(
            logits.flatten(),
            labels.flatten(),
            reduction='mean'
        )

        return loss
```

### 4. Complete SigLIP Model

```python
class SigLIP(NexusModule):
    def __init__(self, config):
        super().__init__(config)

        # Vision encoder
        vision_config = {
            "img_size": config.get("img_size", 224),
            "patch_size": config.get("patch_size", 16),
            "embed_dim": config.get("vision_embed_dim", 768),
            "depth": config.get("vision_depth", 12),
            "num_heads": config.get("vision_num_heads", 12),
            "output_dim": config.get("output_dim", 512),
        }
        self.vision_encoder = SigLIPVisionEncoder(vision_config)

        # Text encoder
        text_config = {
            "vocab_size": config.get("vocab_size", 49408),
            "max_seq_len": config.get("max_seq_len", 77),
            "embed_dim": config.get("text_embed_dim", 512),
            "depth": config.get("text_depth", 12),
            "num_heads": config.get("text_num_heads", 8),
            "output_dim": config.get("output_dim", 512),
        }
        self.text_encoder = SigLIPTextEncoder(text_config)

        # Loss function
        self.loss_fn = SigLIPLoss(
            init_temperature=config.get("init_temperature", 10.0),
            init_bias=config.get("init_bias", -10.0),
            learnable=config.get("learnable_temperature", True)
        )

    def encode_image(self, images):
        return self.vision_encoder(images)

    def encode_text(self, input_ids, attention_mask=None):
        return self.text_encoder(input_ids, attention_mask)

    def forward(self, images, input_ids, attention_mask=None,
                return_loss=True):
        """
        Args:
            images: (B, 3, H, W)
            input_ids: (B, L)
            attention_mask: (B, L)
            return_loss: whether to compute loss

        Returns:
            dict with image_features, text_features, logits, loss
        """
        # Encode
        image_features = self.encode_image(images)
        text_features = self.encode_text(input_ids, attention_mask)

        # Similarity
        logits = image_features @ text_features.T

        output = {
            "image_features": image_features,
            "text_features": text_features,
            "logits": logits
        }

        if return_loss:
            loss = self.loss_fn(image_features, text_features)
            output["loss"] = loss

        return output
```

### 5. Training Loop

```python
def train_siglip(model, dataloader, optimizer, config):
    """SigLIP training loop."""

    for epoch in range(config["epochs"]):
        for images, texts, masks in dataloader:
            images = images.to(device)
            texts = texts.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(
                images,
                texts,
                attention_mask=masks,
                return_loss=True
            )

            loss = outputs["loss"]

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log
            if step % 100 == 0:
                print(f"Loss: {loss.item():.4f}, "
                      f"Temp: {model.loss_fn.temperature.item():.2f}, "
                      f"Bias: {model.loss_fn.bias.item():.2f}")
```

## Optimization Tricks

### 1. Batch Size Scaling

```python
# SigLIP works well with smaller batches than CLIP
# CLIP: 32768 batch size
# SigLIP: 512-2048 batch size

# Can use more frequent gradient updates
config = {
    "batch_size": 1024,      # vs 32K for CLIP
    "accumulation_steps": 4,  # Effective batch: 4096
}
```

### 2. Temperature Initialization

```python
# Initialize temperature high to start with sharp predictions
self.temperature = nn.Parameter(torch.tensor(10.0))

# Optionally clamp during training
self.temperature.data.clamp_(min=1.0, max=100.0)
```

### 3. Gradient Checkpointing

```python
# Save memory on vision encoder
from torch.utils.checkpoint import checkpoint

for block in self.vision_encoder.blocks:
    if self.training:
        x = checkpoint(block, x)
    else:
        x = block(x)
```

### 4. Mixed Precision Training

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

### 5. Efficient Similarity Computation

```python
# For large batches, compute in chunks to save memory
def chunked_similarity(image_feat, text_feat, chunk_size=256):
    B = image_feat.shape[0]
    logits = torch.zeros(B, B, device=image_feat.device)

    for i in range(0, B, chunk_size):
        logits[i:i+chunk_size] = image_feat[i:i+chunk_size] @ text_feat.T

    return logits
```

## Experiments & Results

### ImageNet Zero-Shot Classification

**Setup**:
- Pre-training: WebLI (10B image-text pairs)
- Evaluation: Zero-shot on ImageNet-1K
- Batch sizes: CLIP (32K), SigLIP (2K)

**Results** (Top-1 Accuracy):

| Model | Batch Size | Top-1 |
|-------|-----------|-------|
| CLIP ViT-B/16 | 32768 | 68.3% |
| SigLIP ViT-B/16 | 2048 | **69.1%** |
| CLIP ViT-L/16 | 32768 | 75.4% |
| SigLIP ViT-L/16 | 2048 | **76.8%** |
| SigLIP ViT-So400m | 2048 | **83.2%** |

SigLIP achieves better accuracy with 16× smaller batches!

### Image-Text Retrieval (Flickr30K)

**Recall@1**:

| Model | Image→Text | Text→Image | Avg |
|-------|-----------|------------|-----|
| CLIP-B | 88.0% | 68.7% | 78.4% |
| SigLIP-B | **90.2%** | **72.1%** | **81.2%** |
| CLIP-L | 92.6% | 76.2% | 84.4% |
| SigLIP-L | **94.1%** | **79.3%** | **86.7%** |

### Training Efficiency

**Wall-Clock Time to 70% ImageNet Zero-Shot**:

| Model | Batch Size | GPUs | Time |
|-------|-----------|------|------|
| CLIP | 32768 | 256× A100 | 12 days |
| SigLIP | 2048 | 16× A100 | **10 days** |

16× fewer GPUs, similar or better results!

### Ablation Studies

**Loss Function Comparison**:

| Loss | Batch Size | ImageNet Top-1 |
|------|-----------|---------------|
| Softmax (CLIP) | 32768 | 68.3% |
| Softmax (CLIP) | 2048 | 64.1% ⚠️ |
| Sigmoid (SigLIP) | 2048 | **69.1%** ✓ |
| Sigmoid (SigLIP) | 512 | 67.8% ✓ |

Sigmoid loss is robust to batch size!

**Temperature & Bias**:

| Config | ImageNet Top-1 |
|--------|---------------|
| Fixed t=1, b=0 | 65.2% |
| Fixed t=10, b=-10 | 67.8% |
| Learnable (ours) | **69.1%** |

Learnable parameters important!

### Qualitative Results

**Attention Maps**:
SigLIP learns to focus on discriminative regions without explicit supervision.

**Embedding Space**:
Well-clustered semantic categories, smooth interpolation between concepts.

## Common Pitfalls

### 1. Not Normalizing Features
Wrong: Forgetting L2 normalization
```python
features = self.head(cls_token)  # BUG: Not normalized
return features
```

Correct: Always normalize
```python
features = self.head(cls_token)
features = F.normalize(features, dim=-1)
return features
```

### 2. Wrong Label Shape
Wrong: Using class indices
```python
labels = torch.arange(B)  # BUG: Should be matrix
```

Correct: Identity matrix
```python
labels = torch.eye(B, device=logits.device)
```

### 3. Not Learning Temperature
Wrong: Fixed temperature
```python
self.temperature = 10.0  # Suboptimal
```

Correct: Learnable parameter
```python
self.temperature = nn.Parameter(torch.tensor(10.0))
```

### 4. Large Batch Size Assumption
Wrong: Thinking you need huge batches
```python
# Don't need this for SigLIP!
config = {"batch_size": 32768}  # Unnecessary
```

Correct: Use reasonable batch size
```python
config = {"batch_size": 1024}  # Works great
```

### 5. Forgetting Attention Mask
Wrong: Not using attention mask
```python
text_features = self.text_encoder(input_ids)  # BUG: Pads leak
```

Correct: Pass attention mask
```python
text_features = self.text_encoder(input_ids, attention_mask)
```

## References

### Original Papers
1. **Sigmoid Loss for Language Image Pre-Training**
   Zhai, X., Mustafa, B., Kolesnikov, A., et al., ICCV 2023
   https://arxiv.org/abs/2303.15343

2. **Learning Transferable Visual Models From Natural Language Supervision (CLIP)**
   Radford, A., et al., ICML 2021
   https://arxiv.org/abs/2103.00020

### Analysis
3. **Scaling Laws for Contrastive Language-Image Learning**
   Cherti, M., et al., 2023

### Related Work
4. **ALIGN: Scaling Up Visual and Vision-Language Representation Learning**
   Jia, C., et al., ICML 2021

5. **LiT: Zero-Shot Transfer with Locked-image text Tuning**
   Zhai, X., et al., CVPR 2022

### Applications
6. **PaLI: A Jointly-Scaled Multilingual Language-Image Model**
   Uses SigLIP vision encoder

### Implementation
- Official: https://github.com/google-research/big_vision
- Nexus: `Nexus/nexus/models/cv/siglip.py`
- Hugging Face: https://huggingface.co/google/siglip-base-patch16-224

### Datasets
7. **WebLI: Web-scale Language-Image dataset**
   10B+ image-text pairs
