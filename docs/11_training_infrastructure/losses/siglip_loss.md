# SigLIP Loss: Sigmoid Loss for Language-Image Pre-Training

## Overview

SigLIP (Zhai et al., 2023) replaces CLIP's global softmax with per-pair sigmoid loss, eliminating the need for global normalization and making the loss more scalable for distributed training with huge batch sizes.

## Key Innovation

**CLIP/InfoNCE**: Global softmax over all pairs
$$\\mathcal{L}_{\\text{CLIP}} = -\\log \\frac{\\exp(s_{ii}/\\tau)}{\\sum_j \\exp(s_{ij}/\\tau)}$$

**SigLIP**: Per-pair sigmoid
$$\\mathcal{L}_{\\text{SigLIP}} = -\\sum_{i,j} \\log \\sigma(y_{ij} \\cdot (s_{ij}/\\tau - b))$$

Where $y_{ij} = +1$ for matched pairs, $y_{ij} = -1$ otherwise.

**Benefits**:
1. **No global normalization**: Each pair independent
2. **Better distributed training**: No need to gather all similarities
3. **More stable**: No numerical issues from softmax over large batch
4. **Simpler**: Sigmoid easier than softmax

## Mathematical Details

### Per-Pair Formulation

For each image-text pair $(i, j)$:

$$\\ell_{ij} = -\\log \\sigma(y_{ij} \\cdot z_{ij})$$

Where:
- $z_{ij} = \\text{sim}(\\text{image}_i, \\text{text}_j) / \\tau - b$
- $y_{ij} = +1$ if $i = j$ (matched), $-1$ otherwise
- $b$: Learnable bias term (optional, default 0)

**Binary cross-entropy interpretation**: Treat each pair as binary classification (match vs non-match).

### Full Loss

$$\\mathcal{L} = \\frac{1}{B^2} \\sum_{i=1}^B \\sum_{j=1}^B -\\log \\sigma(y_{ij} \\cdot (s_{ij}/\\tau - b))$$

**Symmetry**: Typically compute both image-to-text and text-to-image.

## Implementation

### Basic Usage

```python
from nexus.training.losses import SigmoidContrastiveLoss

loss_fn = SigmoidContrastiveLoss(temperature=0.1, bias=0.0)

# Image and text embeddings
image_features = image_encoder(images)  # (batch, dim)
text_features = text_encoder(texts)  # (batch, dim)

# Compute loss
loss = loss_fn(image_features, text_features)
```

### CLIP-Style Training

```python
class SigLIPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = VisionTransformer()
        self.text_encoder = TextTransformer()
        self.loss_fn = SigmoidContrastiveLoss(temperature=0.1)
        
    def forward(self, images, texts):
        # Encode
        img_emb = self.image_encoder(images)
        txt_emb = self.text_encoder(texts)
        
        # Normalize
        img_emb = F.normalize(img_emb, dim=-1)
        txt_emb = F.normalize(txt_emb, dim=-1)
        
        # SigLIP loss
        loss = self.loss_fn(img_emb, txt_emb)
        
        return loss

# Training
model = SigLIPModel()
for batch in dataloader:
    images, texts = batch
    loss = model(images, texts)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### With Learnable Bias

```python
# Bias helps with calibration
loss_fn = SigmoidContrastiveLoss(temperature=0.1, bias=0.0)

# Bias is a learned parameter
# Optimized along with model parameters
optimizer = torch.optim.AdamW([
    {'params': model.parameters()},
    {'params': [loss_fn.bias]}  # Include bias in optimization
], lr=1e-4)
```

## Hyperparameters

### Temperature

**Recommended**: 0.1 (slightly higher than CLIP's 0.07)

**Effect**:
- **Lower**: Sharper distinction between matches/non-matches
- **Higher**: Smoother, more tolerant of ambiguity

### Bias

**Default**: 0.0 (learnable parameter)

**Effect**: Shifts the logits, helping with calibration
- Starts at 0, learns optimal offset during training
- Typical final value: -5 to +5

## Advantages Over CLIP Loss

### 1. Scalability

**CLIP**: Requires all-gather of embeddings for global softmax
```
Batch size B across P GPUs:
- Each GPU: B/P samples
- All-gather: Collect all B samples
- Softmax: Over all B samples
```

**SigLIP**: Per-pair sigmoid, no all-gather needed
```
Each GPU: Compute loss on local B/P samples independently
Reduction: Simple average (no gather needed)
```

**Result**: SigLIP scales better to huge batch sizes and many GPUs.

### 2. Numerical Stability

**CLIP**: Softmax over large batch can have numerical issues
- Overflow with exp(large_logit)
- Requires careful scaling

**SigLIP**: Sigmoid numerically stable
- log-sigmoid uses log-sum-exp trick
- No issues with large batches

### 3. Interpretability

**CLIP**: Relative scores (via softmax)
- Hard to interpret individual similarity

**SigLIP**: Absolute scores (via sigmoid)
- Each pair has independent probability
- Easier to calibrate and interpret

## Performance

**Empirical Results** (from paper):

| Model | Loss | ImageNet Acc | Retrieval@1 |
|-------|------|-------------|-------------|
| ViT-B/16 | CLIP | 68.2% | 37.8% |
| ViT-B/16 | SigLIP | 68.5% | 38.4% |

**Key Findings**:
- **Similar or better** performance than CLIP
- **More stable** training with large batches
- **Faster** due to no all-gather overhead

## When to Use

**Use SigLIP over CLIP when**:
- Very large batch sizes (>8K)
- Distributed training across many GPUs
- Numerical stability issues with CLIP
- Want simpler, more interpretable loss

**Use CLIP when**:
- Following exact CLIP setup
- Smaller batch sizes (<1K)
- Established pipeline already using CLIP

## Batch Size Considerations

**SigLIP Less Sensitive to Batch Size**:
- CLIP: Performance degrades with small batch (fewer negatives)
- SigLIP: More robust, still good with smaller batches

**Recommended Batch Sizes**:
- Minimum: 256
- Good: 1K - 4K
- Excellent: 8K+

## Distributed Training

```python
import torch.distributed as dist

# SigLIP naturally distributed (no all-gather!)
class DistributedSigLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = ...
        self.loss_fn = SigmoidContrastiveLoss(temperature=0.1)
    
    def forward(self, images, texts):
        # Encode on local GPU
        img_emb = self.encode_images(images)
        txt_emb = self.encode_texts(texts)
        
        # Compute loss on local pairs (no gather!)
        loss = self.loss_fn(img_emb, txt_emb)
        
        # Backward propagates gradients automatically
        return loss

# Each GPU: Independent loss computation
# Gradients: Averaged via DDP automatically
```

**Advantage**: No communication overhead for loss computation!

## References

**Sigmoid Loss for Language Image Pre-Training**  
Xiaohua Zhai et al., ICCV 2023  
https://arxiv.org/abs/2303.15343

**Key Contribution**: Demonstrated that per-pair sigmoid matches or exceeds CLIP's global softmax while being simpler and more scalable.

**Implementation**: `nexus/training/losses.py` (SigmoidContrastiveLoss class)
