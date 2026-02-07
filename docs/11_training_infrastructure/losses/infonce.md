# InfoNCE Loss: Contrastive Learning with Noise Contrastive Estimation

## Overview

InfoNCE (van den Oord et al., 2018) is a contrastive loss function that learns representations by distinguishing positive pairs from negative pairs. Fundamental to contrastive learning methods like SimCLR, CLIP, and MoCo.

## Mathematical Formulation

For anchor $a$, positive $p$, and $N$ negatives $\\{n_1, \\ldots, n_N\\}$:

$$\\mathcal{L}_{\\text{InfoNCE}} = -\\log \\frac{\\exp(\\text{sim}(a, p) / \\tau)}{\\exp(\\text{sim}(a, p) / \\tau) + \\sum_{i=1}^N \\exp(\\text{sim}(a, n_i) / \\tau)}$$

Where:
- $\\text{sim}(x, y) = x^T y / (\\|x\\| \\|y\\|)$: Cosine similarity
- $\\tau$: Temperature parameter (typically 0.07-0.1)

**Intuition**: Push positive pair together, push negatives apart, all relative to temperature.

## Implementation

### Basic Usage

```python
from nexus.training.losses import InfoNCELoss

loss_fn = InfoNCELoss(temperature=0.07)

# Contrastive batch
anchors = model.encode(images_1)  # (batch, dim)
positives = model.encode(images_2)  # (batch, dim) - augmented versions
negatives = model.encode(negative_images)  # (neg_samples, dim)

loss = loss_fn(anchors, positives, negatives)
```

### Self-Supervised Learning (SimCLR Style)

```python
import torch.nn.functional as F

class SimCLRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNet50()
        self.projector = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 128)
        )
        self.loss_fn = InfoNCELoss(temperature=0.07)
    
    def forward(self, x1, x2):
        # Encode and project
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))
        
        # Normalize
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # In-batch negatives: all other samples
        # Positive: z1[i] with z2[i]
        # Negatives: z1[i] with z2[j] for all j != i
        batch_size = z1.shape[0]
        
        # Compute similarity matrix
        logits = torch.matmul(z1, z2.T) / self.loss_fn.temperature
        
        # Labels: diagonal is positive
        labels = torch.arange(batch_size, device=z1.device)
        
        # Cross-entropy over similarities
        loss = F.cross_entropy(logits, labels)
        
        return loss

# Training
model = SimCLRModel()
for batch in dataloader:
    x1, x2 = augment(batch)  # Two augmented views
    loss = model(x1, x2)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

## Temperature Parameter

**Effect of $\\tau$**:
- **Low $\\tau$ (e.g., 0.01)**: Sharp distribution, hard negatives emphasized
- **High $\\tau$ (e.g., 0.5)**: Smooth distribution, all negatives similar weight
- **Typical**: 0.05 - 0.1 for vision, 0.07 for CLIP

**Intuition**: Temperature controls how much the model focuses on hard negatives vs treating all negatives equally.

## Variants

### NT-Xent (Normalized Temperature-scaled Cross Entropy)

Used in SimCLR, treats both (z1, z2) and (z2, z1) as positive pairs:

```python
# Symmetric loss
loss_1to2 = F.cross_entropy(logits_1to2, labels)
loss_2to1 = F.cross_entropy(logits_2to1, labels)
loss = (loss_1to2 + loss_2to1) / 2
```

### Multi-Positive InfoNCE

Multiple positives per anchor (e.g., multiple augmentations):

$$\\mathcal{L} = -\\log \\frac{\\sum_{p \\in P} \\exp(\\text{sim}(a, p) / \\tau)}{\\sum_{p \\in P} \\exp(\\text{sim}(a, p) / \\tau) + \\sum_{n \\in N} \\exp(\\text{sim}(a, n) / \\tau)}$$

## Performance Characteristics

**Batch Size Dependency**: Larger batch = more negatives = better learning
- **Small batch** (<256): May underperform
- **Medium batch** (256-1024): Good performance
- **Large batch** (>2048): Best results (but needs special techniques)

**Large Batch Techniques**:
1. **Gradient accumulation**: Accumulate over multiple mini-batches
2. **Memory bank**: Store previous embeddings as negatives
3. **MoCo-style queue**: FIFO queue of negatives

## Applications

1. **Self-Supervised Learning**: Learn representations without labels
2. **CLIP**: Align image and text representations
3. **Retrieval**: Learn embeddings for similarity search
4. **Metric Learning**: Learn distance-based embeddings

## When to Use

**Best for**:
- Self-supervised pre-training
- Multi-modal learning (image-text, audio-video)
- Representation learning for downstream tasks

**Alternatives**:
- **Triplet Loss**: When you want explicit margin
- **Supervised Contrastive**: When you have labels
- **SigLIP Loss**: Per-pair sigmoid (no global softmax)

## References

**Representation Learning with Contrastive Predictive Coding**  
van den Oord et al., 2018  
https://arxiv.org/abs/1807.03748

**A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)**  
Chen et al., ICML 2020

**Implementation**: `nexus/training/losses.py` (InfoNCELoss class)
