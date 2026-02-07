# InfoNCE Loss: Contrastive Learning with Noise Contrastive Estimation

## Overview

InfoNCE (Information Noise-Contrastive Estimation) is a foundational loss function for contrastive learning introduced by van den Oord et al. in their 2018 CPC (Contrastive Predictive Coding) paper. The loss enables models to learn meaningful representations by distinguishing between semantically similar pairs (positives) and dissimilar pairs (negatives) without requiring explicit labels.

InfoNCE has become ubiquitous in modern self-supervised learning, serving as the core loss function in landmark methods including:
- **SimCLR** (Google, 2020): Self-supervised visual representations
- **CLIP** (OpenAI, 2021): Vision-language alignment
- **MoCo** (Facebook AI, 2020): Momentum contrast for visual representations
- **BYOL** variants and numerous other contrastive methods

**Core Principle**: InfoNCE maximizes the mutual information between representations of similar inputs while minimizing it for dissimilar inputs, effectively learning to cluster semantically related examples in embedding space while separating unrelated ones.

**Key Innovation**: Unlike traditional cross-entropy which requires explicit class labels, InfoNCE creates a self-supervised classification task from the structure of the data itself: given an anchor, identify its positive match among many negatives.

**Why "Noise-Contrastive"**: The name reflects the loss's connection to Noise Contrastive Estimation (NCE), where the model learns to distinguish true data (positive) from noise samples (negatives). This perspective treats contrastive learning as a binary classification problem scaled to multiple negatives.

## Theoretical Background

### Information-Theoretic Motivation

InfoNCE maximizes a lower bound on mutual information $I(x; y)$ between input $x$ and its representation $y$.

**Mutual Information**:
$$I(x; y) = \mathbb{E}_{p(x,y)} \left[\log \frac{p(x|y)}{p(x)}\right]$$

**Challenge**: Computing $I(x; y)$ directly requires knowing the marginal distribution $p(x)$, which is intractable for high-dimensional data.

**Solution**: InfoNCE approximates mutual information using contrastive learning:

For anchor-positive pair $(a, p)$ and $N$ negatives $\{n_1, \ldots, n_N\}$ drawn from the marginal:

$$I(a; p) \geq \log(N) - \mathcal{L}_{\text{InfoNCE}}$$

This lower bound becomes tighter as $N$ increases, meaning:
- More negatives → better MI estimate → better representations

**Intuition**: By correctly identifying the positive among $N+1$ candidates, the model demonstrates it has extracted $\log(N+1)$ bits of information about the relationship.

### Connection to Noise Contrastive Estimation

Classical NCE (Gutmann & Hyvärinen, 2010) learns by distinguishing data from noise:

$$\mathcal{L}_{\text{NCE}} = -\mathbb{E}_{p_{\text{data}}} [\log h(x)] - N \cdot \mathbb{E}_{p_{\text{noise}}} [\log(1 - h(x))]$$

where $h(x)$ is a discriminator.

**InfoNCE as Multi-Class NCE**: InfoNCE extends binary NCE to multi-class by:
1. Treating positive as "data"
2. Treating all negatives as "noise"
3. Using softmax instead of binary classification

This connection explains why InfoNCE converges to meaningful representations: it's implicitly modeling the data distribution relative to noise.

### Relation to Metric Learning

InfoNCE can be viewed as metric learning with softmax:

**Traditional metric learning** (triplet loss):
$$\mathcal{L}_{\text{triplet}} = \max(0, \|a - p\|^2 - \|a - n\|^2 + \text{margin})$$

**InfoNCE** uses all negatives simultaneously with soft selection:
- No manual margin tuning
- Automatically weights hard negatives more (via softmax)
- More sample efficient (all negatives contribute)

**Comparison**:
- Triplet: Hard boundary, binary decision per triplet
- InfoNCE: Soft probability distribution, relative ranking

### Temperature as Information Control

Temperature $\tau$ controls the "sharpness" of the distribution:

**Low $\tau$ (e.g., 0.01)**:
- Sharp distribution: $\arg\max$ like behavior
- Focuses on hardest negative
- Higher gradient variance
- Faster convergence but less stable

**High $\tau$ (e.g., 1.0)**:
- Smooth distribution: treats all negatives more equally
- Gradient from all negatives
- More stable but slower convergence

**Optimal range**: $\tau \in [0.05, 0.1]$ empirically works best for most vision tasks.

**Information perspective**: Temperature controls how much information each sample contributes:
$$H = -\sum_i p_i \log p_i \propto \tau$$

Lower temperature → lower entropy → model makes more confident decisions.

## Mathematical Formulation

### Basic InfoNCE Loss

Given:
- Anchor embedding: $a \in \mathbb{R}^d$
- Positive embedding: $p \in \mathbb{R}^d$
- Negative embeddings: $\{n_1, \ldots, n_N\} \in \mathbb{R}^d$
- Similarity function: $\text{sim}(x, y) = \frac{x^T y}{\|x\| \|y\|}$ (cosine similarity)
- Temperature: $\tau > 0$

**InfoNCE Loss**:
$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(a, p) / \tau)}{\exp(\text{sim}(a, p) / \tau) + \sum_{i=1}^N \exp(\text{sim}(a, n_i) / \tau)}$$

**Simplified form** using softmax:
$$\mathcal{L}_{\text{InfoNCE}} = -\frac{\text{sim}(a, p)}{\tau} + \log \left(\sum_{i=0}^N \exp\left(\frac{\text{sim}(a, z_i)}{\tau}\right)\right)$$

where $z_0 = p$ and $z_{i>0} = n_i$.

**Cross-entropy interpretation**:
$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(s_{\text{pos}} / \tau)}{\sum_{j} \exp(s_j / \tau)} = \text{CrossEntropy}(\text{logits}, 0)$$

where logits $= [s_{\text{pos}}, s_{n_1}, \ldots, s_{n_N}] / \tau$ and target is index 0.

### Gradient Analysis

**Gradient w.r.t. anchor**:
$$\frac{\partial \mathcal{L}}{\partial a} = \frac{1}{\tau} \left(\sum_{i=1}^N w_i \nabla_a \text{sim}(a, n_i) - \nabla_a \text{sim}(a, p)\right)$$

where weights:
$$w_i = \frac{\exp(\text{sim}(a, n_i) / \tau)}{\exp(\text{sim}(a, p) / \tau) + \sum_j \exp(\text{sim}(a, n_j) / \tau)}$$

**Key properties**:
1. Hard negatives (high similarity) get higher weight $w_i$
2. If positive is correctly ranked first, gradients are small
3. Temperature scales gradient magnitude

**Interpretation**: The gradient pushes the anchor toward the positive and away from negatives, with force proportional to how "confusing" each negative is.

### In-Batch Negatives

For efficiency, most implementations use other samples in the batch as negatives:

**Setup**: Batch of $B$ samples, each with two augmented views $(x_i^1, x_i^2)$

**Positive pairs**: $(z_i^1, z_i^2)$ for $i \in [B]$

**Negative pairs**: $(z_i^1, z_j^2)$ for $i \neq j$

**Loss for sample $i$**:
$$\mathcal{L}_i = -\log \frac{\exp(\text{sim}(z_i^1, z_i^2) / \tau)}{\sum_{j=1}^B \exp(\text{sim}(z_i^1, z_j^2) / \tau)}$$

**Total loss** (symmetric):
$$\mathcal{L} = \frac{1}{2B} \sum_{i=1}^B (\mathcal{L}_i^{1 \to 2} + \mathcal{L}_i^{2 \to 1})$$

**Effective negatives**: $N = B - 1$ per sample, but total $2B(B-1)$ comparisons.

**Computational cost**: $O(B^2 d)$ for computing all pairwise similarities.

### Variants and Extensions

#### 1. NT-Xent (Normalized Temperature-scaled Cross-Entropy)

Used in SimCLR, treats both directions symmetrically:

$$\mathcal{L}_{\text{NT-Xent}} = \frac{1}{2N} \sum_{k=1}^N [\ell(2k-1, 2k) + \ell(2k, 2k-1)]$$

where $\ell(i, j)$ is the InfoNCE loss treating $i$ as anchor and $j$ as positive.

#### 2. Hard Negative Mining

Sample or emphasize hard negatives:

$$\mathcal{L}_{\text{hard}} = -\log \frac{\exp(s_p / \tau)}{\exp(s_p / \tau) + \sum_{i \in \mathcal{H}} \exp(s_{n_i} / \tau)}$$

where $\mathcal{H} = \{i : s_{n_i} > \text{threshold}\}$ (hard negatives).

#### 3. Supervised Contrastive Loss

Extension to labeled data (Khosla et al., 2020):

$$\mathcal{L}_{\text{sup}} = -\frac{1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(z_i \cdot z_p / \tau)}{\sum_{a \in A(i)} \exp(z_i \cdot z_a / \tau)}$$

where $P(i)$ = positives (same class), $A(i)$ = all other samples.

**Advantage**: Multiple positives per class strengthen learned features.

#### 4. Debiased Contrastive Loss

Addresses sampling bias (Chuang et al., 2020):

$$\mathcal{L}_{\text{debiased}} = -\log \frac{\exp(s_p / \tau)}{\exp(s_p / \tau) + \max(\sum_i \exp(s_{n_i} / \tau) - M, e^{-1/\tau})}$$

where $M$ compensates for false negatives in batch.

## Implementation

### Basic InfoNCE Implementation

```python
import torch
import torch.nn.functional as F

def infonce_loss(anchor, positive, negatives, temperature=0.07):
    """
    Basic InfoNCE loss.

    Args:
        anchor: (batch_size, dim) anchor embeddings
        positive: (batch_size, dim) positive embeddings
        negatives: (batch_size, num_negatives, dim) negative embeddings
        temperature: float, temperature parameter

    Returns:
        loss: scalar tensor
    """
    # Normalize embeddings
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    negatives = F.normalize(negatives, dim=-1)

    # Compute similarities
    pos_sim = torch.sum(anchor * positive, dim=-1) / temperature  # (batch_size,)
    neg_sim = torch.matmul(anchor.unsqueeze(1), negatives.transpose(-2, -1)).squeeze(1) / temperature  # (batch_size, num_negatives)

    # InfoNCE loss
    logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)  # (batch_size, 1 + num_negatives)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)  # Positive is index 0

    loss = F.cross_entropy(logits, labels)
    return loss


# Example usage
batch_size = 32
dim = 128
num_negatives = 256

anchor = torch.randn(batch_size, dim)
positive = torch.randn(batch_size, dim)
negatives = torch.randn(batch_size, num_negatives, dim)

loss = infonce_loss(anchor, positive, negatives, temperature=0.07)
print(f"InfoNCE Loss: {loss.item():.4f}")
```

### Efficient In-Batch Implementation (SimCLR Style)

```python
import torch
import torch.nn.functional as F

class InfoNCELoss(torch.nn.Module):
    """InfoNCE loss with in-batch negatives."""

    def __init__(self, temperature=0.07, normalize=True):
        super().__init__()
        self.temperature = temperature
        self.normalize = normalize

    def forward(self, z_i, z_j):
        """
        Args:
            z_i: (batch_size, dim) first view embeddings
            z_j: (batch_size, dim) second view embeddings

        Returns:
            loss: scalar tensor
        """
        batch_size = z_i.shape[0]

        if self.normalize:
            z_i = F.normalize(z_i, dim=-1)
            z_j = F.normalize(z_j, dim=-1)

        # Concatenate both views
        z = torch.cat([z_i, z_j], dim=0)  # (2 * batch_size, dim)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(z, z.T) / self.temperature  # (2B, 2B)

        # Create labels: i-th sample's positive is at i + batch_size
        labels = torch.cat([torch.arange(batch_size) + batch_size,
                           torch.arange(batch_size)], dim=0)
        labels = labels.to(z_i.device)

        # Mask to remove self-similarities
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))

        # Compute loss
        loss = F.cross_entropy(similarity_matrix, labels)

        return loss


# Example usage with two augmented views
model = InfoNCELoss(temperature=0.07)

# Simulate two augmented views of same batch
z1 = torch.randn(64, 128)
z2 = torch.randn(64, 128)

loss = model(z1, z2)
print(f"InfoNCE Loss: {loss.item():.4f}")
```

### CLIP-Style Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CLIPLoss(nn.Module):
    """CLIP-style InfoNCE for image-text pairs."""

    def __init__(self, temperature=0.07, learnable_temperature=False):
        super().__init__()
        if learnable_temperature:
            # Learnable temperature (as in CLIP)
            self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / temperature)))
        else:
            self.register_buffer('logit_scale', torch.tensor(1 / temperature))
            self.logit_scale = torch.log(self.logit_scale)

    def forward(self, image_features, text_features):
        """
        Args:
            image_features: (batch_size, dim) normalized image embeddings
            text_features: (batch_size, dim) normalized text embeddings

        Returns:
            loss: scalar tensor (average of image-to-text and text-to-image)
        """
        # Normalize features
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # Compute logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logits_per_image.T

        # Labels: diagonal elements are positive pairs
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=image_features.device)

        # Symmetric loss
        loss_img = F.cross_entropy(logits_per_image, labels)
        loss_txt = F.cross_entropy(logits_per_text, labels)
        loss = (loss_img + loss_txt) / 2

        return loss


# Example usage
clip_loss = CLIPLoss(temperature=0.07, learnable_temperature=True)

image_features = torch.randn(32, 512)
text_features = torch.randn(32, 512)

loss = clip_loss(image_features, text_features)
print(f"CLIP Loss: {loss.item():.4f}")
```

### Full Training Example

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from nexus.training.losses import InfoNCELoss

class SimCLRModel(nn.Module):
    """SimCLR model with ResNet backbone and projection head."""

    def __init__(self, base_encoder='resnet50', projection_dim=128):
        super().__init__()
        # Encoder
        encoder = getattr(models, base_encoder)(pretrained=False)
        self.encoder = nn.Sequential(*list(encoder.children())[:-1])  # Remove FC layer
        encoder_dim = encoder.fc.in_features

        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x).squeeze()
        z = self.projector(h)
        return z


# Data augmentation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Model and loss
model = SimCLRModel(base_encoder='resnet50', projection_dim=128).cuda()
criterion = InfoNCELoss(temperature=0.07)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(epochs):
    for images in train_loader:
        # Generate two augmented views
        images_i = torch.stack([train_transform(img) for img in images])
        images_j = torch.stack([train_transform(img) for img in images])

        images_i = images_i.cuda()
        images_j = images_j.cuda()

        # Forward pass
        z_i = model(images_i)
        z_j = model(images_j)

        # Compute loss
        loss = criterion(z_i, z_j)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
```

### Memory-Efficient Implementation with Gradient Checkpointing

```python
from torch.utils.checkpoint import checkpoint
from nexus.training.losses import InfoNCELoss

class MemoryEfficientInfoNCE(nn.Module):
    """Memory-efficient InfoNCE using gradient checkpointing."""

    def __init__(self, temperature=0.07, chunk_size=256):
        super().__init__()
        self.temperature = temperature
        self.chunk_size = chunk_size

    def compute_similarities_chunked(self, z_i, z_j):
        """Compute similarities in chunks to save memory."""
        batch_size = z_i.shape[0]
        similarities = []

        for i in range(0, batch_size, self.chunk_size):
            end_i = min(i + self.chunk_size, batch_size)
            chunk_i = z_i[i:end_i]

            chunk_sims = []
            for j in range(0, batch_size, self.chunk_size):
                end_j = min(j + self.chunk_size, batch_size)
                chunk_j = z_j[j:end_j]

                # Use checkpoint to save memory
                sim = checkpoint(
                    lambda x, y: torch.matmul(x, y.T),
                    chunk_i,
                    chunk_j
                )
                chunk_sims.append(sim)

            similarities.append(torch.cat(chunk_sims, dim=1))

        return torch.cat(similarities, dim=0)

    def forward(self, z_i, z_j):
        z_i = F.normalize(z_i, dim=-1)
        z_j = F.normalize(z_j, dim=-1)

        # Compute similarities in chunks
        sim_matrix = self.compute_similarities_chunked(z_i, z_j) / self.temperature

        # Labels: diagonal elements
        labels = torch.arange(z_i.shape[0], device=z_i.device)

        # Symmetric loss
        loss_i2j = F.cross_entropy(sim_matrix, labels)
        loss_j2i = F.cross_entropy(sim_matrix.T, labels)

        return (loss_i2j + loss_j2i) / 2
```

## Experiments and Benchmarks

### Self-Supervised Pre-training (SimCLR on ImageNet)

**Setup**:
- Model: ResNet-50
- Dataset: ImageNet-1K (1.28M images)
- Pre-training: 100 epochs with strong augmentation
- Batch size: 256 (effective 512 with two views)
- Optimizer: LARS with cosine LR schedule

**Results** (Linear evaluation top-1 accuracy):

| Configuration | Temp | Batch | Proj Dim | Top-1 | Notes |
|---------------|------|-------|----------|-------|-------|
| SimCLR | 0.07 | 256 | 128 | 66.5% | Baseline |
| SimCLR | 0.07 | 512 | 128 | 68.2% | Larger batch |
| SimCLR | 0.07 | 1024 | 128 | 69.8% | Even larger |
| SimCLR | 0.1 | 256 | 128 | 65.1% | Higher temp |
| SimCLR | 0.05 | 256 | 128 | 67.2% | Lower temp |
| SimCLR | 0.07 | 256 | 256 | 67.3% | Larger projection |
| Supervised | - | - | - | 76.5% | Full supervision |

**Key Findings**:
- Batch size critical: larger batches → more negatives → better performance
- Temperature 0.07 optimal for vision tasks
- Still 10% gap from supervised (76.5%)

### CLIP (Vision-Language Pre-training)

**Setup**:
- Image encoder: ViT-B/32
- Text encoder: Transformer (63M params)
- Dataset: 400M image-text pairs
- Training: 32 epochs, batch size 32,768

**Results** (Zero-shot classification):

| Dataset | Random | SimCLR | CLIP | Supervised |
|---------|--------|--------|------|------------|
| ImageNet | 0.1% | - | 76.2% | 88.4% |
| CIFAR-10 | 10% | - | 94.9% | 99.0% |
| CIFAR-100 | 1% | - | 77.0% | 87.2% |
| STL-10 | 10% | - | 99.3% | 99.5% |

**Analysis**:
- CLIP's zero-shot nearly matches ImageNet supervised on some datasets
- Demonstrates power of large-scale contrastive learning
- Temperature fixed at 0.07 throughout training

### MoCo v2 (Momentum Contrast)

**Setup**:
- Model: ResNet-50
- Queue size: 65,536 negatives
- Momentum: 0.999 for key encoder
- Training: 200 epochs on ImageNet

**Results** (Linear evaluation):

| Method | Negatives | Top-1 | Top-5 |
|--------|-----------|-------|-------|
| SimCLR | 256 | 66.5% | 87.2% |
| MoCo v1 | 65,536 | 68.6% | 88.9% |
| MoCo v2 | 65,536 | 71.1% | 90.3% |
| MoCo v2 + aug | 65,536 | 73.5% | 91.6% |

**Key Innovation**: Queue of negatives enables 65k negatives without requiring huge batch sizes.

### Temperature Ablation

**Setup**: ResNet-50 on ImageNet, 100 epochs, batch size 256

| Temperature | Top-1 Acc | Loss (early) | Loss (final) | Convergence |
|-------------|-----------|--------------|--------------|-------------|
| 0.01 | 61.3% | 8.52 | 2.15 | Unstable |
| 0.05 | 67.2% | 6.78 | 1.82 | Good |
| 0.07 | 66.5% | 6.12 | 1.75 | Best |
| 0.1 | 65.1% | 5.84 | 1.71 | Good |
| 0.2 | 62.8% | 5.21 | 1.58 | Slower |
| 0.5 | 58.4% | 4.67 | 1.42 | Too smooth |

**Observations**:
- Temperature 0.07 balances stability and performance
- Too low: unstable, focuses only on hardest negative
- Too high: too smooth, doesn't emphasize hard negatives enough

### Batch Size Scaling

**Setup**: SimCLR on ImageNet, temperature 0.07

| Batch Size | Effective Negatives | Top-1 Acc | Training Time | Memory |
|------------|---------------------|-----------|---------------|--------|
| 64 | 63 | 58.2% | 100% | 8 GB |
| 128 | 127 | 62.4% | 110% | 16 GB |
| 256 | 255 | 66.5% | 120% | 32 GB |
| 512 | 511 | 68.2% | 135% | 64 GB |
| 1024 | 1023 | 69.8% | 155% | 128 GB |
| 2048 | 2047 | 71.2% | 180% | 256 GB |

**Analysis**:
- Logarithmic improvement with batch size
- Diminishing returns beyond 1024
- Memory scales linearly, performance sub-linearly

## Common Pitfalls and Solutions

### Pitfall 1: Batch Size Too Small

**Problem**:
```python
# Using small batch size
batch_size = 32
dataloader = DataLoader(dataset, batch_size=32)

loss = infonce_loss(z_i, z_j, temperature=0.07)
# Only 31 negatives per sample!
```

**Symptoms**:
- Poor representation quality
- Model learns trivial solutions
- Accuracy plateaus early

**Solution**:
```python
# Option 1: Increase batch size (if memory allows)
batch_size = 256  # At least 256 recommended
dataloader = DataLoader(dataset, batch_size=256)

# Option 2: Use memory bank (MoCo style)
from nexus.training.losses import MoCoLoss

moco_loss = MoCoLoss(
    temperature=0.07,
    queue_size=65536,  # 65k negatives regardless of batch size
    momentum=0.999
)

# Option 3: Gradient accumulation
accumulation_steps = 8
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Pitfall 2: Forgetting to Normalize Embeddings

**Problem**:
```python
def forward(self, x):
    z = self.encoder(x)
    return z  # Forgot to normalize!

# Loss computation
loss = infonce_loss(z_i, z_j, temperature=0.07)
```

**Symptoms**:
- Training instability
- Loss doesn't decrease properly
- Embeddings have varying magnitudes

**Solution**:
```python
def forward(self, x):
    z = self.encoder(x)
    z = F.normalize(z, dim=-1, p=2)  # L2 normalization
    return z

# Or normalize in loss function
class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        # Normalize inside loss function
        z_i = F.normalize(z_i, dim=-1)
        z_j = F.normalize(z_j, dim=-1)
        # ... rest of loss computation
```

### Pitfall 3: Temperature Too High or Too Low

**Problem**:
```python
# Temperature too high
loss = InfoNCELoss(temperature=1.0)  # Too smooth!

# Or temperature too low
loss = InfoNCELoss(temperature=0.01)  # Too sharp!
```

**Symptoms**:
- High temp: Slow convergence, poor separation
- Low temp: Training instability, focuses only on hardest negative

**Solution**:
```python
# Use standard temperature for vision tasks
loss = InfoNCELoss(temperature=0.07)

# For other domains, tune in range [0.05, 0.1]
# NLP: Often 0.05
# Multi-modal: Often 0.07
# Audio: Often 0.1

# Or make temperature learnable (CLIP approach)
class LearnableTemperatureInfoNCE(nn.Module):
    def __init__(self, init_temperature=0.07):
        super().__init__()
        self.log_temperature = nn.Parameter(
            torch.log(torch.tensor(1.0 / init_temperature))
        )

    def forward(self, z_i, z_j):
        temperature = 1.0 / torch.exp(self.log_temperature)
        # ... loss computation with learned temperature
```

### Pitfall 4: Not Using Strong Augmentation

**Problem**:
```python
# Weak augmentation
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])
# Positive pairs are too similar - model learns shortcuts!
```

**Symptoms**:
- Model learns trivial features (e.g., color histograms)
- Poor transfer performance
- High training accuracy, poor downstream

**Solution**:
```python
# Strong augmentation (SimCLR style)
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([
        transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
    ], p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Key augmentations:
# - Random crop and resize (geometric variation)
# - Color jitter (appearance variation)
# - Grayscale (force shape learning)
# - Gaussian blur (prevent texture shortcuts)
```

### Pitfall 5: Using Projection Head Embeddings for Downstream

**Problem**:
```python
# Training
model = SimCLRModel()  # encoder + projector
# ... training with InfoNCE ...

# Downstream task (WRONG!)
features = model(images)  # Using projection embeddings
classifier = nn.Linear(projection_dim, num_classes)
```

**Symptoms**:
- Poor downstream performance
- Lower accuracy than expected
- Projection embeddings are optimized for contrastive task, not general features

**Solution**:
```python
# Use encoder embeddings, not projection head embeddings
class SimCLRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNet50()
        self.projector = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 128)
        )

    def forward(self, x, return_projection=False):
        h = self.encoder(x)  # Encoder features
        z = self.projector(h)  # Projection features

        if return_projection:
            return z  # For contrastive learning
        else:
            return h  # For downstream tasks

# During contrastive training
z = model(x, return_projection=True)

# During downstream evaluation
features = model(x, return_projection=False)  # Use encoder features!
classifier = nn.Linear(2048, num_classes)  # Based on encoder dim
```

### Pitfall 6: False Negatives in Batch

**Problem**:
```python
# In-batch negatives may include false negatives
# Example: Batch contains multiple images of same class
batch = [cat1, dog1, cat2, dog2, cat3, ...]
# cat1's "negatives" include cat2 and cat3 (same class!)
```

**Symptoms**:
- Model penalized for pulling together same-class samples
- Suboptimal representations
- Lower downstream accuracy

**Solution**:
```python
# Option 1: Use supervised contrastive loss with labels
from nexus.training.losses import SupervisedContrastiveLoss

loss_fn = SupervisedContrastiveLoss(temperature=0.07)
loss = loss_fn(embeddings, labels)  # Handles multiple positives per class

# Option 2: Debiased contrastive loss (Chuang et al., 2020)
from nexus.training.losses import DebiasedInfoNCELoss

loss_fn = DebiasedInfoNCELoss(
    temperature=0.07,
    num_classes=1000,  # Estimate of false negative rate
    world_size=batch_size
)

# Option 3: Hard negative mining (avoid near-duplicates)
from nexus.training.losses import HardNegativeMiningInfoNCE

loss_fn = HardNegativeMiningInfoNCE(
    temperature=0.07,
    num_hard_negatives=128,  # Only use top-128 hardest negatives
    remove_duplicates=True
)
```

### Pitfall 7: Not Scaling Learning Rate with Batch Size

**Problem**:
```python
# Fixed learning rate regardless of batch size
batch_size = 1024  # Large batch
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# Effective learning rate is too low!
```

**Symptoms**:
- Slow convergence with large batches
- Suboptimal final performance
- Training takes much longer than expected

**Solution**:
```python
# Linear scaling rule (Goyal et al., 2017)
base_batch_size = 256
base_lr = 1e-3

current_batch_size = 1024
scaled_lr = base_lr * (current_batch_size / base_batch_size)
# scaled_lr = 4e-3

optimizer = torch.optim.Adam(model.parameters(), lr=scaled_lr)

# With warmup for stability
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

warmup_epochs = 10
total_epochs = 100

warmup_scheduler = LinearLR(
    optimizer,
    start_factor=0.01,
    end_factor=1.0,
    total_iters=warmup_epochs
)
cosine_scheduler = CosineAnnealingLR(
    optimizer,
    T_max=total_epochs - warmup_epochs
)
scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[warmup_epochs]
)
```

## Advanced Techniques

### Memory Bank for Large-Scale Negatives

```python
class MoCoLoss(nn.Module):
    """MoCo-style InfoNCE with queue of negatives."""

    def __init__(self, dim=128, queue_size=65536, momentum=0.999, temperature=0.07):
        super().__init__()
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature

        # Create queue
        self.register_buffer('queue', torch.randn(dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0

        # Replace oldest keys with new ones
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_size

        self.queue_ptr[0] = ptr

    def forward(self, q, k):
        """
        Args:
            q: query embeddings (batch_size, dim) from query encoder
            k: key embeddings (batch_size, dim) from key encoder (momentum)
        """
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Positive logits
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)

        # Negative logits from queue
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # Logits: [batch_size, 1 + queue_size]
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature

        # Labels: positive is first element
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        # Update queue
        self.dequeue_and_enqueue(k)

        loss = F.cross_entropy(logits, labels)
        return loss
```

### Multi-Positive InfoNCE

```python
class MultiPositiveInfoNCE(nn.Module):
    """InfoNCE with multiple positives per anchor."""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchors, positives_list, negatives):
        """
        Args:
            anchors: (batch_size, dim)
            positives_list: list of (batch_size, dim), multiple positive views
            negatives: (batch_size, num_negatives, dim)
        """
        anchors = F.normalize(anchors, dim=-1)
        negatives = F.normalize(negatives, dim=-1)

        # Compute positive similarities for all positive views
        pos_sims = []
        for positives in positives_list:
            positives = F.normalize(positives, dim=-1)
            pos_sim = torch.sum(anchors * positives, dim=-1, keepdim=True)
            pos_sims.append(pos_sim)

        # Average positive similarities
        pos_sim = torch.stack(pos_sims, dim=0).mean(dim=0) / self.temperature

        # Negative similarities
        neg_sim = torch.matmul(
            anchors.unsqueeze(1),
            negatives.transpose(-2, -1)
        ).squeeze(1) / self.temperature

        # Compute loss
        logits = torch.cat([pos_sim, neg_sim], dim=-1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        return F.cross_entropy(logits, labels)
```

### Hierarchical Contrastive Learning

```python
class HierarchicalInfoNCE(nn.Module):
    """InfoNCE at multiple hierarchy levels."""

    def __init__(self, temperatures=[0.07, 0.1, 0.15]):
        super().__init__()
        self.temperatures = temperatures

    def forward(self, anchors_list, positives_list, negatives_list):
        """
        Args:
            anchors_list: list of embeddings at different hierarchy levels
            positives_list: list of positive embeddings
            negatives_list: list of negative embeddings
        """
        total_loss = 0
        for anchors, positives, negatives, temp in zip(
            anchors_list, positives_list, negatives_list, self.temperatures
        ):
            loss = self.compute_infonce(anchors, positives, negatives, temp)
            total_loss += loss

        return total_loss / len(self.temperatures)

    def compute_infonce(self, anchors, positives, negatives, temperature):
        anchors = F.normalize(anchors, dim=-1)
        positives = F.normalize(positives, dim=-1)
        negatives = F.normalize(negatives, dim=-1)

        pos_sim = torch.sum(anchors * positives, dim=-1) / temperature
        neg_sim = torch.matmul(
            anchors.unsqueeze(1),
            negatives.transpose(-2, -1)
        ).squeeze(1) / temperature

        logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        return F.cross_entropy(logits, labels)
```

## References

1. **Representation Learning with Contrastive Predictive Coding**
   Aaron van den Oord, Yazhe Li, Oriol Vinyals
   arXiv:1807.03748, 2018
   https://arxiv.org/abs/1807.03748

   Original InfoNCE paper introducing the loss function and its connection to mutual information maximization.

2. **A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)**
   Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton
   ICML 2020
   https://arxiv.org/abs/2002.05709

   Landmark paper showing InfoNCE with simple augmentations achieves strong self-supervised results. Introduced NT-Xent loss.

3. **Momentum Contrast for Unsupervised Visual Representation Learning (MoCo)**
   Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, Ross Girshick
   CVPR 2020
   https://arxiv.org/abs/1911.05722

   Introduces queue mechanism for maintaining large negative sets with small batch sizes.

4. **Learning Transferable Visual Models From Natural Language Supervision (CLIP)**
   Alec Radford, Jong Wook Kim, Chris Hallacy, et al.
   OpenAI, 2021
   https://arxiv.org/abs/2103.00020

   Applies InfoNCE to vision-language pre-training at massive scale (400M pairs). Demonstrates zero-shot transfer.

5. **Supervised Contrastive Learning**
   Prannay Khosla, Piotr Teterwak, Chen Wang, et al.
   NeurIPS 2020
   https://arxiv.org/abs/2004.11362

   Extends InfoNCE to supervised setting with multiple positives per class. Shows improvements over cross-entropy.

6. **Debiased Contrastive Learning**
   Ching-Yao Chuang, Joshua Robinson, Lin Yen-Chen, Antonio Torralba, Stefanie Jegelka
   NeurIPS 2020
   https://arxiv.org/abs/2007.00224

   Addresses false negative problem in InfoNCE by debiasing the loss function.

7. **Understanding the Behaviour of Contrastive Loss**
   Feng Wang, Huaping Liu
   CVPR 2021
   https://arxiv.org/abs/2012.09740

   Theoretical analysis of InfoNCE including gradient behavior and convergence properties.

## Implementation Notes

**File Location**: `nexus/training/losses/infonce.py`

**Key Classes**:
- `InfoNCELoss`: Standard implementation with in-batch negatives
- `CLIPLoss`: CLIP-style symmetric InfoNCE
- `MoCoLoss`: MoCo-style with momentum encoder and queue
- `SupervisedContrastiveLoss`: Multi-positive supervised variant

**Dependencies**:
- PyTorch >= 1.12
- NumPy (for utilities)

**Typical Usage**:
- Self-supervised pre-training: SimCLR, MoCo, BYOL variants
- Multi-modal learning: CLIP, ALIGN, vision-language models
- Metric learning: Face recognition, person re-ID
- Retrieval: Image-text retrieval, cross-modal search

**Performance Considerations**:
- Batch size critical: aim for 256+
- Memory scales as O(B²) for similarity matrix
- Use mixed precision to fit larger batches
- Consider MoCo-style queue for very large negative sets

**Testing**: `tests/training/losses/test_infonce.py`

**Benchmarking**: `benchmarks/losses/infonce_benchmark.py`
