# Self-Supervised Learning Methods

This directory contains comprehensive documentation for self-supervised learning (SSL) algorithms implemented in Nexus. Self-supervised learning enables models to learn rich representations from unlabeled data by creating pretext tasks from the data itself, eliminating the need for manual annotations.

## Table of Contents

1. [MAE (Masked Autoencoder)](#mae)
2. [I-JEPA (Image Joint-Embedding Predictive Architecture)](#i-jepa)
3. [V-JEPA 2 (Video Joint-Embedding Predictive Architecture)](#v-jepa-2)
4. [DINOv2](#dinov2)
5. [data2vec 2.0](#data2vec-20)
6. [Barlow Twins](#barlow-twins)
7. [VICReg](#vicreg)

## Overview

Self-supervised learning has revolutionized representation learning by enabling models to learn from vast amounts of unlabeled data. Unlike supervised learning which requires expensive labeled datasets, SSL creates learning signals directly from the data structure itself.

### Key Paradigms

Self-supervised learning methods can be categorized into several paradigms:

#### 1. Masked Prediction
Learn by predicting masked portions of the input.
- **Pixel-level**: MAE (predict masked pixels)
- **Feature-level**: I-JEPA, V-JEPA 2, data2vec (predict masked representations)

#### 2. Contrastive Learning
Learn by pulling positive pairs together and pushing negative pairs apart.
- SimCLR, MoCo (not covered here, but foundational)

#### 3. Non-Contrastive Learning
Learn without negative pairs through redundancy reduction or asymmetric architectures.
- **Redundancy Reduction**: Barlow Twins, VICReg
- **Asymmetric**: BYOL, SimSiam (not covered)

#### 4. Self-Distillation
Learn by matching representations with a teacher model.
- DINOv2, data2vec 2.0

## Why Self-Supervised Learning?

### Advantages

1. **No Manual Labels Required**: Learn from raw, unlabeled data
2. **Scalability**: Can leverage internet-scale datasets
3. **Better Representations**: Often outperform supervised pre-training
4. **Transfer Learning**: Pre-trained models transfer well to downstream tasks
5. **Data Efficiency**: Downstream tasks require fewer labeled examples

### Common Applications

- **Computer Vision**: Image classification, object detection, segmentation
- **Video Understanding**: Action recognition, video prediction
- **Medical Imaging**: Where labeled data is scarce and expensive
- **Robotics**: Learning visual representations for control
- **Multi-Modal Learning**: Aligning vision and language

## Algorithm Landscape

### Reconstruction-Based Methods

#### MAE (Masked Autoencoder)
- **File**: [mae.md](mae.md)
- **Difficulty**: Beginner-Intermediate
- **Key Concepts**: Masked image modeling, asymmetric encoder-decoder
- **Training Speed**: Fast (processes only visible patches)
- **Use Case**: General-purpose vision pre-training

### Joint-Embedding Predictive Architectures

#### I-JEPA (Image)
- **File**: [ijepa.md](ijepa.md)
- **Difficulty**: Intermediate
- **Key Concepts**: Representation prediction, EMA target encoder
- **Training Speed**: Medium
- **Use Case**: Semantic image understanding, no augmentation needed

#### V-JEPA 2 (Video)
- **File**: [vjepa2.md](vjepa2.md)
- **Difficulty**: Intermediate-Advanced
- **Key Concepts**: Spatiotemporal prediction, temporal masking
- **Training Speed**: Medium (factorized attention)
- **Use Case**: Video understanding, world models, robotics

### Self-Distillation Methods

#### DINOv2
- **File**: [dinov2.md](dinov2.md)
- **Difficulty**: Advanced
- **Key Concepts**: Self-distillation, centering, sharpening
- **Training Speed**: Medium-Slow
- **Use Case**: Strong general-purpose features, zero-shot transfer

#### data2vec 2.0
- **File**: [data2vec.md](data2vec.md)
- **Difficulty**: Advanced
- **Key Concepts**: Multimodal learning, contextualized targets, inverse masking
- **Training Speed**: Fast (2x speedup over v1)
- **Use Case**: Unified framework for vision, speech, and text

### Redundancy Reduction Methods

#### Barlow Twins
- **File**: [barlow_twins.md](barlow_twins.md)
- **Difficulty**: Intermediate
- **Key Concepts**: Cross-correlation matrix, decorrelation
- **Training Speed**: Fast (no momentum encoder)
- **Use Case**: Simple, stable training without negative pairs

#### VICReg
- **File**: [vicreg.md](vicreg.md)
- **Difficulty**: Intermediate
- **Key Concepts**: Variance, invariance, covariance regularization
- **Training Speed**: Fast (no momentum encoder)
- **Use Case**: Small batch sizes, stable training

## Comparison Table

| Method | Paradigm | Augmentation | Momentum Encoder | Negative Pairs | Batch Size | Multimodal |
|--------|----------|--------------|------------------|----------------|------------|------------|
| MAE | Masked Prediction | ❌ | ❌ | ❌ | Medium | ❌ |
| I-JEPA | Masked Prediction | ❌ | ✅ | ❌ | Medium | ❌ |
| V-JEPA 2 | Masked Prediction | ❌ | ✅ | ❌ | Medium | ❌ |
| DINOv2 | Self-Distillation | ✅ | ✅ | ❌ | Large | ❌ |
| data2vec 2.0 | Self-Distillation | Varies | ✅ | ❌ | Medium | ✅ |
| Barlow Twins | Redundancy Reduction | ✅ | ❌ | ❌ | Large | ❌ |
| VICReg | Redundancy Reduction | ✅ | ❌ | ❌ | Small-Medium | ❌ |

## Performance Comparison

### ImageNet-1K Linear Probing (Top-1 Accuracy)

| Method | ViT-Base/16 | ViT-Large/16 | ViT-Huge/14 |
|--------|-------------|--------------|-------------|
| MAE | 67.8% | 75.5% | 76.6% |
| I-JEPA | 75.3% | 80.3% | - |
| DINOv2 | 79.0% | 82.1% | 83.5% |
| data2vec | 74.2% | 78.8% | - |

Note: Performance varies based on pre-training data scale and training epochs.

## When to Use Each Method

### Use MAE when:
- You want fast, simple pre-training
- Computational efficiency is important
- You're working with images
- You want a strong baseline

### Use I-JEPA when:
- You want semantic features without augmentation
- You don't want to predict pixels
- You need generalizable representations
- You're fine with moderate training cost

### Use V-JEPA 2 when:
- You're working with video data
- You need spatiotemporal understanding
- You're building world models
- Robotics or action prediction is your goal

### Use DINOv2 when:
- Maximum performance is needed
- You have large-scale data
- Zero-shot transfer is important
- You can afford training cost

### Use data2vec 2.0 when:
- You need multimodal learning
- You want a unified framework for vision/audio/text
- Training efficiency is important
- You need contextualized representations

### Use Barlow Twins when:
- You want simplicity
- You don't want momentum encoders
- Training stability is important
- You have standard augmentation pipelines

### Use VICReg when:
- You have small batch sizes
- You want explicit variance control
- Simple, stable training is needed
- You don't want momentum encoders

## Core Concepts

### Masking Strategies

1. **Random Masking** (MAE, data2vec)
   - Mask 60-75% of patches randomly
   - Simple and effective

2. **Block Masking** (I-JEPA, V-JEPA)
   - Mask contiguous blocks of patches
   - Encourages semantic understanding

3. **Temporal Masking** (V-JEPA 2)
   - Mask future frames
   - Enables prediction of dynamics

### Avoiding Collapse

SSL models must avoid "representational collapse" where all inputs map to the same output. Different methods use different strategies:

1. **Momentum Encoder** (I-JEPA, V-JEPA, DINOv2, data2vec)
   - Slowly-updated teacher provides stable targets
   - Prevents shortcuts

2. **Redundancy Reduction** (Barlow Twins, VICReg)
   - Explicitly decorrelate features
   - Maintain variance along dimensions

3. **Centering + Sharpening** (DINOv2)
   - Center teacher outputs
   - Sharpen with temperature

### Target Representations

1. **Pixel-Level** (MAE)
   - Reconstruct masked pixels
   - Normalized pixel values

2. **Feature-Level** (I-JEPA, V-JEPA, data2vec)
   - Predict latent representations
   - More semantic, less tied to low-level details

3. **Soft Targets** (DINOv2)
   - Teacher's probability distribution
   - Enables knowledge distillation

## Implementation Patterns

All SSL methods in Nexus follow consistent patterns:

### Basic Usage

```python
from nexus.models.ssl import MAE, JEPAModel, VICRegModel

# Configure model
config = {
    "img_size": 224,
    "patch_size": 16,
    "encoder_dim": 768,
    "mask_ratio": 0.75
}

# Initialize
model = MAE(config)

# Training loop
for images in dataloader:
    loss, reconstructed, mask = model(images)
    loss.backward()
    optimizer.step()

# Fine-tuning on downstream task
features = model.encode(images)  # Extract representations
```

### Multi-View Methods

```python
from nexus.models.ssl import VICRegModel, BarlowTwinsModel

# Multi-view augmentation
view1 = augment(images)
view2 = augment(images)

# Training
loss, metrics = model(view1, view2)
```

### Video Methods

```python
from nexus.models.ssl import VJEPAModel

# Video input (B, C, T, H, W)
video = torch.randn(4, 3, 16, 224, 224)

# Training
loss, metrics = model(video)
```

## Training Best Practices

### Data Augmentation

Different methods have different augmentation requirements:

1. **No Augmentation**: I-JEPA, V-JEPA, MAE
2. **Strong Augmentation**: DINOv2, Barlow Twins, VICReg
   - Random crops
   - Color jittering
   - Gaussian blur
   - Solarization

### Optimizer Settings

Most methods work well with:
- **Optimizer**: AdamW
- **Learning Rate**: 1e-4 to 1e-3 (with warmup)
- **Weight Decay**: 0.04 to 0.1
- **Batch Size**: 256-4096 (depending on method)

### Learning Rate Scheduling

```python
# Warmup + Cosine decay (standard)
warmup_epochs = 40
total_epochs = 800

lr_schedule = cosine_schedule_with_warmup(
    base_lr=1e-3,
    warmup_epochs=warmup_epochs,
    total_epochs=total_epochs
)
```

### EMA Updates

For methods with momentum encoders:
- Start momentum: 0.996
- Can use cosine schedule to increase to 1.0

```python
# Momentum update
momentum = 0.996
for target_param, online_param in zip(target_encoder.parameters(),
                                       online_encoder.parameters()):
    target_param.data = momentum * target_param.data + (1 - momentum) * online_param.data
```

## Evaluation Protocols

### Linear Probing

Freeze encoder, train linear classifier on top:

```python
# Freeze encoder
for param in model.encoder.parameters():
    param.requires_grad = False

# Train linear head
linear_head = nn.Linear(encoder_dim, num_classes)
# ... train on labeled data
```

### Fine-Tuning

Unfreeze and fine-tune entire model:

```python
# Unfreeze encoder
for param in model.encoder.parameters():
    param.requires_grad = True

# Fine-tune with lower learning rate
optimizer = AdamW(model.parameters(), lr=1e-5)
```

### k-NN Evaluation

Measure representation quality without training:

```python
# Extract features
train_features = model.encode(train_images)
test_features = model.encode(test_images)

# k-NN classification
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(train_features, train_labels)
accuracy = knn.score(test_features, test_labels)
```

## Common Pitfalls

### 1. Representational Collapse
**Symptoms**: Loss goes to zero, all embeddings identical
**Solutions**:
- Check EMA momentum (not too high)
- Verify variance regularization (VICReg)
- Add centering (DINOv2)

### 2. Training Instability
**Symptoms**: Loss spikes, NaN values
**Solutions**:
- Lower learning rate
- Add gradient clipping
- Check batch normalization
- Reduce mask ratio

### 3. Poor Downstream Performance
**Symptoms**: Low linear probing accuracy
**Solutions**:
- Train longer (SSL needs more epochs than supervised)
- Check masking strategy
- Verify target normalization
- Try different evaluation protocols

### 4. Memory Issues
**Symptoms**: OOM errors
**Solutions**:
- Use gradient checkpointing
- Process only visible patches (MAE, I-JEPA)
- Reduce batch size
- Use mixed precision training

### 5. Slow Training
**Symptoms**: Very slow iterations
**Solutions**:
- Use inverse masking (data2vec)
- Factorized attention (V-JEPA)
- Reduce decoder size (MAE)
- Optimize data loading

## Key Papers

### Foundational

1. **SimCLR**: [A Simple Framework for Contrastive Learning](https://arxiv.org/abs/2002.05709) (Chen et al., 2020)
2. **MoCo**: [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722) (He et al., 2020)
3. **BYOL**: [Bootstrap Your Own Latent](https://arxiv.org/abs/2006.07733) (Grill et al., 2020)

### Covered Methods

4. **MAE**: [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) (He et al., 2022)
5. **I-JEPA**: [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243) (Assran et al., 2023)
6. **V-JEPA**: [Revisiting Feature Prediction for Learning Visual Representations from Video](https://arxiv.org/abs/2404.08471) (Meta AI, 2024)
7. **DINOv2**: [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) (Oquab et al., 2023)
8. **data2vec 2.0**: [Efficient Self-supervised Learning with Contextualized Target Representations](https://arxiv.org/abs/2212.07525) (Baevski et al., 2022)
9. **Barlow Twins**: [Barlow Twins: Self-Supervised Learning via Redundancy Reduction](https://arxiv.org/abs/2103.03230) (Zbontar et al., 2021)
10. **VICReg**: [VICReg: Variance-Invariance-Covariance Regularization](https://arxiv.org/abs/2105.04906) (Bardes et al., 2022)

## Additional Resources

### Tutorials
- [Self-Supervised Learning Course (Hugging Face)](https://huggingface.co/blog/self-supervised-learning)
- [Lil'Log: Self-Supervised Learning](https://lilianweng.github.io/posts/2021-05-31-contrastive/)
- [MAE Explained](https://jalammar.github.io/illustrated-transformer/)

### Implementations
- [VISSL](https://github.com/facebookresearch/vissl): Meta's SSL library
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab SSL toolkit
- [Lightly](https://github.com/lightly-ai/lightly): Self-supervised learning library

### Benchmarks
- [VTAB](https://github.com/google-research/task_adaptation): Visual Task Adaptation Benchmark
- [ImageNet](https://www.image-net.org/): Standard pre-training dataset

## File Structure

```
12_self_supervised_learning/
├── README.md              # This file
├── mae.md                # Masked Autoencoder
├── ijepa.md              # Image JEPA
├── vjepa2.md             # Video JEPA 2
├── dinov2.md             # DINOv2
├── data2vec.md           # data2vec 2.0
├── barlow_twins.md       # Barlow Twins
└── vicreg.md             # VICReg
```

## Getting Started

### Recommended Learning Path

1. **Start with MAE** ([mae.md](mae.md))
   - Understand masked image modeling
   - Learn about asymmetric encoder-decoder
   - Implement your first SSL model

2. **Study I-JEPA** ([ijepa.md](ijepa.md))
   - Learn representation-level prediction
   - Understand EMA target encoders
   - See why predicting features > predicting pixels

3. **Explore Redundancy Reduction** ([vicreg.md](vicreg.md) or [barlow_twins.md](barlow_twins.md))
   - Understand non-contrastive learning
   - Learn about collapse prevention
   - See simple, stable training

4. **Deep Dive into Advanced Methods**
   - [dinov2.md](dinov2.md) for state-of-the-art features
   - [vjepa2.md](vjepa2.md) for video understanding
   - [data2vec.md](data2vec.md) for multimodal learning

### Quick Start Example

```python
import torch
from nexus.models.ssl import MAE

# Simple MAE pre-training
config = {
    "img_size": 224,
    "patch_size": 16,
    "encoder_dim": 768,
    "decoder_dim": 512,
    "mask_ratio": 0.75
}

model = MAE(config).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4)

# Training loop
for images in train_loader:
    images = images.cuda()

    # Forward pass
    loss, reconstructed, mask = model(images)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Loss: {loss.item():.4f}")

# Extract features for downstream tasks
with torch.no_grad():
    features = model.encode(test_images)
```

Each algorithm documentation includes:
- Detailed theory and motivation
- Mathematical formulations
- Step-by-step implementation guide
- Code walkthrough with explanations
- Optimization tricks and hyperparameters
- Experimental results and ablations
- Common pitfalls and debugging tips
- References to original papers

Happy learning!
