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

## Historical Evolution of SSL

Self-supervised learning has evolved rapidly over the past few years. Understanding this evolution helps contextualize current methods:

### Timeline

**2018-2019: Contrastive Era**
- SimCLR (2020): Large batch contrastive learning
- MoCo (2019): Momentum contrast with queue
- BYOL (2020): No negative pairs needed
- SwAV (2020): Clustering-based approach

**2020-2021: Masked Image Modeling**
- BEiT (2021): Visual tokenization + BERT
- MAE (2022): Simple pixel reconstruction
- SimMIM (2021): Direct feature prediction

**2022-2023: Joint-Embedding Predictive Architectures**
- I-JEPA (2023): Feature prediction in latent space
- data2vec (2022): Multimodal unified framework
- DINOv2 (2023): Self-distillation at scale

**2024: Video and Scaling**
- V-JEPA 2 (2024): Video world models
- Scaling to billions of images
- Zero-shot transfer to robotics

### Key Insights from Evolution

1. **Negative pairs not necessary**: BYOL, SimSiam, Barlow Twins showed collapse can be avoided without negatives
2. **Masking is powerful**: MAE showed masking + reconstruction works amazingly well
3. **Predict features, not pixels**: I-JEPA, data2vec2 show latent prediction > pixel prediction
4. **Scale matters**: DINOv2 demonstrated importance of data scale
5. **Unification**: data2vec showed one framework can work across modalities

## Detailed Method Comparison

### Architecture Comparison

| Method | Encoder | Decoder | Target Encoder | Special Components |
|--------|---------|---------|----------------|-------------------|
| MAE | ViT | Lightweight Transformer | ❌ | Asymmetric design |
| I-JEPA | ViT | Lightweight Predictor | ✅ (EMA) | Block masking |
| V-JEPA 2 | Video ViT | Predictor | ✅ (EMA) | Factorized space-time |
| DINOv2 | ViT | ❌ | ✅ (EMA) | Centering, sharpening |
| data2vec 2.0 | Modality-agnostic | Fast Conv | ✅ (EMA) | Multi-masking |
| Barlow Twins | ResNet/ViT | ❌ | ❌ | Correlation matrix |
| VICReg | ResNet/ViT | ❌ | ❌ | Explicit regularization |

### Training Requirements

| Method | Batch Size | Epochs | GPUs (typical) | Training Time (Base) |
|--------|------------|--------|----------------|---------------------|
| MAE | 4096 | 800 | 64 | 3 days |
| I-JEPA | 2048 | 600 | 32 | 4 days |
| V-JEPA 2 | 512-1024 | 400 | 16-32 | 7 days |
| DINOv2 | 1024 | 800 | 64+ | 5 days |
| data2vec 2.0 | 1024 | 400 | 32 | 2 days (2× faster) |
| Barlow Twins | 2048 | 1000 | 32 | 4 days |
| VICReg | 256-1024 | 1000 | 16-32 | 4 days |

### Computational Efficiency

| Method | Pre-training Speed | Memory Usage | Inference Speed |
|--------|-------------------|--------------|-----------------|
| MAE | Fast | Low | Fast |
| I-JEPA | Medium | Medium | Fast |
| V-JEPA 2 | Medium | High (video) | Medium |
| DINOv2 | Medium-Slow | High | Fast |
| data2vec 2.0 | Fast (2× v1) | Medium | Fast |
| Barlow Twins | Fast | Low | Fast |
| VICReg | Fast | Low | Fast |

### Downstream Performance Characteristics

| Method | Classification | Detection | Segmentation | Few-Shot | Zero-Shot |
|--------|----------------|-----------|--------------|----------|-----------|
| MAE | Good | Excellent | Excellent | Good | Poor |
| I-JEPA | Excellent | Excellent | Excellent | Excellent | Good |
| V-JEPA 2 | Good (video) | Good | Good | Good | Excellent (robotics) |
| DINOv2 | Excellent | Excellent | Excellent | Excellent | Excellent |
| data2vec 2.0 | Good | Good | Good | Good | Good (multimodal) |
| Barlow Twins | Good | Good | Good | Medium | Poor |
| VICReg | Good | Good | Good | Medium | Poor |

## Decision Trees for Method Selection

### Based on Data Type

```
What kind of data?
├─ Images (static)
│  ├─ Need best performance? → DINOv2
│  ├─ Want simplicity + speed? → MAE
│  ├─ No augmentation available? → I-JEPA
│  └─ Small batch size? → VICReg
├─ Video
│  ├─ Action recognition? → V-JEPA 2
│  ├─ World modeling? → V-JEPA 2
│  └─ Fast training? → VideoMAE
├─ Multimodal (vision + audio + text)
│  └─ → data2vec 2.0
└─ Any modality with limited resources
   └─ → data2vec 2.0 (efficient)
```

### Based on Downstream Task

```
What's your downstream task?
├─ Image Classification
│  ├─ ImageNet-scale? → DINOv2 or I-JEPA
│  ├─ Fast fine-tuning? → MAE
│  └─ Medical imaging? → I-JEPA (no augmentation)
├─ Object Detection
│  ├─ Best mAP? → MAE or I-JEPA
│  └─ Fast training? → MAE
├─ Semantic Segmentation
│  ├─ Dense prediction? → MAE
│  └─ Few-shot? → I-JEPA or DINOv2
├─ Video Understanding
│  └─ → V-JEPA 2
├─ Robotics
│  ├─ Visual control? → V-JEPA 2
│  └─ Manipulation? → I-JEPA or V-JEPA 2
└─ Zero-shot Transfer
   ├─ Vision? → DINOv2
   └─ Robotics? → V-JEPA 2
```

### Based on Computational Budget

```
What's your compute budget?
├─ Limited (< 8 GPUs)
│  ├─ Small batch OK? → VICReg or MAE
│  ├─ Need multimodal? → data2vec 2.0
│  └─ Video? → V-JEPA 2 (with small batches)
├─ Medium (8-32 GPUs)
│  ├─ Want simplicity? → MAE or Barlow Twins
│  ├─ Best performance? → I-JEPA or DINOv2
│  └─ Multimodal? → data2vec 2.0
└─ Large (64+ GPUs)
   ├─ State-of-the-art? → DINOv2
   ├─ Fast iteration? → MAE (800 epochs in 3 days)
   └─ Video at scale? → V-JEPA 2
```

## Advanced Training Strategies

### Curriculum Learning for SSL

Start with easier tasks, gradually increase difficulty:

```python
# Progressive masking (MAE/I-JEPA)
def get_mask_ratio(epoch, total_epochs):
    start_ratio = 0.5
    end_ratio = 0.75
    progress = epoch / total_epochs
    return start_ratio + (end_ratio - start_ratio) * progress

# Progressive resolution (all methods)
def get_image_size(epoch):
    if epoch < 100:
        return 112
    elif epoch < 400:
        return 224
    else:
        return 384  # High-res fine-tuning

# Progressive clip length (V-JEPA)
def get_num_frames(epoch):
    return min(8 + epoch // 50, 32)
```

### Multi-Stage Pre-training

Combine multiple SSL methods sequentially:

```python
# Stage 1: Fast pre-training with MAE (800 epochs)
model_mae = MAE(config)
train(model_mae, epochs=800)

# Stage 2: Refinement with I-JEPA (200 epochs)
model_ijepa = IJEPA(config)
model_ijepa.encoder.load_state_dict(model_mae.encoder.state_dict())
train(model_ijepa, epochs=200)

# Stage 3: Self-distillation with DINO (100 epochs)
model_dino = DINOv2(config)
model_dino.student.load_state_dict(model_ijepa.encoder.state_dict())
train(model_dino, epochs=100)

# Result: Best of all three methods!
```

### Cross-Modal Bootstrap

Use one modality to bootstrap another:

```python
# Pre-train on images (abundant data)
model_vision = Data2Vec(config, modality='vision')
train(model_vision, image_dataset, epochs=800)

# Transfer to audio (less data)
model_audio = Data2Vec(config, modality='audio')
model_audio.encoder.load_state_dict(
    model_vision.encoder.state_dict(),
    strict=False  # Allow size mismatch for input projection
)
train(model_audio, audio_dataset, epochs=200)

# Result: Better audio representations with less audio data!
```

## Debugging SSL Training

### Loss Patterns

**Healthy training**:
```
Epoch 0-10: Loss decreases rapidly (0.8 → 0.5)
Epoch 10-100: Steady decrease (0.5 → 0.3)
Epoch 100-800: Slow decrease (0.3 → 0.2)
```

**Problematic patterns**:
```
Flat loss: Check learning rate
Increasing loss: Reduce LR or check EMA momentum
Spiky loss: Increase batch size or add gradient clipping
Loss → 0 too fast: Representational collapse (check EMA/regularization)
```

### Monitoring Metrics

Beyond loss, track these:
```python
metrics = {
    # Feature statistics
    'feature_std': features.std(),  # Should be ~1.0
    'feature_mean': features.mean(),  # Should be ~0.0
    
    # For methods with EMA teacher
    'teacher_student_similarity': F.cosine_similarity(
        student_features, teacher_features
    ).mean(),  # Should be 0.8-0.95
    
    # For masked methods
    'mask_ratio_actual': mask.float().mean(),  # Should match config
    
    # Gradient statistics
    'grad_norm': torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf')),
    
    # For contrastive/distillation methods
    'temperature': model.temperature,  # Should be stable
}
```

### Common Warning Signs

1. **All features identical**: Collapse, check regularization
2. **Features have huge std**: Unstable, reduce LR or add normalization
3. **Grad norm > 10**: Add gradient clipping
4. **Teacher-student cosine sim > 0.99**: Teacher stuck, reduce EMA momentum
5. **Loss suddenly jumps**: Learning rate too high or batch norm issue

## Integration with Nexus

All SSL methods follow consistent Nexus patterns:

```python
from nexus.models.ssl import MAE, IJEPA, DINOv2, VICReg, BarlowTwins, Data2Vec, VJEPAModel

# Unified interface
models = {
    'mae': MAE(config),
    'ijepa': IJEPA(config),
    'dinov2': DINOv2(config),
    'vicreg': VICReg(config),
    'barlow_twins': BarlowTwins(config),
    'data2vec': Data2Vec(config),
    'vjepa': VJEPAModel(config)
}

# Same training loop works for all
for name, model in models.items():
    optimizer = AdamW(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        for batch in train_loader:
            loss, metrics = model(batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Log
            log_metrics(name, epoch, metrics)
    
    # Same evaluation
    features = model.encode(test_images)
    evaluate_downstream(features, test_labels)
```

### Configuration Inheritance

```python
# Base SSL config
base_config = {
    'img_size': 224,
    'patch_size': 16,
    'embed_dim': 768,
    'num_layers': 12,
    'num_heads': 12,
    'mlp_ratio': 4.0,
}

# Method-specific overrides
mae_config = {
    **base_config,
    'decoder_dim': 512,
    'decoder_layers': 8,
    'mask_ratio': 0.75,
}

ijepa_config = {
    **base_config,
    'predictor_dim': 384,
    'predictor_layers': 6,
    'ema_momentum': 0.996,
}

# Easy to experiment with different configs
```

## Extended Best Practices

### Data Loading Optimization

```python
# Optimal dataloader for SSL
train_loader = DataLoader(
    dataset,
    batch_size=256,
    num_workers=16,  # High parallelism
    pin_memory=True,  # Faster GPU transfer
    prefetch_factor=4,  # Prefetch multiple batches
    persistent_workers=True,  # Reuse workers
    drop_last=True,  # Consistent batch sizes
)

# For multi-GPU
sampler = torch.utils.data.distributed.DistributedSampler(dataset)
train_loader = DataLoader(
    dataset,
    batch_size=256 // world_size,
    sampler=sampler,
    num_workers=16 // world_size,
    pin_memory=True,
)
```

### Memory Optimization

```python
# Gradient checkpointing (trade compute for memory)
model.enable_gradient_checkpointing()

# Mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    loss = model(images)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# Gradient accumulation (simulate large batch)
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Hyperparameter Sweeps

```python
# Grid search for SSL
import itertools

mask_ratios = [0.5, 0.6, 0.75]
learning_rates = [1e-4, 1.5e-4, 2e-4]
weight_decays = [0.04, 0.05, 0.06]

for mask_ratio, lr, wd in itertools.product(mask_ratios, learning_rates, weight_decays):
    config = {
        'mask_ratio': mask_ratio,
        'learning_rate': lr,
        'weight_decay': wd,
    }
    
    model = MAE(config)
    train(model, epochs=400)  # Shorter for search
    acc = evaluate(model)
    
    print(f"mask_ratio={mask_ratio}, lr={lr}, wd={wd}: {acc:.2f}%")
```

## Future Directions

### Open Research Questions

1. **Optimal masking strategies**: Block vs random vs semantic
2. **Scaling laws**: How do SSL methods scale with model size and data?
3. **Multi-modal fusion**: Best ways to combine vision, audio, text
4. **Efficient architectures**: Lighter models with same performance
5. **Theoretical understanding**: Why does SSL work so well?

### Emerging Trends

1. **Video world models**: V-JEPA leading to model-based RL
2. **Multimodal foundation models**: Unified vision-language-audio
3. **Billion-scale pre-training**: Instagram-3.5B, JFT-3B datasets
4. **Zero-shot transfer**: Pre-trained models work without fine-tuning
5. **Efficient SSL**: Faster training, smaller models

### Recommended Reading

**Foundational Papers**:
1. "Momentum Contrast for Unsupervised Visual Representation Learning" (MoCo)
2. "Bootstrap Your Own Latent" (BYOL)
3. "Masked Autoencoders Are Scalable Vision Learners" (MAE)

**Advanced Topics**:
4. "A Path Towards Autonomous Machine Intelligence" (Yann LeCun's vision)
5. "Scaling Vision Transformers" (Scaling laws)
6. "Multimodal Learning with Transformers" (Survey)

**Implementation Guides**:
7. "Self-Supervised Learning in Practice" (Practical guide)
8. "Efficient Training of SSL Models" (Engineering practices)

## Extended Additional Resources

### Libraries and Frameworks

**General SSL**:
- [VISSL](https://github.com/facebookresearch/vissl): Meta's self-supervised learning library
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab's SSL toolkit  
- [Lightly](https://github.com/lightly-ai/lightly): SSL library with focus on practicality
- [solo-learn](https://github.com/vturrisi/solo-learn): Unified SSL codebase

**Model Hubs**:
- [Hugging Face](https://huggingface.co/models): Pre-trained SSL models
- [TIMM](https://github.com/rwightman/pytorch-image-models): PyTorch Image Models
- [TorchVision](https://pytorch.org/vision/stable/models.html): Standard pre-trained models

### Tutorials and Courses

**Beginner**:
- [Self-Supervised Learning Tutorial](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html)
- [Lil'Log: Contrastive Representation Learning](https://lilianweng.github.io/posts/2021-05-31-contrastive/)

**Intermediate**:
- [Stanford CS231n](http://cs231n.stanford.edu/): Computer Vision course
- [FastAI Deep Learning Course](https://course.fast.ai/): Practical deep learning

**Advanced**:
- [SSL Workshop at NeurIPS](https://sslneuips22.github.io/): Latest research
- [CVPR SSL Tutorial](https://github.com/facebookresearch/vissl/tree/main/tutorials): Detailed walkthroughs

### Benchmarks and Datasets

**Image Datasets**:
- [ImageNet-1K](https://www.image-net.org/): Standard benchmark
- [ImageNet-21K](https://www.image-net.org/download.php): Larger vocabulary
- [iNaturalist](https://github.com/visipedia/inat_comp): Fine-grained
- [Places365](http://places2.csail.mit.edu/): Scene understanding

**Video Datasets**:
- [Kinetics](https://www.deepmind.com/open-source/kinetics): Action recognition
- [Something-Something](https://developer.qualcomm.com/software/ai-datasets/something-something): Temporal reasoning
- [ActivityNet](http://activity-net.org/): Untrimmed videos

**Evaluation Benchmarks**:
- [VTAB](https://github.com/google-research/task_adaptation): Visual Task Adaptation
- [ELEVATER](https://github.com/Computer-Vision-in-the-Wild/Elevater_Toolkit_IC): Vision benchmarks
- [BigBench](https://github.com/google/BIG-bench): Large-scale benchmark

Happy learning! The field of self-supervised learning is rapidly evolving, and these methods represent the current state-of-the-art. Experiment with different approaches to find what works best for your specific use case.
