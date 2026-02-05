# Vision Transformers

This directory contains comprehensive documentation for Vision Transformer architectures and their variants.

## Contents

### Core Architectures

- **[ViT](vit.md)** - Vision Transformer: The original pure transformer for image classification
- **[Swin Transformer](swin_transformer.md)** - Hierarchical vision transformer with shifted windows
- **[HiViT](hivit.md)** - Hierarchical Vision Transformer with multi-scale processing

### CNN Architectures

- **[ResNet/VGG](resnet_vgg.md)** - Classic convolutional architectures
- **[EfficientNet](efficientnet.md)** - Compound scaling and mobile-optimized CNNs

### Self-Supervised & Foundation Models

- **[DINOv2](dinov2.md)** - Self-supervised visual features without labels
- **[SigLIP](siglip.md)** - Sigmoid loss for language-image pre-training
- **[EVA-02](eva02.md)** - Enhanced Vision-language pre-training
- **[InternVL](internvl.md)** - Intern Vision-Language model

## Quick Comparison

| Model | Type | Key Feature | Best For |
|-------|------|-------------|----------|
| ViT | Pure Transformer | Patch-based attention | Large-scale pre-training |
| Swin | Hierarchical ViT | Shifted windows | Dense prediction tasks |
| HiViT | Multi-scale ViT | Hierarchical features | Multi-resolution analysis |
| ResNet | CNN | Residual connections | Baseline comparisons |
| EfficientNet | CNN | Compound scaling | Mobile/edge deployment |
| DINOv2 | Self-supervised ViT | No labels needed | Transfer learning |
| SigLIP | Vision-Language | Sigmoid loss | Image-text matching |
| EVA-02 | Vision-Language | MIM + CLIP | Multimodal tasks |
| InternVL | Vision-Language | Cross-modal fusion | VQA, captioning |

## Model Selection Guide

### For Image Classification

- **Small datasets**: DINOv2 → fine-tune
- **Large datasets**: ViT-B/16 or Swin-B
- **Mobile/Edge**: EfficientNet-B0 to B3

### For Dense Prediction (Detection, Segmentation)

- **Best accuracy**: Swin-L or EVA-02-L
- **Balanced**: Swin-B or HiViT-B
- **Fast inference**: ResNet-50 + FPN

### For Transfer Learning

- **General vision**: DINOv2-g or EVA-02-L
- **Vision-language**: SigLIP-L or InternVL
- **Few-shot**: DINOv2 with linear probing

### For Multi-modal Applications

- **Image-text retrieval**: SigLIP-L
- **Visual question answering**: InternVL
- **Zero-shot classification**: EVA-02-CLIP

## Implementation Notes

All models follow the Nexus framework patterns:

- Inherit from NexusModule
- Use unified configuration dictionaries
- Support feature extraction and fine-tuning
- Include weight initialization utilities

See individual model documentation for detailed references and paper citations.
