# MAE: Masked Autoencoder

## Overview & Motivation

Masked Autoencoders (MAE) is a simple yet powerful self-supervised learning approach that learns visual representations by masking random patches of images and reconstructing the missing pixels. Inspired by masked language modeling in NLP (BERT), MAE demonstrates that a high mask ratio (75%) combined with an asymmetric encoder-decoder architecture can learn excellent visual representations.

### Key Innovation

**Asymmetric encoder-decoder design**: 
- Large encoder processes only visible patches (computational efficiency)
- Lightweight decoder reconstructs full image from encoded visible patches + mask tokens
- Mask ratio of 75% makes pre-training fast and effective

## Theoretical Background

### Learning Paradigm: Masked Reconstruction

MAE learns by solving a denoising autoencoding task in pixel space:

```
minimize ||reconstruct(encode(mask(x))) - x||²
```

### Why High Mask Ratio Works

1. **Information Asymmetry**: Masking 75% creates a difficult task requiring semantic understanding
2. **Computational Efficiency**: Encoder processes only 25% of patches
3. **Redundancy Removal**: Forces model to learn high-level representations, not just copy pixels
4. **Better Generalization**: Prevents overfitting to low-level statistics

### Autoencoding vs MAE

Traditional autoencoding compresses all information. MAE is different:
- **Random sampling**: Removes random patches (not compression)
- **High mask ratio**: 75% vs typical 15-30%
- **Asymmetric**: Encoder much larger than decoder
- **Normalized pixels**: Reconstruct normalized patch statistics

## Mathematical Formulation

### Loss Function

Mean squared error on masked patches only:

```
L = (1/|M|) Σ_{i∈M} ||x_i - x̂_i||²
```

Where:
- M: Set of masked patch indices
- x_i: Original pixel values for patch i (normalized)
- x̂_i: Reconstructed pixel values

### Pixel Normalization

Normalize each patch independently:

```
x_norm = (x - μ_patch) / √(σ²_patch + ε)
```

This prevents shortcuts based on overall image statistics.

### Encoder Forward Pass

```
1. Patchify image: x → patches (B, N, P²C)
2. Linear projection: patches → embeddings (B, N, D)
3. Add positional embedding: embeddings + pos_embed
4. Remove masked patches: embeddings[:, visible_mask]
5. Prepend CLS token
6. Transform: ViT encoder
7. Output: encoded_visible (B, N_vis+1, D)
```

### Decoder Forward Pass

```
1. Project encoder output to decoder dim
2. Add mask tokens at masked positions
3. Add positional embeddings for all positions
4. Transform: lightweight transformer decoder
5. Remove CLS token
6. Linear head to pixel space: (B, N, P²C)
7. Reshape to patches: (B, N, patch_dim)
```

## High-Level Intuition

Think of MAE like a jigsaw puzzle solver:

1. **Masking**: Someone removes 75% of puzzle pieces
2. **Encoder**: You look at the remaining 25% and understand the scene
3. **Decoder**: You imagine what the missing pieces look like
4. **Learning**: You get better by comparing your imagined pieces to the real ones

**Key insight**: With only 25% of pieces, you can't just memorize pixel patterns. You must understand the semantic content (e.g., "this is a forest scene, so the missing parts probably have trees and leaves").

## Implementation Details

### Network Architecture

**Encoder (ViT-Base)**:
- Patch size: 16×16
- Embedding dim: 768
- Layers: 12
- Attention heads: 12
- Processes only visible patches (25%)
- Has CLS token

**Decoder (Lightweight)**:
- Embedding dim: 512 (smaller than encoder)
- Layers: 8 (fewer than encoder)
- Attention heads: 16
- Processes all patches (visible + masked)

### Masking Implementation

```python
def random_masking(x, mask_ratio=0.75):
    """
    Random masking by per-sample shuffling.
    
    Args:
        x: Input sequence (B, N, D)
        mask_ratio: Fraction of patches to mask
        
    Returns:
        x_masked: Visible patches (B, N*(1-mask_ratio), D)
        mask: Binary mask (B, N), 1 is keep, 0 is remove
        ids_restore: Indices to restore original order
    """
    B, N, D = x.shape
    len_keep = int(N * (1 - mask_ratio))
    
    # Random permutation for each sample
    noise = torch.rand(B, N, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    
    # Keep visible patches
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    
    # Generate binary mask: 1 is keep, 0 is remove
    mask = torch.ones([B, N], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    
    return x_masked, mask, ids_restore
```

### Code Reference

See `nexus/models/ssl/mae.py` for full implementation:

```python
from nexus.models.ssl import MAE

config = {
    "img_size": 224,
    "patch_size": 16,
    "encoder_dim": 768,
    "decoder_dim": 512,
    "encoder_layers": 12,
    "decoder_layers": 8,
    "mask_ratio": 0.75,
    "norm_pix_loss": True
}

model = MAE(config)
loss, reconstructed, mask = model(images)
```

## Optimization Tricks

### 1. Normalized Pixel Loss

Normalize targets by patch-wise mean and variance:

```python
if norm_pix_loss:
    mean = target.mean(dim=-1, keepdim=True)
    var = target.var(dim=-1, keepdim=True)
    target = (target - mean) / (var + 1e-6).sqrt()
```

This improves representation quality by preventing shortcuts.

### 2. Learning Rate Scaling

Scale learning rate with batch size:

```python
base_lr = 1.5e-4
actual_lr = base_lr * batch_size / 256
```

### 3. Warmup Schedule

```python
warmup_epochs = 40
# Linear warmup then cosine decay
```

### 4. Weight Decay

High weight decay works well:

```python
weight_decay = 0.05  # Higher than typical supervised learning
```

### 5. Mixed Precision Training

MAE benefits greatly from mixed precision:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    loss, _, _ = model(images)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## Experiments & Results

### ImageNet-1K Pre-training

| Model | Encoder Params | Pre-train Epochs | Linear Probe Acc | Fine-tune Acc |
|-------|----------------|------------------|------------------|---------------|
| ViT-B/16 | 86M | 800 | 67.8% | 83.6% |
| ViT-L/16 | 304M | 800 | 75.5% | 85.9% |
| ViT-H/14 | 632M | 1600 | 76.6% | 86.9% |

### Ablation Studies

**Mask Ratio**:
| Mask Ratio | Linear Probe Acc |
|------------|------------------|
| 15% | 61.2% |
| 50% | 65.7% |
| 75% | **67.8%** |
| 90% | 64.3% |

**Decoder Depth**:
| Decoder Layers | Linear Probe Acc | Training Time |
|----------------|------------------|---------------|
| 1 | 65.2% | Fast |
| 4 | 66.9% | Medium |
| 8 | **67.8%** | Slow |

## Common Pitfalls

### 1. Low Mask Ratio
**Problem**: Mask ratio < 50% leads to poor representations
**Solution**: Use 75% mask ratio (optimal in most cases)

### 2. Decoder Too Large
**Problem**: Large decoder makes training slow without much benefit
**Solution**: Use lightweight decoder (8 layers, 512 dim for ViT-B)

### 3. No Pixel Normalization
**Problem**: Model learns to predict mean color without understanding semantics
**Solution**: Enable `norm_pix_loss=True`

### 4. Evaluating on Training Task
**Problem**: Reconstruction quality ≠ representation quality
**Solution**: Evaluate via linear probing or fine-tuning on downstream tasks

### 5. Wrong Masking Order
**Problem**: Masking after positional encoding leaks information
**Solution**: Mask before adding positional embeddings

## References

```bibtex
@article{he2022masked,
  title={Masked autoencoders are scalable vision learners},
  author={He, Kaiming and Chen, Xinlei and Xie, Saining and Li, Yanghao and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={CVPR},
  year={2022}
}
```

**Official Code**: https://github.com/facebookresearch/mae
**Nexus Implementation**: `nexus/models/ssl/mae.py`

### Reconstruction Target Details

Per-patch normalization improves representations significantly:

```python
def compute_normalized_targets(images, patches, patch_size=16):
    # Compute per-patch statistics
    mean = patches.mean(dim=-1, keepdim=True)  # (B, N, 1)
    var = patches.var(dim=-1, keepdim=True)    # (B, N, 1)
    
    # Normalize
    normalized_patches = (patches - mean) / (var + 1e-6).sqrt()
    
    return normalized_patches, mean, var
```

This prevents shortcuts where the model just predicts the mean color without understanding content.

## Extended Implementation Examples

### Complete MAE Training Pipeline

```python
from nexus.models.ssl import MAE
import torch
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR

# Configuration
config = {
    "img_size": 224,
    "patch_size": 16,
    "encoder_dim": 768,
    "decoder_dim": 512,
    "encoder_layers": 12,
    "decoder_layers": 8,
    "encoder_heads": 12,
    "decoder_heads": 16,
    "mask_ratio": 0.75,
    "norm_pix_loss": True
}

# Initialize
model = MAE(config).cuda()

# Optimizer with weight decay
optimizer = AdamW(
    model.parameters(),
    lr=1.5e-4 * 4096 / 256,  # Scale with batch size
    betas=(0.9, 0.95),
    weight_decay=0.05
)

# Cosine scheduler with warmup
scheduler = CosineAnnealingLR(optimizer, T_max=800, eta_min=1e-6)

# Mixed precision scaler
scaler = GradScaler()

# Training loop
global_step = 0
for epoch in range(800):
    model.train()
    epoch_loss = 0
    
    for batch_idx, images in enumerate(train_loader):
        images = images.cuda()
        
        # Forward with mixed precision
        with autocast():
            loss, pred, mask = model(images)
        
        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += loss.item()
        global_step += 1
        
        # Logging
        if global_step % 100 == 0:
            print(f"Epoch {epoch}, Step {global_step}: Loss = {loss.item():.4f}")
    
    scheduler.step()
    
    # Checkpoint saving
    if epoch % 100 == 0:
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'loss': epoch_loss / len(train_loader)
        }, f'checkpoints/mae_epoch{epoch}.pth')

print("Training completed!")
```

### Visualization Tools

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_mae_predictions(model, dataloader, num_samples=4):
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    
    with torch.no_grad():
        images = next(iter(dataloader))[:num_samples].cuda()
        loss, pred, mask = model(images)
        
        # Unpatchify predictions
        pred_images = model.unpatchify(pred)
        mask_expanded = mask.unsqueeze(-1).repeat(1, 1, 16*16*3)
        mask_expanded = model.unpatchify(mask_expanded)
        
        for i in range(num_samples):
            # Original
            orig = images[i].cpu().permute(1, 2, 0).numpy()
            orig = (orig - orig.min()) / (orig.max() - orig.min())
            axes[i, 0].imshow(orig)
            axes[i, 0].set_title('Original')
            axes[i, 0].axis('off')
            
            # Masked
            masked = orig.copy()
            mask_img = mask_expanded[i].cpu().permute(1, 2, 0).numpy()
            masked[mask_img[:,:,0] > 0.5] = 0.7
            axes[i, 1].imshow(masked)
            axes[i, 1].set_title('Masked (75%)')
            axes[i, 1].axis('off')
            
            # Reconstruction
            recon = pred_images[i].cpu().permute(1, 2, 0).numpy()
            recon = (recon - recon.min()) / (recon.max() - recon.min())
            axes[i, 2].imshow(recon)
            axes[i, 2].set_title('Reconstruction')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('mae_visualizations.png', dpi=150)
    plt.show()

# Use it
visualize_mae_predictions(model, val_loader)
```

### Linear Probing Script

```python
def evaluate_linear_probe(encoder, train_loader, val_loader, num_classes=1000):
    # Freeze encoder
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()
    
    # Linear classifier
    classifier = nn.Linear(768, num_classes).cuda()
    optimizer = AdamW(classifier.parameters(), lr=0.001, weight_decay=0.0)
    
    best_acc = 0
    for epoch in range(90):
        # Training
        classifier.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.cuda(), labels.cuda()
            
            with torch.no_grad():
                features = encoder(images)[:, 0]  # CLS token
            
            logits = classifier(features)
            loss = F.cross_entropy(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        classifier.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.cuda(), labels.cuda()
                features = encoder(images)[:, 0]
                logits = classifier(features)
                pred = logits.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        
        acc = 100.0 * correct / total
        if acc > best_acc:
            best_acc = acc
            torch.save(classifier.state_dict(), 'best_linear_probe.pth')
        
        print(f"Epoch {epoch}: Train Loss = {train_loss/len(train_loader):.4f}, "
              f"Val Acc = {acc:.2f}%")
    
    return best_acc

# Evaluate
acc = evaluate_linear_probe(model.encoder, train_loader, val_loader)
print(f"Best Linear Probe Accuracy: {acc:.2f}%")
```

### Full Fine-Tuning Script

```python
def fine_tune_mae(encoder, train_loader, val_loader, num_classes=1000):
    # Classification head
    classifier = nn.Sequential(
        nn.LayerNorm(768),
        nn.Dropout(0.1),
        nn.Linear(768, num_classes)
    ).cuda()
    
    # Unfreeze all
    for param in encoder.parameters():
        param.requires_grad = True
    
    # Layer-wise learning rate decay
    def get_layer_wise_lr(layer_id, base_lr=5e-5, num_layers=12, decay=0.65):
        return base_lr * (decay ** (num_layers - layer_id))
    
    param_groups = []
    for i, layer in enumerate(encoder.blocks):
        param_groups.append({
            'params': layer.parameters(),
            'lr': get_layer_wise_lr(i)
        })
    param_groups.append({'params': classifier.parameters(), 'lr': 1e-3})
    
    optimizer = AdamW(param_groups, weight_decay=0.05)
    scheduler = CosineAnnealingLR(optimizer, T_max=50)
    
    best_acc = 0
    for epoch in range(50):
        # Train
        encoder.train()
        classifier.train()
        for images, labels in train_loader:
            images, labels = images.cuda(), labels.cuda()
            
            features = encoder(images)[:, 0]
            logits = classifier(features)
            loss = F.cross_entropy(logits, labels)
            
            # Label smoothing
            loss = loss * 0.9 + 0.1 * (-F.log_softmax(logits, dim=1).mean())
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimizer.step()
        
        scheduler.step()
        
        # Validate
        encoder.eval()
        classifier.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.cuda(), labels.cuda()
                features = encoder(images)[:, 0]
                logits = classifier(features)
                pred = logits.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        
        acc = 100.0 * correct / total
        best_acc = max(best_acc, acc)
        print(f"Epoch {epoch}: Val Acc = {acc:.2f}%, Best = {best_acc:.2f}%")
    
    return best_acc

acc = fine_tune_mae(model.encoder, train_loader, val_loader)
print(f"Fine-tuning Accuracy: {acc:.2f}%")
```

## Extended Ablation Studies

### Effect of Encoder-Decoder Size Ratio

```python
configs = [
    {'encoder_dim': 768, 'decoder_dim': 192, 'ratio': 4.0},
    {'encoder_dim': 768, 'decoder_dim': 384, 'ratio': 2.0},
    {'encoder_dim': 768, 'decoder_dim': 512, 'ratio': 1.5},
    {'encoder_dim': 768, 'decoder_dim': 768, 'ratio': 1.0},
]

for cfg in configs:
    model = MAE(cfg).cuda()
    train(model, epochs=400)
    acc = evaluate(model)
    print(f"Ratio {cfg['ratio']}: {acc:.2f}%")
```

### Effect of Normalized Pixel Loss

```python
for norm_pix_loss in [False, True]:
    config['norm_pix_loss'] = norm_pix_loss
    model = MAE(config).cuda()
    train(model, epochs=400)
    acc = evaluate(model)
    print(f"Norm pix loss={norm_pix_loss}: {acc:.2f}%")

# Results: True is better (67.8% vs 65.2%)
```

### Effect of Training Length

```python
epochs_list = [100, 200, 400, 800, 1600]

for epochs in epochs_list:
    model = MAE(config).cuda()
    train(model, epochs=epochs)
    acc = evaluate(model)
    print(f"Epochs {epochs}: {acc:.2f}%")

# Results: 800 epochs is optimal
```

## Transfer Learning Experiments

### COCO Object Detection

```python
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer

# Use MAE encoder as backbone
cfg = get_cfg()
cfg.MODEL.BACKBONE.NAME = "build_mae_vit_backbone"
cfg.MODEL.WEIGHTS = "mae_pretrained.pth"
cfg.DATASETS.TRAIN = ("coco_2017_train",)
cfg.DATASETS.TEST = ("coco_2017_val",)

# Train Mask R-CNN
trainer = DefaultTrainer(cfg)
trainer.train()

# Results
# MAE pre-trained: 49.2 AP
# Supervised: 47.6 AP
# Improvement: +1.6 AP
```

### ADE20K Semantic Segmentation

```python
class SegmentationHead(nn.Module):
    def __init__(self, in_dim=768, num_classes=150):
        super().__init__()
        self.fpn = nn.ModuleList([
            nn.ConvTranspose2d(in_dim, 512, 2, stride=2),
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ConvTranspose2d(128, num_classes, 2, stride=2),
        ])
    
    def forward(self, x):
        # x: (B, 14, 14, 768)
        x = x.permute(0, 3, 1, 2)
        for layer in self.fpn:
            x = layer(x)
            if layer != self.fpn[-1]:
                x = F.relu(x)
        return x  # (B, num_classes, 224, 224)

# Train segmentation
encoder = model.encoder
seg_head = SegmentationHead().cuda()
train_segmentation(encoder, seg_head, ade20k_loader)

# Results
# MAE pre-trained: 48.1 mIoU
# Supervised: 45.8 mIoU
# Improvement: +2.3 mIoU
```

## Additional Common Pitfalls

### 6. Incorrect Positional Embedding Handling

**Problem**: Adding positional embeddings before masking leaks position information
**Solution**:
```python
# WRONG
x = patch_embed(img) + pos_embed  # Add first
x_masked, mask = random_masking(x)  # Then mask - LEAKS INFO!

# CORRECT
x = patch_embed(img)  # Embed
x_masked, mask, ids_restore = random_masking(x)  # Mask first
x_masked = x_masked + pos_embed[:, :len_keep]  # Then add pos embed
```

### 7. Wrong Decoder Size

**Problem**: Decoder too large wastes computation, too small underfits
**Solution**:
```python
# Optimal decoder for ViT-Base
decoder_config = {
    'dim': 512,     # 2/3 of encoder (768)
    'depth': 8,     # 2/3 of encoder (12)
    'heads': 16,    # More heads ok (cheap)
}
```

### 8. Batch Size Too Small

**Problem**: MAE works best with large batches
**Solution**:
```python
# Target effective batch size: 4096
batch_per_gpu = 64
num_gpus = 8
grad_accum = 8
effective_batch = 64 * 8 * 8 = 4096

# Scale learning rate
lr = 1.5e-4 * effective_batch / 256
```

### 9. Using Strong Augmentation

**Problem**: MAE doesn't need (and is hurt by) strong augmentation
**Solution**:
```python
# MAE augmentation (minimal)
transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# NO ColorJitter, NO RandAugment, NO Mixup!
```

### 10. Stopping Training Too Early

**Problem**: MAE needs 800 epochs
**Solution**:
```python
# Full training schedule
total_epochs = 800
warmup_epochs = 40

# Don't stop at 100 or 200 epochs!
# Performance keeps improving up to 800
```

## Production Deployment

### Model Export

```python
# Export encoder only (no decoder needed for inference)
encoder = model.encoder
encoder.eval()

# TorchScript
example = torch.randn(1, 3, 224, 224)
traced = torch.jit.trace(encoder, example)
traced.save('mae_encoder.pt')

# ONNX
torch.onnx.export(
    encoder, example, 'mae_encoder.onnx',
    input_names=['image'],
    output_names=['features'],
    dynamic_axes={'image': {0: 'batch'}}
)
```

### Quantization

```python
# Post-training quantization
encoder_fp32 = model.encoder
encoder_fp32.eval()

# Prepare for quantization
encoder_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
encoder_prepared = torch.quantization.prepare(encoder_fp32)

# Calibrate
with torch.no_grad():
    for images, _ in calibration_loader:
        encoder_prepared(images)

# Convert
encoder_int8 = torch.quantization.convert(encoder_prepared)

# Save
torch.save(encoder_int8.state_dict(), 'mae_encoder_int8.pth')

# Size reduction: ~4x smaller, minimal accuracy loss
```

## Extended References

```bibtex
@article{he2022masked,
  title={Masked Autoencoders Are Scalable Vision Learners},
  author={He, Kaiming and Chen, Xinlei and Xie, Saining and Li, Yanghao and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={CVPR},
  year={2022}
}

@article{dosovitskiy2020vit,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and others},
  journal={ICLR},
  year={2021}
}

@article{bao2021beit,
  title={BEiT: BERT Pre-Training of Image Transformers},
  author={Bao, Hangbo and Dong, Li and Wei, Furu},
  journal={ICLR},
  year={2022}
}

@article{xie2022simmim,
  title={SimMIM: A Simple Framework for Masked Image Modeling},
  author={Xie, Zhenda and Zhang, Zheng and Cao, Yue and others},
  journal={CVPR},
  year={2022}
}

@article{feichtenhofer2022masked,
  title={Masked Autoencoders As Spatiotemporal Learners},
  author={Feichtenhofer, Christoph and Li, Yanghao and He, Kaiming and others},
  journal={NeurIPS},
  year={2022}
}

@article{chen2021empirical,
  title={An Empirical Study of Training Self-Supervised Vision Transformers},
  author={Chen, Xinlei and Xie, Saining and He, Kaiming},
  journal={ICCV},
  year={2021}
}

@article{tong2022videomae,
  title={VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training},
  author={Tong, Zhan and Song, Yibing and Wang, Jue and Wang, Limin},
  journal={NeurIPS},
  year={2022}
}

@article{goyal2021self,
  title={Self-supervised Pretraining of Visual Features in the Wild},
  author={Goyal, Priya and Caron, Mathilde and Lefaudeux, Benjamin and others},
  journal={arXiv preprint arXiv:2103.01988},
  year={2021}
}
```

**Official Code**: https://github.com/facebookresearch/mae
**Nexus Implementation**: `nexus/models/ssl/mae.py`

## Additional Resources

- **MAE Blog**: [Masked Autoencoders](https://ai.facebook.com/blog/masked-autoencoders/)
- **Demo Notebook**: [Interactive MAE](https://github.com/facebookresearch/mae/blob/main/DEMO.ipynb)
- **Hugging Face**: [Pre-trained models](https://huggingface.co/models?search=mae)
- **Papers with Code**: [Leaderboard](https://paperswithcode.com/method/mae)
- **TIMM Library**: [PyTorch implementations](https://github.com/rwightman/pytorch-image-models)
