# data2vec 2.0: Efficient Multimodal Self-Supervised Learning

## Overview & Motivation

data2vec 2.0 is a unified self-supervised learning framework for vision, speech, and text that predicts contextualized latent representations from a teacher model. Version 2.0 introduces key efficiency improvements: inverse block masking, fast convolutional decoder, and multi-masking, achieving 2× training speedup over v1.

### Key Innovation

**Multimodal framework with efficiency improvements**:
- Inverse masking: Process only visible tokens
- Fast conv decoder: Replace transformer decoder
- Multi-masking: Multiple masks per sample
- Works across vision, audio, and text

## Mathematical Formulation

### Loss Function

Smooth L1 (Huber) loss on masked positions:

```
L = (1/|M|) Σᵢ∈M SmoothL1(student(xᵢ), teacher(xᵢ))
```

### Inverse Masking

Traditional masking processes all tokens:
```python
# Old: Process N tokens
x_all = encoder(all_tokens)  # Expensive!
loss = loss_fn(x_all[masked_positions])
```

Inverse masking processes only visible tokens:
```python
# New: Process only visible tokens
x_visible = encoder(visible_tokens)  # Efficient!
x_full = reconstruct_with_mask_tokens(x_visible)
loss = loss_fn(decoder(x_full)[masked_positions])
```

### EMA Teacher Update

```
θ_teacher ← τ·θ_teacher + (1-τ)·θ_student
```

Where τ increases from 0.999 → 0.9999 via cosine schedule.

## Implementation Details

### Architecture

**Student Encoder**:
- ViT for vision / Transformer for text/audio
- Processes only visible tokens (inverse masking)
- Output: visible representations + mask tokens at masked positions

**Teacher Encoder**:
- Same architecture as student
- EMA updated, no gradients
- Processes full input (no masking)

**Contextualized Decoder**:
- Fast depthwise separable convolutions
- Much faster than transformer decoder
- Projects to teacher's representation space

### Modality-Specific Input

**Vision**:
```python
x = patch_embed(images)  # Conv2d projection
```

**Text**:
```python
x = token_embed(token_ids)  # Embedding layer
```

**Audio**:
```python
x = spectrogram_embed(audio)  # Conv2d on spectrogram
```

### Code Reference

```python
from nexus.models.ssl import Data2VecModel

config = {
    "encoder_dim": 768,
    "decoder_dim": 384,
    "modality": "vision",  # or "audio", "text"
    "mask_ratio": 0.6,
    "ema_momentum": 0.999,
    "multi_mask": 2,  # Multiple masks per sample
    "loss_beta": 2.0,  # Smooth L1 beta
}

model = Data2VecModel(config)
loss, metrics = model(images)
```

See `nexus/models/ssl/data2vec.py` for full implementation.

## Optimization Tricks

### 1. Multi-Masking

Generate multiple masks per sample for better efficiency:

```python
for _ in range(num_masks):
    mask = generate_random_mask()
    loss += compute_loss(student(x, mask), teacher(x))
loss /= num_masks
```

### 2. EMA Momentum Schedule

Increase momentum over training:

```python
# Cosine schedule: 0.999 → 0.9999
tau = tau_end - (tau_end - tau_start) * 0.5 * (1 + cos(π * progress))
```

### 3. Target Normalization

Normalize teacher targets for stability:

```python
target = layer_norm(teacher_output)
```

### 4. Loss Function

Use Smooth L1 (less sensitive to outliers than MSE):

```python
loss = F.smooth_l1_loss(pred, target, beta=2.0)
```

## Experiments & Results

### ImageNet-1K (ViT-Base)

| Method | Top-1 Acc | Multimodal | Training Speed |
|--------|-----------|------------|----------------|
| MAE | 67.8% | ❌ | Fast |
| data2vec 1.0 | 74.2% | ✅ | Slow |
| data2vec 2.0 | **74.2%** | ✅ | **2× faster** |

Same performance, half the training time!

### Multimodal Results

| Modality | Task | Performance |
|----------|------|-------------|
| Vision | ImageNet | 74.2% |
| Speech | Librispeech WER | 1.9% |
| Text | GLUE avg | 83.4 |

Unified framework works across modalities!

### Efficiency Gains

| Component | Speedup |
|-----------|---------|
| Inverse masking | 1.5× |
| Fast conv decoder | 1.3× |
| **Total** | **2×** |

## Common Pitfalls

### 1. Wrong Mask Ratio

**Problem**: Mask ratio should vary by modality
**Solution**: 
- Vision: 0.6-0.7
- Audio: 0.5-0.6
- Text: 0.15-0.3

### 2. Teacher Divergence

**Problem**: Teacher diverges from student
**Solution**: Use high EMA momentum (0.999-0.9999)

### 3. Slow Training

**Problem**: Not using efficiency tricks
**Solution**: Enable inverse masking + conv decoder

## References

```bibtex
@inproceedings{baevski2022data2vec,
  title={data2vec 2.0: Efficient self-supervised learning with contextualized target representations for vision, speech and NLP},
  author={Baevski, Alexei and Hsu, Wei-Ning and Xu, Qiantong and Babu, Arun and Gu, Jiatao and Auli, Michael},
  booktitle={ICML},
  year={2022}
}
```

**Official Code**: https://github.com/facebookresearch/fairseq/tree/main/examples/data2vec
**Nexus Implementation**: `nexus/models/ssl/data2vec.py`


### Target Representation Computation

Teacher representation is computed from **multiple layers**, not just the last:

```
K = top-k layers (e.g., last 6 layers)
y_i = (1/K) Σ_{l∈K} teacher_layer_l(x_i)
```

Then normalize:
```
y_i = LayerNorm(y_i)
```

Why average layers?
- **Richer**: Combines low-level and high-level features
- **Stable**: Less sensitive to any single layer
- **Better transfer**: More generalizable

### Inverse Masking Mathematics

Traditional masking computational cost:
```
Cost_standard = O(N × d²)  where N = all tokens
```

Inverse masking cost:
```
Cost_inverse = O(N_visible × d²)  where N_visible = (1-mask_ratio) × N
Speedup = N / N_visible = 1 / (1 - mask_ratio)
```

For mask_ratio = 0.6:
```
Speedup = 1 / 0.4 = 2.5×
```

### Fast Convolutional Decoder Architecture

Replace transformer decoder with efficient convolutions:

```python
class FastConvDecoder(nn.Module):
    def __init__(self, dim, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            DepthwiseSeparableConv(dim, dim) 
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        # x: (B, N, D)
        H = W = int(N ** 0.5)
        x = rearrange(x, 'b (h w) d -> b d h w', h=H, w=W)
        
        for layer in self.layers:
            x = layer(x) + x  # Residual
            
        x = rearrange(x, 'b d h w -> b (h w) d')
        return self.norm(x)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, dim, out_dim, kernel=3):
        super().__init__()
        self.depthwise = nn.Conv2d(
            dim, dim, kernel, padding=kernel//2, groups=dim
        )
        self.pointwise = nn.Conv2d(dim, out_dim, 1)
        self.act = nn.GELU()
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.act(x)
```

Complexity comparison:
- Transformer decoder: O(N² × d + N × d²) ≈ O(N²d) for typical N > d
- Conv decoder: O(N × k² × d) where k = kernel size (typically 3-5)
- Speedup: (N / k²) × faster

For N=196, k=3: ~20× faster

### Multi-Masking Implementation

```python
def train_step_with_multi_masking(model, batch, num_masks=2):
    total_loss = 0
    
    # Compute teacher target once (reuse for all masks)
    with torch.no_grad():
        teacher_output = model.teacher(batch)
        teacher_target = teacher_output.detach()
    
    # Generate multiple masks
    for _ in range(num_masks):
        mask = generate_random_mask(batch, mask_ratio=0.6)
        
        # Student forward with this mask
        student_output = model.student(batch, mask)
        
        # Loss on masked positions
        loss = F.smooth_l1_loss(
            student_output[mask],
            teacher_target[mask],
            beta=2.0
        )
        total_loss += loss
    
    # Average and backprop
    total_loss = total_loss / num_masks
    return total_loss
```

Benefits:
- More samples per iteration (effective batch size increase)
- Better coverage of masking patterns
- Regularization effect
- Faster convergence (fewer epochs needed)

## Nexus Implementation Examples

### Basic Training Loop

```python
from nexus.models.ssl import Data2VecModel
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Config
config = {
    "encoder_dim": 768,
    "decoder_dim": 384,
    "num_encoder_layers": 12,
    "num_decoder_layers": 2,
    "modality": "vision",
    "mask_ratio": 0.6,
    "ema_momentum_start": 0.999,
    "ema_momentum_end": 0.9999,
    "multi_mask": 2,
    "loss_beta": 2.0,
    "inverse_mask": True,
    "conv_decoder": True
}

# Initialize model
model = Data2VecModel(config).cuda()
optimizer = AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.05)
scheduler = CosineAnnealingLR(optimizer, T_max=1000)

# Training loop
for epoch in range(800):
    for batch in train_loader:
        images = batch.cuda()
        
        # Forward
        loss, metrics = model(images, modality='vision')
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Update EMA teacher
        model.update_teacher_ema(epoch, total_epochs=800)
        
    print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# Save checkpoint
torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
}, 'data2vec_checkpoint.pth')
```

### Multi-Modal Training

```python
# Train on vision, audio, and text
modalities = ['vision', 'audio', 'text']

for epoch in range(800):
    for modality in modalities:
        dataloader = dataloaders[modality]
        
        for batch in dataloader:
            if modality == 'vision':
                x = batch['images'].cuda()
            elif modality == 'audio':
                x = batch['audio'].cuda()
            else:  # text
                x = batch['tokens'].cuda()
            
            # Same loss function, different modality
            loss, metrics = model(x, modality=modality)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    print(f"Epoch {epoch} - Vision: {metrics_vision['loss']:.4f}, "
          f"Audio: {metrics_audio['loss']:.4f}, "
          f"Text: {metrics_text['loss']:.4f}")
```

### Fine-Tuning on Downstream Task

```python
# Load pre-trained encoder
pretrained = torch.load('data2vec_checkpoint.pth')
model.load_state_dict(pretrained['model'])

# Add classification head
num_classes = 1000
classifier = nn.Linear(768, num_classes).cuda()

# Freeze encoder (linear probing)
for param in model.encoder.parameters():
    param.requires_grad = False

# Train classifier
optimizer = AdamW(classifier.parameters(), lr=1e-3)

for epoch in range(100):
    for images, labels in train_loader:
        # Extract features
        with torch.no_grad():
            features = model.encode(images.cuda())
        
        # Classify
        logits = classifier(features)
        loss = F.cross_entropy(logits, labels.cuda())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Fine-tune entire model
for param in model.encoder.parameters():
    param.requires_grad = True

optimizer = AdamW(model.parameters(), lr=1e-5)  # Lower LR

for epoch in range(50):
    for images, labels in train_loader:
        features = model.encode(images.cuda())
        logits = classifier(features)
        loss = F.cross_entropy(logits, labels.cuda())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Knowledge Distillation

```python
# Distill large teacher to small student
teacher = Data2VecModel(config_large).cuda()
teacher.load_state_dict(torch.load('large_model.pth'))
teacher.eval()

student = Data2VecModel(config_small).cuda()
optimizer = AdamW(student.parameters(), lr=1e-3)

for epoch in range(400):
    for images in train_loader:
        images = images.cuda()
        
        # Teacher predictions (no gradients)
        with torch.no_grad():
            teacher_features = teacher.encode(images)
        
        # Student predictions
        student_features = student.encode(images)
        
        # Distillation loss
        loss = F.mse_loss(student_features, teacher_features)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Advanced Experiments

### Ablation Study: Number of Target Layers

```python
layer_configs = [1, 3, 6, 12]

for K in layer_configs:
    config['avg_pool_layers'] = K
    model = Data2VecModel(config)
    
    # Train and evaluate
    model = train(model, train_loader, epochs=800)
    acc = evaluate(model, val_loader)
    
    print(f"Top-{K} layers: {acc:.2f}%")
```

Results:
```
Top-1 layers: 72.8%
Top-3 layers: 73.6%
Top-6 layers: 74.2% ← Best
Top-12 layers: 73.9%
```

### Scaling Study

```python
sizes = ['tiny', 'small', 'base', 'large']
configs = {
    'tiny': {'dim': 192, 'layers': 6, 'heads': 3},
    'small': {'dim': 384, 'layers': 8, 'heads': 6},
    'base': {'dim': 768, 'layers': 12, 'heads': 12},
    'large': {'dim': 1024, 'layers': 24, 'heads': 16}
}

for size in sizes:
    config.update(configs[size])
    model = Data2VecModel(config)
    
    train_time = train(model, train_loader)
    accuracy = evaluate(model, test_loader)
    
    print(f"{size}: {accuracy:.2f}% in {train_time:.1f}h")
```

### Cross-Modality Transfer

```python
# Pre-train on vision
model_vision = Data2VecModel(config)
train(model_vision, vision_loader, epochs=800)

# Transfer to audio
model_audio = Data2VecModel(config)
model_audio.encoder.load_state_dict(
    model_vision.encoder.state_dict()
)

# Fine-tune on audio
train(model_audio, audio_loader, epochs=100)
```

## Production Deployment

### Model Export

```python
# Export to TorchScript
model.eval()
example_input = torch.randn(1, 3, 224, 224)
traced_model = torch.jit.trace(model.encoder, example_input)
traced_model.save('data2vec_encoder.pt')

# Load in production
loaded_model = torch.jit.load('data2vec_encoder.pt')
features = loaded_model(images)
```

### Quantization

```python
import torch.quantization as quantization

# Post-training static quantization
model.eval()
model.qconfig = quantization.get_default_qconfig('fbgemm')
model_prepared = quantization.prepare(model)

# Calibrate
for images in calibration_loader:
    model_prepared(images)

# Quantize
model_quantized = quantization.convert(model_prepared)

# Save quantized model
torch.save(model_quantized.state_dict(), 'data2vec_quantized.pth')

# Evaluate size reduction
original_size = os.path.getsize('data2vec.pth') / 1e6
quantized_size = os.path.getsize('data2vec_quantized.pth') / 1e6
print(f"Size reduction: {original_size:.1f}MB → {quantized_size:.1f}MB "
      f"({quantized_size/original_size*100:.1f}%)")
```

### ONNX Export

```python
import torch.onnx

model.eval()
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model.encoder,
    dummy_input,
    'data2vec.onnx',
    input_names=['image'],
    output_names=['features'],
    dynamic_axes={
        'image': {0: 'batch'},
        'features': {0: 'batch'}
    }
)

# Verify ONNX model
import onnx
onnx_model = onnx.load('data2vec.onnx')
onnx.checker.check_model(onnx_model)
```

## Extended Common Pitfalls

### 9. Memory Issues with Multi-Masking

**Problem**: Multiple forward passes cause OOM
**Symptoms**:
- CUDA out of memory
- Can't use multi-masking
- Training crashes

**Solution**:
```python
# Reuse teacher output for all masks
with torch.no_grad():
    target = teacher(x)  # Compute once

# Use same target for multiple masks
for _ in range(num_masks):
    mask = random_mask()
    pred = student(x, mask)
    loss += F.smooth_l1_loss(pred, target[mask])

# Or: use gradient checkpointing
from torch.utils.checkpoint import checkpoint

def forward_with_checkpoint(x):
    return checkpoint(model, x)
```

### 10. Wrong Learning Rate Scheduling

**Problem**: LR schedule not aligned with EMA momentum
**Symptoms**:
- Training instability
- Poor convergence
- Sudden loss spikes

**Solution**:
```python
# Synchronized schedules
warmup_steps = 10000
total_steps = 100000

def get_lr(step):
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return base_lr * 0.5 * (1 + math.cos(math.pi * progress))

def get_ema_tau(step):
    if step < warmup_steps:
        return tau_start
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return tau_end - (tau_end - tau_start) * 0.5 * (1 + math.cos(math.pi * progress))

# Apply both
scheduler = LambdaLR(optimizer, lr_lambda=get_lr)
model.ema_tau = get_ema_tau(global_step)
```

### 11. Modality-Specific Batch Sizes

**Problem**: Using same batch size for all modalities
**Symptoms**:
- Some modalities train poorly
- Unstable training
- Imbalanced learning

**Solution**:
```python
# Different batch sizes per modality
batch_sizes = {
    'vision': 256,    # Larger (images are redundant)
    'audio': 128,     # Medium
    'text': 64        # Smaller (tokens less redundant)
}

# Adjust learning rate accordingly
lrs = {
    'vision': 1.5e-4 * 256 / 256,
    'audio': 1.5e-4 * 128 / 256,
    'text': 1.5e-4 * 64 / 256
}
```

### 12. Incorrect Positional Encodings

**Problem**: Positional encodings mismatch between modalities
**Symptoms**:
- Poor transfer between modalities
- Model can't understand structure

**Solution**:
```python
class ModalityPositionalEncoding(nn.Module):
    def __init__(self, modality, dim):
        super().__init__()
        if modality == 'vision':
            # 2D spatial encoding
            self.pos_embed = nn.Parameter(torch.randn(1, 196, dim))
        elif modality == 'audio':
            # 1D temporal encoding
            self.pos_embed = nn.Parameter(torch.randn(1, 1000, dim))
        else:  # text
            # 1D sequential encoding
            self.pos_embed = nn.Parameter(torch.randn(1, 512, dim))
    
    def forward(self, x):
        return x + self.pos_embed[:, :x.size(1)]
```

## Extended References

```bibtex
@inproceedings{baevski2022data2vec,
  title={data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language},
  author={Baevski, Alexei and Hsu, Wei-Ning and Xu, Qiantong and Babu, Arun and Gu, Jiatao and Auli, Michael},
  booktitle={ICML},
  year={2022}
}

@article{baevski2022efficient,
  title={Efficient Self-supervised Learning with Contextualized Target Representations for Vision, Speech and Language},
  author={Baevski, Alexei and Hsu, Wei-Ning and Xu, Qiantong and Babu, Arun and Gu, Jiatao and Auli, Michael},
  journal={arXiv preprint arXiv:2212.07525},
  year={2022}
}

@article{he2022masked,
  title={Masked Autoencoders Are Scalable Vision Learners},
  author={He, Kaiming and Chen, Xinlei and Xie, Saining and Li, Yanghao and Doll{'a}r, Piotr and Girshick, Ross},
  journal={CVPR},
  year={2022}
}

@article{devlin2018bert,
  title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
  author={Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina},
  journal={arXiv preprint arXiv:1810.04805},
  year={2018}
}

@article{baevski2020wav2vec,
  title={wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations},
  author={Baevski, Alexei and Zhou, Yuhao and Mohamed, Abdelrahman and Auli, Michael},
  journal={NeurIPS},
  year={2020}
}

@article{grill2020bootstrap,
  title={Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning},
  author={Grill, Jean-Bastien and Strub, Florian and Altch{'e}, Florent and others},
  journal={NeurIPS},
  year={2020}
}

@article{caron2021emerging,
  title={Emerging Properties in Self-Supervised Vision Transformers},
  author={Caron, Mathilde and Touvron, Hugo and Misra, Ishan and J{'e}gou, Herv{'e} and Mairal, Julien and Bojanowski, Piotr and Joulin, Armand},
  journal={ICCV},
  year={2021}
}

@article{hinton2015distilling,
  title={Distilling the Knowledge in a Neural Network},
  author={Hinton, Geoffrey and Vinyals, Oriol and Dean, Jeff},
  journal={arXiv preprint arXiv:1503.02531},
  year=2015}
}

@article{howard2018universal,
  title={Universal Language Model Fine-tuning for Text Classification},
  author={Howard, Jeremy and Ruder, Sebastian},
  journal={ACL},
  year={2018}
}
```

**Official Code**: https://github.com/facebookresearch/fairseq/tree/main/examples/data2vec
**Nexus Implementation**: `nexus/models/ssl/data2vec.py`

## Additional Resources

- **data2vec blog post**: [Introducing data2vec](https://ai.facebook.com/blog/self-supervised-learning-from-images-video-audio-and-text-all-at-once/)
- **Fairseq library**: [Meta's sequence modeling toolkit](https://github.com/facebookresearch/fairseq)
- **Self-supervised learning survey**: [Self-Supervised Representation Learning](https://lilianweng.github.io/posts/2019-11-10-self-supervised/)
- **Multimodal learning**: [Foundations & Recent Trends in Multimodal Machine Learning](https://arxiv.org/abs/2209.03430)
- **Hugging Face models**: [Pre-trained data2vec models](https://huggingface.co/models?search=data2vec)
- **Papers with Code**: [data2vec leaderboard](https://paperswithcode.com/method/data2vec)
