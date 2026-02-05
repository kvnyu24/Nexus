# DINOv2: Learning Robust Visual Features without Supervision

## Overview & Motivation

DINOv2 is a self-distillation method that learns robust visual features through a combination of self-supervised learning at scale. It trains on a curated dataset of 142M images and achieves state-of-the-art performance on many downstream tasks with zero-shot transfer or simple linear probing.

### Key Innovation

**Self-distillation with image-level and patch-level objectives**:
- Student: Predicts teacher's output (softmax distribution)
- Teacher: EMA of student, with centering and sharpening
- Multi-crop training: Global + local views
- Curated pre-training data (142M images)

## Mathematical Formulation

### Loss Function

Cross-entropy between student and teacher distributions:

```
L = -Σᵢ Pₜₑₐcₕₑᵣ(xᵢ) log Pₛₜᵤdₑₙₜ(xᵢ)
```

### Teacher Output Processing

```
# Student output
s = student(x) / τₛ
P_student = softmax(s)

# Teacher output with centering and sharpening
with no_grad():
    t = (teacher(x) - center) / τₜ
    P_teacher = softmax(t)
```

Where:
- τₛ = 0.1 (student temperature)
- τₜ = 0.04-0.07 (teacher temperature, sharpens distribution)
- center: Running mean of teacher outputs

### Centering

Prevents collapse by removing batch mean:

```
center ← m·center + (1-m)·mean(teacher_outputs)
output_centered = teacher_output - center
```

### Multi-Crop Training

Train on multiple views of different resolutions:
- 2 global views (224×224)
- 8 local views (96×96)

## Implementation Details

### Architecture

**Student & Teacher**: ViT-L/14 or ViT-g/14
- Large models (300M-1B parameters)
- Vision Transformer architecture
- Student updated via gradients
- Teacher updated via EMA

### Training Details

**Dataset**: 142M images (curated from web)
**Batch size**: 4096
**Training length**: ~150 epochs
**Optimizer**: AdamW
**Learning rate**: 1e-4 (with warmup)

### Code Outline

```python
# Pseudo-code for DINOv2 training

student = ViT(...)
teacher = ViT(...)  # EMA copy
teacher.requires_grad_(False)

center = torch.zeros(output_dim)

for images in dataloader:
    # Multi-crop augmentation
    global_views = [augment_global(img) for img in images]  # 2 views
    local_views = [augment_local(img) for img in images]    # 8 views
    views = global_views + local_views
    
    # Student forward on all views
    student_outputs = [student(view) / temp_student for view in views]
    student_probs = [softmax(out) for out in student_outputs]
    
    # Teacher forward on global views only
    with torch.no_grad():
        teacher_outputs = [teacher(view) for view in global_views]
        # Center and sharpen
        teacher_outputs = [(out - center) / temp_teacher 
                          for out in teacher_outputs]
        teacher_probs = [softmax(out) for out in teacher_outputs]
        
        # Update center
        center = m * center + (1 - m) * mean(teacher_outputs)
    
    # Cross-entropy loss
    loss = 0
    for s_prob in student_probs:
        for t_prob in teacher_probs:
            loss += cross_entropy(s_prob, t_prob)
    
    loss.backward()
    optimizer.step()
    
    # EMA update teacher
    for t_param, s_param in zip(teacher.parameters(), student.parameters()):
        t_param.data = tau * t_param.data + (1 - tau) * s_param.data
```

## Optimization Tricks

### 1. Temperature Scheduling

Teacher temperature increases over training:

```python
temp_teacher = 0.04 + (0.07 - 0.04) * (epoch / total_epochs)
temp_student = 0.1  # Fixed
```

### 2. Centering Momentum

```python
center_momentum = 0.9  # Typical value
```

### 3. EMA Momentum

```python
tau = 0.996  # Can use cosine schedule → 1.0
```

### 4. Multi-Crop Strategy

- Global crops: 224×224 (force global understanding)
- Local crops: 96×96 (force local feature learning)

## Experiments & Results

### ImageNet-1K Linear Probing

| Model | Params | Top-1 Acc |
|-------|--------|-----------|
| ViT-B/14 | 86M | 79.0% |
| ViT-L/14 | 304M | 82.1% |
| ViT-g/14 | 1B | **83.5%** |

State-of-the-art without labels!

### Zero-Shot Transfer

DINOv2 features work out-of-the-box:

| Task | Performance |
|------|-------------|
| k-NN ImageNet | 81.2% |
| Semantic Seg (ADE20K) | 47.2 mIoU |
| Depth Estimation | 0.077 RMSE |

No fine-tuning needed!

## Common Pitfalls

### 1. Collapse

**Problem**: All outputs become identical
**Solution**: Use centering + appropriate temperatures

### 2. Large Batch Requirements

**Problem**: DINOv2 needs large batches
**Solution**: Use batch size ≥ 1024 (distributed training)

### 3. Data Quality

**Problem**: Training on uncurated data gives worse results
**Solution**: Use curated, diverse dataset

## References

```bibtex
@article{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timoth{\'e}e and Moutakanni, Th{\'e}o and Vo, Huy V and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and others},
  journal={arXiv preprint arXiv:2304.07193},
  year={2023}
}
```

**Official Code**: https://github.com/facebookresearch/dinov2
**Nexus Implementation**: Would be in `nexus/models/ssl/dinov2.py` (placeholder for now)
