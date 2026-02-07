# Segmentation Models

Documentation for state-of-the-art segmentation models, focusing on promptable and foundation models.

## Contents

### Foundation Segmentation Models

- **[SAM](sam.md)** - Segment Anything Model: Universal image segmentation
- **[SAM 2](sam2.md)** - Segment Anything Model 2: Video and image segmentation
- **[MedSAM](medsam.md)** - Medical image segmentation with SAM

## Model Overview

| Model | Type | Prompts | Real-Time | Key Feature |
|-------|------|---------|-----------|-------------|
| SAM | Image | Points, Boxes, Masks | ✗ | Zero-shot segmentation |
| SAM 2 | Video + Image | Points, Boxes, Masks | ✓ | Temporal consistency |
| MedSAM | Medical Image | Boxes | ✗ | Medical imaging |

## Key Concepts

### Promptable Segmentation

Unlike traditional segmentation models that predict fixed categories, promptable models accept various input prompts:

**Prompt Types:**

1. **Point Prompts**: Click to indicate foreground/background
2. **Box Prompts**: Bounding box around object
3. **Mask Prompts**: Rough mask to refine
4. **Text Prompts**: Natural language description (in some variants)

### Architecture Pattern

```
Input Image
    ↓
Image Encoder (ViT-based)
    ↓
Image Embeddings (Cached)
    ↓
Prompt Encoder ← User Prompts (points, boxes, masks)
    ↓
Prompt Embeddings
    ↓
Mask Decoder (Lightweight Transformer)
    ↓
Masks + IoU Scores
```

## Model Selection Guide

### For General Image Segmentation

- **Interactive segmentation**: SAM
- **Automatic segmentation**: SAM with grid prompts
- **High-resolution**: SAM with ViT-H encoder

### For Video Segmentation

- **Object tracking**: SAM 2
- **Consistent masks**: SAM 2 with memory
- **Real-time video**: SAM 2 with small encoder

### For Medical Imaging

- **CT/MRI scans**: MedSAM
- **Large 3D volumes**: MedSAM with box prompts
- **Fine structures**: SAM with fine-tuning

### For Edge Deployment

- **Mobile**: SAM with MobileSAM encoder
- **Web**: SAM with ViT-B encoder
- **Server**: SAM with ViT-H for best quality

## Usage Patterns

### Interactive Segmentation

```python
from nexus.models.cv import SAM

# Load SAM
sam = SAM.from_pretrained("sam_vit_h")

# Encode image once
image_embedding = sam.image_encoder(image)

# Multiple prompts without re-encoding
while True:
    prompt = get_user_click()  # Point or box
    mask = sam.predict(
        image_embedding=image_embedding,
        prompt=prompt
    )
    display(mask)
```

### Automatic Segmentation

```python
# Generate masks for everything in image
sam_auto = SAMAutomaticMaskGenerator(sam)
masks = sam_auto.generate(image)

# Returns list of masks with scores
for mask in masks:
    display(mask["segmentation"])
    print(f"Confidence: {mask['stability_score']}")
```

### Video Tracking

```python
from nexus.models.cv import SAM2

sam2 = SAM2.from_pretrained("sam2_hiera_large")

# Initialize with first frame
first_frame_mask = sam2.predict(
    image=video[0],
    prompt=initial_box
)

# Propagate through video
for frame in video[1:]:
    mask = sam2.track(
        image=frame,
        prev_mask=prev_mask,
        memory=sam2.memory
    )
    prev_mask = mask
```

## Training & Fine-Tuning

### SAM Fine-Tuning

```python
# Fine-tune decoder only (efficient)
for param in sam.image_encoder.parameters():
    param.requires_grad = False

optimizer = AdamW(
    sam.mask_decoder.parameters(),
    lr=1e-4,
    weight_decay=0.01
)

# Train on domain-specific data
for images, prompts, masks in dataloader:
    pred_masks = sam(images, prompts)
    loss = dice_loss(pred_masks, masks) + focal_loss(pred_masks, masks)
    loss.backward()
    optimizer.step()
```

### Data Augmentation

```python
# Augmentation for promptable segmentation
transforms = [
    RandomFlip(prob=0.5),
    RandomRotate(degrees=15),
    RandomScale(scale_range=(0.8, 1.2)),
    ColorJitter(brightness=0.2, contrast=0.2),
    # Augment prompts along with image
    AugmentPrompts(),  # Transforms boxes/points with image
]
```

## Performance Metrics

### Segmentation Quality

- **IoU (Intersection over Union)**: Primary metric
- **Boundary F-score**: Edge accuracy
- **Stability Score**: Prediction confidence

### Efficiency Metrics

- **Encoder Time**: One-time cost per image
- **Decoder Time**: Per-prompt cost (should be fast)
- **Memory**: Image embedding storage

## Optimization Tricks

### 1. Cache Image Embeddings

```python
# Encode once, use many times
image_embedding = sam.image_encoder(image)  # Slow
for prompt in prompts:
    mask = sam.predict_with_embedding(image_embedding, prompt)  # Fast
```

### 2. Batch Prompts

```python
# Process multiple prompts in parallel
prompts = [point1, point2, box1, box2]
masks = sam.predict_batch(image_embedding, prompts)
```

### 3. Quantization

```python
# INT8 quantization for deployment
from torch.quantization import quantize_dynamic
sam_int8 = quantize_dynamic(sam, {nn.Linear}, dtype=torch.qint8)
```

### 4. ONNX Export

```python
# Export for cross-platform deployment
torch.onnx.export(
    sam.mask_decoder,
    (image_embedding, prompt_embedding),
    "sam_decoder.onnx",
    opset_version=17
)
```

## Benchmarks

### SAM Zero-Shot Performance

| Dataset | IoU | Boundary F | Description |
|---------|-----|------------|-------------|
| COCO | 46.5 | - | General objects |
| LVIS | 44.7 | - | Long-tail objects |
| ADE20K | 42.3 | - | Scene parsing |
| Cityscapes | 38.1 | - | Street scenes |

### SAM 2 Video Performance

| Dataset | J&F | Description |
|---------|-----|-------------|
| DAVIS 2017 | 76.2 | Video object segmentation |
| YouTube-VOS | 72.8 | YouTube videos |

### MedSAM Performance

| Modality | DSC | Description |
|----------|-----|-------------|
| CT | 85.3 | Computed tomography |
| MRI | 82.7 | Magnetic resonance |
| Microscopy | 79.4 | Cell images |

## Common Use Cases

### Interactive Annotation

- **Manual labeling**: Click to refine masks
- **Data generation**: Create training data
- **Quality control**: Verify/correct predictions

### Automatic Analysis

- **Cell counting**: Segment and count cells
- **Damage assessment**: Identify damaged regions
- **Object removal**: Generate masks for inpainting

### Video Understanding

- **Object tracking**: Follow objects in video
- **Action recognition**: Segment actors/objects
- **Video editing**: Remove/replace objects

## Implementation Resources

- **SAM**: `Nexus/nexus/models/cv/sam.py`
- **SAM 2**: `Nexus/nexus/models/cv/sam2.py`
- **MedSAM**: `Nexus/nexus/models/cv/medsam.py`

## References

See individual model documentation for detailed papers and citations.
