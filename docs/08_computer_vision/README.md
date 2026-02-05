# Computer Vision Models

Comprehensive documentation for computer vision models implemented in Nexus, covering vision transformers, object detection, segmentation, and 3D reconstruction.

## Directory Structure

### [Vision Transformers](vision_transformers/)

State-of-the-art transformer architectures for image understanding:

- **ViT** - Original Vision Transformer
- **Swin Transformer** - Hierarchical windows
- **HiViT** - Multi-scale hierarchical processing
- **DINOv2** - Self-supervised foundation model
- **SigLIP** - Sigmoid loss vision-language
- **EVA-02** - Enhanced vision-language
- **InternVL** - Cross-modal fusion
- **EfficientNet** - Compound scaling CNNs
- **ResNet/VGG** - Classic CNN baselines

### [Object Detection](object_detection/)

Modern object detection systems from R-CNN to YOLO:

- **DETR** - Transformer-based detection
- **RT-DETR** - Real-time DETR
- **Faster R-CNN** - Two-stage with RPN
- **Cascade R-CNN** - Progressive refinement
- **Mask R-CNN** - Instance segmentation
- **Keypoint R-CNN** - Pose estimation
- **Grounding DINO** - Open-vocabulary detection
- **YOLO-World** - Open-vocabulary YOLO
- **YOLOv10** - Latest YOLO with NMS-free training

### [Segmentation](segmentation/)

Foundation models for promptable segmentation:

- **SAM** - Segment Anything Model
- **SAM 2** - Video segmentation
- **MedSAM** - Medical imaging

### [NeRF & 3D](nerf_3d/)

Neural radiance fields and 3D reconstruction (see separate docs)

## Quick Start

### Image Classification

```python
from nexus.models.cv import VisionTransformer, DINOv2

# Vision Transformer
vit = VisionTransformer({
    "image_size": 224,
    "patch_size": 16,
    "num_classes": 1000,
    "embed_dim": 768,
    "num_layers": 12,
    "num_heads": 12,
})

# Or use pre-trained DINOv2
dinov2 = DINOv2.from_pretrained("dinov2_vitb14")
features = dinov2(images)
```

### Object Detection

```python
from nexus.models.cv.rcnn import FasterRCNN
from nexus.models.cv import RTDETR

# Faster R-CNN (two-stage)
detector = FasterRCNN({
    "in_channels": 3,
    "num_classes": 80,
    "backbone": "resnet50",
})

# RT-DETR (real-time transformer)
rtdetr = RTDETR({
    "num_classes": 80,
    "backbone": "resnet50",
})

outputs = detector(images)
boxes = outputs["boxes"]
scores = outputs["scores"]
```

### Segmentation

```python
from nexus.models.cv import SAM

# Load SAM
sam = SAM({
    "encoder_embed_dim": 768,
    "encoder_depth": 12,
    "encoder_num_heads": 12,
})

# Encode image once
embedding = sam.image_encoder(image)

# Multiple prompts
for prompt in prompts:
    mask = sam.predict(
        image_embedding=embedding,
        prompt=prompt  # points, boxes, or masks
    )
```

## Model Selection Matrix

### By Task

| Task | Recommended Models | Alternatives |
|------|-------------------|--------------|
| Image Classification | DINOv2, ViT, Swin | EfficientNet, ResNet |
| Object Detection | RT-DETR, YOLOv10 | Faster R-CNN, DETR |
| Instance Segmentation | Mask R-CNN | SAM + detection |
| Semantic Segmentation | SAM, SAM 2 | - |
| Pose Estimation | Keypoint R-CNN | - |
| Open-Vocabulary Detection | Grounding DINO | YOLO-World |
| Video Segmentation | SAM 2 | - |

### By Constraints

| Constraint | Best Choice | Notes |
|------------|-------------|-------|
| Real-time inference | YOLOv10, RT-DETR | >30 FPS |
| High accuracy | Swin-L, Cascade R-CNN | Slower but better |
| Mobile deployment | EfficientNet | Optimized for edge |
| Zero-shot capability | DINOv2, Grounding DINO | No fine-tuning needed |
| Few-shot learning | DINOv2 | Excellent transfer |
| Interactive annotation | SAM | Promptable masks |

### By Data Availability

| Data Amount | Recommended Approach | Model |
|-------------|---------------------|-------|
| None (zero-shot) | Pre-trained foundation | DINOv2, SAM |
| 10-100 samples | Few-shot transfer | DINOv2 + linear probe |
| 1K-10K samples | Fine-tune backbone | ViT, Swin |
| 100K+ samples | Train from scratch | Any architecture |

## Performance Benchmarks

### ImageNet-1K Classification (Top-1 Accuracy)

| Model | Params | Accuracy | Throughput |
|-------|--------|----------|------------|
| ResNet-50 | 25M | 76.5% | 1200 img/s |
| EfficientNet-B3 | 12M | 81.6% | 800 img/s |
| ViT-B/16 | 86M | 84.1% | 650 img/s |
| Swin-B | 88M | 83.5% | 580 img/s |
| DINOv2-B | 86M | 84.5% | 600 img/s |

### COCO Detection (mAP)

| Model | Backbone | mAP | FPS |
|-------|----------|-----|-----|
| Faster R-CNN | ResNet-50 | 40.2 | 7 |
| Cascade R-CNN | ResNet-50 | 43.0 | 5 |
| DETR | ResNet-50 | 42.0 | 28 |
| RT-DETR-R50 | ResNet-50 | 53.1 | 108 |
| YOLOv10-L | CSPNet | 54.4 | 80 |

### COCO Instance Segmentation (mask mAP)

| Model | Backbone | mAP | FPS |
|-------|----------|-----|-----|
| Mask R-CNN | ResNet-50 | 37.1 | 5 |
| SAM (ViT-H) | ViT-H | 46.5* | - |

*Zero-shot performance

## Training Best Practices

### General Guidelines

1. **Data Augmentation**
   - Classification: RandAugment + MixUp + CutMix
   - Detection: Mosaic + RandomFlip + ColorJitter
   - Segmentation: RandomCrop + Flip + Scale

2. **Learning Rate**
   - From scratch: 1e-3 with warmup
   - Fine-tuning: 1e-5 to 5e-5
   - Linear probe: 1e-2 to 1e-1

3. **Batch Size**
   - Scale with model size
   - Use gradient accumulation if needed
   - Linear LR scaling with batch size

4. **Regularization**
   - DropPath for transformers
   - Weight decay: 0.01-0.1
   - Label smoothing: 0.1

### Common Pitfalls

Wrong: Training ViT from scratch on small datasets
```python
model = ViT({...})  # Will underfit on <100K images
```

Correct: Use pre-trained models
```python
model = DINOv2.from_pretrained("dinov2_vitb14")
model.head = nn.Linear(768, num_classes)
```

Wrong: No learning rate warmup for transformers
```python
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
```

Correct: Warmup then cosine decay
```python
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)
```

## Implementation Details

### Code Organization

```
nexus/models/cv/
├── vit.py                  # Vision Transformer
├── swin_transformer.py     # Swin Transformer
├── dinov2.py              # DINOv2
├── siglip.py              # SigLIP
├── eva02.py               # EVA-02
├── intern_vl.py           # InternVL
├── efficient_net.py       # EfficientNet
├── resnet.py              # ResNet
├── vgg.py                 # VGG
├── detr.py                # DETR
├── rt_detr.py             # RT-DETR
├── grounding_dino.py      # Grounding DINO
├── yolo_world.py          # YOLO-World
├── yolov10.py             # YOLOv10
├── sam.py                 # SAM
├── sam2.py                # SAM 2
├── medsam.py              # MedSAM
├── rcnn/                  # R-CNN family
│   ├── faster_rcnn.py
│   ├── cascade_rcnn.py
│   ├── mask_rcnn.py
│   └── keypoint_rcnn.py
├── hivit/                 # HiViT
└── nerf/                  # NeRF family
```

### Design Patterns

All models follow Nexus conventions:

1. **Inherit from NexusModule**
2. **Config-driven initialization**
3. **Dict-based outputs**
4. **Weight initialization mixins**
5. **Feature extraction support**

Example:

```python
class MyModel(WeightInitMixin, NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Build model
        self.init_weights_vision()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "logits": logits,
            "features": features,
            "embeddings": embeddings
        }
```

## Optimization Techniques

### Memory Optimization

- **Gradient Checkpointing**: Trade compute for memory
- **Flash Attention**: 3-5x memory reduction
- **Mixed Precision**: FP16/BF16 training
- **Activation Checkpointing**: Recompute instead of store

### Speed Optimization

- **Compile**: `torch.compile()` for 20-30% speedup
- **TensorRT**: INT8 quantization for inference
- **ONNX Export**: Cross-platform deployment
- **Model Pruning**: Remove redundant parameters

### Distributed Training

- **DDP**: Data parallel across GPUs
- **FSDP**: Fully sharded data parallel
- **DeepSpeed**: ZeRO optimizer stages
- **Pipeline Parallelism**: For very large models

## Visualization & Analysis

### Attention Maps

```python
# Visualize ViT attention
attentions = model(image, output_attentions=True)
plot_attention_maps(attentions, num_layers=4)
```

### Feature Maps

```python
# Visualize intermediate features
outputs = model(image)
features = outputs["features"]
visualize_feature_maps(features[6])  # Layer 6
```

### Gradients (GradCAM)

```python
# Class activation maps
from nexus.visualization import GradCAM
gradcam = GradCAM(model, target_layer="blocks.11")
heatmap = gradcam(image, target_class=281)  # 'cat' class
```

## Resources

### Papers

See individual model documentation for paper references.

### Pre-trained Weights

- **Hugging Face**: https://huggingface.co/models
- **timm**: https://github.com/huggingface/pytorch-image-models
- **Official repos**: See model-specific docs

### Datasets

- **ImageNet**: https://image-net.org/
- **COCO**: https://cocodataset.org/
- **ADE20K**: https://groups.csail.mit.edu/vision/datasets/ADE20K/
- **Cityscapes**: https://www.cityscapes-dataset.com/

## Contributing

When adding new models:

1. Follow Nexus design patterns
2. Add comprehensive docstrings
3. Include configuration examples
4. Provide pre-trained weights if available
5. Add model documentation to this section

## Citation

If you use these models in your research, please cite the original papers. See individual model documentation for BibTeX entries.
