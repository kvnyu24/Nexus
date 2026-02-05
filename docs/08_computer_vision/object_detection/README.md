# Object Detection

Comprehensive documentation for object detection models, from two-stage detectors to modern transformer-based approaches.

## Contents

### Transformer-Based Detectors

- **[DETR](detr.md)** - Detection Transformer: End-to-end object detection with transformers
- **[RT-DETR](rt_detr.md)** - Real-Time DETR: Fast transformer detector
- **[Grounding DINO](grounding_dino.md)** - Open-set detection with language grounding
- **[YOLO-World](yolo_world.md)** - Open-vocabulary YOLO detector

### Two-Stage Detectors (R-CNN Family)

- **[Faster R-CNN](faster_rcnn.md)** - Region-based CNN with Region Proposal Network
- **[Cascade R-CNN](cascade_rcnn.md)** - Multi-stage refinement for better localization
- **[Mask R-CNN](mask_rcnn.md)** - Instance segmentation via RoI masking
- **[Keypoint R-CNN](keypoint_rcnn.md)** - Human pose estimation extension

### Single-Stage Detectors

- **[YOLOv10](yolov10.md)** - Latest YOLO with NMS-free training

## Model Comparison

| Model | Type | Speed (FPS) | mAP | Key Feature |
|-------|------|-------------|-----|-------------|
| Faster R-CNN | Two-stage | 7 | 42.0 | RPN + RoI pooling |
| Cascade R-CNN | Two-stage | 5 | 44.9 | Progressive refinement |
| Mask R-CNN | Two-stage | 5 | 43.2 | Instance segmentation |
| DETR | Transformer | 28 | 42.0 | Set prediction |
| RT-DETR | Transformer | 108 | 53.1 | Real-time |
| Grounding DINO | Transformer | 15 | 52.5 | Open-vocabulary |
| YOLO-World | Single-stage | 52 | 35.4 | Zero-shot detection |
| YOLOv10 | Single-stage | 80 | 54.4 | NMS-free |

## Selection Guide

### For Real-Time Applications

- **High speed**: YOLOv10-N/S or RT-DETR-R18
- **Balanced**: RT-DETR-R50 or YOLOv10-M
- **Best accuracy**: RT-DETR-R101

### For High Accuracy

- **Closed-set**: Cascade R-CNN + Swin-L backbone
- **Open-vocabulary**: Grounding DINO-L
- **With segmentation**: Mask R-CNN + FPN

### For Zero-Shot/Open-Vocabulary

- **Text prompts**: Grounding DINO
- **Category names**: YOLO-World
- **Phrase grounding**: Grounding DINO with BERT

### For Specific Tasks

- **Instance segmentation**: Mask R-CNN
- **Keypoint detection**: Keypoint R-CNN
- **Crowd detection**: Cascade R-CNN (handles overlap well)
- **Small objects**: FPN backbone + multi-scale training

## Common Architectures

### Two-Stage Pipeline

```
Image → Backbone → FPN → RPN → RoI Align → Detection Head → Boxes + Classes
                           ↓
                    Region Proposals
```

### Transformer Pipeline

```
Image → Backbone → Transformer Encoder → Object Queries → Decoder → Boxes + Classes
```

### Single-Stage Pipeline

```
Image → Backbone → Neck (FPN/PAN) → Detection Heads → Boxes + Classes
```

## Training Tips

### Data Augmentation

```python
# Standard augmentation for detectors
transforms = [
    RandomFlip(prob=0.5),
    RandomResize(scales=[0.8, 1.0, 1.2]),
    RandomCrop(size=640),
    ColorJitter(brightness=0.2, contrast=0.2),
    Mosaic(prob=0.5),  # For YOLO
]
```

### Learning Rate Schedules

- **Two-stage**: 1e-3 with step decay at 8, 11 epochs (12 total)
- **DETR**: 1e-4 with drop at 100 epochs (150 total)
- **YOLO**: Cosine decay with warmup

### Loss Functions

- **Classification**: Focal Loss (single-stage), Cross Entropy (two-stage)
- **Localization**: GIoU Loss or CIoU Loss
- **Matching**: Hungarian matching (DETR), Max IoU (R-CNN)

## Implementation Resources

All implementations available in `/Users/kevinyu/Projects/Nexus/nexus/models/cv/`:

- DETR: `detr.py`
- R-CNN family: `rcnn/` directory
- RT-DETR: `rt_detr.py`
- Grounding DINO: `grounding_dino.py`
- YOLO-World: `yolo_world.py`
- YOLOv10: `yolov10.py`

## Benchmarks

### COCO val2017 Results

**Two-Stage Detectors:**

| Model | Backbone | mAP | AP50 | AP75 |
|-------|----------|-----|------|------|
| Faster R-CNN | ResNet-50 | 40.2 | 61.0 | 43.8 |
| Cascade R-CNN | ResNet-50 | 43.0 | 61.2 | 46.3 |
| Mask R-CNN | ResNet-50 | 41.0 | 61.7 | 44.9 |

**Transformer Detectors:**

| Model | Backbone | mAP | AP50 | AP75 | FPS |
|-------|----------|-----|------|------|-----|
| DETR | ResNet-50 | 42.0 | 62.4 | 44.2 | 28 |
| RT-DETR-R50 | ResNet-50 | 53.1 | 71.3 | 57.6 | 108 |
| Grounding DINO-T | Swin-T | 48.4 | 67.2 | 52.1 | 15 |

**Single-Stage Detectors:**

| Model | Input Size | mAP | FPS |
|-------|------------|-----|-----|
| YOLOv10-N | 640 | 38.5 | 142 |
| YOLOv10-S | 640 | 46.3 | 120 |
| YOLOv10-M | 640 | 51.1 | 92 |
| YOLOv10-L | 640 | 54.4 | 80 |

## References

See individual model documentation for detailed papers and implementations.
