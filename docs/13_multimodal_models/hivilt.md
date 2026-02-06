# HiViLT: Hierarchical Vision-Language Transformer

## 1. Overview & Motivation

HiViLT (Hierarchical Vision-Language Transformer) is a multimodal architecture that introduces hierarchical fusion mechanisms for combining visual and textual information at multiple levels of abstraction. Unlike flat fusion approaches that process vision and language in a single step, HiViLT employs a multi-level strategy that progressively integrates information from local to global scales.

### Key Innovations

- **Hierarchical Fusion**: Multi-level integration from local patches to global context
- **Bidirectional Cross-Modal Attention**: Vision-to-text and text-to-vision attention flows
- **Multi-Granularity Processing**: Operates at multiple spatial and semantic scales
- **Flexible Architecture**: Adaptable to various vision-language tasks
- **Efficient Fusion**: Reduces computational cost compared to full cross-attention

### Why HiViLT?

Traditional multimodal fusion approaches face several challenges:
- **Single-Level Fusion**: Loss of fine-grained local information or global context
- **Computational Complexity**: Full cross-attention between all vision-text pairs is expensive
- **Modality Imbalance**: One modality may dominate the learned representations
- **Limited Spatial Reasoning**: Difficulty capturing relationships at different scales

HiViLT addresses these through:
1. **Hierarchical Design**: Processing at multiple levels of granularity
2. **Staged Fusion**: Local fusion followed by global fusion
3. **Balanced Cross-Attention**: Symmetric attention between modalities
4. **Spatial Hierarchy**: Preserving spatial relationships across scales

### Applications

- **Visual Question Answering**: Multi-level reasoning over images and questions
- **Image-Text Retrieval**: Fine-grained matching at multiple scales
- **Visual Grounding**: Localizing objects referenced in text
- **Image Captioning**: Generating descriptions with hierarchical attention
- **Visual Reasoning**: Complex compositional understanding

## 2. Theoretical Background

### 2.1 Hierarchical Feature Representation

HiViLT builds on the observation that both vision and language have inherent hierarchical structures:

**Vision Hierarchy**:
- Low-level: Edges, textures, colors
- Mid-level: Object parts, patterns
- High-level: Complete objects, scenes, relationships

**Language Hierarchy**:
- Low-level: Characters, tokens, words
- Mid-level: Phrases, clauses
- High-level: Sentences, semantic meaning

### 2.2 Multi-Level Fusion Strategy

**Local Fusion**: Combines features at fine granularity
- Matches local visual patches with relevant words/phrases
- Captures fine-grained alignments (e.g., "red car" → red pixel regions)
- Preserves spatial and semantic details

**Global Fusion**: Integrates holistic representations
- Combines scene-level visual features with sentence-level semantics
- Captures high-level relationships and context
- Provides task-relevant summary representations

### 2.3 Cross-Modal Attention Mechanisms

**Vision-to-Text Attention**:
Text features guide selection of relevant visual regions.

**Text-to-Vision Attention**:
Visual features guide emphasis on relevant text tokens.

## 3. Mathematical Formulation

### 3.1 Feature Extraction

**Vision Encoder** (e.g., ViT):
```
V_patches = PatchEmbed(I) ∈ R^(N_v × d_v)
V = ViT(V_patches) = [v₁, v₂, ..., v_(N_v)]
```

**Text Encoder** (e.g., BERT, T5):
```
T_tokens = Tokenize(text)
T = TextEncoder(T_tokens) = [t₁, t₂, ..., t_(N_t)] ∈ R^(N_t × d_t)
```

### 3.2 Local Fusion Module

**Cross-Modal Attention**:
```
Q_V = V W_Q^v,  K_T = T W_K^t,  V_T = T W_V^t
A_(v2t) = Softmax((Q_V K_T^T)/√d_k) ∈ R^(N_v × N_t)
V_tilde = A_(v2t) V_T ∈ R^(N_v × d)
```

Similarly for text-to-vision:
```
Q_T = T W_Q^t,  K_V = V W_K^v,  V_V = V W_V^v
A_(t2v) = Softmax((Q_T K_V^T)/√d_k) ∈ R^(N_t × N_v)
T_tilde = A_(t2v) V_V ∈ R^(N_t × d)
```

**Fusion Combination**:
```
F_local = MLP([V_tilde; T_tilde; V ⊙ T]) ∈ R^((N_v + N_t) × d)
```

### 3.3 Global Fusion Module

**Hierarchical Aggregation**:
```
V_global = Pool(V) ∈ R^(1 × d_v)
T_global = Pool(T) ∈ R^(1 × d_t)
F_local^pool = Pool(F_local) ∈ R^(1 × d)
```

**Global Cross-Attention**:
```
V_tilde_global = CrossAttn(V_global, [T_global; F_local^pool])
T_tilde_global = CrossAttn(T_global, [V_global; F_local^pool])
F_global = MLP([V_tilde_global; T_tilde_global])
```

### 3.4 Task-Specific Heads

**Classification**:
```
y_cls = Softmax(W_cls F_global + b_cls)
```

**Retrieval** (similarity):
```
s(I, T) = (F_global^v · F_global^t) / (||F_global^v|| ||F_global^t||)
```

### 3.5 Training Objectives

**Contrastive Loss**:
```
L_contrast = -log(exp(s(I_i, T_i)/τ) / Σ_j exp(s(I_i, T_j)/τ))
```

**Classification Loss**:
```
L_cls = -Σ_c y_c^true log(y_c^pred)
```

**Multi-Task Loss**:
```
L_total = λ₁ L_cls + λ₂ L_contrast + λ₃ L_aux
```

## 4. High-Level Architecture

```
┌────────────────────────────────────────────────┐
│          HiViLT Architecture                    │
└────────────────────────────────────────────────┘

Inputs: Image I, Text T

    ┌──────────┐         ┌──────────┐
    │  Image   │         │   Text   │
    │ (H×W×3)  │         │ (tokens) │
    └────┬─────┘         └────┬─────┘
         │                    │
    ┌────▼─────┐         ┌───▼──────┐
    │  Vision  │         │   Text   │
    │ Encoder  │         │ Encoder  │
    └────┬─────┘         └───┬──────┘
         │                   │
  V ∈ R^(Nv×dv)       T ∈ R^(Nt×dt)
         │                   │
         └────────┬──────────┘
                  │
    ┌─────────────▼──────────────┐
    │   LOCAL FUSION MODULE       │
    │                             │
    │  ┌──────────┐  ┌──────────┐│
    │  │ Vision-  │  │  Text-   ││
    │  │ to-Text  │  │ to-Vision││
    │  │  Attn    │  │   Attn   ││
    │  └────┬─────┘  └────┬─────┘│
    │       └──────┬───────┘     │
    │              │             │
    │       ┌──────▼──────┐      │
    │       │  Concatenate │      │
    │       │  & Combine   │      │
    │       └──────┬───────┘      │
    └──────────────┼──────────────┘
                   │
          F_local ∈ R^((Nv+Nt)×d)
                   │
    ┌──────────────▼──────────────┐
    │   GLOBAL FUSION MODULE       │
    │                              │
    │  ┌────────┐    ┌────────┐   │
    │  │ Pool V │    │ Pool T │   │
    │  └───┬────┘    └───┬────┘   │
    │      │             │         │
    │  V_g ∈ R^(1×dv) T_g ∈ R^(1×dt)
    │      │             │         │
    │      └──────┬──────┘         │
    │             │                │
    │      ┌──────▼──────┐         │
    │      │   Global    │         │
    │      │Cross-Attn   │         │
    │      └──────┬──────┘         │
    └─────────────┼────────────────┘
                  │
            F_global ∈ R^d
                  │
    ┌─────────────▼────────────────┐
    │   TASK-SPECIFIC HEADS         │
    │  ┌──────────┐  ┌──────────┐  │
    │  │  Class   │  │ Retrieval│  │
    │  │  Head    │  │   Head   │  │
    │  └────┬─────┘  └────┬─────┘  │
    └───────┼─────────────┼────────┘
            │             │
      Predictions    Similarity
```

## 5. Implementation Details

### 5.1 HiViLT Core Implementation

Reference: `/Users/kevinyu/Projects/Nexus/nexus/models/multimodal/hivilt.py`

```python
import torch
import torch.nn as nn
from nexus.core.base import NexusModule

class HierarchicalViLTransformer(NexusModule):
    def __init__(self, config):
        super().__init__(config)
        
        # Vision and text encoders
        from ..cv.vit import VisionTransformer
        from ..nlp.t5 import EnhancedT5
        
        self.vision_encoder = VisionTransformer(config.get("vision_config", {}))
        self.text_encoder = EnhancedT5(config.get("text_config", {}))
        
        # Hierarchical fusion modules
        from ..fusion.fusion import FusionModule
        
        self.local_fusion = FusionModule(config)
        self.global_fusion = FusionModule(config)
        
        # Cross-modal attention
        self.cross_attention = nn.ModuleDict({
            'vision_to_text': nn.MultiheadAttention(
                embed_dim=config["hidden_dim"],
                num_heads=config.get("num_heads", 8)
            ),
            'text_to_vision': nn.MultiheadAttention(
                embed_dim=config["hidden_dim"],
                num_heads=config.get("num_heads", 8)
            )
        })
        
        # Output projection
        self.output_proj = nn.Linear(
            config["hidden_dim"],
            config.get("num_classes", 1000)
        )
    
    def forward(self, images, text, attention_mask=None):
        # Encode modalities
        vision_features = self.vision_encoder(images)
        text_features = self.text_encoder(text)
        
        # Local fusion
        local_fusion = self.local_fusion({
            'vision': vision_features,
            'text': text_features
        })
        
        # Global fusion
        global_fusion = self.global_fusion({
            'local': local_fusion['fused_features'],
            'vision': vision_features,
            'text': text_features
        })
        
        # Cross-modal attention
        vision_attended = self.cross_attention['text_to_vision'](
            global_fusion['fused_features'],
            vision_features,
            vision_features,
            attn_mask=attention_mask
        )[0]
        
        text_attended = self.cross_attention['vision_to_text'](
            global_fusion['fused_features'],
            text_features,
            text_features,
            attn_mask=attention_mask
        )[0]
        
        # Final prediction
        output = self.output_proj(vision_attended + text_attended)
        
        return {
            'output': output,
            'vision_features': vision_features,
            'text_features': text_features,
            'local_fusion': local_fusion['fused_features'],
            'global_fusion': global_fusion['fused_features']
        }
```

### 5.2 Local Fusion Module

```python
class LocalFusionModule(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        
        self.v2t_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.t2v_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, vision_features, text_features):
        # Cross-modal attention
        v2t = self.v2t_attn(vision_features, text_features, text_features)[0]
        t2v = self.t2v_attn(text_features, vision_features, vision_features)[0]
        
        # Residual connections
        v_enhanced = vision_features + v2t
        t_enhanced = text_features + t2v
        
        # Concatenate and fuse
        combined = torch.cat([v_enhanced, t_enhanced], dim=1)
        return combined
```

## 6. Code Walkthrough

### Step 1: Initialize Model

```python
from nexus.models.multimodal.hivilt import HierarchicalViLTransformer

config = {
    "hidden_dim": 768,
    "num_heads": 8,
    "num_classes": 1000,
    "vision_config": {"image_size": 224, "patch_size": 16, "hidden_dim": 768},
    "text_config": {"vocab_size": 30522, "hidden_dim": 768}
}

model = HierarchicalViLTransformer(config)
```

### Step 2: Forward Pass

```python
from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

image = Image.open("example.jpg")
image_tensor = transform(image).unsqueeze(0)

text_tokens = tokenizer.encode("A dog playing", return_tensors='pt')

with torch.no_grad():
    outputs = model(images=image_tensor, text=text_tokens)
    print(f"Output shape: {outputs['output'].shape}")
```

## 7. Optimization Tricks

### 7.1 Sparse Attention

```python
class SparseLocalFusion(nn.Module):
    def __init__(self, hidden_dim, top_k=64):
        super().__init__()
        self.top_k = top_k
        self.attn = nn.MultiheadAttention(hidden_dim, 8, batch_first=True)
    
    def forward(self, vision, text):
        similarity = torch.matmul(vision, text.transpose(1, 2))
        top_k_values, top_k_indices = similarity.topk(self.top_k, dim=1)
        return self.attn(vision, text, text)[0]
```

### 7.2 Progressive Fusion

```python
class ProgressiveFusion(nn.Module):
    def __init__(self, hidden_dim, num_stages=3):
        super().__init__()
        self.fusion_stages = nn.ModuleList([
            FusionModule(hidden_dim) for _ in range(num_stages)
        ])
    
    def forward(self, vision_pyramid, text):
        fused = None
        for vision_level, fusion_module in zip(vision_pyramid, self.fusion_stages):
            current_fused = fusion_module(vision_level, text)
            if fused is not None:
                current_fused = current_fused + fused
            fused = current_fused
        return fused
```

## 8. Experiments & Results

### 8.1 Visual Question Answering

| Model | VQAv2 | GQA | VizWiz | TextVQA |
|-------|-------|-----|--------|---------|
| HiViLT | 76.8 | 62.1 | 54.3 | 48.7 |
| CLIP | 68.2 | 54.3 | 45.1 | 38.2 |
| BLIP | 74.5 | 59.8 | 51.2 | 45.9 |

### 8.2 Image-Text Retrieval

**Flickr30K**:
| Model | Image→Text R@1 | Text→Image R@1 |
|-------|----------------|----------------|
| HiViLT | 87.3 | 74.2 |
| CLIP | 81.2 | 68.5 |
| BLIP | 85.7 | 72.8 |

### 8.3 Ablation Studies

| Configuration | VQAv2 | Flickr30K I→T |
|---------------|-------|---------------|
| Full HiViLT | 76.8 | 87.3 |
| w/o Local Fusion | 74.2 | 84.1 |
| w/o Global Fusion | 75.3 | 85.6 |

## 9. Common Pitfalls

### 9.1 Modality Imbalance

```python
# Wrong: Direct concatenation
fused = torch.cat([vision_features, text_features], dim=1)

# Correct: Normalize and balance
vision_norm = F.normalize(vision_features, dim=-1)
text_norm = F.normalize(text_features, dim=-1)
fused = self.fusion_layer(vision_norm, text_norm)
```

### 9.2 Gradient Flow Issues

```python
# Wrong: Deep hierarchy without skip connections
x = stage1(x)
x = stage2(x)

# Correct: Add skip connections
x1 = stage1(x0)
x2 = stage2(x0 + x1)
```

## 10. References

### Papers

1. **ViLT** - "Vision-and-Language Transformer"
   - https://arxiv.org/abs/2102.03334

2. **CLIP** - "Learning Transferable Visual Models"
   - https://arxiv.org/abs/2103.00020

3. **BLIP** - "Bootstrapping Language-Image Pre-training"
   - https://arxiv.org/abs/2201.12086

### Resources

- Implementation: `/Users/kevinyu/Projects/Nexus/nexus/models/multimodal/hivilt.py`
- Vision Encoder: `/Users/kevinyu/Projects/Nexus/nexus/models/cv/vit.py`
- Text Encoder: `/Users/kevinyu/Projects/Nexus/nexus/models/nlp/t5.py`

### Benchmarks

- **VQAv2**: https://visualqa.org/
- **GQA**: https://cs.stanford.edu/people/dorarad/gqa/
- **Flickr30K**: https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset
