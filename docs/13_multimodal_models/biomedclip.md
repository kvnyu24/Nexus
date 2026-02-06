# BiomedCLIP: Biomedical Vision-Language Pre-training

## Overview & Motivation

BiomedCLIP represents Microsoft's pioneering effort to bring contrastive vision-language pre-training to the biomedical domain. Building on the success of CLIP for natural images, BiomedCLIP adapts this framework to medical imaging where accurate image-text alignment is critical for clinical decision support.

**Key Achievement**: First large-scale biomedical vision-language model trained on diverse medical imaging modalities, enabling zero-shot classification and cross-modal retrieval for clinical applications.

**The Medical AI Challenge**:

Traditional medical AI faces unique obstacles:
- Data scarcity due to privacy and expert annotation requirements  
- Domain specificity: general models fail on medical images
- Complex medical terminology
- Diverse imaging modalities (X-ray, CT, MRI, pathology)
- High stakes requiring exceptional accuracy

**BiomedCLIP's Solution**:
1. Domain-specific training on PubMed medical images
2. Multi-modality coverage (radiology, pathology, dermatology, etc.)
3. Zero-shot medical image classification
4. Cross-modal retrieval for clinical decision support
5. Open foundation for medical AI applications

**Model Architecture**:
- Image Encoder: ViT-B/16 (vision transformer)
- Text Encoder: BioClinicalBERT  
- Projection Dim: 512
- Training Data: PMC-OA (15M image-caption pairs)
- Medical Domains: 100+ anatomical regions, 1000+ procedures

**Performance Highlights**:
- 85%+ accuracy on radiology classification (zero-shot)
- State-of-the-art medical image-text retrieval
- Outperforms general CLIP by 25-40% on medical images

## Theoretical Background

### Contrastive Learning for Medical Images

BiomedCLIP adapts contrastive learning to medical imaging characteristics:

**CLIP Framework**:
```
Goal: Learn joint vision-language representations
Method: Maximize similarity between matching pairs
Training: Web-scale data (400M+ pairs)
```

**BiomedCLIP Adaptation**:
```
Goal: Medical-specific representations  
Method: Same contrastive framework
Training: PubMed medical literature (15M pairs)
Challenge: Medical terminology, specialized images
```

**Key Differences**:
- Visual: Grayscale medical images vs. RGB natural photos
- Text: Technical clinical descriptions vs. everyday captions
- Semantics: Clinical relationships vs. general hierarchies

### PMC-OA Dataset  

**Dataset Characteristics**:
- Size: 15 million image-caption pairs
- Source: PubMed Central open-access articles
- Domains: Radiology, pathology, dermatology, ophthalmology  
- Image Types: X-ray, CT, MRI, ultrasound, microscopy
- Captions: Expert-written from peer-reviewed literature

**Data Quality**:
1. Expert-written by medical professionals
2. Peer-reviewed scientific articles
3. Diverse medical specialties and conditions
4. Detailed technical descriptions
5. Structured clinical terminology

### Clinical Applications

1. Computer-Aided Diagnosis: Support radiologist interpretation
2. Medical Image Search: Find similar cases for reference
3. Report Generation: Assist in radiological report writing
4. Medical Education: Teaching and self-study resources

## Mathematical Formulation

### Contrastive Loss Function

**InfoNCE Loss**:

Given batch of N image-text pairs:
```
Images: {I₁, I₂, ..., I_N}
Texts: {T₁, T₂, ..., T_N}

Encode:
v_i = ImageEncoder(I_i) ∈ ℝ^d  
t_i = TextEncoder(T_i) ∈ ℝ^d

Normalize:
v̂_i = v_i / ||v_i||₂
t̂_i = t_i / ||t_i||₂

Similarity:
s_{ij} = τ · (v̂_i^T t̂_j)

Loss for image i:
L_i2t = -log(exp(s_{ii}) / Σⱼ exp(s_{ij}))

Loss for text i:
L_t2i = -log(exp(s_{ii}) / Σⱼ exp(s_{ji}))

Total loss:
L = (1/2N) Σᵢ (L_i2t + L_t2i)
```

### Zero-Shot Classification

Given test image I and class names C = {c₁, ..., c_K}:

```
Step 1: Create prompts
prompt_k = template(c_k)
# e.g., "chest X-ray showing {c_k}"

Step 2: Encode
v = ImageEncoder(I)
{t₁, ..., t_K} = TextEncoder({prompt₁, ..., prompt_K})

Step 3: Compute similarities  
score_k = v^T t_k

Step 4: Softmax probabilities
p_k = exp(score_k) / Σⱼ exp(score_j)

Prediction: argmax_k p_k
```

**Template Engineering**:

Medical templates outperform generic ones:
- "chest X-ray showing {class}"
- "pathology image demonstrating {class}"  
- "CT scan with evidence of {class}"
- "medical image consistent with {class}"

### Cross-Modal Retrieval

**Image-to-Text**:
```
Query: Image I_q
Database: Texts {T₁, ..., T_M}

v_q = ImageEncoder(I_q)
{t₁, ..., t_M} = TextEncoder({T₁, ..., T_M})

scores = [v_q^T t₁, ..., v_q^T t_M]
results = top-K by score
```

**Text-to-Image** (symmetric):
```
Query: Text T_q  
Database: Images {I₁, ..., I_M}

t_q = TextEncoder(T_q)
{v₁, ..., v_M} = ImageEncoder({I₁, ..., I_M})

scores = [t_q^T v₁, ..., t_q^T v_M]
results = top-K images
```

## High-Level Intuition

### The Medical Library Analogy

**Traditional Search** (keyword matching):
```
Query: "pneumonia"
Result: Images with "pneumonia" in filename
Problem: Misses "infiltrates", "consolidation"
```

**BiomedCLIP** (semantic understanding):
```
Query: "pneumonia"
Understanding: Links to related terms
Result: All relevant images
Benefit: True semantic search
```

### Zero-Shot Intelligence

**Without BiomedCLIP**:
```
Task: Classify pneumonia
Requirements:
- Collect 10,000+ labeled X-rays  
- Train specialized model
- Model only works for pneumonia
Time: Weeks, expensive annotation
```

**With BiomedCLIP**:
```
Task: Classify pneumonia
Requirements:
- Define text prompts
- Compare image to descriptions
Time: Minutes, no training needed
Bonus: Works for any describable disease
```

## Implementation Details

### Model Components

From `nexus/models/multimodal/biomedclip.py`:

**Biomedical Image Encoder**:
```python
class BiomedicalImageEncoder(NexusModule):
    """ViT encoder for medical imaging."""
    
    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 768,
        num_layers: int = 12,
        patch_size: int = 16,
    ):
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, hidden_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # CLS token
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, hidden_dim) * 0.02
        )
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(
            torch.randn(1, 1 + 196, hidden_dim) * 0.02
        )
        
        # Transformer encoder
        self.encoder = nn.TransformerEncoder(...)
        
        # Projection head
        self.projection = nn.Linear(hidden_dim, hidden_dim)
```

**Biomedical Text Encoder**:
```python
class BiomedicalTextEncoder(NexusModule):
    """Text encoder for medical terminology."""
    
    def __init__(
        self,
        vocab_size: int = 49408,
        hidden_dim: int = 512,
        num_layers: int = 12,
    ):
        # Medical vocabulary embeddings
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Parameter(...)
        self.encoder = nn.TransformerEncoder(...)
        self.projection = nn.Linear(hidden_dim, hidden_dim)
```

### Usage Examples

**Zero-Shot Classification**:
```python
from nexus.models.multimodal import BiomedCLIP

model = BiomedCLIP(
    image_encoder_dim=768,
    text_encoder_dim=512,
    projection_dim=512
)

# Load medical image
chest_xray = load_medical_image("xray.jpg")

# Define diagnoses
diagnoses = [
    "normal chest X-ray",
    "pneumonia",
    "pulmonary edema",
    "pleural effusion",
]

# Create prompts
prompts = [f"chest X-ray showing {d}" for d in diagnoses]

# Forward pass
input_ids = tokenize(prompts)
output = model(images=chest_xray.repeat(4,1,1,1), 
               input_ids=input_ids)

# Get probabilities
logits = output['logits_per_image'][0]
probs = torch.softmax(logits, dim=0)

for diag, prob in zip(diagnoses, probs):
    print(f"{diag}: {prob:.1%}")
```

**Medical Image Retrieval**:
```python
# Text-to-image search
query = "intracranial hemorrhage cases"

text_embeds = model.encode_text(tokenize([query]))
image_embeds = model.encode_image(image_database)

similarities = text_embeds @ image_embeds.T
top_indices = similarities[0].argsort(descending=True)[:10]

for idx in top_indices:
    display_image(image_database[idx])
```

## Code Walkthrough

### Forward Pass

```python
def forward(
    self,
    images: Optional[torch.Tensor] = None,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    return_loss: bool = False
) -> Dict[str, torch.Tensor]:
    """Forward pass through BiomedCLIP."""
    
    outputs = {}
    
    # Encode images
    if images is not None:
        image_features = self.encode_image(images)
        outputs['image_features'] = image_features
    
    # Encode text
    if input_ids is not None:
        text_features = self.encode_text(input_ids, attention_mask)
        outputs['text_features'] = text_features
    
    # Compute similarity
    if images is not None and input_ids is not None:
        logit_scale = self.logit_scale.exp()
        
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logits_per_image.T
        
        outputs['logits_per_image'] = logits_per_image
        outputs['logits_per_text'] = logits_per_text
        
        # Contrastive loss
        if return_loss:
            batch_size = images.shape[0]
            labels = torch.arange(batch_size, device=images.device)
            
            loss_i = F.cross_entropy(logits_per_image, labels)
            loss_t = F.cross_entropy(logits_per_text, labels)
            loss = (loss_i + loss_t) / 2
            
            outputs['loss'] = loss
    
    return outputs
```

### Zero-Shot Classification

```python
def zero_shot_classify(
    model,
    image: torch.Tensor,
    class_names: List[str],
    templates: List[str] = None
) -> torch.Tensor:
    """Classify using zero-shot learning."""
    
    if templates is None:
        templates = [
            "medical image showing {}",
            "this demonstrates {}",
            "consistent with {}",
        ]
    
    # Encode image
    image_features = model.encode_image(image)
    
    # Create and encode prompts
    all_prompts = []
    for class_name in class_names:
        for template in templates:
            all_prompts.append(template.format(class_name))
    
    text_inputs = tokenize(all_prompts)
    text_features = model.encode_text(text_inputs)
    
    # Compute similarities
    similarities = image_features @ text_features.T
    
    # Average across templates
    similarities = similarities.reshape(len(class_names), len(templates))
    avg_similarities = similarities.mean(dim=1)
    
    # Convert to probabilities
    probabilities = torch.softmax(avg_similarities, dim=0)
    
    return probabilities
```

## Optimization Tricks

### Medical Image Preprocessing

```python
def preprocess_medical_image(image, modality='xray'):
    """Modality-specific preprocessing."""
    
    if modality == 'xray':
        # Clinical window normalization
        image = clip_and_normalize(image, 
                                   window_center=40, 
                                   window_width=400)
    
    elif modality == 'ct':
        # Hounsfield unit normalization
        image = (image + 1000) / 2000
    
    elif modality == 'mri':
        # Percentile normalization
        p1, p99 = np.percentile(image, [1, 99])
        image = np.clip((image - p1) / (p99 - p1), 0, 1)
    
    # Convert grayscale to RGB if needed
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    
    return torch.FloatTensor(image).permute(2, 0, 1)
```

### Template Engineering

```python
# Domain-specific templates improve performance

RADIOLOGY_TEMPLATES = [
    "chest X-ray showing {}",
    "radiograph demonstrating {}",
    "imaging findings consistent with {}",
]

PATHOLOGY_TEMPLATES = [
    "histopathology slide showing {}",
    "microscopic image demonstrating {}",
    "tissue section with {}",
]

def get_templates_for_modality(modality):
    template_map = {
        'xray': RADIOLOGY_TEMPLATES,
        'pathology': PATHOLOGY_TEMPLATES,
        'dermatology': ["clinical photograph of {}"],
    }
    return template_map.get(modality, ["medical image showing {}"])
```

## Experiments & Results

### Zero-Shot Classification

**Radiology Tasks**:

| Dataset | Classes | BiomedCLIP | General CLIP | Improvement |
|---------|---------|------------|--------------|-------------|
| ChestX-ray14 | 14 | 68.2% | 42.5% | +60.5% |
| CheXpert | 5 | 73.1% | 51.2% | +42.8% |
| MIMIC-CXR | 8 | 71.5% | 48.3% | +48.0% |

**Pathology Tasks**:

| Dataset | Classes | BiomedCLIP | General CLIP |
|---------|---------|------------|--------------|
| NCT-CRC | 9 | 82.3% | 61.4% |
| PatchCamelyon | 2 | 88.5% | 72.1% |
| Kather | 8 | 79.2% | 58.7% |

### Cross-Modal Retrieval

**Image-to-Text**:

| Dataset | R@1 | R@5 | R@10 |
|---------|-----|-----|------|
| ROCO | 42.3% | 71.2% | 82.5% |
| MedICaT | 38.7% | 68.4% | 79.1% |
| PMC-OA | 45.1% | 73.8% | 84.2% |

**Text-to-Image**:

| Dataset | R@1 | R@5 | R@10 |
|---------|-----|-----|------|
| ROCO | 39.5% | 69.3% | 80.7% |
| MedICaT | 36.2% | 65.8% | 77.3% |

### Ablation Studies

**Impact of Medical Pre-training**:

| Variant | ChestX-ray14 | Pathology |
|---------|--------------|-----------|
| CLIP (ImageNet) | 42.5% | 61.4% |
| CLIP (PMC-100K) | 55.3% | 69.1% |
| BiomedCLIP (PMC-15M) | 68.2% | 82.3% |

Conclusion: Scale and domain-specificity both critical

## Common Pitfalls

### 1. Inappropriate Templates

**Bad**:
```python
template = "a photo of {}"  # Generic
```

**Good**:
```python
templates = [
    "chest X-ray showing {}",
    "radiological findings of {}",
]  # Medical-specific
```

### 2. Ignoring Medical Preprocessing

Medical images need modality-specific handling:

```python
# CT scans need Hounsfield unit normalization
# X-rays need clinical windowing
# MRI needs percentile normalization

def preprocess_ct(ct_image):
    return apply_window(ct_image, center=40, width=400)
```

### 3. Class Imbalance

```python
# Calibrate with disease prevalence
def calibrate_with_priors(predictions, priors):
    adjusted = predictions * priors
    return adjusted / adjusted.sum()

priors = {
    'normal': 0.60,
    'pneumonia': 0.15,
    'rare_disease': 0.01,
}
```

## References

### Original Papers

1. **BiomedCLIP: Biomedical Vision-Language Foundation Model**
   - Authors: Sheng Zhang et al., Microsoft Research
   - Link: https://arxiv.org/abs/2303.00915 (2023)

2. **PMC-OA Dataset Paper**
   - 15M image-caption pairs from PubMed Central

3. **CLIP: Learning Transferable Visual Models**
   - Link: https://arxiv.org/abs/2103.00020
   - Original contrastive learning framework

### Medical Datasets

4. **ChestX-ray14**: 14 thoracic diseases
5. **MIMIC-CXR**: Large chest X-ray dataset
6. **PatchCamelyon**: Histopathology classification

### Implementation

7. **Hugging Face Model**: microsoft/BiomedCLIP-PubMedBERT
8. **Medical Image Tools**: MONAI framework
9. **Clinical Applications**: CAD literature reviews
10. **Medical Education**: Interactive image search tools
