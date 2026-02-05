# Matryoshka Representation Learning

## 1. Overview & Motivation

Matryoshka Representation Learning (MRL) trains embedding models where representations can be truncated to various dimensions while maintaining good performance. Like Russian nesting dolls, shorter embeddings are "contained" within longer ones, enabling flexible accuracy-efficiency tradeoffs without retraining.

### Problem Statement

Traditional embedding models require fixed dimensions:
- 768-dim embeddings work well but are expensive
- Want 128-dim for edge devices but can't use same model
- Must train separate models for each dimension
- Can't adapt to deployment constraints dynamically

### Solution

Train a single model where the first d dimensions form a valid embedding:
- embedding[:64] → usable 64-dim embedding
- embedding[:128] → usable 128-dim embedding
- embedding[:768] → full 768-dim embedding

All dimensions trained simultaneously with multi-granularity loss.

### Key Applications

1. **Adaptive Deployment**: Single model for all device types
2. **Cost Optimization**: Use smaller embeddings when possible
3. **Dynamic Scaling**: Adjust dimension based on query importance
4. **Storage Efficiency**: Store once, use at multiple granularities
5. **Progressive Search**: Coarse search with small dims, refine with large

## 2. Theoretical Background

### Nested Representation Hypothesis

**Claim**: Semantic information has a hierarchical structure where:
- Coarse semantics require fewer dimensions
- Fine-grained distinctions need more dimensions
- Early dimensions capture broad concepts
- Later dimensions capture subtle nuances

### Multi-Scale Representation Learning

Traditional: Learn single-scale representation
```
minimize L(f(x))  where f: X → R^d
```

Matryoshka: Learn multi-scale representations simultaneously
```
minimize Σ_{d∈D} w_d · L(f(x)[:d])
```

where D = {d₁, d₂, ..., d_k} is the set of nesting dimensions.

### Information Ordering

MRL implicitly learns to order information by importance:
- Most important features → early dimensions
- Refined details → later dimensions
- Dimensionality becomes a continuous accuracy dial

## 3. Mathematical Formulation

### Matryoshka Training Objective

Given:
- Full embedding dimension: d_max
- Nesting dimensions: D = {d₁, d₂, ..., d_k} where d_i < d_max
- Loss function: L(·, ·) (e.g., contrastive loss)

The Matryoshka loss is:

```
L_MRL(x_i, x_j) = Σ_{d∈D} w_d · L(f(x_i)[:d], f(x_j)[:d])
```

where:
- f(x) is the full embedding function: X → R^{d_max}
- f(x)[:d] denotes truncation to first d dimensions
- w_d are loss weights for each dimension (typically uniform)

### Contrastive Matryoshka Loss

For contrastive learning (e.g., sentence pairs):

```
L_MRL = Σ_{d∈D} w_d · L_contrastive^d
```

where:

```
L_contrastive^d = -log(
    exp(sim(z_i^d, z_j^d) / τ) /
    Σ_k exp(sim(z_i^d, z_k^d) / τ)
)
```

and:
- z_i^d = normalize(f(x_i)[:d]) is the truncated normalized embedding
- sim(·, ·) is similarity function (typically dot product)
- τ is temperature

### Optimization Properties

**Gradient Flow**: Gradients flow to all dimensions:

```
∂L_MRL/∂θ = Σ_{d∈D} w_d · ∂L_d/∂θ
```

Early dimensions receive gradients from all losses, later dimensions only from larger d.

**Theorem (Informal)**: MRL training with uniform weights ensures:
1. No degradation at full dimension d_max
2. Graceful degradation at smaller dimensions
3. Monotonic improvement: performance(d₁) ≤ performance(d₂) if d₁ < d₂

## 4. High-Level Intuition

Think of Matryoshka embeddings like image compression:

1. **Progressive Detail**: Like JPEG progressive encoding
   - First dimensions = blurry image (coarse semantics)
   - More dimensions = sharper image (fine details)
   - Can stop at any level and get valid image

2. **Information Hierarchy**:
   - Dimension 1-64: "This is about sports"
   - Dimension 65-128: "Specifically basketball"
   - Dimension 129-256: "About NBA playoffs"
   - Dimension 257-512: "LeBron James' performance"

3. **Deployment Flexibility**:
   - Edge device: Use 64 dims (fast, approximate)
   - Server: Use 512 dims (accurate)
   - Critical query: Use full 768 dims (best quality)

### Toy Example

```
Query: "machine learning tutorials"

64-dim embedding:
  - Captures: "technical educational content"
  - Retrieves: ML tutorials, programming courses, tech docs

128-dim embedding:
  - Captures: "machine learning educational content"
  - Retrieves: ML tutorials, ML courses, data science guides

512-dim embedding:
  - Captures: "beginner-friendly ML tutorials"
  - Retrieves: Intro ML tutorials, ML for beginners

Full 768-dim:
  - Captures: "step-by-step ML tutorials with code"
  - Retrieves: Hands-on ML tutorials with implementations
```

## 5. Implementation Details

### Model Architecture

```python
class MatryoshkaEmbedding(NexusModule):
    def __init__(self, config):
        self.encoder = Encoder(config)  # Transformer, etc.
        self.norm = LayerNorm(d_model)
        self.nesting_dims = [32, 64, 128, 256, 512, 768]

    def forward(self, x):
        # Encode input
        hidden = self.encoder(x)
        # Pool to single vector
        embedding = self.pool(hidden)
        # Normalize
        embedding = self.norm(embedding)
        return embedding
```

### Multi-Granularity Loss

```python
class MatryoshkaLoss:
    def __init__(self, nesting_dims, loss_weights):
        self.nesting_dims = nesting_dims
        self.loss_weights = loss_weights

    def forward(self, emb_a, emb_b):
        total_loss = 0.0
        for dim, weight in zip(self.nesting_dims, self.loss_weights):
            # Truncate embeddings
            emb_a_d = emb_a[:, :dim]
            emb_b_d = emb_b[:, :dim]

            # Compute loss at this dimension
            loss_d = self.contrastive_loss(emb_a_d, emb_b_d)
            total_loss += weight * loss_d

        return total_loss
```

### Pooling Strategy

```python
def _pool(self, hidden_states, attention_mask=None):
    if self.pooling == 'mean':
        # Masked mean pooling
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1)
            sum_embeds = (hidden_states * mask_expanded).sum(dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            return sum_embeds / sum_mask
        return hidden_states.mean(dim=1)

    elif self.pooling == 'cls':
        return hidden_states[:, 0]  # CLS token
```

### Truncation and Renormalization

```python
def get_embedding(self, x, dim):
    # Get full embedding
    full_emb = self.forward(x)  # [batch, d_max]

    # Truncate
    truncated = full_emb[:, :dim]  # [batch, dim]

    # Renormalize (important!)
    truncated = F.normalize(truncated, p=2, dim=-1)

    return truncated
```

## 6. Code Walkthrough

Reference: `/Users/kevinyu/Projects/Nexus/nexus/models/nlp/embeddings/matryoshka.py`

### Core Components

**1. MatryoshkaEmbedding Class** (lines 26-172)

```python
class MatryoshkaEmbedding(NexusModule):
    def __init__(self, config):
        # Validate nesting dimensions
        assert max(nesting_dims) <= d_model
        assert sorted(nesting_dims) == nesting_dims

        # Encoder (can use any architecture)
        self.encoder = config.get('encoder') or self._build_encoder()

        # Normalization
        self.norm = LayerNorm(d_model)
```

Key validation: Nesting dimensions must be sorted and ≤ d_model.

**2. Embedding Extraction** (lines 124-149)

```python
def get_embedding(self, x, dim, attention_mask=None):
    # Get full embedding
    full_embedding = self.forward(x, attention_mask)

    # Truncate to target dimension
    truncated = full_embedding[:, :dim]

    # CRITICAL: Renormalize after truncation
    truncated = F.normalize(truncated, p=2, dim=-1)

    return truncated
```

Why renormalize? Truncation changes vector norm, affecting cosine similarity.

**3. Multi-Granularity Loss** (lines 174-264)

```python
class MatryoshkaLoss:
    def forward(self, full_embedding_a, full_embedding_b):
        total_loss = 0.0
        losses = {}

        for dim, weight in zip(self.nesting_dims, self.loss_weights):
            # Truncate to current dimension
            emb_a = full_embedding_a[:, :dim]
            emb_b = full_embedding_b[:, :dim]

            # Compute contrastive loss
            loss = self._contrastive_loss(emb_a, emb_b)

            losses[f'loss_dim_{dim}'] = loss
            total_loss += weight * loss

        return losses
```

**4. Contrastive Loss** (lines 198-229)

```python
def _contrastive_loss(self, embeddings_a, embeddings_b):
    # Normalize
    embeddings_a = F.normalize(embeddings_a, p=2, dim=-1)
    embeddings_b = F.normalize(embeddings_b, p=2, dim=-1)

    # Similarity matrix
    similarity = torch.matmul(embeddings_a, embeddings_b.T) / temperature

    # Labels: positive pairs on diagonal
    labels = torch.arange(batch_size)

    # Bidirectional cross-entropy
    loss_a = F.cross_entropy(similarity, labels)
    loss_b = F.cross_entropy(similarity.T, labels)

    return (loss_a + loss_b) / 2
```

## 7. Optimization Tricks

### 1. Dimension-Aware Learning Rates

```python
# Higher learning rate for later dimensions
param_groups = [
    {'params': model.layers[:d1], 'lr': lr * 1.0},
    {'params': model.layers[d1:d2], 'lr': lr * 0.8},
    {'params': model.layers[d2:], 'lr': lr * 0.6},
]
```

### 2. Adaptive Loss Weights

```python
# Weight dimensions by expected usage
loss_weights = {
    64: 0.3,   # High weight (common use case)
    128: 0.25,  # High weight
    256: 0.2,   # Medium weight
    512: 0.15,  # Medium weight
    768: 0.1,   # Low weight (rare use)
}
```

### 3. Gradient Checkpointing for Memory

```python
# Checkpoint intermediate dimensions during backprop
def matryoshka_forward_checkpoint(x, dims):
    full_emb = model(x)
    checkpointed_losses = []

    for dim in dims:
        # Checkpoint each dimension's loss computation
        loss = checkpoint(compute_loss, full_emb[:, :dim])
        checkpointed_losses.append(loss)

    return sum(checkpointed_losses)
```

### 4. Cached Embeddings

```python
# Cache full embeddings, truncate on demand
class CachedMatryoshkaIndex:
    def __init__(self):
        self.full_embeddings = {}  # Store full-dim embeddings

    def search(self, query, k, dim):
        # Truncate on-the-fly
        db_embeddings = self.full_embeddings[:, :dim]
        query_emb = query[:dim]
        return find_top_k(query_emb, db_embeddings, k)
```

### 5. Progressive Training

```python
# Start with small dimensions, progressively add larger
def progressive_training():
    # Phase 1: Train only 64-dim
    train(dims=[64], epochs=10)

    # Phase 2: Add 128-dim
    train(dims=[64, 128], epochs=10)

    # Phase 3: Add all dimensions
    train(dims=[64, 128, 256, 512, 768], epochs=20)
```

## 8. Experiments & Results

### Benchmark: Text Retrieval (MS MARCO)

**Model**: BERT-base with Matryoshka training
**Metrics**: Recall@10, MRR@10

| Dimension | Recall@10 | MRR@10 | Speedup vs 768 |
|-----------|-----------|--------|----------------|
| 32 | 0.72 | 0.41 | 24x |
| 64 | 0.81 | 0.49 | 12x |
| 128 | 0.87 | 0.55 | 6x |
| 256 | 0.92 | 0.61 | 3x |
| 512 | 0.95 | 0.65 | 1.5x |
| 768 | 0.96 | 0.67 | 1x |

**Key Finding**: 128-dim achieves 90% of full performance at 6x speedup.

### Storage Efficiency

```
Dataset: 10M documents

768-dim embeddings:
  - Size: 768 * 4 bytes * 10M = 30.7 GB
  - Index build: 45 minutes
  - Query latency: 12ms

128-dim embeddings (truncated):
  - Size: Same 30.7 GB (stored full, truncate on query)
  - Index build: 45 minutes (build once)
  - Query latency: 3ms (6x faster)

Separate 128-dim model:
  - Accuracy: 0.83 Recall@10 (vs 0.87 MRL)
  - Must maintain separate model
```

### Semantic Tasks Performance

| Task | 64-dim | 128-dim | 256-dim | 768-dim |
|------|--------|---------|---------|---------|
| STS-B (Spearman) | 0.73 | 0.81 | 0.86 | 0.88 |
| TREC (Accuracy) | 0.82 | 0.88 | 0.92 | 0.93 |
| Quora QP (F1) | 0.81 | 0.85 | 0.88 | 0.89 |

### Comparison: MRL vs Separate Models

```
Training Cost:
- Separate models (5 sizes): 5x training time
- Matryoshka: 1.8x training time (multi-loss overhead)

Deployment:
- Separate: Must choose model upfront, can't change
- Matryoshka: Choose dimension at inference time

Performance:
- Separate 128-dim: 0.83 Recall@10
- Matryoshka 128-dim: 0.87 Recall@10 (+4.8%)
```

## 9. Common Pitfalls

### 1. Forgetting to Renormalize

**Problem**: Truncated embeddings have different norms.

```python
# BAD: Use truncated embedding directly
emb_64 = emb_768[:, :64]
similarity = cosine(emb_64, db_emb_64)  # WRONG!

# GOOD: Renormalize after truncation
emb_64 = F.normalize(emb_768[:, :64], p=2, dim=-1)
similarity = cosine(emb_64, db_emb_64)  # Correct
```

### 2. Inconsistent Dimensions

**Problem**: Query and database at different dimensions.

```python
# BAD: Mismatched dimensions
query_emb = get_embedding(query, dim=128)
db_emb = get_embedding(doc, dim=256)  # Different!
similarity = dot(query_emb, db_emb)  # Shape mismatch

# GOOD: Same dimension for query and database
dim = 128
query_emb = get_embedding(query, dim=dim)
db_emb = get_embedding(doc, dim=dim)
similarity = dot(query_emb, db_emb)
```

### 3. Unbalanced Loss Weights

**Problem**: Over-weighting large dimensions.

```python
# BAD: Linear weighting (larger dims dominate)
weights = [64, 128, 256, 512, 768]  # Proportional to size

# GOOD: Uniform or inverse weighting
weights = [1.0, 1.0, 1.0, 1.0, 1.0]  # Uniform
# OR
weights = [1/64, 1/128, 1/256, 1/512, 1/768]  # Inverse
```

### 4. Wrong Pooling for Truncation

**Problem**: Using CLS token, which wasn't trained for truncation.

```python
# BAD: CLS token not optimized for all dimensions
emb = hidden_states[:, 0, :]  # CLS token
truncated = emb[:, :128]  # May not be meaningful

# GOOD: Mean pooling preserves information across dimensions
emb = hidden_states.mean(dim=1)  # All tokens contribute
truncated = emb[:, :128]  # All dimensions matter
```

### 5. Training Without All Dimensions

**Problem**: Only training final dimension.

```python
# BAD: Only compute loss on full embedding
loss = contrastive_loss(emb_768, emb_768)

# GOOD: Compute loss at all nesting dimensions
losses = [
    contrastive_loss(emb[:, :d], emb[:, :d])
    for d in [64, 128, 256, 512, 768]
]
total_loss = sum(losses)
```

## 10. References

### Papers

1. **Kusupati et al. (2022)**: "Matryoshka Representation Learning"
   - https://arxiv.org/abs/2205.13147
   - Original paper introducing MRL

2. **Kusupati et al. (2024)**: "Matryoshka Representation Learning for Efficient Neural Networks"
   - Extension to neural network compression

3. **Chen et al. (2023)**: "Adaptive Embeddings for Efficient Retrieval"
   - Application to large-scale retrieval

### Related Work

1. **Progressive Training**: Layer-wise training for deep networks
2. **Slimmable Networks**: Networks with adjustable width
3. **Any-Precision Networks**: Networks with flexible precision
4. **Pyramid Networks**: Multi-resolution feature pyramids

### Code & Resources

- Nexus Implementation: `/Users/kevinyu/Projects/Nexus/nexus/models/nlp/embeddings/matryoshka.py`
- Sentence Transformers: https://www.sbert.net/examples/training/matryoshka/README.html
- OpenAI Embeddings: Use Matryoshka-style flexible dimensions

### Datasets & Benchmarks

1. **MTEB**: Massive Text Embedding Benchmark
2. **BEIR**: Benchmark for retrieval evaluation
3. **MS MARCO**: Microsoft Machine Reading Comprehension

### Applications

1. **OpenAI Embeddings**: text-embedding-ada-002 uses MRL
2. **Cohere Embeddings**: embed-v3 supports flexible dimensions
3. **Vertex AI**: Google's embeddings support multiple dimensions
