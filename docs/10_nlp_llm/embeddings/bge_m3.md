# BGE-M3: Multi-Functionality Multi-Granularity Multi-Lingual Embeddings

## 1. Overview & Motivation

BGE-M3 unifies three retrieval paradigms—dense, sparse, and multi-vector—in a single model, achieving state-of-the-art performance across 100+ languages. The "M3" refers to Multi-Functionality (3 retrieval modes), Multi-Granularity (token to passage level), and Multi-Lingual support.

### Problem Statement

Traditional embedding models support only one retrieval mode:
- **Dense**: Good for semantic matching, but misses lexical overlap
- **Sparse** (BM25): Good for exact matches, but misses semantics
- **Multi-vector** (ColBERT): Best quality, but expensive

Users must choose one approach or maintain multiple models, each with tradeoffs.

### Solution

Train a single model with three output heads:
1. **Dense Head**: Traditional dense embeddings (CLS pooling)
2. **Sparse Head**: Learned sparse representations (SPLADE-style)
3. **ColBERT Head**: Token-level embeddings for late interaction

Enable hybrid retrieval combining all three modes with learned or manual weights.

### Key Applications

1. **Universal Retrieval**: Single model for all retrieval scenarios
2. **Multilingual Search**: Cross-lingual information retrieval
3. **Hybrid Search**: Combine semantic and lexical matching
4. **Quality-Critical Tasks**: Maximum accuracy from multi-mode fusion
5. **Research & Experimentation**: Compare retrieval paradigms fairly

## 2. Theoretical Background

### Three Retrieval Paradigms

**1. Dense Retrieval**
- Single vector per document/query
- Similarity: dot product or cosine
- Fast but may miss exact matches

**2. Sparse Retrieval**
- Sparse vector over vocabulary (like TF-IDF)
- Learned importance scores (not just term frequency)
- Captures lexical matching

**3. Multi-Vector Retrieval**
- Vector per token (query and document)
- Late interaction: MaxSim scoring
- Expensive but highest quality

### Unified Architecture

```
Input Text
    ↓
Transformer Encoder
    ↓
Hidden States (batch, seq_len, hidden)
    ↓
    ├─→ Dense Head → CLS embedding
    ├─→ Sparse Head → Vocabulary weights
    └─→ ColBERT Head → Token embeddings
```

### Self-Knowledge Distillation

BGE-M3 uses self-distillation where stronger modes teach weaker ones:
1. Train all three heads jointly
2. Use ColBERT scores as soft labels for dense/sparse
3. Improves consistency across modes

## 3. Mathematical Formulation

### Dense Similarity

```
s_dense(q, d) = <e_q^dense, e_d^dense>
```

where e_q^dense, e_d^dense ∈ R^d are normalized dense embeddings.

### Sparse Similarity

Sparse embeddings: e^sparse ∈ R^|V| (vocabulary size)

```
e_i^sparse = max_{t=1}^{T_i} log(1 + ReLU(w_t))
```

where w_t ∈ R^|V| are token-level vocabulary logits.

Similarity (sparse dot product):
```
s_sparse(q, d) = Σ_{v∈V} e_q^sparse[v] · e_d^sparse[v]
```

### Multi-Vector Similarity (MaxSim)

Token embeddings: e^colbert ∈ R^{T×d'}

```
s_colbert(q, d) = Σ_{i=1}^{T_q} max_{j=1}^{T_d} <e_q^colbert[i], e_d^colbert[j]>
```

For each query token, find most similar document token, then sum.

### Hybrid Similarity

```
s_hybrid(q, d) = w_1·s_dense(q,d) + w_2·s_sparse(q,d) + w_3·s_colbert(q,d)
```

where w_1, w_2, w_3 are mode weights (typically w_1=0.4, w_2=0.3, w_3=0.3).

### Training Objective

Multi-task learning with contrastive losses:

```
L = L_dense + L_sparse + L_colbert + L_distill
```

where:
- L_dense: Dense embedding contrastive loss
- L_sparse: Sparse embedding contrastive loss
- L_colbert: ColBERT MaxSim contrastive loss
- L_distill: Self-distillation from ColBERT to dense/sparse

## 4. High-Level Intuition

Think of BGE-M3 as a "Swiss Army knife" for retrieval:

### Dense Mode (Fast Search)
Like using a GPS to find "restaurants nearby":
- Understands "restaurants" semantically (includes cafes, eateries)
- Fast approximate search
- May miss if you asked for exact term "restaurant"

### Sparse Mode (Exact Matches)
Like Ctrl+F search in a document:
- Finds exact term matches
- Good for names, IDs, specific phrases
- Misses synonyms and paraphrases

### ColBERT Mode (Maximum Quality)
Like reading and comparing sentence-by-sentence:
- Token-level alignment
- Handles both semantic and lexical
- Slower but most accurate

### Hybrid (Best of All)
Like using GPS + map + asking locals:
- Dense finds general area (semantic)
- Sparse confirms key terms present (lexical)
- ColBERT validates detailed match (fine-grained)
- Combined confidence from all three

### Example Query

Query: "machine learning optimization algorithms"

**Dense**: Finds documents about:
- ML algorithms, optimization techniques, gradient descent
- (Semantic understanding)

**Sparse**: Finds documents containing:
- Exact terms: "machine", "learning", "optimization", "algorithms"
- (Lexical matching)

**ColBERT**: Finds documents where:
- "machine learning" aligns with "ML"
- "optimization" aligns with "optimize"
- "algorithms" aligns with specific algorithm names
- (Token-level fine-grained matching)

**Hybrid**: Combines all three for best results

## 5. Implementation Details

### Architecture Components

```python
class BGEM3Embedder(NexusModule):
    def __init__(self, config, encoder=None):
        # Transformer encoder (BERT-style)
        self.encoder = encoder or self._build_transformer()

        # Three output heads
        self.dense_head = DenseEmbeddingHead(hidden_size, dense_dim)
        self.sparse_head = SparseEmbeddingHead(hidden_size, vocab_size)
        self.colbert_head = ColBERTEmbeddingHead(hidden_size, colbert_dim)
```

### Dense Head

```python
class DenseEmbeddingHead(nn.Module):
    def __init__(self, hidden_size, output_dim):
        self.linear = nn.Linear(hidden_size, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, hidden_states):
        # Extract CLS token
        cls_embedding = hidden_states[:, 0, :]

        # Project and normalize
        dense_emb = self.linear(cls_embedding)
        dense_emb = self.layer_norm(dense_emb)
        dense_emb = F.normalize(dense_emb, p=2, dim=-1)

        return dense_emb
```

### Sparse Head (SPLADE-style)

```python
class SparseEmbeddingHead(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden_states, attention_mask, top_k=100):
        # Project to vocabulary space
        logits = self.linear(hidden_states)  # (batch, seq, vocab)

        # SPLADE activation: log(1 + ReLU(x))
        sparse_weights = torch.log1p(F.relu(logits))

        # Apply mask
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1)
            sparse_weights = sparse_weights * mask_expanded

        # Max pool over sequence
        sparse_weights, _ = torch.max(sparse_weights, dim=1)

        # Keep top-k activations (efficiency)
        top_values, top_indices = torch.topk(sparse_weights, k=top_k)

        return top_indices, top_values
```

### ColBERT Head

```python
class ColBERTEmbeddingHead(nn.Module):
    def __init__(self, hidden_size, output_dim):
        self.linear = nn.Linear(hidden_size, output_dim)

    def forward(self, hidden_states, attention_mask=None):
        # Project each token
        token_embeddings = self.linear(hidden_states)

        # Normalize each token
        token_embeddings = F.normalize(token_embeddings, p=2, dim=-1)

        # Apply mask
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1)
            token_embeddings = token_embeddings * mask_expanded

        return token_embeddings
```

### Hybrid Similarity

```python
def compute_similarity(self, query_output, doc_output, mode="hybrid", weights=None):
    if weights is None:
        weights = {"dense": 0.4, "sparse": 0.3, "colbert": 0.3}

    scores = []

    # Dense similarity
    if "dense_embeddings" in query_output:
        dense_sim = (query_output['dense_embeddings'] *
                    doc_output['dense_embeddings']).sum(dim=-1)
        scores.append(weights["dense"] * dense_sim)

    # Sparse similarity
    if "sparse_indices" in query_output:
        sparse_sim = self._sparse_dot_product(
            query_output['sparse_indices'],
            query_output['sparse_values'],
            doc_output['sparse_indices'],
            doc_output['sparse_values']
        )
        scores.append(weights["sparse"] * sparse_sim)

    # ColBERT similarity (MaxSim)
    if "colbert_embeddings" in query_output:
        colbert_sim = self._maxsim_score(
            query_output['colbert_embeddings'],
            doc_output['colbert_embeddings']
        )
        scores.append(weights["colbert"] * colbert_sim)

    return sum(scores)
```

### MaxSim Computation

```python
def _maxsim_score(self, q_emb, d_emb):
    # q_emb: (batch, q_len, dim)
    # d_emb: (batch, d_len, dim)

    # Compute pairwise similarities
    similarities = torch.bmm(q_emb, d_emb.transpose(1, 2))
    # (batch, q_len, d_len)

    # For each query token, take max over document tokens
    maxsim_scores = similarities.max(dim=-1)[0]  # (batch, q_len)

    # Sum over query tokens
    return maxsim_scores.sum(dim=-1)  # (batch,)
```

## 6. Code Walkthrough

Reference: `Nexus/nexus/models/nlp/embeddings/bge_m3.py`

### Forward Pass

```python
def forward(self, input_ids, attention_mask=None,
            return_dense=True, return_sparse=True, return_colbert=True):
    # 1. Encode input
    hidden_states = self.encoder(input_ids)  # (batch, seq, hidden)

    outputs = {}

    # 2. Dense embeddings
    if return_dense and self.config.enable_dense:
        dense_emb = self.dense_head(hidden_states)
        outputs['dense_embeddings'] = dense_emb

    # 3. Sparse embeddings
    if return_sparse and self.config.enable_sparse:
        sparse_indices, sparse_values = self.sparse_head(
            hidden_states, attention_mask, top_k=self.config.sparse_top_k
        )
        outputs['sparse_indices'] = sparse_indices
        outputs['sparse_values'] = sparse_values

    # 4. ColBERT embeddings
    if return_colbert and self.config.enable_colbert:
        colbert_emb = self.colbert_head(hidden_states, attention_mask)
        outputs['colbert_embeddings'] = colbert_emb

    return outputs
```

### Key Components

1. **BGEM3Config** (lines 27-54): Configuration for all modes
2. **Dense Head** (lines 57-84): CLS-based dense embeddings
3. **Sparse Head** (lines 87-134): SPLADE-style sparse vectors
4. **ColBERT Head** (lines 137-172): Token-level embeddings
5. **BGEM3Embedder** (lines 175-258): Main model class
6. **Similarity Computation** (lines 260-343): Multi-mode scoring

## 7. Optimization Tricks

### 1. Sparse Vector Quantization

```python
# Store sparse vectors efficiently
class QuantizedSparseVector:
    def __init__(self, indices, values):
        # Quantize values to uint8 (256 levels)
        self.indices = indices  # int32
        self.values = (values * 255).byte()  # uint8

    def dot(self, other):
        # Dequantize and compute
        v1 = self.values.float() / 255.0
        v2 = other.values.float() / 255.0
        return (v1 * v2).sum()
```

### 2. Two-Stage Retrieval

```python
# Stage 1: Dense retrieval (fast, recall-focused)
candidates = dense_index.search(query_dense, k=1000)

# Stage 2: Rerank with ColBERT (accurate)
reranked = []
for doc_id in candidates:
    score = colbert_score(query_colbert, doc_colbert[doc_id])
    reranked.append((doc_id, score))

return sorted(reranked, key=lambda x: x[1], reverse=True)[:k]
```

### 3. Adaptive Mode Selection

```python
# Choose mode based on query characteristics
def select_mode(query):
    if has_named_entities(query):
        return "sparse"  # Exact matching important
    elif is_short(query):
        return "colbert"  # Precision matters
    else:
        return "dense"  # Efficient semantic search
```

### 4. Index Compression

```python
# Use different indices for different modes
class HybridIndex:
    def __init__(self):
        self.dense_index = HNSWIndex()  # Fast ANN
        self.sparse_index = InvertedIndex()  # Sparse lookup
        self.colbert_index = None  # On-demand loading

    def search(self, query, mode="hybrid"):
        if mode == "dense":
            return self.dense_index.search(query.dense)
        elif mode == "hybrid":
            # Retrieve with dense, rerank with sparse
            candidates = self.dense_index.search(query.dense, k=100)
            return self.sparse_rerank(query.sparse, candidates)
```

### 5. Batch Processing

```python
# Process all three modes in parallel
def batch_embed(texts, batch_size=32):
    dense_embs, sparse_embs, colbert_embs = [], [], []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        outputs = model(batch, return_all=True)

        dense_embs.append(outputs['dense_embeddings'])
        sparse_embs.append(outputs['sparse_embeddings'])
        colbert_embs.append(outputs['colbert_embeddings'])

    return {
        'dense': torch.cat(dense_embs),
        'sparse': concatenate_sparse(sparse_embs),
        'colbert': torch.cat(colbert_embs)
    }
```

## 8. Experiments & Results

### BEIR Benchmark (14 datasets)

**Model**: BGE-M3 vs single-mode baselines

| Model | NDCG@10 | Recall@100 | Latency (ms) |
|-------|---------|------------|--------------|
| Dense (BGE-base) | 0.531 | 0.842 | 12 |
| Sparse (SPLADE++) | 0.498 | 0.819 | 45 |
| ColBERT | 0.562 | 0.871 | 180 |
| BGE-M3 (dense) | 0.538 | 0.849 | 15 |
| BGE-M3 (sparse) | 0.521 | 0.836 | 48 |
| BGE-M3 (colbert) | 0.571 | 0.878 | 185 |
| **BGE-M3 (hybrid)** | **0.589** | **0.891** | 190 |

**Key Finding**: Hybrid mode outperforms any single mode significantly.

### Multilingual Performance (MIRACL)

**18 languages**, NDCG@10:

| Model | Avg | English | Chinese | Arabic | Hindi |
|-------|-----|---------|---------|--------|-------|
| mBERT | 0.412 | 0.489 | 0.441 | 0.387 | 0.362 |
| XLM-R | 0.438 | 0.512 | 0.468 | 0.401 | 0.389 |
| mContriever | 0.456 | 0.531 | 0.492 | 0.418 | 0.407 |
| **BGE-M3** | **0.523** | **0.598** | **0.562** | **0.489** | **0.471** |

### Mode Contribution Analysis

Query type → Best mode:

```
Named entity queries:
  - Dense: 0.52
  - Sparse: 0.61  ← Best
  - ColBERT: 0.58

Semantic queries:
  - Dense: 0.67   ← Best
  - Sparse: 0.51
  - ColBERT: 0.69 (slightly better)

Long queries (>15 words):
  - Dense: 0.58
  - Sparse: 0.54
  - ColBERT: 0.71 ← Best

Hybrid (all types): 0.74 ← Consistently best
```

### Storage & Latency

```
Per-document storage (1M docs):

Dense only: 1024 * 4 bytes * 1M = 4 GB
Sparse only: ~100 * 4 bytes * 1M = 0.4 GB (avg 100 active dims)
ColBERT only: 128 * 128 * 4 bytes * 1M = 65 GB (avg 128 tokens)

All three: 4 + 0.4 + 65 = 69.4 GB

Latency (per query):
Dense: 12ms
Sparse: 45ms
ColBERT: 180ms
Hybrid (sequential): 237ms
Hybrid (parallel): 185ms (max of three)
```

## 9. Common Pitfalls

### 1. Inconsistent Normalization

**Problem**: Forgetting to normalize dense embeddings.

```python
# BAD: Unnormalized dense embeddings
dense_emb = self.dense_head(hidden_states)
similarity = dense_emb @ doc_emb.T  # Magnitudes affect score

# GOOD: Always normalize
dense_emb = F.normalize(self.dense_head(hidden_states), p=2, dim=-1)
similarity = dense_emb @ doc_emb.T  # Pure cosine similarity
```

### 2. Sparse Vector Memory Explosion

**Problem**: Storing full vocabulary-size sparse vectors.

```python
# BAD: Store full sparse vectors (30K+ dimensions)
sparse_emb = torch.zeros(batch_size, vocab_size)

# GOOD: Store only top-k indices and values
top_k = 100
sparse_indices, sparse_values = torch.topk(sparse_logits, k=top_k)
```

### 3. Incorrect MaxSim Implementation

**Problem**: Taking max over wrong dimension.

```python
# BAD: Max over query tokens (incorrect)
maxsim = (q_emb @ d_emb.T).max(dim=1).sum(dim=1)

# GOOD: Max over doc tokens, sum over query tokens
similarities = q_emb @ d_emb.T  # (q_len, d_len)
maxsim = similarities.max(dim=1)[0].sum()  # Max over d_len, sum over q_len
```

### 4. Unbalanced Hybrid Weights

**Problem**: One mode dominates others.

```python
# BAD: Dense mode dominates
weights = {"dense": 0.8, "sparse": 0.1, "colbert": 0.1}

# GOOD: Balanced or task-specific weights
weights = {"dense": 0.4, "sparse": 0.3, "colbert": 0.3}

# BETTER: Learn weights from validation data
weights = learn_optimal_weights(val_data)
```

### 5. Missing Attention Masking

**Problem**: Including padding tokens in embeddings.

```python
# BAD: Ignore padding in sparse/ColBERT
sparse_weights = torch.log1p(F.relu(logits))
# Padding tokens contribute!

# GOOD: Apply attention mask
if attention_mask is not None:
    mask_expanded = attention_mask.unsqueeze(-1)
    sparse_weights = sparse_weights * mask_expanded
```

## 10. References

### Papers

1. **Chen et al. (2024)**: "BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings"
   - https://arxiv.org/abs/2402.03216
   - Original BGE-M3 paper

2. **Formal et al. (2021)**: "SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking"
   - https://arxiv.org/abs/2107.05720
   - Sparse retrieval foundation

3. **Khattab & Zaharia (2020)**: "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT"
   - https://arxiv.org/abs/2004.12832
   - ColBERT multi-vector retrieval

4. **Xiao et al. (2023)**: "C-Pack: Packaged Resources For General Chinese Embeddings"
   - BGE-base foundation model

### Code & Libraries

- Nexus Implementation: `Nexus/nexus/models/nlp/embeddings/bge_m3.py`
- Official BGE: https://github.com/FlagOpen/FlagEmbedding
- SPLADE: https://github.com/naver/splade
- ColBERT: https://github.com/stanford-futuredata/ColBERT

### Benchmarks

1. **BEIR**: https://github.com/beir-cellar/beir
   - 14 diverse retrieval tasks
2. **MIRACL**: https://github.com/project-miracl/miracl
   - Multilingual retrieval benchmark
3. **MTEB**: https://github.com/embeddings-benchmark/mteb
   - Massive embedding benchmark

### Related Work

1. **Dense Retrieval**: DPR, ANCE, SimCSE
2. **Sparse Retrieval**: DeepCT, DocT5Query, SPLADE
3. **Multi-Vector**: ColBERT, Poly-encoders
4. **Hybrid**: uniCOIL, COIL, SPLADEv2

### Applications

- **Elasticsearch**: Hybrid search with multiple retrievers
- **Weaviate**: Multi-vector search support
- **Vespa**: Hybrid ranking with multiple signals
- **Milvus**: Multi-index hybrid search
