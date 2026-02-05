# Retrieval-Augmented Generation (RAG) Methods

This directory contains comprehensive documentation on advanced RAG techniques that enhance LLM generation with retrieved external knowledge.

## Overview

Retrieval-Augmented Generation (RAG) addresses the fundamental limitations of standalone LLMs:
- **Knowledge cutoff**: Models don't know about recent information
- **Hallucination**: Models generate plausible but incorrect information
- **Domain specificity**: Limited performance on specialized domains
- **Attribution**: Difficulty citing sources

RAG solves these by retrieving relevant documents and conditioning generation on them.

## RAG Landscape

### Evolution of RAG

```
Standard RAG (2020)
    ↓
Self-RAG (2023) ← Adaptive retrieval + self-reflection
    ↓
CRAG (2024) ← Corrective retrieval with quality assessment
    ↓
GraphRAG (2024) ← Knowledge graph-based retrieval
    ↓
RAPTOR (2024) ← Hierarchical tree-based retrieval
```

### Core Components

All RAG systems share these building blocks:

1. **Document Encoder**: Embed documents into dense vectors
2. **Retriever**: Find relevant documents for a query
3. **Context Fusion**: Integrate retrieved docs with query
4. **Generator**: Produce output conditioned on context

## Method Comparison

| Method | Retrieval Strategy | Key Innovation | When to Use |
|--------|-------------------|----------------|-------------|
| **Basic RAG** | Dense retrieval | Foundation | Simple Q&A, knowledge lookup |
| **Self-RAG** | Adaptive | Self-reflection tokens | Quality-critical applications |
| **CRAG** | Corrective | Web search fallback | Unreliable knowledge bases |
| **GraphRAG** | Graph-based | Community summaries | Global sensemaking queries |
| **RAPTOR** | Hierarchical tree | Multi-level abstraction | Long documents, varying query scope |
| **Adaptive RAG** | Query-dependent | Dynamic strategy | Mixed workloads |

---

## RAG Methods

### 1. Standard RAG Module

**Core Idea**: Encode documents, retrieve top-k, attend to retrieved context, generate.

**Pipeline**:
```
Query → Encode → Retrieve top-k docs → Cross-attention fusion → Generate
```

**Strengths**:
- Simple, interpretable
- Works well for factoid Q&A
- Efficient with pre-built indices

**Limitations**:
- No quality assessment of retrievals
- Fixed retrieval strategy
- Single-hop reasoning only

**Reference**: [RAG: Retrieval-Augmented Generation (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)

---

### 2. Self-RAG: Learning to Retrieve, Generate, and Critique

**Core Idea**: Model learns to:
- **Decide** when to retrieve (not every query needs retrieval)
- **Critique** retrieval quality via reflection tokens
- **Self-correct** using relevance/support assessments

**Reflection Tokens**:
- `[Retrieve]`: Should I retrieve passages?
- `[IsRelevant]`: Is this passage relevant to my query?
- `[IsSupported]`: Is my generation supported by the passage?
- `[IsUseful]`: Is my response useful overall?

**When to Use**:
- Quality-critical applications (medical, legal)
- When retrieval is expensive (reduce unnecessary calls)
- Need for interpretable decision-making

**Strengths**:
- Adaptive retrieval (only when needed)
- Self-assessment improves reliability
- Controllable via reflection token thresholds

**Limitations**:
- Requires training on reflection labels
- More complex than standard RAG

**Reference**: [Self-RAG (Asai et al., 2023)](https://arxiv.org/abs/2310.11511)

---

### 3. CRAG: Corrective Retrieval Augmented Generation

**Core Idea**: Evaluate retrieval quality, then take corrective action:
- **Correct**: High-quality retrievals → filter and use
- **Incorrect**: Poor retrievals → fallback to web search
- **Ambiguous**: Mixed quality → combine filtered docs + web search

**Pipeline**:
```
Retrieve → Evaluate confidence → {Correct, Incorrect, Ambiguous}
                                      ↓           ↓            ↓
                                   Filter     Web Search   Combine
                                      ↓           ↓            ↓
                                   ──────────  Generate  ───────────
```

**Key Components**:
1. **Retrieval Evaluator**: Scores relevance (Correct/Ambiguous/Incorrect)
2. **Document Filter**: Decompose → Score strips → Recompose
3. **Web Search Fallback**: When local retrieval fails

**When to Use**:
- Unreliable or incomplete knowledge bases
- Need for factual accuracy (journalism, fact-checking)
- Combining internal KB + external search

**Strengths**:
- Robust to retrieval failures
- Decompose-recompose removes irrelevant content
- Combines multiple knowledge sources

**Limitations**:
- Higher latency (evaluation + potential web search)
- Requires web search API

**Reference**: [CRAG (Yan et al., 2024)](https://arxiv.org/abs/2401.15884)

---

### 4. GraphRAG: Graph-Based Retrieval Augmented Generation

**Core Idea**: Build knowledge graph from documents, detect communities, pre-summarize communities, retrieve summaries.

**Pipeline**:
```
Documents → Extract entities/relations → Build KG → Community detection
                                                           ↓
Query ← Retrieve relevant communities ← Summarize communities
  ↓
Generate
```

**Key Innovations**:
- **Entity extraction**: Identify entities and relationships
- **Community detection**: Hierarchical clustering (Leiden algorithm)
- **Community summaries**: Pre-compute summaries for each cluster
- **Global queries**: Answer questions requiring synthesis across many docs

**When to Use**:
- **Global sensemaking**: "What are the main themes?" "Summarize the dataset"
- Large document collections (1000s of documents)
- Queries requiring multi-hop reasoning over entities

**Strengths**:
- Handles global queries (not just local facts)
- Hierarchical communities enable multi-scale reasoning
- Structured knowledge representation

**Limitations**:
- Expensive graph construction
- Requires entity extraction quality
- Best for static document collections

**Reference**: [GraphRAG (Edge et al., 2024)](https://arxiv.org/abs/2404.16130)

---

### 5. RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval

**Core Idea**: Build hierarchical tree of summaries by recursive clustering and abstraction. Retrieve from any level (leaves = details, root = high-level summary).

**Pipeline**:
```
Text chunks (leaves)
    ↓ Cluster
Level 1 summaries
    ↓ Cluster
Level 2 summaries
    ↓ Cluster
Root summary

Query → Retrieve from all levels → Combine → Generate
```

**Key Features**:
- **Multi-level retrieval**: Get both details and abstractions
- **Recursive summarization**: Bottom-up tree construction
- **Soft clustering**: Chunks can belong to multiple clusters

**When to Use**:
- Long documents (books, reports, codebases)
- Queries with varying scope (detailed vs high-level)
- Need for both fine-grained and abstract information

**Strengths**:
- Retrieves at optimal abstraction level
- Scalable to very long documents
- Flexible retrieval (any tree level)

**Limitations**:
- Expensive tree construction
- Summarization quality critical
- Optimal for offline indexing

**Reference**: [RAPTOR (Sarthi et al., 2024)](https://arxiv.org/abs/2401.18059)

---

### 6. Adaptive RAG

**Core Idea**: Dynamically select retrieval strategy based on query complexity.

**Strategies**:
- **Simple queries**: No retrieval (LLM knowledge sufficient)
- **Factoid queries**: Single-hop RAG
- **Multi-hop queries**: Iterative RAG
- **Complex queries**: GraphRAG or RAPTOR

**Decision Logic**:
```python
def select_strategy(query):
    complexity = classify_query(query)
    if complexity == "simple":
        return direct_generation()
    elif complexity == "factoid":
        return standard_rag()
    elif complexity == "multi_hop":
        return iterative_rag()
    else:  # complex
        return graph_rag()
```

**When to Use**: Production systems with diverse query types

---

## Component Deep Dives

### Document Encoder

Transforms text into dense embeddings:

```python
class DocumentEncoder(NexusModule):
    def __init__(self, config):
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = TransformerEncoder(num_layers=4)

    def forward(self, input_ids):
        embeddings = self.token_embedding(input_ids)
        hidden_states = self.transformer(embeddings)
        return hidden_states.mean(dim=1)  # Pool to single vector
```

**Optimization**:
- Use pre-trained encoders (BERT, RoBERTa)
- Late interaction (ColBERT) for better recall
- Matryoshka embeddings for flexible dimensions

### Retriever

Finds top-k relevant documents:

```python
class EfficientRetriever(NexusModule):
    def __init__(self, config):
        self.index = AnnoyIndex(hidden_size, metric='dot')

    def build_index(self, doc_embeddings):
        for i, emb in enumerate(doc_embeddings):
            self.index.add_item(i, emb)
        self.index.build(num_trees=10)

    def retrieve(self, query_embedding, k=5):
        indices, distances = self.index.get_nns_by_vector(
            query_embedding, k, include_distances=True
        )
        return indices, distances
```

**Optimization**:
- Approximate nearest neighbor (FAISS, Annoy, ScaNN)
- Quantization for memory efficiency
- Caching for repeated queries

### Context Fusion

Integrates query + retrieved documents:

**Cross-Attention**:
```python
class CrossAttentionFusion(NexusModule):
    def forward(self, query_states, doc_states):
        # Query attends to documents
        fused = self.attention(
            query_states,
            key_value_states=doc_states
        )
        return self.norm(fused + query_states)
```

**Concatenation**:
```python
# Simple but effective
context = torch.cat([query_embedding, doc_embeddings.mean(dim=0)])
```

## Performance Guidelines

### Latency

| Method | Index Build | Query Time | Scalability |
|--------|-------------|------------|-------------|
| Basic RAG | Fast (1x) | Fast (1x) | Excellent |
| Self-RAG | Fast (1x) | Moderate (1.5x) | Good |
| CRAG | Fast (1x) | Slow (2-3x) | Good |
| GraphRAG | Very slow (100x) | Fast (1x) | Moderate |
| RAPTOR | Slow (10x) | Fast (1.2x) | Good |

### Quality

| Method | Factoid Q&A | Multi-hop | Global Queries | Long Docs |
|--------|-------------|-----------|----------------|-----------|
| Basic RAG | ⭐⭐⭐⭐ | ⭐⭐ | ⭐ | ⭐⭐ |
| Self-RAG | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| CRAG | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| GraphRAG | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| RAPTOR | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## Common Pitfalls

### 1. Retrieval Precision

**Problem**: Top-k documents aren't relevant

**Solutions**:
- Fine-tune encoders on domain data
- Use hybrid retrieval (sparse + dense)
- Increase k and rerank
- Implement CRAG-style filtering

### 2. Context Length Overflow

**Problem**: Retrieved documents exceed LLM context window

**Solutions**:
- Summarize documents before feeding to LLM
- Use extraction-then-abstraction (RAPTOR)
- Sliding window with overlap
- Hierarchical retrieval (coarse-to-fine)

### 3. Hallucination Despite Retrieval

**Problem**: LLM ignores retrieved context

**Solutions**:
- Use Self-RAG reflection tokens
- Instruction tuning on RAG tasks
- Constrained generation (force attribution)
- Post-hoc verification

### 4. Expensive Retrieval

**Problem**: Retrieval dominates latency

**Solutions**:
- Cache frequent queries
- Adaptive retrieval (Self-RAG)
- Approximate nearest neighbor search
- Batch retrieval for multiple queries

## Best Practices

### Index Construction

```python
# 1. Chunk documents appropriately
chunks = split_documents(docs, chunk_size=512, overlap=50)

# 2. Encode in batches
embeddings = []
for batch in batched(chunks, batch_size=32):
    emb = encoder(batch)
    embeddings.append(emb)

# 3. Build efficient index
retriever.build_index(embeddings)
retriever.save_index("index.ann")
```

### Query Processing

```python
# 1. Encode query
query_emb = encoder(query)

# 2. Retrieve with margin
k_retrieved = k_desired * 2  # Retrieve more for reranking

# 3. Rerank by cross-encoder
docs, scores = retriever.retrieve(query_emb, k_retrieved)
reranked = cross_encoder.rerank(query, docs)
top_k = reranked[:k_desired]

# 4. Fuse and generate
context = fuse(query_emb, top_k)
output = generator(context)
```

## References

### Foundational Papers
1. **RAG: Retrieval-Augmented Generation** (Lewis et al., 2020) - https://arxiv.org/abs/2005.11401
2. **REALM: Retrieval-Augmented Language Model Pre-Training** (Guu et al., 2020) - https://arxiv.org/abs/2002.08909

### Advanced Methods
3. **Self-RAG** (Asai et al., 2023) - https://arxiv.org/abs/2310.11511
4. **CRAG** (Yan et al., 2024) - https://arxiv.org/abs/2401.15884
5. **GraphRAG** (Edge et al., 2024) - https://arxiv.org/abs/2404.16130
6. **RAPTOR** (Sarthi et al., 2024) - https://arxiv.org/abs/2401.18059

### Retrieval Techniques
7. **Dense Passage Retrieval** (Karpukhin et al., 2020) - https://arxiv.org/abs/2004.04906
8. **ColBERT** (Khattab & Zaharia, 2020) - https://arxiv.org/abs/2004.12832

## Documentation Files

- [RAG Module](./rag_module.md) - Standard RAG implementation
- [Document Encoder](./document_encoder.md) - Encoding documents to embeddings
- [Retriever](./retriever.md) - Efficient similarity search
- [Self-RAG](./self_rag.md) - Adaptive retrieval with self-reflection
- [CRAG](./crag.md) - Corrective retrieval with quality assessment
- [GraphRAG](./graph_rag.md) - Knowledge graph-based retrieval
- [RAPTOR](./raptor.md) - Hierarchical tree-based retrieval
- [Adaptive RAG](./adaptive_rag.md) - Query-dependent strategy selection
