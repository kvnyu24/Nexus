# Embeddings

Text embeddings convert sequences (words, sentences, documents) into dense vector representations that capture semantic meaning. These vectors enable similarity search, clustering, classification, and retrieval tasks.

## Overview

Modern embedding models go beyond simple word vectors to capture contextual meaning, multi-lingual semantics, and even multiple retrieval modalities in a single representation.

## When to Use Embeddings

Use embeddings when you need:

1. **Semantic Search**: Find documents similar to a query
2. **Clustering**: Group similar texts together
3. **Classification**: Use embeddings as features for classifiers
4. **Recommendation**: Find similar items based on descriptions
5. **Retrieval-Augmented Generation (RAG)**: Retrieve relevant context for LLMs

## Approaches

### 1. Matryoshka Representation Learning (MRL)

Creates nested embeddings where shorter truncations maintain good performance, enabling flexible accuracy-efficiency tradeoffs.

**Strengths:**
- Single model supports multiple dimensions
- No retraining for different sizes
- Graceful performance degradation
- Storage and compute efficiency

**Weaknesses:**
- Requires special training procedure
- Slightly lower performance at very low dimensions
- Need to choose dimensions at deployment

**Use when:** You need flexible embedding sizes for different deployment scenarios (edge devices, servers) or want to optimize storage/compute costs.

See: [matryoshka_representation_learning.md](./matryoshka_representation_learning.md)

### 2. BGE-M3

Unified model supporting three retrieval modes: dense, sparse (learned), and multi-vector (ColBERT-style).

**Strengths:**
- Multi-functionality (3 retrieval modes in one)
- Multi-lingual (100+ languages)
- State-of-the-art retrieval performance
- Hybrid search capabilities

**Weaknesses:**
- Larger model size
- Higher computational cost
- More complex deployment
- Requires index support for all modes

**Use when:** You need maximum retrieval quality, multi-lingual support, or want to experiment with hybrid retrieval strategies.

See: [bge_m3.md](./bge_m3.md)

## Comparison Matrix

| Feature | Matryoshka | BGE-M3 |
|---------|-----------|--------|
| Flexibility | Variable dimensions | Multiple retrieval modes |
| Performance | Good (single mode) | Excellent (hybrid) |
| Efficiency | High (truncatable) | Medium (multi-head) |
| Training Complexity | Medium | High |
| Deployment Complexity | Low | High |
| Multi-lingual | Depends on base model | Native (100+ langs) |
| Best Use Case | Efficiency-focused | Quality-focused |

## Best Practices

### Matryoshka Embeddings

1. **Dimension Selection**: Test multiple dimensions on your dataset to find the optimal tradeoff
2. **Training Strategy**: Use uniform loss weighting initially, then adjust based on target dimensions
3. **Normalization**: Always normalize embeddings before similarity computation
4. **Benchmarking**: Evaluate at all nesting dimensions during development

### BGE-M3 Embeddings

1. **Mode Selection**: Start with dense-only, add sparse/ColBERT for difficult queries
2. **Weight Tuning**: Adjust hybrid weights (dense/sparse/colbert) based on your data
3. **Index Design**: Use specialized indices (HNSW for dense, inverted index for sparse)
4. **Query Analysis**: Route queries to appropriate modes based on characteristics

## General Guidelines

1. **Batch Processing**: Process embeddings in batches for efficiency
2. **Caching**: Cache embeddings for frequently accessed documents
3. **Normalization**: Normalize embeddings for cosine similarity (dot product with normalized vectors)
4. **Dimensionality**: Higher dimensions aren't always better - validate on your task
5. **Fine-tuning**: Fine-tune on domain-specific data when possible

## Embedding Quality Metrics

- **Retrieval Accuracy**: Recall@k, MRR, NDCG
- **Clustering Quality**: Silhouette score, Davies-Bouldin index
- **Semantic Similarity**: Correlation with human judgments
- **Cross-lingual Transfer**: Performance on multi-lingual tasks

## Common Pitfalls

1. **Over-dimensioning**: Using unnecessarily large embeddings
2. **No Fine-tuning**: Generic embeddings may underperform on domain-specific tasks
3. **Ignoring Normalization**: Forgetting to normalize before similarity computation
4. **Batch Size Mismatch**: Training/inference batch size differences affecting quality
5. **Query-Document Asymmetry**: Using same encoder for queries and documents when asymmetric may be better

## Resources

- Sentence Transformers: [https://www.sbert.net/](https://www.sbert.net/)
- MTEB Leaderboard: [https://huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- BGE Models: [https://github.com/FlagOpen/FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)
- Matryoshka Paper: [https://arxiv.org/abs/2205.13147](https://arxiv.org/abs/2205.13147)

## Example Use Cases

1. **Semantic Search Engine**: Build Google-like search over documents
2. **Question Answering**: Retrieve relevant passages for RAG systems
3. **Duplicate Detection**: Find near-duplicate content
4. **Content Recommendation**: Recommend similar articles/products
5. **Multi-lingual Search**: Cross-lingual information retrieval
