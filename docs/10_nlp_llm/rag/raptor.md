# RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval

## Overview

RAPTOR builds a hierarchical tree of document summaries through recursive clustering and summarization. At query time, retrieval occurs at any level, allowing access to both fine-grained details (leaves) and high-level abstractions (root).

**Key Idea**: Different queries require different levels of abstraction. "What's the book about?" needs high-level summaries, while "What did character X say on page 42?" needs leaf-level details.

## Pipeline

Reference: `/Users/kevinyu/Projects/Nexus/nexus/models/nlp/rag/raptor.py`

```
Text Chunks (Leaves)
       │
       ▼
   Cluster by similarity
       │
       ▼
Level 1: Cluster Summaries
       │
       ▼
   Cluster again
       │
       ▼
Level 2: Higher-level Summaries
       │
       ▼
   Cluster again
       │
       ▼
Root: Global Summary

Query → Retrieve from ALL levels → Combine → Generate
```

## Key Components

### 1. Text Clusterer

Soft clustering of text chunks:

```python
class TextClusterer(NexusModule):
    def forward(self, chunk_embeddings):
        """Cluster chunks by semantic similarity"""

        # Project into clustering space
        projected = self.projection(chunk_embeddings)

        # Compute similarities to learned centroids
        similarities = cosine_similarity(projected, self.centroids)

        # Predict number of clusters
        num_clusters = self.num_cluster_predictor(chunk_embeddings.mean(dim=0))

        # Soft assignment via softmax
        assignments = F.softmax(similarities / temperature, dim=-1)

        # Compute cluster embeddings
        cluster_embeddings = torch.matmul(
            assignments.t(), chunk_embeddings
        ) / assignments.sum(dim=0).unsqueeze(-1)

        return {
            "assignments": assignments,
            "cluster_embeddings": cluster_embeddings,
            "num_clusters": num_clusters
        }
```

### 2. Recursive Summarizer

Summarizes clusters via cross-attention:

```python
class RecursiveSummarizer(NexusModule):
    def forward(self, cluster_embeddings, cluster_mask=None):
        """Summarize cluster members"""

        # Learnable summary tokens
        summary = self.summary_tokens.expand(num_clusters, -1, -1)

        # Cross-attention: summary tokens attend to cluster members
        for cross_attn, ffn in zip(self.cross_attn_layers, self.ffn_layers):
            attended, attn_weights = cross_attn(
                summary, cluster_embeddings, cluster_embeddings,
                key_padding_mask=cluster_mask
            )
            summary = summary + attended
            summary = summary + ffn(summary)

        # Pool to single embedding per cluster
        pooled = self.output_projection(summary.reshape(num_clusters, -1))

        return {
            "summary_embeddings": pooled,
            "summary_tokens": summary
        }
```

### 3. Tree Retriever

Retrieves from multiple tree levels:

```python
class TreeRetriever(NexusModule):
    def forward(self, query_embedding, tree_nodes):
        """Retrieve from all tree levels"""

        all_scores = []

        for level_idx, level_nodes in enumerate(tree_nodes):
            # Level-specific projection
            keys = self.level_key_projs[level_idx](level_nodes)

            # Similarity scores
            scores = torch.matmul(
                query_embedding, keys.t()
            ) / sqrt(hidden_size)

            # Weight by level importance
            weighted_scores = scores * self.level_weights[level_idx]
            all_scores.append(weighted_scores)

        # Concatenate scores across levels
        concat_scores = torch.cat(all_scores, dim=-1)

        # Select top-k across all levels
        top_scores, top_indices = torch.topk(concat_scores, num_retrieved)

        return {
            "retrieved_embeddings": gather_by_indices(tree_nodes, top_indices),
            "retrieval_scores": top_scores,
            "level_indices": map_to_level(top_indices)
        }
```

## Code Example

```python
from nexus.models.nlp.rag.raptor import RAPTOR

config = {
    "hidden_size": 768,
    "vocab_size": 50257,
    "max_depth": 3,              # Tree depth
    "max_clusters": 50,          # Clusters per level
    "num_retrieved": 10,         # Nodes to retrieve
    "num_summary_tokens": 4      # Summary tokens per cluster
}

raptor = RAPTOR(config)

# Build tree (offline, one-time)
tree_nodes = raptor.build_tree(chunk_embeddings)

# Query time
outputs = raptor(
    query_embedding=query_emb,
    tree_nodes=tree_nodes
)

print(f"Retrieved from levels: {outputs['level_indices']}")
print(f"Tree depth: {outputs['tree_depth']}")
```

## Tree Construction Algorithm

```python
def build_tree(chunk_embeddings):
    tree_levels = [chunk_embeddings]
    current_embeddings = chunk_embeddings

    for depth in range(max_depth):
        if current_embeddings.size(0) <= 1:
            break

        # Cluster current level
        cluster_out = clusterer(current_embeddings)
        assignments = cluster_out["hard_assignments"]

        # Group by cluster
        cluster_groups = [
            current_embeddings[assignments == c]
            for c in range(cluster_out["num_clusters"])
        ]

        # Summarize each cluster
        summaries = summarizer(pad(cluster_groups))

        tree_levels.append(summaries)
        current_embeddings = summaries

    return tree_levels
```

## Results (Sarthi et al., 2024)

**QuALITY (Long Document QA)**:
- Baseline RAG: 47.2%
- RAPTOR: **55.7%** (+8.5%)

**NarrativeQA (Book Understanding)**:
- Baseline RAG: 23.1%
- RAPTOR: **30.8%** (+7.7%)

**Qasper (Scientific Papers)**:
- Baseline RAG: 29.4%
- RAPTOR: **35.1%** (+5.7%)

## Optimization Tricks

1. **Caching**: Cache tree construction for static documents
```python
# Build tree once, reuse for all queries
tree_cache = {}
doc_hash = hash(documents)
if doc_hash not in tree_cache:
    tree_cache[doc_hash] = raptor.build_tree(documents)
```

2. **Level Weighting**: Adjust based on query type
```python
# Abstract query → weight higher levels more
if is_abstract_query(query):
    level_weights = [0.5, 1.0, 1.5]  # Root weighted highest
else:
    level_weights = [1.5, 1.0, 0.5]  # Leaves weighted highest
```

## When to Use

**Good For**:
- Long documents (books, reports, codebases)
- Queries with varying abstraction levels
- Need for both detail and high-level understanding

**Not Good For**:
- Short documents (tree adds overhead)
- Real-time indexing (tree construction is slow)
- Purely factoid queries (simple RAG sufficient)

## References

1. **RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval**
   Sarthi et al., 2024
   https://arxiv.org/abs/2401.18059
