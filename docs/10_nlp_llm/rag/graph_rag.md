# GraphRAG: Graph-Based Retrieval Augmented Generation

## Overview

GraphRAG constructs a knowledge graph from documents by extracting entities and relationships, then uses hierarchical community detection to partition the graph. Each community is pre-summarized, and at query time, relevant community summaries are retrieved.

**Key Insight**: Global sensemaking queries ("What are the main themes?") can't be answered by retrieving a few documents. We need to aggregate information across the entire corpus.

## Pipeline

Reference: `/Users/kevinyu/Projects/Nexus/nexus/models/nlp/rag/graph_rag.py`

```
Documents
    ↓
┌─────────────────────┐
│ Entity Extraction   │ → Extract entities & relationships
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Knowledge Graph     │ → Build entity-relation graph
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Community Detection │ → Hierarchical clustering (Leiden)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│ Community Summaries │ → Pre-compute summaries
└─────────┬───────────┘
          │
    [Index Built]
          │
Query     │
   │      │
   ▼      ▼
┌─────────────────────┐
│ Retrieve Communities│ → Find relevant summaries
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│    Generate         │ → Answer using summaries
└─────────────────────┘
```

## Key Components

### 1. Entity Extractor

Identifies entities and relationships using NER + relation classification:

```python
class EntityExtractor(NexusModule):
    def forward(self, input_ids, attention_mask):
        """Extract entities and relations from text"""

        # Encode text
        hidden_states = self.encoder(input_ids)

        # BIO tagging for entity spans
        entity_logits = self.entity_classifier(hidden_states)
        # (batch, seq_len, num_entity_types * 3)

        # Bilinear relation scoring between all token pairs
        head_repr = self.relation_head_proj(hidden_states)
        tail_repr = self.relation_tail_proj(hidden_states)
        relation_logits = self.relation_bilinear(head_repr, tail_repr)
        # (batch, seq_len, seq_len, num_relation_types)

        return {
            "entity_logits": entity_logits,
            "relation_logits": relation_logits,
            "hidden_states": hidden_states
        }
```

### 2. Knowledge Graph

Maintains entity embeddings and performs message passing:

```python
class KnowledgeGraph(NexusModule):
    def forward(self, entity_indices, edge_index, edge_type):
        """Message passing over knowledge graph"""

        node_features = self.entity_embeddings(entity_indices)

        # Graph Neural Network layers
        for layer in self.gnn_layers:
            # Compute messages from neighbors
            source_features = node_features[edge_index[0]]
            rel_features = self.relation_embeddings(edge_type)

            messages = layer["message_proj"](
                torch.cat([source_features, rel_features], dim=-1)
            )

            # Attention-weighted aggregation
            attn_weights = layer["attention"](messages)
            aggregated = scatter_add(messages * attn_weights, edge_index[1])

            # Update node features
            node_features = layer["update"](aggregated, node_features)

        return {"entity_embeddings": node_features}
```

### 3. Community Detection

Hierarchical clustering using differentiable pooling:

```python
class CommunityDetector(NexusModule):
    def forward(self, node_embeddings, adjacency=None):
        """Hierarchical community detection"""

        all_assignments = []
        all_community_embeddings = []

        current_embeddings = node_embeddings
        current_adjacency = adjacency

        for level in range(community_levels):
            # Soft assignment to clusters
            assignment_logits = self.assignment_layers[level](current_embeddings)
            assignment = F.softmax(assignment_logits, dim=-1)

            # Coarsen: S^T * X (pool embeddings)
            community_embeddings = torch.matmul(assignment.t(), current_embeddings)

            # Coarsen adjacency: S^T * A * S
            coarsened_adjacency = torch.matmul(
                torch.matmul(assignment.t(), current_adjacency),
                assignment
            )

            all_assignments.append(assignment)
            all_community_embeddings.append(community_embeddings)

            current_embeddings = community_embeddings
            current_adjacency = coarsened_adjacency

        return {
            "assignments": all_assignments,
            "community_embeddings": all_community_embeddings
        }
```

### 4. Community Summarizer

Generates summaries for each community:

```python
class CommunitySummarizer(NexusModule):
    def forward(self, community_embeddings, entity_embeddings=None):
        """Generate community summaries via cross-attention"""

        # Learnable summary queries
        queries = self.summary_queries.expand(num_communities, -1, -1)

        # Cross-attention to community entities
        summary, attn_weights = self.cross_attention(
            queries, entity_embeddings, entity_embeddings
        )

        # Feedforward refinement
        summary = summary + self.ffn(summary)

        # Pool to single vector per community
        pooled = summary.mean(dim=1)

        return {
            "summary_embeddings": summary,
            "pooled_summaries": pooled
        }
```

## Code Example

```python
from nexus.models.nlp.rag.graph_rag import GraphRAGPipeline

config = {
    "hidden_size": 768,
    "vocab_size": 50257,
    "num_entity_types": 10,
    "num_relation_types": 20,
    "max_entities": 10000,
    "num_communities": 10,
    "community_levels": 3,
    "num_retrieved": 5
}

graph_rag = GraphRAGPipeline(config)

# Build index offline (expensive, one-time)
# Extract entities from documents
# Build graph
# Detect communities
# Compute summaries

# Query time (fast)
outputs = graph_rag(
    query_embedding=query_emb,
    community_summaries=precomputed_summaries
)

print(f"Retrieved summaries: {outputs['retrieved_summaries'].shape}")
print(f"Retrieval scores: {outputs['retrieval_scores']}")
```

## Results (Edge et al., 2024)

**Podcast Transcripts Dataset**:
- Baseline RAG: Unable to answer global queries
- GraphRAG: **Comprehensiveness +34%, Diversity +24%**

**News Articles Dataset**:
- Baseline RAG: 41% answer quality
- GraphRAG: **67% answer quality** (+26%)

GraphRAG excels at queries like:
- "What are the main themes in this dataset?"
- "Summarize the key entities and their relationships"
- "What are the different perspectives on topic X?"

## When to Use

**Good For**:
- Global sensemaking queries
- Large document collections (1000s)
- Multi-hop reasoning over entities
- Synthesizing information across corpus

**Not Good For**:
- Factoid Q&A (overkill)
- Real-time indexing (expensive graph construction)
- Streaming data (graph is static)

## References

1. **From Local to Global: A Graph RAG Approach to Query-Focused Summarization**
   Edge et al., 2024
   https://arxiv.org/abs/2404.16130
