# CRAG: Corrective Retrieval Augmented Generation

## Overview

CRAG addresses unreliable retrieval by adding a lightweight retrieval evaluator that assesses document quality before generation. Based on the assessment, CRAG takes corrective action.

**Three-Way Decision**:
1. **Correct**: Retrievals are good → filter and use them
2. **Incorrect**: Retrievals are bad → fallback to web search
3. **Ambiguous**: Mixed quality → combine filtered docs + web search

## Pipeline

Reference: `/Users/kevinyu/Projects/Nexus/nexus/models/nlp/rag/crag.py`

```
┌──────────────┐
│    Query     │
└──────┬───────┘
       │
       ▼
┌──────────────────┐
│ Retrieve top-k   │
└──────┬───────────┘
       │
       ▼
┌──────────────────────────────┐
│ Evaluate Retrieval Confidence│
│  (Correct/Ambiguous/Incorrect)│
└──────┬───────────────────────┘
       │
   ┌───┴────┬─────────────┐
   │        │             │
Correct  Ambiguous    Incorrect
   │        │             │
   ▼        ▼             ▼
┌─────┐  ┌─────┐      ┌──────────┐
│Filter│  │Filter│      │Web Search│
│ Docs │  │ + │      └──────────┘
└──┬──┘  │Web  │             │
   │     │Search│             │
   │     └──┬──┘              │
   └────────┼─────────────────┘
            │
            ▼
    ┌──────────────┐
    │   Generate   │
    └──────────────┘
```

## Key Components

### 1. Retrieval Evaluator

Classifies documents as Correct/Ambiguous/Incorrect:

```python
class RetrievalEvaluator(NexusModule):
    def forward(self, query_embedding, document_embeddings):
        """Score relevance of each document"""

        # Cross-attention between query and documents
        interaction = self.cross_attention(
            query, documents, documents
        )

        # Continuous relevance score
        relevance_scores = self.relevance_scorer(interaction)

        # Classify based on thresholds
        confidence_labels = torch.where(
            relevance_scores >= confidence_threshold,  # e.g., 0.7
            torch.zeros_like(scores, dtype=torch.long),  # Correct
            torch.where(
                relevance_scores >= ambiguity_threshold,  # e.g., 0.3
                torch.ones_like(scores, dtype=torch.long),  # Ambiguous
                torch.full_like(scores, 2, dtype=torch.long)  # Incorrect
            )
        )

        return {
            "confidence_labels": confidence_labels,
            "relevance_scores": relevance_scores,
            "actions": determine_action(relevance_scores)
        }
```

### 2. Document Filter (Decompose-Recompose)

Removes irrelevant content from documents:

```python
class DocumentFilter(NexusModule):
    def forward(self, query_embedding, document_embeddings):
        """Decompose docs into strips, filter, recompose"""

        # Decompose into knowledge strips
        strips = self.decomposer(document_embeddings)
        # (batch, num_docs, num_strips, hidden_size)

        # Score each strip against query
        strip_scores = self.strip_scorer(
            torch.cat([query_expanded, strips], dim=-1)
        )

        # Filter: keep strips above threshold
        strip_mask = (strip_scores >= strip_threshold)

        # Recompose: weighted sum of retained strips
        filtered = self.recomposer(
            (strips * strip_scores * strip_mask).sum(dim=2)
        )

        return {
            "filtered_embeddings": filtered,
            "retention_ratio": strip_mask.mean()
        }
```

### 3. Web Search Fallback

Generates synthetic search results when retrieval fails:

```python
class WebSearchFallback(NexusModule):
    def forward(self, query_embedding):
        """Generate web search results"""

        # Reformulate query for search
        reformulated = self.query_reformulator(query_embedding)

        # Generate search result embeddings
        search_results = self.search_result_generator(reformulated)

        # Score results
        result_scores = self.result_scorer(
            torch.cat([query_embedding, search_results], dim=-1)
        )

        return {
            "search_results": search_results,
            "result_scores": result_scores
        }
```

## Code Example

```python
from nexus.models.nlp.rag.crag import CRAGPipeline

config = {
    "hidden_size": 768,
    "vocab_size": 50257,
    "confidence_threshold": 0.7,   # "Correct" threshold
    "ambiguity_threshold": 0.3,    # "Incorrect" threshold
    "num_search_results": 5,
    "num_strips": 8,                # Knowledge strips per document
    "strip_threshold": 0.5          # Minimum strip relevance
}

crag = CRAGPipeline(config)

outputs = crag(
    query_embedding=query_emb,
    document_embeddings=retrieved_docs
)

print(f"Action: {outputs['action_taken']}")
print(f"Context source: {outputs['context_source']}")
print(f"Filter retention: {outputs['filter_retention']}")
```

## Results (Yan et al., 2024)

| Dataset | Standard RAG | CRAG | Gain |
|---------|-------------|------|------|
| PopQA | 55.2% | **63.5%** | +8.3% |
| Biography | 81.9% | **88.7%** | +6.8% |
| PubHealth | 72.1% | **87.3%** | +15.2% |

CRAG shows largest gains on tasks requiring factual accuracy.

## Optimization Tricks

1. **Adaptive Thresholds**: Adjust based on domain
```python
if domain == "medical":
    confidence_threshold = 0.8  # Higher bar for medical facts
elif domain == "general":
    confidence_threshold = 0.6
```

2. **Parallel Evaluation**: Score all documents simultaneously
```python
# Instead of sequential
for doc in documents:
    score = evaluate(query, doc)

# Parallel
scores = evaluate(query, torch.stack(documents))
```

## References

1. **Corrective Retrieval Augmented Generation**
   Yan et al., 2024
   https://arxiv.org/abs/2401.15884
