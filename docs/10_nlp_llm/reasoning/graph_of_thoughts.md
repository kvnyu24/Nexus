# Graph of Thoughts: Solving Elaborate Problems with LLMs

## Overview

Graph of Thoughts (GoT) extends Tree of Thoughts by allowing arbitrary graph structures over thoughts, enabling:

- **Generate**: Create new thought nodes
- **Aggregate**: Merge multiple thoughts into one
- **Refine**: Iteratively improve a thought
- **Score**: Evaluate thought quality

This supports more expressive reasoning patterns like merging parallel chains, iterative refinement loops, and decomposition-aggregation.

## Operations

Reference: `Nexus/nexus/models/nlp/reasoning/graph_of_thoughts.py`

### 1. Generate Operation

Create new thoughts from existing ones:

```python
class GenerateOperation(NexusModule):
    def forward(self, source_embedding):
        """Generate num_generated new thoughts from source"""
        generated_flat = self.generator(source_embedding)
        generated = generated_flat.view(-1, num_generated, hidden_size)

        # Diversity loss encourages distinct thoughts
        diversity_loss = encourage_orthogonality(generated)

        return {"generated": generated, "diversity_loss": diversity_loss}
```

### 2. Aggregate Operation

Merge multiple thoughts:

```python
class AggregateOperation(NexusModule):
    def forward(self, thought_embeddings):
        """Merge multiple thoughts via attention"""
        # Learnable query for aggregation
        query = self.agg_query.expand(batch_size, -1, -1)

        # Attention over all thoughts
        aggregated, attn = self.aggregate_attention(
            query, thought_embeddings, thought_embeddings
        )

        return {"aggregated": aggregated, "attention_weights": attn}
```

### 3. Refine Operation

Iteratively improve thoughts:

```python
class RefineOperation(NexusModule):
    def forward(self, thought_embedding, context_embedding):
        """Refine thought given context"""
        combined = torch.cat([thought_embedding, context_embedding], dim=-1)
        refined_candidate = self.refine_network(combined)

        # Gating: how much to update
        gate_values = self.gate(combined)
        refined = gate_values * refined_candidate + (1 - gate_values) * thought_embedding

        return {"refined": refined, "gate_values": gate_values}
```

### 4. Score Operation

Evaluate thought quality:

```python
class ScoreOperation(NexusModule):
    def forward(self, thought_embedding, problem_embedding):
        """Score thought in context of problem"""
        combined = torch.cat([thought_embedding, problem_embedding], dim=-1)
        score = self.scorer(combined)  # Output in [0, 1]
        return {"score": score}
```

## Example: Decompose-Aggregate Pattern

```
          ┌────────────┐
          │  Problem   │
          └──────┬─────┘
                 │
        ┌────────┴────────┐
        │   Generate (3)  │
        └────────┬────────┘
                 │
      ┌──────────┼──────────┐
      │          │          │
      ▼          ▼          ▼
  ┌─────┐    ┌─────┐    ┌─────┐
  │Sub 1│    │Sub 2│    │Sub 3│
  └──┬──┘    └──┬──┘    └──┬──┘
     │          │          │
     │  ┌───────┴───────┐  │
     └──►   Aggregate   ◄──┘
        └───────┬───────┘
                │
        ┌───────▼───────┐
        │    Refine     │
        └───────┬───────┘
                │
        ┌───────▼───────┐
        │     Score     │
        └───────────────┘
```

## Code Example

```python
from nexus.models.nlp.reasoning.graph_of_thoughts import GoTController

config = {
    "hidden_size": 768,
    "vocab_size": 50257,
    "max_nodes": 20,
    "num_generated": 3,
    "max_refinements": 3,
    "operations": ["generate", "aggregate", "refine", "score"]
}

got = GoTController(config)

outputs = got(problem_embedding=problem_emb)

print(f"Best score: {outputs['best_score']}")
print(f"Nodes created: {outputs['num_nodes']}")
print(f"Operation sequence: {outputs['operation_history']}")
```

## Results (Besta et al., 2024)

| Task | CoT | ToT | GoT | Gain |
|------|-----|-----|-----|------|
| Sorting | 12% | 39% | **62%** | +50% vs CoT |
| Keyword Counting | 61% | 78% | **91%** | +30% vs CoT |
| Set Operations | 44% | 56% | **83%** | +39% vs CoT |

## References

1. **Graph of Thoughts: Solving Elaborate Problems with Large Language Models**
   Besta et al., 2024
   https://arxiv.org/abs/2308.09687
