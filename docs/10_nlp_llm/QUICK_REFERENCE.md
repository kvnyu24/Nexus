# NLP Methods Quick Reference Guide

## Reasoning Methods Cheat Sheet

### Chain-of-Thought (CoT)
```python
# Use when: Arithmetic, logical reasoning, step-by-step problems
# Cost: 1x (single forward pass)
# Gain: +40% on math tasks

prompt = "Question\n\nLet's think step by step:"
answer = model.generate(prompt)
```

### Self-Consistency
```python
# Use when: Need robustness, reduce variance
# Cost: 10-40x (multiple samples)
# Gain: +15% over CoT

answers = [model.generate(prompt, temp=0.7) for _ in range(40)]
final = most_common(answers)
```

### Tree of Thoughts (ToT)
```python
# Use when: Planning, search problems, need backtracking
# Cost: 30-50x (tree search)
# Gain: +60% on Game of 24

tot = TreeOfThoughts({
    "max_depth": 3,
    "branching_factor": 3,
    "search_method": "bfs"
})
result = tot.solve(problem)
```

### Graph of Thoughts (GoT)
```python
# Use when: Complex workflows, iterative refinement
# Cost: 20-100x (graph operations)
# Gain: +50% on sorting tasks

got = GoTController({
    "operations": ["generate", "aggregate", "refine", "score"]
})
result = got(problem_embedding)
```

### ReAct
```python
# Use when: Need external knowledge, tool use
# Cost: 2-10x (tool calls)
# Gain: +40% on ALFWorld

agent = ReActAgent({
    "tools": ["Search", "Lookup", "Calculator", "Finish"]
})
result = agent(input_ids)  # Interleaves Thought-Action-Observation
```

---

## RAG Methods Cheat Sheet

### Self-RAG
```python
# Use when: Quality-critical, want adaptive retrieval
# Cost: 1.5x (reflection tokens)
# Gain: +20% on PopQA

model = SelfRAGModel({
    "retrieve_threshold": 0.5,  # When to retrieve
    "relevance_weight": 1.0,
    "support_weight": 1.0,
    "utility_weight": 0.5
})
outputs = model(input_ids, document_embeddings)
```

### CRAG
```python
# Use when: Unreliable knowledge base, need robustness
# Cost: 2-3x (evaluation + web search)
# Gain: +15% on PubHealth

crag = CRAGPipeline({
    "confidence_threshold": 0.7,  # "Correct" threshold
    "ambiguity_threshold": 0.3,   # "Incorrect" threshold
    "num_strips": 8               # Document filtering
})
outputs = crag(query_embedding, document_embeddings)
```

### GraphRAG
```python
# Use when: Global queries, need synthesis across corpus
# Cost: 1x query, 100x index build
# Gain: +25% on global sensemaking

# Offline: Build graph and communities
graph_rag = GraphRAGPipeline(config)
# ... extract entities, build graph, detect communities, summarize

# Online: Retrieve community summaries
outputs = graph_rag(query_embedding, community_summaries)
```

### RAPTOR
```python
# Use when: Long documents, varying query abstraction levels
# Cost: 1.2x query, 10x index build
# Gain: +8% on long document QA

# Offline: Build hierarchical tree
raptor = RAPTOR(config)
tree_nodes = raptor.build_tree(chunk_embeddings)

# Online: Retrieve from multiple levels
outputs = raptor(query_embedding, tree_nodes)
```

---

## Decision Trees

### Which Reasoning Method?

```
Start
  │
  ├─ Need tool use? ──Yes──► ReAct
  │                   No
  │                    │
  ├─ Planning/Search? ──Yes──► Tree of Thoughts
  │                    No
  │                     │
  ├─ Complex workflow? ──Yes──► Graph of Thoughts
  │                     No
  │                      │
  ├─ Need robustness? ──Yes──► Self-Consistency
  │                     No
  │                      │
  └─────────────────────► Chain-of-Thought
```

### Which RAG Method?

```
Start
  │
  ├─ Global query? ──Yes──► GraphRAG
  │                 No
  │                  │
  ├─ Long document? ──Yes──► RAPTOR
  │                  No
  │                   │
  ├─ Quality critical? ──Yes──► Self-RAG
  │                      No
  │                       │
  ├─ Unreliable KB? ──Yes──► CRAG
  │                     No
  │                      │
  └─────────────────────► Standard RAG
```

---

## Performance Matrix

### Reasoning Methods

| Method | Cost | Math | Planning | Reasoning | Tools |
|--------|------|------|----------|-----------|-------|
| CoT | 1x | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ❌ |
| Self-Consistency | 20x | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | ❌ |
| ToT | 40x | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ❌ |
| GoT | 50x | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ❌ |
| ReAct | 5x | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ |

### RAG Methods

| Method | Query Cost | Index Cost | Factoid | Multi-hop | Global |
|--------|-----------|------------|---------|-----------|--------|
| Standard | 1x | 1x | ⭐⭐⭐⭐ | ⭐⭐ | ⭐ |
| Self-RAG | 1.5x | 1x | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| CRAG | 2.5x | 1x | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| GraphRAG | 1x | 100x | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| RAPTOR | 1.2x | 10x | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## Common Configurations

### Reasoning

```python
# Development (fast iteration)
config = {
    "max_depth": 2,
    "branching_factor": 2,
    "num_samples": 5
}

# Production (best quality)
config = {
    "max_depth": 4,
    "branching_factor": 5,
    "num_samples": 40
}

# Balanced
config = {
    "max_depth": 3,
    "branching_factor": 3,
    "num_samples": 20
}
```

### RAG

```python
# Fast retrieval
config = {
    "num_retrieved": 3,
    "retrieve_threshold": 0.4,
    "confidence_threshold": 0.6
}

# High quality
config = {
    "num_retrieved": 10,
    "retrieve_threshold": 0.6,
    "confidence_threshold": 0.8
}

# Balanced
config = {
    "num_retrieved": 5,
    "retrieve_threshold": 0.5,
    "confidence_threshold": 0.7
}
```

---

## Troubleshooting

### Reasoning

**Problem**: Poor reasoning quality
- ✅ Increase `max_depth` or `branching_factor`
- ✅ Use better evaluation prompts
- ✅ Try Self-Consistency for robustness

**Problem**: Too slow
- ✅ Reduce `num_samples` or `branching_factor`
- ✅ Use DFS instead of BFS
- ✅ Add aggressive pruning

**Problem**: Repetitive thoughts
- ✅ Increase temperature
- ✅ Add diversity loss (GoT)
- ✅ Implement duplicate detection

### RAG

**Problem**: Low retrieval quality
- ✅ Fine-tune document encoder
- ✅ Use CRAG for quality assessment
- ✅ Increase `num_retrieved` and rerank

**Problem**: Context overflow
- ✅ Use RAPTOR for hierarchical retrieval
- ✅ Implement document filtering (CRAG)
- ✅ Summarize before feeding to LLM

**Problem**: Unnecessary retrieval
- ✅ Use Self-RAG adaptive retrieval
- ✅ Lower `retrieve_threshold`
- ✅ Classify queries first

---

## Implementation Paths

### Basic → Advanced Reasoning

```
1. Start: Chain-of-Thought prompting
2. Add: Self-Consistency for robustness
3. Upgrade: ToT for search problems
4. Advanced: GoT for complex workflows
5. Augment: ReAct for tool use
```

### Basic → Advanced RAG

```
1. Start: Standard RAG with dense retrieval
2. Add: Self-RAG for adaptive retrieval
3. Upgrade: CRAG for quality control
4. Specialize: GraphRAG (global) or RAPTOR (long docs)
```

---

## Key Takeaways

### Reasoning
- **CoT is the foundation**: Always start here
- **Self-Consistency is cheap insurance**: 20x cost, significant gains
- **ToT for search**: When you need to explore alternatives
- **GoT for complexity**: Most flexible, highest ceiling
- **ReAct for grounding**: Reduce hallucination with tools

### RAG
- **Not all queries need retrieval**: Self-RAG saves 30-60% of calls
- **Quality matters more than quantity**: CRAG filtering > more docs
- **Structure enables reasoning**: GraphRAG for synthesis, RAPTOR for abstraction
- **Cache aggressively**: Index build is expensive, reuse it
- **Domain-specific tuning**: Adjust thresholds per domain

---

## File Locations

**Code Implementations**:
- Reasoning: `Nexus/nexus/models/nlp/reasoning/`
- RAG: `Nexus/nexus/models/nlp/rag/`

**Documentation**:
- Reasoning: `Nexus/docs/10_nlp_llm/reasoning/`
- RAG: `Nexus/docs/10_nlp_llm/rag/`

**Usage Examples**:
- See individual method documentation for detailed examples
- Check `README.md` files for method comparison and selection guidance
