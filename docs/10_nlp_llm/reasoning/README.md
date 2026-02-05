# Reasoning Methods for Large Language Models

This directory contains comprehensive documentation on advanced reasoning techniques that enable LLMs to solve complex problems through structured thought processes.

## Overview

Reasoning methods enhance LLMs' ability to tackle complex problems by introducing structured approaches to thought generation, evaluation, and refinement. Unlike standard prompting, these techniques enable models to:

- **Decompose** complex problems into manageable sub-steps
- **Explore** multiple solution paths systematically
- **Evaluate** and select the most promising reasoning chains
- **Ground** reasoning in observable evidence through tool use

## Reasoning Landscape

### 1. Chain-of-Thought (CoT)
**Core Idea**: Generate intermediate reasoning steps sequentially before producing a final answer.

**When to Use**:
- Arithmetic and mathematical reasoning
- Multi-hop question answering
- Problems requiring step-by-step logical deduction

**Key Advantage**: Simple to implement, significant performance gains with few-shot prompting

**Reference**: [Chain-of-Thought Prompting (Wei et al., 2022)](https://arxiv.org/abs/2201.11903)

---

### 2. Tree of Thoughts (ToT)
**Core Idea**: Maintain a tree of reasoning states, branching at each step to explore multiple candidate thoughts, with search algorithms (BFS/DFS) to find optimal paths.

**When to Use**:
- Planning and search problems (e.g., Game of 24, crosswords)
- Tasks requiring deliberate exploration and backtracking
- Problems where intermediate steps can be evaluated independently

**Key Advantage**: Systematic exploration with pruning of low-quality branches

**Reference**: [Tree of Thoughts (Yao et al., NeurIPS 2023)](https://arxiv.org/abs/2305.10601)

---

### 3. Graph of Thoughts (GoT)
**Core Idea**: Extend ToT to arbitrary graph structures, enabling operations like:
- **Generate**: Create new thought nodes
- **Aggregate**: Merge multiple thoughts
- **Refine**: Iteratively improve thoughts
- **Score**: Evaluate thought quality

**When to Use**:
- Complex reasoning requiring feedback loops
- Tasks benefiting from merging parallel reasoning chains
- Iterative refinement workflows

**Key Advantage**: Most flexible reasoning structure, supports decomposition-aggregation patterns

**Reference**: [Graph of Thoughts (Besta et al., 2024)](https://arxiv.org/abs/2308.09687)

---

### 4. Self-Consistency
**Core Idea**: Sample multiple diverse reasoning paths (via temperature sampling), then aggregate final answers via majority voting.

**When to Use**:
- Arithmetic and commonsense reasoning
- Tasks where multiple valid reasoning paths exist
- Improving robustness of CoT

**Key Advantage**: Simple inference-time technique with no training required

**Reference**: [Self-Consistency (Wang et al., ICLR 2023)](https://arxiv.org/abs/2203.11171)

---

### 5. ReAct
**Core Idea**: Interleave reasoning (Thought) and acting (Action) in a loop, grounding reasoning in real observations from tool use.

**When to Use**:
- Question answering requiring external knowledge
- Tasks needing API calls, search, or computation
- Reducing hallucination through factual grounding

**Key Advantage**: Combines reasoning with real-world interaction

**Reference**: [ReAct (Yao et al., 2023)](https://arxiv.org/abs/2210.03629)

---

## Comparison Matrix

| Method | Structure | Search Strategy | Tool Use | Training Required | Best For |
|--------|-----------|----------------|----------|-------------------|----------|
| **Chain-of-Thought** | Sequential | Greedy | No | No | General reasoning, arithmetic |
| **Self-Consistency** | Ensemble | Sample-then-vote | No | No | Robust CoT, reducing variance |
| **Tree of Thoughts** | Tree | BFS/DFS | No | Optional | Planning, search problems |
| **Graph of Thoughts** | Graph | Custom | No | Yes | Complex workflows, refinement |
| **ReAct** | Sequential | Greedy | Yes | Optional | Grounded reasoning, tool use |

## Implementation Hierarchy

```
Reasoning Methods
├── Chain-of-Thought (Base)
│   └── Self-Consistency (Ensemble over CoT)
├── Tree of Thoughts (Structured Search)
│   └── Graph of Thoughts (Generalized Structure)
└── ReAct (Tool-Augmented Reasoning)
```

## Performance Guidelines

### Computational Cost
- **Lowest**: Chain-of-Thought (single forward pass)
- **Low-Medium**: ReAct (sequential tool calls)
- **Medium**: Self-Consistency (multiple samples)
- **High**: Tree of Thoughts (explores multiple branches)
- **Highest**: Graph of Thoughts (complex graph operations)

### Sample Efficiency
- **Best**: Self-Consistency (improves with more samples)
- **Good**: Tree of Thoughts (explores strategically)
- **Moderate**: Graph of Thoughts (requires careful design)

### Interpretability
- **Best**: ReAct (explicit tool actions), Tree of Thoughts (visible search tree)
- **Good**: Chain-of-Thought (readable reasoning steps)
- **Moderate**: Graph of Thoughts (complex graph structure)

## Getting Started

1. **Start Simple**: Begin with Chain-of-Thought prompting
2. **Add Robustness**: Implement Self-Consistency for critical applications
3. **Enable Search**: Use Tree of Thoughts for planning/search problems
4. **Add Tools**: Integrate ReAct for knowledge-grounded tasks
5. **Advanced Workflows**: Apply Graph of Thoughts for complex pipelines

## Common Pitfalls

1. **Over-engineering**: Don't use complex methods (ToT, GoT) when simple CoT suffices
2. **Poor evaluation**: Define clear metrics for thought quality in ToT/GoT
3. **Tool selection**: Ensure ReAct tools are reliable and well-described
4. **Prompt quality**: All methods benefit from clear problem formulation
5. **Computational budget**: Consider token costs for multi-sample/multi-path methods

## References

1. Wei et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
2. Wang et al. (2023). "Self-Consistency Improves Chain of Thought Reasoning in Language Models"
3. Yao et al. (2023). "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"
4. Besta et al. (2024). "Graph of Thoughts: Solving Elaborate Problems with Large Language Models"
5. Yao et al. (2023). "ReAct: Synergizing Reasoning and Acting in Language Models"

## Documentation Files

- [Chain-of-Thought](./chain_of_thought.md) - Sequential reasoning with intermediate steps
- [Self-Consistency](./self_consistency.md) - Ensemble method over multiple reasoning paths
- [Tree of Thoughts](./tree_of_thoughts.md) - Search-based reasoning over thought trees
- [Graph of Thoughts](./graph_of_thoughts.md) - Graph-structured reasoning with flexible operations
- [ReAct](./react.md) - Tool-augmented reasoning through action-observation loops
