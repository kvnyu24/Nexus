# NLP & LLM Methods Documentation Summary

This document provides an overview of the comprehensive documentation created for advanced NLP methods in the Nexus framework.

## Documentation Structure

```
docs/10_nlp_llm/
├── reasoning/
│   ├── README.md                    # Reasoning landscape overview
│   ├── chain_of_thought.md          # Sequential reasoning
│   ├── tree_of_thoughts.md          # Search-based reasoning
│   ├── graph_of_thoughts.md         # Graph-structured reasoning
│   ├── self_consistency.md          # Ensemble reasoning
│   └── react.md                     # Tool-augmented reasoning
│
└── rag/
    ├── README.md                    # RAG landscape overview
    ├── self_rag.md                  # Adaptive retrieval with self-reflection
    ├── crag.md                      # Corrective retrieval
    ├── graph_rag.md                 # Knowledge graph-based retrieval
    └── raptor.md                    # Hierarchical tree-based retrieval
```

## Reasoning Methods

### Chain-of-Thought (CoT)
**Location**: `reasoning/chain_of_thought.md`

Sequential reasoning that breaks down complex problems into intermediate steps. The foundation for all advanced reasoning methods.

**Key Features**:
- Step-by-step reasoning
- Emergent capability in large models
- Simple prompting technique
- 40%+ accuracy gains on GSM8K

**Implementation**: `Nexus/nexus/models/nlp/chain_of_thoughts.py`

---

### Tree of Thoughts (ToT)
**Location**: `reasoning/tree_of_thoughts.md`

Explores multiple reasoning paths in a tree structure with BFS/DFS search.

**Key Features**:
- Branching and backtracking
- Systematic exploration
- Thought evaluation and pruning
- 74% accuracy on Game of 24

**Implementation**: `Nexus/nexus/models/nlp/reasoning/tree_of_thoughts.py`

---

### Graph of Thoughts (GoT)
**Location**: `reasoning/graph_of_thoughts.md`

Arbitrary graph structures over thoughts with operations: Generate, Aggregate, Refine, Score.

**Key Features**:
- Most flexible reasoning structure
- Supports decomposition-aggregation
- Iterative refinement loops
- 62% accuracy on sorting tasks

**Implementation**: `Nexus/nexus/models/nlp/reasoning/graph_of_thoughts.py`

---

### Self-Consistency
**Location**: `reasoning/self_consistency.md`

Samples multiple CoT paths and aggregates via majority voting.

**Key Features**:
- No training required
- Simple inference-time technique
- +16.7% gain on GSM8K
- Temperature-based diversity

**Implementation**: `Nexus/nexus/models/nlp/reasoning/self_consistency.py`

---

### ReAct
**Location**: `reasoning/react.md`

Interleaves reasoning (Thought) and acting (Action) with tool use.

**Key Features**:
- Tool-augmented reasoning
- Grounded in observations
- Reduces hallucination
- +44% gain on ALFWorld

**Implementation**: `Nexus/nexus/models/nlp/reasoning/react.py`

---

## RAG Methods

### Self-RAG
**Location**: `rag/self_rag.md`

Adaptive retrieval with self-reflection tokens for quality control.

**Key Features**:
- Decides when to retrieve
- Self-assessment via reflection tokens
- [Retrieve], [IsRelevant], [IsSupported], [IsUseful]
- +20.9% gain on PopQA

**Implementation**: `Nexus/nexus/models/nlp/rag/self_rag.py`

---

### CRAG (Corrective RAG)
**Location**: `rag/crag.md`

Evaluates retrieval quality and takes corrective action.

**Key Features**:
- Three-way decision (Correct/Ambiguous/Incorrect)
- Document filtering via decompose-recompose
- Web search fallback
- +15.2% gain on PubHealth

**Implementation**: `Nexus/nexus/models/nlp/rag/crag.py`

---

### GraphRAG
**Location**: `rag/graph_rag.md`

Knowledge graph-based retrieval with community detection.

**Key Features**:
- Entity and relationship extraction
- Hierarchical community detection
- Community pre-summarization
- Excels at global sensemaking queries

**Implementation**: `Nexus/nexus/models/nlp/rag/graph_rag.py`

---

### RAPTOR
**Location**: `rag/raptor.md`

Hierarchical tree-based retrieval via recursive clustering and summarization.

**Key Features**:
- Multi-level abstraction
- Retrieves from any tree level
- Recursive summarization
- +8.5% gain on QuALITY

**Implementation**: `Nexus/nexus/models/nlp/rag/raptor.py`

---

## Quick Reference

### Reasoning Method Selection

| Task Type | Recommended Method | Reasoning |
|-----------|-------------------|-----------|
| Arithmetic | Chain-of-Thought | Simple, effective |
| Planning/Search | Tree of Thoughts | Systematic exploration |
| Complex workflows | Graph of Thoughts | Flexible structure |
| Need robustness | Self-Consistency | Multiple paths |
| Requires tools | ReAct | Grounded reasoning |

### RAG Method Selection

| Query Type | Recommended Method | Reasoning |
|-----------|-------------------|-----------|
| Factoid Q&A | Standard RAG | Simple, fast |
| Quality-critical | Self-RAG | Adaptive + self-assessment |
| Unreliable KB | CRAG | Corrective retrieval |
| Global queries | GraphRAG | Community summaries |
| Long documents | RAPTOR | Multi-level retrieval |

## Implementation Patterns

### Common Architecture Components

All implementations follow consistent patterns:

1. **Module Structure**: Inherit from `NexusModule`
2. **Configuration**: Dict-based config with sensible defaults
3. **Forward Pass**: Returns dict with multiple outputs
4. **Intermediate States**: Exposed for debugging/visualization

### Example Usage Pattern

```python
# 1. Import module
from nexus.models.nlp.reasoning.tree_of_thoughts import TreeOfThoughts

# 2. Configure
config = {
    "model": language_model,
    "max_depth": 3,
    "branching_factor": 3,
    "search_method": "bfs"
}

# 3. Initialize
tot = TreeOfThoughts(config)

# 4. Run
result = tot.solve("Problem statement")

# 5. Access outputs
print(result["answer"])
print(result["reasoning_path"])
```

## Documentation Features

Each method documentation includes:

1. **Overview & Motivation**: Why this method exists
2. **Theoretical Background**: Mathematical formulation
3. **High-Level Intuition**: Flow diagrams and examples
4. **Implementation Details**: Prompting strategies, architectures
5. **Code Walkthrough**: Usage examples with references to actual code
6. **Optimization Tricks**: Performance improvements
7. **Experiments & Results**: Benchmark performance
8. **Common Pitfalls**: Known issues and solutions
9. **References**: Original papers and related work

## Key Insights

### Reasoning Methods

1. **Emergence**: CoT reasoning emerges at scale (>100B params)
2. **Exploration vs Exploitation**: ToT balances systematic search vs efficiency
3. **Flexibility**: GoT is most flexible but requires careful design
4. **Ensemble**: Self-Consistency is simple yet effective
5. **Grounding**: ReAct reduces hallucination via tool use

### RAG Methods

1. **Adaptive Retrieval**: Not all queries need retrieval (Self-RAG)
2. **Quality Matters**: Evaluating retrieval quality is crucial (CRAG)
3. **Structure**: Knowledge graphs enable global reasoning (GraphRAG)
4. **Abstraction**: Multiple levels serve different query types (RAPTOR)
5. **Cost-Quality Trade-off**: Advanced methods improve quality but increase latency

## Performance Guidelines

### Computational Cost (Relative)

**Reasoning**:
- CoT: 1x (baseline)
- Self-Consistency: 10-40x (multiple samples)
- ToT: 30-50x (tree search)
- GoT: 20-100x (graph operations)
- ReAct: 2-10x (tool calls)

**RAG**:
- Standard RAG: 1x (baseline)
- Self-RAG: 1.5x (reflection tokens)
- CRAG: 2-3x (evaluation + potential web search)
- GraphRAG: 1x query, 100x index build
- RAPTOR: 1.2x query, 10x index build

### Quality Improvements

**Reasoning** (GSM8K Math):
- Baseline: ~17%
- CoT: ~58% (+41%)
- Self-Consistency: ~72% (+55%)
- ToT: ~74% (+57%)

**RAG** (PopQA):
- Baseline LLM: ~20%
- Standard RAG: ~35%
- Self-RAG: ~56% (+36% vs LLM)
- CRAG: ~64% (+44% vs LLM)

## References

### Key Papers

**Reasoning**:
1. Wei et al. (2022) - Chain-of-Thought Prompting
2. Wang et al. (2023) - Self-Consistency
3. Yao et al. (2023) - Tree of Thoughts
4. Besta et al. (2024) - Graph of Thoughts
5. Yao et al. (2023) - ReAct

**RAG**:
1. Lewis et al. (2020) - RAG: Retrieval-Augmented Generation
2. Asai et al. (2023) - Self-RAG
3. Yan et al. (2024) - CRAG
4. Edge et al. (2024) - GraphRAG
5. Sarthi et al. (2024) - RAPTOR

## Future Directions

### Reasoning
- **Hybrid Methods**: Combining ToT search with ReAct tool use
- **Learned Strategies**: Meta-learning which reasoning method to use
- **Multimodal Reasoning**: Extending to vision, audio
- **Efficient Search**: Faster tree/graph exploration

### RAG
- **Active Learning**: Improving retrievers via user feedback
- **Dynamic Graphs**: Real-time knowledge graph updates
- **Multimodal Retrieval**: Images, tables, code
- **Federated RAG**: Retrieval across distributed sources

## Getting Started

1. **Read Overview**: Start with `reasoning/README.md` or `rag/README.md`
2. **Choose Method**: Use decision tables to select appropriate method
3. **Study Example**: Read detailed documentation for chosen method
4. **Run Code**: Use provided examples with actual implementations
5. **Optimize**: Apply tricks from "Optimization" sections
6. **Avoid Pitfalls**: Review "Common Pitfalls" sections

## Contributing

When adding new methods:

1. Follow the established documentation structure
2. Include mathematical formulations where applicable
3. Provide working code examples
4. Add benchmark results from papers
5. Document optimization tricks and pitfalls
6. Link to reference implementations in `/nexus/models/`

---

**Total Documentation**: 10 comprehensive markdown files covering 9 advanced NLP methods with complete theory, implementation details, and practical guidance.
