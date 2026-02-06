# Graph of Thoughts (GoT): Solving Elaborate Problems with LLMs

## 1. Overview & Motivation

Graph of Thoughts (GoT) extends Tree of Thoughts by allowing arbitrary graph structures over thoughts, enabling more expressive reasoning patterns beyond trees. GoT introduces four fundamental operations:

- **Generate**: Create new thought nodes from existing ones
- **Aggregate**: Merge multiple thoughts into a single unified thought
- **Refine**: Iteratively improve a thought through feedback loops
- **Score**: Evaluate thought quality for selection and pruning

**Key Insight**: Many complex reasoning tasks require patterns that trees cannot express - such as merging parallel reasoning chains, iterative refinement loops, and decomposition-aggregation workflows.

### Why Graph of Thoughts?

Tree of Thoughts limitations:
- Cannot merge thoughts from different branches
- No iterative refinement (forward-only)
- Fixed acyclic structure

GoT addresses these by allowing:
1. **Aggregation nodes**: Combine insights from multiple reasoning paths
2. **Refinement cycles**: Iteratively improve solutions
3. **Flexible topology**: Any directed acyclic graph (DAG) structure

### When to Use GoT

GoT excels at tasks requiring:
- **Decomposition-Aggregation**: Break problem into sub-problems, then merge solutions
- **Iterative Refinement**: Draft → Critique → Revise loops
- **Multi-perspective Fusion**: Combine different approaches or viewpoints
- **Hierarchical Planning**: Multi-level decomposition with cross-level dependencies

## 2. Theory: Graph Operations & Prompting Strategies

### Graph Structure

A thought graph $G = (V, E, O)$ consists of:
- $V$: Set of thought nodes
- $E \subseteq V \times V$: Directed edges representing dependencies
- $O$: Set of operations (Generate, Aggregate, Refine, Score)

Each node $v \in V$ has:
- Content $c_v$ (the thought text/embedding)
- Operation type $o_v \in O$
- Value score $s_v \in [0, 1]$
- Parent nodes $\text{parents}(v) \subseteq V$

### Four Core Operations

**1. Generate Operation**

Creates $k$ new thoughts from a source thought:

$$
\text{Generate}(v, k) = \{v_1, \ldots, v_k\} \text{ where } v_i \sim p(\cdot | v)
$$

Prompting strategy:
```
Given: {source_thought}
Task: Generate {k} diverse next steps/sub-problems
Output: {k} distinct thoughts
```

**2. Aggregate Operation**

Merges multiple thoughts into one:

$$
\text{Aggregate}(V') = v_{\text{merged}} \text{ where } c_{v_{\text{merged}}} = f(c_{v_1}, \ldots, c_{v_n})
$$

for $V' = \{v_1, \ldots, v_n\}$.

Prompting strategy:
```
Given: {thought_1}, {thought_2}, ..., {thought_n}
Task: Synthesize these into a unified solution
Output: Aggregated thought combining all insights
```

**3. Refine Operation**

Improves a thought based on feedback:

$$
\text{Refine}(v, f) = v' \text{ where } v' \sim p(\cdot | v, f)
$$

where $f$ is feedback/context.

Prompting strategy:
```
Given: {current_thought}
Feedback: {critique or context}
Task: Improve the thought addressing the feedback
Output: Refined thought
```

**4. Score Operation**

Evaluates thought quality:

$$
\text{Score}(v, P) = s \in [0, 1]
$$

where $P$ is the original problem.

Prompting strategy:
```
Given: Problem: {P}, Thought: {v}
Task: Rate this thought's quality on [0, 1]
Criteria: Correctness, completeness, relevance
Output: Score
```

### Graph Execution Strategies

**Topological Execution**: Process nodes in topological order (dependencies first)

$$
\text{for } v \in \text{TopologicalSort}(G): \text{ Execute } o_v(v)
$$

**Iterative Refinement**: Execute refinement subgraphs until convergence

$$
\text{while } \Delta s > \epsilon: v \leftarrow \text{Refine}(v, \text{feedback})
$$

## 3. Mathematical Formulation: Sampling & Aggregation

### Generate Operation - Sampling

Sample $k$ thoughts with diversity:

$$
\{t_1, \ldots, t_k\} = \text{top-}k\left(\left\{t \sim p(t | s, T) : t \in \mathcal{T}\right\}\right)
$$

with temperature $T$ and diversity penalty:

$$
\text{score}(t_i) = \log p(t_i | s) - \lambda \sum_{j<i} \text{sim}(t_i, t_j)
$$

### Aggregate Operation - Merging

Attention-based aggregation:

$$
c_{\text{agg}} = \text{Attention}(Q, K, V)
$$

where:
- $Q$ = learnable aggregation query
- $K, V$ = keys/values from thoughts $\{c_{v_1}, \ldots, c_{v_n}\}$

Weighted averaging:

$$
c_{\text{agg}} = \sum_{i=1}^n w_i c_{v_i}, \quad w_i = \frac{\exp(s_{v_i})}{\sum_j \exp(s_{v_j})}
$$

### Refine Operation - Update Rule

Gated update:

$$
c_{v'}  = \alpha \cdot c_{\text{refined}} + (1 - \alpha) \cdot c_v
$$

where $\alpha = \sigma(g(c_v, f))$ is learned gate.

### Score Operation - Quality Estimation

Multi-criteria scoring:

$$
s_v = \sum_{i=1}^m \beta_i \cdot \text{score}_i(v)
$$

where $\text{score}_i$ evaluates criterion $i$ (e.g., correctness, relevance).

### Graph-Level Aggregation

Final answer from multiple terminal nodes:

$$
a^* = \text{Aggregate}\left(\arg\max_{v \in \text{Terminals}(G)} s_v\right)
$$

## 4. Intuition

### Decompose-Aggregate Pattern

```
          ┌────────────┐
          │  Problem   │
          │  "Write    │
          │   essay"   │
          └──────┬─────┘
                 │
        ┌────────▼────────┐
        │   Generate (3)  │  ← Decompose into sub-problems
        └────────┬────────┘
                 │
      ┌──────────┼──────────┐
      │          │          │
      ▼          ▼          ▼
  ┌─────┐    ┌─────┐    ┌─────┐
  │Intro│    │Body │    │Concl│  ← Solve independently
  └──┬──┘    └──┬──┘    └──┬──┘
     │          │          │
     │  ┌───────┴───────┐  │
     └──►   Aggregate   ◄──┘  ← Merge solutions
        └───────┬───────┘
                │
        ┌───────▼───────┐
        │    Refine     │      ← Improve coherence
        └───────┬───────┘
                │
        ┌───────▼───────┐
        │     Score     │      ← Evaluate quality
        └───────────────┘
```

### Iterative Refinement Loop

```
┌─────────┐
│ Initial │
│ Thought │
└────┬────┘
     │
     ▼
┌────────────┐
│   Score    │ ──┐
└────┬───────┘   │
     │           │
     ▼           │
┌────────────┐   │
│  Refine    │   │ Loop until
└────┬───────┘   │ score > threshold
     │           │ or max iterations
     │           │
     └───────────┘
     │
     ▼
┌─────────┐
│  Final  │
└─────────┘
```

### Multi-Path Fusion

```
        ┌─────────┐
        │ Problem │
        └────┬────┘
             │
    ┌────────┴────────┐
    │                 │
    ▼                 ▼
┌────────┐      ┌────────┐
│Approach│      │Approach│
│   A    │      │   B    │
└───┬────┘      └───┬────┘
    │               │
    │   ┌───────────┘
    └───►  Aggregate
        └─────┬──────┘
              │
              ▼
        ┌──────────┐
        │  Best of │
        │   Both   │
        └──────────┘
```

## 5. Implementation Details

### Graph Construction

```python
class ThoughtGraph:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def add_node(self, thought, operation, parents=None):
        node = {
            'id': len(self.nodes),
            'thought': thought,
            'operation': operation,
            'parents': parents or [],
            'score': None
        }
        self.nodes.append(node)
        return node['id']

    def add_edge(self, from_id, to_id):
        self.edges.append((from_id, to_id))

    def topological_sort(self):
        # Kahn's algorithm
        in_degree = {i: 0 for i in range(len(self.nodes))}
        for from_id, to_id in self.edges:
            in_degree[to_id] += 1

        queue = [i for i in in_degree if in_degree[i] == 0]
        result = []

        while queue:
            node_id = queue.pop(0)
            result.append(node_id)

            for from_id, to_id in self.edges:
                if from_id == node_id:
                    in_degree[to_id] -= 1
                    if in_degree[to_id] == 0:
                        queue.append(to_id)

        return result
```

### Operation Implementations

**Generate**:
```python
def generate_operation(source_thought, k, temperature=0.8):
    prompt = f"""Given: {source_thought}

Generate {k} distinct next steps or sub-problems.
Each should be different and explore a unique direction.

Steps:"""

    response = model.generate(
        prompt,
        temperature=temperature,
        n=k,
        stop=['\n\n']
    )

    thoughts = parse_numbered_list(response)
    return thoughts[:k]
```

**Aggregate**:
```python
def aggregate_operation(thoughts):
    thoughts_str = '\n'.join([f"{i+1}. {t}" for i, t in enumerate(thoughts)])

    prompt = f"""Synthesize the following thoughts into one unified solution:

{thoughts_str}

Unified solution:"""

    aggregated = model.generate(prompt, temperature=0.3)
    return aggregated
```

**Refine**:
```python
def refine_operation(thought, feedback=None, max_iterations=3):
    current = thought

    for i in range(max_iterations):
        if feedback:
            prompt = f"""Current: {current}
Feedback: {feedback}

Improve the current thought based on feedback:"""
        else:
            prompt = f"""Current: {current}

Critically analyze and improve this thought:"""

        refined = model.generate(prompt, temperature=0.5)

        # Check if improvement is sufficient
        if score_operation(refined) > score_operation(current) + 0.1:
            current = refined
        else:
            break

    return current
```

**Score**:
```python
def score_operation(thought, problem):
    prompt = f"""Problem: {problem}
Thought: {thought}

Rate this thought on a scale from 0.0 to 1.0 based on:
- Correctness (is it accurate?)
- Completeness (does it address all aspects?)
- Relevance (is it on-topic?)

Score (0.0-1.0):"""

    response = model.generate(prompt, temperature=0.0)
    score = extract_float(response, default=0.5)
    return clip(score, 0.0, 1.0)
```

## 6. Code Walkthrough

Reference: `/Users/kevinyu/Projects/Nexus/nexus/models/nlp/reasoning/graph_of_thoughts.py`

### Basic Usage

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

# Execute graph on problem
outputs = got(problem_embedding=problem_emb)

print(f"Best score: {outputs['best_score']}")
print(f"Nodes created: {outputs['num_nodes']}")
print(f"Operation sequence: {outputs['operation_history']}")
```

### Neural Operation Modules

**GenerateOperation**:
```python
class GenerateOperation(NexusModule):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_generated = config["num_generated"]

        self.generator = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size * self.num_generated)
        )

    def forward(self, source_embedding):
        """Generate num_generated new thoughts from source"""
        generated_flat = self.generator(source_embedding)
        generated = generated_flat.view(-1, self.num_generated, self.hidden_size)

        # Diversity loss encourages distinct thoughts
        diversity_loss = self.encourage_orthogonality(generated)

        return {
            "generated": generated,
            "diversity_loss": diversity_loss
        }

    def encourage_orthogonality(self, embeddings):
        """Penalize similar embeddings"""
        # embeddings: (batch, num_generated, hidden_size)
        normalized = F.normalize(embeddings, dim=-1)
        similarity = torch.bmm(normalized, normalized.transpose(1, 2))

        # Penalize off-diagonal elements (want orthogonal)
        mask = 1 - torch.eye(self.num_generated, device=embeddings.device)
        loss = (similarity * mask).abs().mean()

        return loss
```

**AggregateOperation**:
```python
class AggregateOperation(NexusModule):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]

        self.aggregate_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            batch_first=True
        )

        self.agg_query = nn.Parameter(
            torch.randn(1, 1, self.hidden_size)
        )

    def forward(self, thought_embeddings):
        """Merge multiple thoughts via attention"""
        batch_size = thought_embeddings.size(0)

        # Learnable query for aggregation
        query = self.agg_query.expand(batch_size, -1, -1)

        # Attention over all thoughts
        aggregated, attn = self.aggregate_attention(
            query, thought_embeddings, thought_embeddings
        )

        return {
            "aggregated": aggregated.squeeze(1),
            "attention_weights": attn
        }
```

**RefineOperation**:
```python
class RefineOperation(NexusModule):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]

        self.refine_network = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size * 2),
            nn.GELU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size)
        )

        # Gate controls how much to update
        self.gate = nn.Sequential(
            nn.Linear(self.hidden_size * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, thought_embedding, context_embedding):
        """Refine thought given context"""
        combined = torch.cat([thought_embedding, context_embedding], dim=-1)

        refined_candidate = self.refine_network(combined)

        # Gating: how much to update
        gate_values = self.gate(combined)
        refined = gate_values * refined_candidate + (1 - gate_values) * thought_embedding

        return {
            "refined": refined,
            "gate_values": gate_values
        }
```

**ScoreOperation**:
```python
class ScoreOperation(NexusModule):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]

        self.scorer = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )

    def forward(self, thought_embedding, problem_embedding):
        """Score thought in context of problem"""
        combined = torch.cat([thought_embedding, problem_embedding], dim=-1)
        score = self.scorer(combined)

        return {"score": score}
```

### Full Graph Execution

```python
class GoTController(NexusModule):
    def __init__(self, config):
        super().__init__()
        self.generate_op = GenerateOperation(config)
        self.aggregate_op = AggregateOperation(config)
        self.refine_op = RefineOperation(config)
        self.score_op = ScoreOperation(config)

    def forward(self, problem_embedding):
        # Build graph structure
        graph = ThoughtGraph()

        # Root: problem
        root_id = graph.add_node(problem_embedding, "root")

        # Generate initial thoughts
        gen_outputs = self.generate_op(problem_embedding)
        thought_ids = []

        for i in range(gen_outputs["generated"].size(1)):
            thought_emb = gen_outputs["generated"][:, i, :]
            thought_id = graph.add_node(thought_emb, "generate", parents=[root_id])
            thought_ids.append(thought_id)

        # Aggregate thoughts
        all_thoughts = torch.stack([
            graph.nodes[tid]["thought"] for tid in thought_ids
        ], dim=1)

        agg_outputs = self.aggregate_op(all_thoughts)
        agg_id = graph.add_node(agg_outputs["aggregated"], "aggregate", parents=thought_ids)

        # Refine
        refine_outputs = self.refine_op(
            agg_outputs["aggregated"],
            problem_embedding
        )
        refined_id = graph.add_node(refine_outputs["refined"], "refine", parents=[agg_id])

        # Score
        score_outputs = self.score_op(
            refine_outputs["refined"],
            problem_embedding
        )

        return {
            "final_thought": refine_outputs["refined"],
            "best_score": score_outputs["score"],
            "num_nodes": len(graph.nodes),
            "graph": graph
        }
```

## 7. Optimization Tricks: Temperature & Aggregation Methods

### 1. Adaptive Temperature

Vary temperature based on operation:

```python
temperature_map = {
    "generate": 0.8,    # High diversity for generation
    "aggregate": 0.3,   # Low temperature for merging
    "refine": 0.5,      # Medium for refinement
    "score": 0.0        # Greedy for scoring
}
```

### 2. Weighted Aggregation

Weight thoughts by their scores:

```python
def weighted_aggregate(thoughts, scores):
    weights = F.softmax(scores / temperature, dim=0)
    aggregated = torch.sum(weights.unsqueeze(-1) * thoughts, dim=0)
    return aggregated
```

### 3. Hierarchical Aggregation

Aggregate in stages for many thoughts:

```python
def hierarchical_aggregate(thoughts):
    while len(thoughts) > 1:
        # Pair and aggregate
        paired = []
        for i in range(0, len(thoughts), 2):
            if i + 1 < len(thoughts):
                pair = aggregate_operation([thoughts[i], thoughts[i+1]])
            else:
                pair = thoughts[i]
            paired.append(pair)
        thoughts = paired

    return thoughts[0]
```

### 4. Refinement Scheduling

Decrease refinement intensity over iterations:

```python
def refinement_gate_schedule(iteration, max_iterations):
    # Start with large updates, decrease over time
    return 1.0 - (iteration / max_iterations) * 0.7

gate_value = refinement_gate_schedule(current_iter, max_iters)
refined = gate_value * new_thought + (1 - gate_value) * old_thought
```

### 5. Parallel Graph Execution

Execute independent subgraphs in parallel:

```python
import asyncio

async def execute_graph_parallel(graph):
    levels = graph.get_levels()  # Group by dependency level

    for level in levels:
        # Execute all nodes at this level in parallel
        tasks = [execute_node(node_id) for node_id in level]
        await asyncio.gather(*tasks)
```

### 6. Thought Pruning

Remove low-scoring thoughts to save computation:

```python
def prune_thoughts(thoughts, scores, keep_ratio=0.5):
    k = max(1, int(len(thoughts) * keep_ratio))
    top_k_indices = torch.topk(scores, k).indices
    return [thoughts[i] for i in top_k_indices]
```

## 8. Experiments: GSM8K & MMLU Benchmarks

### Results from Besta et al. (2024)

**Sorting Task**:
| Method | Accuracy | LLM Calls |
|--------|----------|-----------|
| IO Prompting | 12% | 1 |
| CoT | 18% | 1 |
| CoT-SC (n=10) | 24% | 10 |
| ToT (b=3) | 39% | ~30 |
| GoT | **62%** | ~25 |

**Keyword Counting**:
| Method | Accuracy | LLM Calls |
|--------|----------|-----------|
| IO | 61% | 1 |
| CoT | 68% | 1 |
| ToT | 78% | ~30 |
| GoT | **91%** | ~20 |

**Set Operations**:
| Method | Accuracy | LLM Calls |
|--------|----------|-----------|
| CoT | 44% | 1 |
| ToT | 56% | ~30 |
| GoT | **83%** | ~25 |

### GSM8K Math Word Problems

| Method | GSM8K | Avg. LLM Calls |
|--------|-------|----------------|
| CoT | 57.2% | 1 |
| Self-Consistency (n=40) | 74.4% | 40 |
| ToT (BFS) | 78.9% | ~50 |
| GoT (Decomp-Agg) | **81.3%** | ~30 |

GoT achieves better accuracy with fewer calls by using aggregation efficiently.

### MMLU Complex Reasoning

| Subject | CoT | ToT | GoT |
|---------|-----|-----|-----|
| Abstract Algebra | 38.2% | 52.3% | **58.7%** |
| Formal Logic | 42.1% | 58.7% | **65.3%** |
| Philosophy | 55.6% | 62.1% | **69.8%** |

### Ablation Studies

**Operation Importance**:
| Operations Used | Accuracy (Sorting) |
|----------------|-------------------|
| Generate only | 18% |
| Generate + Score | 32% |
| Generate + Aggregate | 48% |
| Generate + Aggregate + Refine | 56% |
| All (+ Score) | **62%** |

**Aggregation Methods**:
| Aggregation | Accuracy |
|-------------|----------|
| Random selection | 24% |
| Concatenation | 38% |
| Attention-based | **62%** |
| Weighted average | 58% |

**Refinement Iterations**:
| Iterations | Accuracy | Time |
|------------|----------|------|
| 0 | 52% | 1x |
| 1 | 58% | 1.3x |
| 3 | **62%** | 1.8x |
| 5 | 62% | 2.4x |

Diminishing returns after 3 iterations.

## 9. Pitfalls

### 1. Over-complicated Graphs

**Problem**: Creating overly complex graph structures.

**Solution**:
- Start simple (linear chains or single decompose-aggregate)
- Add complexity only when needed
- Most tasks need ≤10 nodes

### 2. Weak Aggregation

**Problem**: Aggregation loses information from individual thoughts.

**Solution**:
```python
# Include all source information
aggregated_prompt = f"""
Merge these thoughts WITHOUT losing key information:
{thought_1}
{thought_2}

Merged (preserve all important details):"""
```

### 3. Infinite Refinement Loops

**Problem**: Refinement never converges.

**Solution**:
```python
max_iterations = 3
convergence_threshold = 0.01

for i in range(max_iterations):
    refined = refine(current)
    if abs(score(refined) - score(current)) < convergence_threshold:
        break
    current = refined
```

### 4. Inconsistent Scoring

**Problem**: Scores don't reflect actual quality.

**Solution**:
- Use multiple criteria (correctness, completeness, relevance)
- Normalize scores across nodes
- Calibrate with ground truth examples

### 5. Memory Explosion

**Problem**: Large graphs consume excessive memory.

**Solution**:
```python
# Limit graph size
max_nodes = 20

# Prune after each operation
if len(graph.nodes) > max_nodes:
    prune_lowest_scored_nodes(graph, keep=max_nodes // 2)
```

### 6. Sequential Execution Bottleneck

**Problem**: Executing graph sequentially is slow.

**Solution**:
- Identify independent nodes and execute in parallel
- Use asynchronous operations
- Batch operations where possible

### 7. Poor Graph Topology

**Problem**: Graph structure doesn't match problem structure.

**Solution**:
- For decomposable problems: Use decompose-aggregate
- For iterative tasks: Use refinement loops
- For multi-strategy: Use parallel paths + aggregation

## 10. References

1. **Graph of Thoughts: Solving Elaborate Problems with Large Language Models**
   Besta et al., 2024
   https://arxiv.org/abs/2308.09687

2. **Tree of Thoughts: Deliberate Problem Solving with Large Language Models**
   Yao et al., NeurIPS 2023
   https://arxiv.org/abs/2305.10601

3. **Beyond Chain-of-Thought: A Survey of Chain-of-X**
   Liu et al., 2023
   https://arxiv.org/abs/2305.08291

4. **Graph Neural Prompting for Large Language Models**
   Zhang et al., 2024
   https://arxiv.org/abs/2309.15427

5. **Reasoning with Language Model is Planning with World Model**
   Hao et al., EMNLP 2023
   https://arxiv.org/abs/2305.14992

## Related Methods

- **Tree of Thoughts**: Restricted to tree structure (GoT generalizes)
- **Chain-of-Thought**: Linear reasoning (special case of GoT)
- **Self-Consistency**: Parallel generation + voting (GoT with aggregation)
- **Least-to-Most**: Sequential decomposition (GoT with linear graph)
- **Reflexion**: Iterative refinement with feedback (GoT with cycles)
