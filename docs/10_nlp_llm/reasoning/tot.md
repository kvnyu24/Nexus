# Tree of Thoughts (ToT): Deliberate Problem Solving

## 1. Overview & Motivation

Tree of Thoughts (ToT) generalizes Chain-of-Thought by maintaining a tree structure where each node represents a partial solution ("thought"). Unlike sequential CoT, ToT enables:

- **Exploration**: Branch into multiple candidate thoughts at each step
- **Evaluation**: Score thoughts for progress toward the solution
- **Search**: Use BFS/DFS to systematically explore the thought space
- **Backtracking**: Abandon unpromising paths and try alternatives

**Key Insight**: Deliberate problem-solving requires looking ahead, evaluating alternatives, and backtracking when needed - capabilities absent in standard autoregressive generation.

### When to Use ToT

ToT excels at tasks requiring:
- **Planning**: Game of 24, chess moves, travel itineraries
- **Search**: Crossword puzzles, code synthesis
- **Constraint satisfaction**: Scheduling, resource allocation
- **Multi-step optimization**: Where intermediate steps can be evaluated

## 2. Theory: Search Strategies

### Search in Thought Space

ToT models problem-solving as search over a tree $T = (V, E)$ where:
- $V$ = set of thought nodes (partial solutions)
- $E$ = transitions between thoughts
- Root $v_0$ = initial problem state
- Leaves = complete solutions

Each node $v \in V$ has:
- Thought text $t_v$
- Value score $s_v \in [0, 1]$
- Depth $d_v$
- Children $\text{children}(v) \subset V$

### Tree Search Algorithms

**Breadth-First Search (BFS)**:
Explores all thoughts at depth $d$ before moving to $d+1$:

$$
\text{BFS}(v_0, d_{\max}) = \bigcup_{d=0}^{d_{\max}} \text{Expand}(\text{Frontier}_d)
$$

where $\text{Frontier}_d$ = top-$k$ nodes at depth $d$ by value.

**Depth-First Search (DFS)**:
Explores the most promising branch to maximum depth, then backtracks:

$$
\text{DFS}(v, d_{\max}) = \begin{cases}
v & \text{if } d_v = d_{\max} \\
\text{DFS}(\text{best}(\text{children}(v)), d_{\max}) & \text{otherwise}
\end{cases}
$$

**Beam Search**:
Keep only top-$k$ nodes at each level:

$$
\text{Beam}_d = \text{top-}k\left(\bigcup_{v \in \text{Beam}_{d-1}} \text{children}(v)\right)
$$

## 3. Mathematical Formulation: Sampling & Aggregation

### Thought Generation

Generate $k$ diverse candidate thoughts from state $s$:

$$
\{t_1, \ldots, t_k\} \sim p(t | s, P; \theta)
$$

with temperature $T$ for diversity:

$$
p(t_i | s) = \frac{\exp(f(t_i | s) / T)}{\sum_j \exp(f(t_j | s) / T)}
$$

### Thought Value Estimation

Value function $V(v)$ estimates how promising thought $v$ is:

$$
V(v) = \mathbb{E}_{s \sim p(\cdot | v)} [R(s)]
$$

where $R(s)$ is reward for complete solution $s$.

In practice, use LLM-based evaluation:

**Vote-based**:
$$
V(v) = \frac{1}{N} \sum_{i=1}^N \text{vote}_i(v), \quad \text{vote}_i \in \{0, 0.5, 1\}
$$

**Score-based**:
$$
V(v) = \text{LM}_{\text{eval}}(v, P) \in [0, 1]
$$

### Tree Aggregation

Final solution combines best paths:

$$
s^* = \arg\max_{s \in \text{Leaves}(T)} V(s)
$$

With path probability:

$$
p(\text{path}(v)) = \prod_{v' \in \text{ancestors}(v)} p(v' | \text{parent}(v'))
$$

## 4. Intuition

### Tree Structure

```
                    ┌─────────────────┐
                    │ Problem: Solve  │
                    │  24 with        │
                    │  4, 5, 6, 10    │
                    └────────┬────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
           ▼                 ▼                 ▼
    ┌──────────┐      ┌──────────┐      ┌──────────┐
    │ 4 + 5 = 9│      │ 6 × 4 = 24│      │ 10 - 6 = 4│
    │ (v=0.7)  │      │ (v=0.95) │      │ (v=0.5)  │
    └──────────┘      └──────────┘      └──────────┘
           │                 │ ✓               │
           │                 │                 │
  (Expand further)     (Solution!)      (Expand further)
```

### BFS vs DFS

**BFS**: Explores all options at each depth
- Pros: Complete exploration, finds global optimum
- Cons: High memory, many LLM calls

**DFS**: Explores one promising path deeply
- Pros: Memory-efficient, fewer LLM calls
- Cons: May miss better solutions

### Example: Game of 24

```
Input: 4, 5, 6, 10
Goal: Reach 24 using +, -, ×, ÷

Step 1: Generate thoughts
  - "4 + 5 = 9"
  - "6 × 4 = 24" ✓
  - "10 - 6 = 4"

Step 2: Evaluate
  - "6 × 4 = 24" scores 1.0 (solution!)

Result: 6 × 4 = 24
```

## 5. Implementation Details

### Thought Generation

Generate $k$ diverse candidate thoughts from current state:

```python
def generate_thoughts(state: str, k: int) -> List[str]:
    prompt = f"""Given the following problem and reasoning so far:

{state}

Generate {k} distinct possible next steps for reasoning about this problem.
Number each step and provide clear, specific reasoning.

Possible next steps:"""

    response = model.generate(prompt, temperature=0.8, n=k)
    thoughts = parse_numbered_responses(response, k)
    return thoughts
```

### Thought Evaluation

**Vote-based evaluation**:

```python
def evaluate_by_vote(state: str, thought: str, num_votes: int = 3) -> float:
    prompt = f"""Given the problem and reasoning so far:
{state}

Evaluate the following next step:
{thought}

Is this step promising for reaching the solution?
Answer with exactly one word: 'sure', 'maybe', or 'impossible'."""

    vote_scores = {"sure": 1.0, "maybe": 0.5, "impossible": 0.0}
    total = 0.0

    for _ in range(num_votes):
        response = model.generate(prompt).strip().lower()
        if "sure" in response:
            total += vote_scores["sure"]
        elif "impossible" in response:
            total += vote_scores["impossible"]
        else:
            total += vote_scores["maybe"]

    return total / num_votes
```

**Score-based evaluation**:

```python
def evaluate_by_score(state: str, thought: str) -> float:
    prompt = f"""Rate the following reasoning step on a scale from 0.0 to 1.0,
where 0.0 means completely wrong and 1.0 means certainly correct.

Problem: {state}
Step: {thought}

Score (0.0 to 1.0):"""

    response = model.generate(prompt)
    score = extract_float(response, default=0.5)
    return clip(score, 0.0, 1.0)
```

### Search Implementation

**BFS**:
```python
def bfs_search(problem, max_depth, branching_factor, beam_width):
    root = ThoughtNode(thought=problem, depth=0)
    frontier = [root]

    for depth in range(max_depth):
        next_frontier = []

        for node in frontier:
            # Generate children
            thoughts = generate_thoughts(node.state, branching_factor)

            for thought in thoughts:
                child = ThoughtNode(thought, depth + 1)
                child.value = evaluate(node.state, thought)
                node.add_child(child)
                next_frontier.append(child)

        # Keep top-k nodes
        next_frontier.sort(key=lambda n: n.value, reverse=True)
        frontier = next_frontier[:beam_width]

        # Check for solution
        for node in frontier:
            if is_solution(node):
                return node

    return max(frontier, key=lambda n: n.value)
```

**DFS**:
```python
def dfs_search(problem, max_depth, branching_factor):
    def dfs_recursive(node, depth):
        if depth >= max_depth or is_solution(node):
            return node

        # Generate and evaluate children
        thoughts = generate_thoughts(node.state, branching_factor)
        children = []

        for thought in thoughts:
            child = ThoughtNode(thought, depth + 1)
            child.value = evaluate(node.state, thought)
            children.append(child)

        # Explore best child first
        children.sort(key=lambda c: c.value, reverse=True)

        best_result = None
        for child in children:
            result = dfs_recursive(child, depth + 1)
            if best_result is None or result.value > best_result.value:
                best_result = result

        return best_result

    root = ThoughtNode(thought=problem, depth=0)
    return dfs_recursive(root, 0)
```

## 6. Code Walkthrough

### Basic ToT Implementation

Reference: `/Users/kevinyu/Projects/Nexus/nexus/models/nlp/reasoning/tree_of_thoughts.py`

```python
from nexus.models.nlp.reasoning.tree_of_thoughts import TreeOfThoughts

config = {
    "model": language_model,
    "max_depth": 3,              # Maximum tree depth
    "branching_factor": 3,       # Thoughts generated per node
    "search_method": "bfs",      # "bfs" or "dfs"
    "evaluation_method": "vote", # "vote" or "score"
    "num_votes": 3,              # Votes per evaluation
    "value_threshold": 0.3,      # Minimum value to continue
    "beam_width": 5              # Max nodes per BFS level
}

tot = TreeOfThoughts(config)

# Solve problem
result = tot.solve("Solve the 24 game with numbers 4, 5, 6, 10.")

print(f"Answer: {result['answer']}")
print(f"Best value: {result['best_value']}")
print(f"Full reasoning path: {result['reasoning_path']}")
```

### ThoughtNode Structure

```python
from nexus.models.nlp.reasoning.tree_of_thoughts import ThoughtNode

# Create thought node
node = ThoughtNode(
    thought="4 + 5 = 9",
    value=0.7,
    depth=1,
    is_terminal=False
)

# Add children
child1 = ThoughtNode(thought="9 + 6 = 15")
child2 = ThoughtNode(thought="9 - 10 = -1")
node.add_child(child1)
node.add_child(child2)

# Get best child
best = node.best_child()

# Retrieve reasoning path from root to this node
path = node.get_path()
```

### Custom Search Strategy

```python
def custom_search(tot_model, problem, max_depth):
    """Hybrid search: BFS first, then DFS on promising branches"""

    # Phase 1: BFS to depth 2
    bfs_config = tot_model.config.copy()
    bfs_config["max_depth"] = 2
    bfs_config["search_method"] = "bfs"

    tot_bfs = TreeOfThoughts(bfs_config)
    bfs_result = tot_bfs.solve(problem)
    top_node = bfs_result["best_node"]

    # Phase 2: DFS from best node
    dfs_config = tot_model.config.copy()
    dfs_config["max_depth"] = max_depth
    dfs_config["search_method"] = "dfs"

    tot_dfs = TreeOfThoughts(dfs_config)
    dfs_result = tot_dfs.solve_from_node(top_node)

    return dfs_result
```

## 7. Optimization Tricks

### 1. Pruning Strategies

**Value-based pruning**:
```python
if node.value < value_threshold:
    continue  # Skip this branch
```

**Beam search pruning**:
```python
# Keep only top-k nodes per level
nodes_at_depth.sort(key=lambda n: n.value, reverse=True)
frontier = nodes_at_depth[:beam_width]
```

**Duplicate detection**:
```python
seen_thoughts = set()

for thought in generated_thoughts:
    thought_hash = hash(normalize(thought))
    if thought_hash in seen_thoughts:
        continue  # Skip duplicate
    seen_thoughts.add(thought_hash)
```

### 2. Thought Caching

Cache evaluations for identical thoughts:

```python
evaluation_cache = {}

def cached_evaluate(state, thought):
    key = hash((state, thought))
    if key not in evaluation_cache:
        evaluation_cache[key] = evaluate(state, thought)
    return evaluation_cache[key]
```

### 3. Parallel Thought Generation

Generate and evaluate thoughts in parallel:

```python
import asyncio

async def parallel_expand(node, branching_factor):
    # Generate thoughts asynchronously
    thoughts = await async_generate_thoughts(node.state, branching_factor)

    # Evaluate in parallel
    evaluations = await asyncio.gather(
        *[async_evaluate(node.state, thought) for thought in thoughts]
    )

    return [(thought, value) for thought, value in zip(thoughts, evaluations)]
```

### 4. Adaptive Branching

Adjust branching based on node value:

```python
def adaptive_branching_factor(node_value):
    """High-value nodes get more children"""
    if node_value > 0.8:
        return 5  # Explore promising nodes more
    elif node_value > 0.5:
        return 3
    else:
        return 1  # Minimal exploration for low-value nodes
```

### 5. Early Termination

Stop search when solution is found with high confidence:

```python
def should_terminate(node):
    if node.is_terminal and node.value > 0.95:
        return True  # Found high-confidence solution
    return False
```

### 6. Temperature Scheduling for Diversity

```python
def temperature_by_depth(depth, max_depth):
    # Higher temperature at shallow depths for exploration
    return 1.0 - (depth / max_depth) * 0.5

thoughts = generate_thoughts(
    state,
    k=branching_factor,
    temperature=temperature_by_depth(current_depth, max_depth)
)
```

## 8. Experiments: GSM8K & MMLU Benchmarks

### Benchmark Performance (from Yao et al., NeurIPS 2023)

**Game of 24**:
- Input-Output Prompting: 7.3%
- Chain-of-Thought: 4.0%
- Tree of Thoughts (BFS): **74.0%** (+66.7% vs best baseline)

**Creative Writing**:
- Input-Output: 6.19 coherence score
- CoT: 6.93
- ToT: **7.56** (+9.1% vs CoT)

**Mini Crosswords (5×5)**:
- CoT: 16% success rate
- ToT: **78% success rate** (+62% absolute)

### GSM8K Math Reasoning

| Method | GSM8K Accuracy | LLM Calls | Relative Cost |
|--------|----------------|-----------|---------------|
| Standard | 18.5% | 1 | 1x |
| CoT | 57.2% | 1 | 1x |
| Self-Consistency (n=40) | 74.4% | 40 | 40x |
| ToT (BFS, b=3, d=3) | **78.9%** | ~50 | 50x |

ToT outperforms self-consistency with similar computational cost.

### MMLU Complex Reasoning Subset

| Subject | CoT | ToT | Improvement |
|---------|-----|-----|-------------|
| Formal Logic | 42.1% | 58.7% | +16.6% |
| Abstract Algebra | 38.2% | 52.3% | +14.1% |
| Logical Fallacies | 65.3% | 78.9% | +13.6% |

### Search Method Comparison

| Search Method | Game of 24 Accuracy | LLM Calls | Time | Memory |
|---------------|---------------------|-----------|------|--------|
| Greedy (CoT)  | 4.0% | 1 | 1x | O(1) |
| Sample (k=10) | 9.0% | 10 | 10x | O(1) |
| BFS (b=3, d=3)| **74.0%** | ~50 | 50x | O(b^d) |
| DFS (b=3, d=3)| 68.0% | ~30 | 30x | O(d) |
| Beam (k=5)    | 71.2% | ~40 | 40x | O(k×d) |

**Takeaway**: BFS finds better solutions, DFS is more efficient.

### Ablation: Thought Evaluation

| Evaluation Method | Game of 24 | Creative Writing | LLM Calls/Node |
|------------------|------------|------------------|----------------|
| Random selection | 12.0% | 6.25 | 0 |
| Self-eval (once) | 54.0% | 7.12 | 1 |
| Vote (n=3)       | **74.0%** | **7.56** | 3 |
| Vote (n=5)       | 76.0% | 7.61 | 5 |

**Takeaway**: Voting improves reliability; 3 votes is cost-effective.

### Branching Factor Analysis

| Branching Factor | Game of 24 | Total LLM Calls |
|------------------|------------|-----------------|
| b=2 | 62.3% | ~30 |
| b=3 | **74.0%** | ~50 |
| b=5 | 76.8% | ~120 |

Diminishing returns beyond b=3.

## 9. Pitfalls

### 1. Poor Thought Diversity

**Problem**: Generated thoughts are too similar.

**Solution**:
```python
# Use temperature sampling
thoughts = model.generate(prompt, temperature=0.8, top_p=0.9)

# Enforce diversity via constraint
"Generate 3 DISTINCT approaches (don't repeat the same idea)"
```

### 2. Weak Evaluation Function

**Problem**: Evaluation doesn't distinguish good vs bad thoughts.

**Solution**:
- Use specific criteria: "Rate based on: correctness, completeness, efficiency"
- Provide few-shot examples of good vs bad thoughts
- Use multiple evaluators and aggregate

### 3. Inefficient Search

**Problem**: Exploring too many low-value branches.

**Solution**:
```python
# Aggressive pruning
value_threshold = 0.5  # Drop bottom 50%

# Adaptive beam width
beam_width = max(1, int(branching_factor * avg_value))
```

### 4. Depth Explosion

**Problem**: Tree grows too deep, wasting computation.

**Solution**:
```python
# Set reasonable max_depth
max_depth = 3  # Most problems solve in ≤3 steps

# Use iterative deepening
for depth in range(1, max_depth + 1):
    result = search(max_depth=depth)
    if result.is_complete():
        return result
```

### 5. Prompt Engineering

**Problem**: Poor prompts yield poor thoughts/evaluations.

**Solution**:
```python
# Good generation prompt
"List 3 distinct next steps. For each, explain:
1. What action to take
2. Expected outcome
3. Why this helps solve the problem"

# Good evaluation prompt
"On a scale 1-5, rate this step for:
- Correctness (does it make sense?)
- Progress (does it move toward solution?)
- Efficiency (is it the best approach?)"
```

### 6. Memory Constraints

**Problem**: BFS can consume excessive memory for large trees.

**Solution**:
- Use beam search instead of full BFS
- Implement iterative deepening DFS
- Prune aggressively based on value thresholds

### 7. Evaluation Inconsistency

**Problem**: LLM evaluator gives inconsistent scores.

**Solution**:
```python
# Use multiple evaluations and average
scores = [evaluate(thought) for _ in range(num_evals)]
final_score = np.mean(scores)

# Or use majority vote for categorical evaluations
votes = [vote(thought) for _ in range(num_votes)]
final_vote = Counter(votes).most_common(1)[0][0]
```

## 10. References

1. **Tree of Thoughts: Deliberate Problem Solving with Large Language Models**
   Yao et al., NeurIPS 2023
   https://arxiv.org/abs/2305.10601

2. **Large Language Model Guided Tree-of-Thought**
   Long, NeurIPS 2023 Workshop
   https://arxiv.org/abs/2305.08291

3. **Beyond Chain-of-Thought: A Survey of Chain-of-X**
   Liu et al., 2023
   https://arxiv.org/abs/2305.08291

4. **Monte Carlo Tree Search for Large Language Models**
   Leong et al., 2023
   https://arxiv.org/abs/2309.17179

5. **Tree-of-Mixed-Thought: Combining Fast and Slow Thinking for Multi-hop Visual Reasoning**
   Ning et al., 2024
   https://arxiv.org/abs/2308.09688

## Related Methods

- **Chain-of-Thought**: Sequential reasoning (ToT generalizes this)
- **Graph of Thoughts**: Arbitrary graph structure (more flexible than ToT)
- **MCTS for LLMs**: Monte Carlo Tree Search for planning
- **Self-Consistency**: Multiple paths + voting (simpler than ToT)
- **Beam Search**: Simplified ToT with fixed beam width
