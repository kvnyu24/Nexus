# Self-Consistency: Improving Chain of Thought Reasoning

## 1. Overview & Motivation

Self-Consistency improves Chain-of-Thought by sampling multiple diverse reasoning paths using temperature sampling, then aggregating final answers via majority voting. This simple technique significantly improves performance on arithmetic, commonsense, and symbolic reasoning tasks.

**Key Insight**: Complex reasoning problems often have multiple valid solution paths. Marginalizing over these paths produces more reliable answers than any single path.

### Why Self-Consistency?

Chain-of-Thought with greedy decoding has limitations:
- Single reasoning path may contain errors
- Sensitive to prompt phrasing
- No uncertainty quantification

Self-Consistency addresses these by:
1. **Exploring multiple reasoning paths**: Sample diverse solutions
2. **Aggregating answers**: Use majority voting to select most consistent answer
3. **Improving robustness**: Reduce sensitivity to individual path errors
4. **Estimating confidence**: Voting distribution indicates certainty

### When to Use Self-Consistency

Self-Consistency excels when:
- **Multiple valid paths exist**: Problems solvable through different approaches
- **Robustness is critical**: Need to reduce variance in predictions
- **Computational budget allows**: Can afford multiple forward passes
- **Discrete answer space**: Clear final answers to vote over

## 2. Theory: Prompting Strategies

### Decoding Strategies Comparison

**Greedy Decoding (Standard CoT)**:
```
Q: Problem
A: [Single reasoning path] → Answer
```

**Sampling-based Self-Consistency**:
```
Q: Problem
A1: [Path 1] → Answer_1
A2: [Path 2] → Answer_2
A3: [Path 3] → Answer_3
...
An: [Path n] → Answer_n

Final: majority_vote([Answer_1, ..., Answer_n])
```

### Prompting Best Practices

**1. Use CoT Prompting as Base**:
```
Q: {problem}
A: Let's think step by step.
```

**2. Enable Diversity via Temperature**:
- Set temperature T ∈ [0.5, 1.0]
- Higher T → more diverse paths
- Sweet spot often T = 0.7

**3. Maintain Consistent Format**:
```
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
   Each can has 3 tennis balls. How many tennis balls does he have now?

A: Roger started with 5 balls.
   He bought 2 cans with 3 balls each.
   2 × 3 = 6 new balls.
   Total: 5 + 6 = 11 balls.

Therefore, the answer is: 11
```

**4. Extract Final Answer Clearly**:
Use consistent markers:
- "Therefore, the answer is: X"
- "Final Answer: X"
- "The result is X"

## 3. Mathematical Formulation: Sampling & Aggregation

### Standard CoT (Greedy Decoding)

$$
\hat{a} = \text{argmax}_a \; p(a | \text{CoT}(q))
$$

where $\text{CoT}(q)$ is the greedy reasoning chain.

### Self-Consistency (Marginalization)

Ideally, marginalize over all reasoning paths:

$$
p(a | q) = \sum_{r \in \mathcal{R}} p(a | r, q) p(r | q)
$$

Approximate via Monte Carlo sampling:

$$
\hat{a} = \text{argmax}_a \sum_{i=1}^m \mathbb{1}[a_i = a]
$$

where:
- $m$ = number of sampled paths
- $r_i \sim p(\cdot | q; T)$ with temperature $T > 0$
- $a_i = \text{extract\_answer}(r_i)$

### Sampling Process

Sample reasoning paths with temperature:

$$
p(r | q; T) = \prod_{t=1}^{|r|} p(r_t | r_{<t}, q; T)
$$

where:

$$
p(r_t | r_{<t}, q; T) = \frac{\exp(f(r_t | r_{<t}, q) / T)}{\sum_{r'} \exp(f(r' | r_{<t}, q) / T)}
$$

### Majority Voting

Simple majority vote:

$$
\hat{a} = \text{mode}(\{a_1, a_2, \ldots, a_m\})
$$

Weighted voting (by path probability):

$$
\hat{a} = \text{argmax}_a \sum_{i: a_i = a} w_i
$$

where $w_i = p(r_i | q)$.

### Confidence Estimation

Confidence based on voting distribution:

$$
\text{confidence}(\hat{a}) = \frac{\text{count}(\hat{a})}{m}
$$

Entropy-based confidence:

$$
\text{confidence} = 1 - H(p), \quad H(p) = -\sum_a p(a) \log p(a)
$$

## 4. Intuition

### Visual Representation

```
                    ┌─────────────┐
                    │   Problem   │
                    └──────┬──────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
    ┌────────┐        ┌────────┐        ┌────────┐
    │ Path 1 │        │ Path 2 │        │ Path 3 │
    │  T=0.7 │        │  T=0.7 │        │  T=0.7 │
    └───┬────┘        └───┬────┘        └───┬────┘
        │                 │                 │
        ▼                 ▼                 ▼
    Answer: 42       Answer: 42       Answer: 43
        │                 │                 │
        └─────────────────┼─────────────────┘
                          │
                   ┌──────▼──────┐
                   │Majority Vote│
                   └──────┬──────┘
                          │
                          ▼
                    Final: 42 (2/3 votes)
```

### Example: Math Problem

**Problem**: "Janet's ducks lay 16 eggs per day. She eats three for breakfast and bakes muffins with four. She sells the rest at the farmers' market. How many does she sell?"

**Path 1** (Correct):
- Starts with 16 eggs
- Eats 3, has 13 left
- Uses 4 for muffins, has 9 left
- Sells 9
→ Answer: 9

**Path 2** (Correct, different ordering):
- Total consumed: 3 + 4 = 7
- Remaining: 16 - 7 = 9
- Sells 9
→ Answer: 9

**Path 3** (Error in arithmetic):
- Eats 3: 16 - 3 = 13
- Uses 4: 13 - 4 = 8  [Error!]
- Sells 8
→ Answer: 8

**Majority Vote**: 9 (2 out of 3) → **Correct Answer**

### Why It Works

1. **Error Correction**: Individual errors are outvoted
2. **Uncertainty Reduction**: Multiple paths provide confidence estimates
3. **Robustness**: Less sensitive to single-path failures
4. **Diversity**: Temperature sampling explores solution space

## 5. Implementation Details

### Basic Algorithm

```python
def self_consistency(question, num_samples=40, temperature=0.7):
    reasoning_paths = []
    answers = []

    # Sample diverse reasoning paths
    for i in range(num_samples):
        prompt = f"{question}\n\nLet's think step by step:"

        reasoning = model.generate(
            prompt,
            temperature=temperature,
            top_p=0.9,
            max_tokens=256
        )

        reasoning_paths.append(reasoning)

        # Extract final answer
        answer = extract_answer(reasoning)
        answers.append(answer)

    # Majority vote
    answer_counts = Counter(answers)
    final_answer, count = answer_counts.most_common(1)[0]

    confidence = count / num_samples

    return {
        'answer': final_answer,
        'confidence': confidence,
        'distribution': answer_counts,
        'reasoning_paths': reasoning_paths
    }
```

### Answer Extraction

```python
def extract_answer(reasoning_text):
    """Extract final answer from reasoning chain"""

    # Look for common patterns
    patterns = [
        r"[Tt]he answer is:?\s*(.+)",
        r"[Ff]inal [Aa]nswer:?\s*(.+)",
        r"[Tt]herefore,?\s+(.+)",
        r"[Ss]o the answer is:?\s*(.+)"
    ]

    for pattern in patterns:
        match = re.search(pattern, reasoning_text)
        if match:
            answer = match.group(1).strip()
            # Clean up
            answer = answer.rstrip('.')
            return normalize_answer(answer)

    # Fallback: last line
    return reasoning_text.split('\n')[-1].strip()

def normalize_answer(answer):
    """Normalize answer for comparison"""
    # Remove units, normalize numbers, etc.
    answer = answer.lower().strip()
    answer = re.sub(r'[,\s]+', '', answer)  # Remove commas/spaces
    return answer
```

## 6. Code Walkthrough

Reference: `/Users/kevinyu/Projects/Nexus/nexus/models/nlp/reasoning/self_consistency.py`

### Basic Usage

```python
from nexus.models.nlp.reasoning.self_consistency import SelfConsistency

config = {
    'model': language_model,
    'num_samples': 40,        # Number of reasoning paths
    'temperature': 0.7,       # Sampling temperature
    'aggregation': 'majority' # 'majority' or 'weighted'
}

sc = SelfConsistency(config)

result = sc.solve(
    "What is 15% of 200?",
    return_details=True
)

print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Distribution: {result['answer_distribution']}")
```

### Weighted Voting

```python
class WeightedSelfConsistency:
    def solve(self, question, num_samples=40):
        samples = []

        for _ in range(num_samples):
            reasoning = self.model.generate(question, temperature=0.7)
            answer = extract_answer(reasoning)

            # Compute path probability as weight
            log_prob = self.model.score(reasoning)
            weight = math.exp(log_prob)

            samples.append((answer, weight))

        # Weighted voting
        vote_weights = defaultdict(float)
        for answer, weight in samples:
            vote_weights[answer] += weight

        final_answer = max(vote_weights.items(), key=lambda x: x[1])[0]

        return final_answer
```

### Adaptive Sampling

```python
def adaptive_self_consistency(
    question,
    min_samples=10,
    max_samples=100,
    confidence_threshold=0.9
):
    """Sample until confidence threshold is reached"""

    answers = []

    for i in range(max_samples):
        reasoning = model.generate(question, temperature=0.7)
        answer = extract_answer(reasoning)
        answers.append(answer)

        # Check confidence after min_samples
        if i >= min_samples:
            counts = Counter(answers)
            majority_count = counts.most_common(1)[0][1]
            confidence = majority_count / len(answers)

            if confidence >= confidence_threshold:
                print(f"Converged after {i+1} samples")
                break

    # Final vote
    return Counter(answers).most_common(1)[0][0]
```

## 7. Optimization Tricks: Temperature & Aggregation

### 1. Temperature Tuning

```python
# Task-specific temperature
temperature_map = {
    'arithmetic': 0.7,      # Moderate diversity
    'commonsense': 0.8,     # Higher diversity
    'logic': 0.5,           # Lower diversity
    'open_ended': 0.9       # Maximum diversity
}

temp = temperature_map.get(task_type, 0.7)
```

### 2. Top-p (Nucleus) Sampling

```python
# Combine temperature with top-p
reasoning = model.generate(
    prompt,
    temperature=0.7,
    top_p=0.9,  # Sample from top 90% probability mass
    top_k=None
)
```

### 3. Clustered Voting

```python
def cluster_answers(answers):
    """Group similar answers before voting"""
    # Normalize and cluster
    clusters = defaultdict(list)

    for answer in answers:
        normalized = normalize(answer)
        clusters[normalized].append(answer)

    # Vote on clusters
    cluster_counts = {k: len(v) for k, v in clusters.items()}
    winner = max(cluster_counts.items(), key=lambda x: x[1])[0]

    return winner
```

### 4. Confidence-Weighted Sampling

```python
def confidence_weighted_vote(samples):
    """Weight votes by model's confidence in each path"""

    weighted_votes = defaultdict(float)

    for reasoning, answer in samples:
        # Get model's confidence
        logprobs = model.score(reasoning)
        confidence = math.exp(logprobs.mean())

        weighted_votes[answer] += confidence

    return max(weighted_votes.items(), key=lambda x: x[1])[0]
```

### 5. Early Stopping

```python
def early_stopping_sc(question, max_samples=100, min_samples=10):
    """Stop when answer stabilizes"""

    answers = []
    prev_majority = None
    stable_count = 0

    for i in range(max_samples):
        answer = sample_and_extract(question)
        answers.append(answer)

        if i >= min_samples:
            current_majority = Counter(answers).most_common(1)[0][0]

            if current_majority == prev_majority:
                stable_count += 1
                if stable_count >= 5:  # Stable for 5 iterations
                    break
            else:
                stable_count = 0

            prev_majority = current_majority

    return Counter(answers).most_common(1)[0][0]
```

### 6. Hierarchical Aggregation

```python
def hierarchical_aggregation(answers, num_levels=2):
    """Aggregate in multiple stages"""

    current = answers

    for level in range(num_levels):
        # Group into batches
        batch_size = len(current) // (2 ** level)
        batches = [current[i:i+batch_size]
                   for i in range(0, len(current), batch_size)]

        # Vote within each batch
        current = [Counter(batch).most_common(1)[0][0]
                   for batch in batches if batch]

    # Final vote
    return Counter(current).most_common(1)[0][0]
```

## 8. Experiments: GSM8K & MMLU Benchmarks

### GSM8K Math Word Problems

From Wang et al. (ICLR 2023):

| Model | CoT (Greedy) | Self-Consistency (n=40) | Gain |
|-------|--------------|-------------------------|------|
| UL2 20B | 4.4% | 11.4% | +7.0% |
| LaMDA 137B | 41.4% | 58.1% | +16.7% |
| GPT-3 175B | 40.7% | 57.4% | +16.7% |
| PaLM 540B | 56.9% | **74.4%** | +17.5% |

### MMLU Reasoning Subjects

| Subject | CoT | Self-Consistency | Improvement |
|---------|-----|------------------|-------------|
| Abstract Algebra | 38.2% | 48.7% | +10.5% |
| Astronomy | 52.1% | 63.8% | +11.7% |
| College Mathematics | 34.5% | 47.2% | +12.7% |
| Formal Logic | 42.1% | 56.3% | +14.2% |

### Additional Benchmarks

**SVAMP (Math)**:
- CoT: 68.9%
- Self-Consistency: **78.7%**
- Gain: +9.8%

**AQuA (Math)**:
- CoT: 35.8%
- Self-Consistency: **50.3%**
- Gain: +14.5%

**CommonsenseQA**:
- CoT: 72.5%
- Self-Consistency: **81.2%**
- Gain: +8.7%

**StrategyQA**:
- CoT: 66.1%
- Self-Consistency: **75.6%**
- Gain: +9.5%

### Scaling with Number of Samples

| Num Samples | GSM8K (PaLM 540B) | Relative Cost |
|-------------|-------------------|---------------|
| 1 (greedy)  | 56.9% | 1x |
| 5           | 67.2% | 5x |
| 10          | 70.8% | 10x |
| 20          | 73.1% | 20x |
| 40          | **74.4%** | 40x |
| 80          | 74.7% | 80x |

**Takeaway**: Diminishing returns after ~40 samples.

### Temperature Analysis

| Temperature | GSM8K | Path Diversity |
|-------------|-------|----------------|
| 0.3 | 68.2% | Low |
| 0.5 | 71.5% | Medium-Low |
| 0.7 | **74.4%** | Medium |
| 1.0 | 73.1% | High |
| 1.5 | 69.8% | Very High |

**Sweet spot**: T = 0.7

### Ablation Studies

**Aggregation Methods**:
| Method | GSM8K |
|--------|-------|
| Random selection | 58.3% |
| Mean answer | 65.1% |
| Majority vote | **74.4%** |
| Weighted vote | 73.8% |

**Prompt Format Impact**:
| Format | Accuracy |
|--------|----------|
| Direct answer | 56.9% |
| "Let's think step by step" | 74.4% |
| "Solve this carefully" | 72.1% |
| Custom task-specific | **75.2%** |

## 9. Pitfalls

### 1. Insufficient Samples

**Problem**: Too few samples lead to unreliable voting.

**Solution**:
- Use at least 10-20 samples
- Optimal: 20-40 samples for most tasks
- Monitor confidence metrics

### 2. Poor Answer Extraction

**Problem**: Cannot extract consistent answers from reasoning chains.

**Solution**:
```python
# Enforce structured output
prompt = """Solve step by step, then provide your final answer in this format:
Final Answer: [your answer here]

Problem: {problem}"""
```

### 3. Answer Space Issues

**Problem**: Answers vary in format (e.g., "42", "forty-two", "42.0").

**Solution**:
```python
def normalize_answer(answer):
    # Convert to lowercase
    answer = answer.lower().strip()

    # Normalize numbers
    answer = text_to_number(answer)  # "forty-two" → "42"

    # Remove units
    answer = re.sub(r'\s*(dollars|eggs|items)', '', answer)

    # Round floats
    try:
        num = float(answer)
        answer = str(int(num) if num.is_integer() else round(num, 2))
    except:
        pass

    return answer
```

### 4. Temperature Too Low/High

**Problem**:
- Too low: All paths are identical (no diversity)
- Too high: Paths are nonsensical

**Solution**:
```python
# Validate diversity
def check_diversity(reasoning_paths):
    unique = len(set(reasoning_paths))
    if unique < len(reasoning_paths) * 0.5:
        print("Warning: Low diversity, increase temperature")
```

### 5. Computational Cost

**Problem**: Many samples can be expensive.

**Solution**:
```python
# Adaptive sampling
if easy_problem(question):
    num_samples = 10
elif medium_problem(question):
    num_samples = 20
else:
    num_samples = 40

# Or use early stopping
result = adaptive_sc(question, min_samples=10, max_samples=40)
```

### 6. Majority Vote Ties

**Problem**: Multiple answers with same vote count.

**Solution**:
```python
def break_tie(tied_answers, reasoning_paths):
    # Use answer that appeared first
    # Or use weighted voting based on path quality
    weights = [score_path(p) for p in reasoning_paths]
    return weighted_vote(tied_answers, weights)
```

### 7. Overfitting to Prompt

**Problem**: All paths follow same flawed reasoning.

**Solution**:
- Use diverse prompts:
```python
prompts = [
    "Let's think step by step:",
    "Let's solve this carefully:",
    "Breaking this down:",
]

# Sample from different prompts
for prompt in prompts:
    samples = generate(question + prompt, n=num_samples // len(prompts))
```

## 10. References

1. **Self-Consistency Improves Chain of Thought Reasoning in Language Models**
   Wang et al., ICLR 2023
   https://arxiv.org/abs/2203.11171

2. **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models**
   Wei et al., NeurIPS 2022
   https://arxiv.org/abs/2201.11903

3. **On the Advance of Making Language Models Better Reasoners**
   Huang et al., 2023
   https://arxiv.org/abs/2206.02336

4. **Diverse Demonstrations Improve In-Context Compositional Generalization**
   Levy et al., ACL 2023
   https://arxiv.org/abs/2212.06800

5. **The Unreliability of Explanations in Few-Shot Prompting for Textual Reasoning**
   Ye & Durrett, NeurIPS 2022
   https://arxiv.org/abs/2205.03401

## Related Methods

- **Chain-of-Thought**: Base method (Self-Consistency extends this)
- **Tree of Thoughts**: Structured exploration (more complex than SC)
- **Universal Self-Consistency**: Apply to any reasoning method
- **Self-Refine**: Iterative refinement (different from sampling)
- **Ensemble Methods**: General aggregation approaches
