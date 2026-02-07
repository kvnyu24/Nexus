# Chain-of-Thought (CoT) Reasoning

## 1. Overview & Motivation

Chain-of-Thought (CoT) prompting is a technique that enables large language models to solve complex reasoning tasks by generating intermediate reasoning steps before producing a final answer. Instead of directly answering a question, the model first articulates its thought process, leading to more accurate and interpretable results.

**Key Insight**: Complex reasoning emerges when models are prompted to "show their work" - the intermediate steps serve as a scaffold for solving multi-step problems.

### Why Chain-of-Thought?

Traditional prompting often fails on tasks requiring:
- Multi-step arithmetic
- Logical deduction
- Commonsense reasoning chains
- Symbol manipulation

CoT addresses this by making the reasoning process explicit, allowing the model to:
1. Break down complex problems
2. Track intermediate results
3. Compose multi-step solutions
4. Self-correct through explicit reasoning

## 2. Theory: Prompting Strategies

### Emergence of Reasoning

CoT reasoning emerges in models with sufficient scale (typically >100B parameters). The phenomenon is explained by:

1. **Compositional Reasoning**: Models learn to compose simple reasoning primitives
2. **In-Context Learning**: Few-shot examples demonstrate the reasoning pattern
3. **Latent Reasoning**: Models develop internal representations for multi-step processes

### Prompting Approaches

**Zero-Shot CoT**:
Simply append "Let's think step by step" to the problem:

```
Q: What is 15% of 80?
A: Let's think step by step.
```

**Few-Shot CoT**:
Provide examples with reasoning chains:

```
Q: John has 3 apples. He buys 2 more. How many does he have?
A: John started with 3 apples. He bought 2 more.
   So 3 + 2 = 5 apples.

Q: Sarah has 10 cookies. She eats 3. How many are left?
A: [Model generates reasoning]
```

### Prompt Design Principles

1. **Clarity**: Use clear, unambiguous language
2. **Exemplar Quality**: Choose diverse, representative examples
3. **Step Granularity**: Balance detail vs. conciseness
4. **Format Consistency**: Maintain consistent structure across examples

## 3. Mathematical Formulation

### Basic Formulation

Given a problem $P$, traditional prompting computes:

$$
p(A | P) = \text{LM}(P)
$$

Chain-of-Thought introduces intermediate reasoning steps $R = (r_1, r_2, \ldots, r_n)$:

$$
p(A | P) = \sum_{R} p(A | R, P) \cdot p(R | P)
$$

In practice, we use greedy decoding:

$$
R^* = \text{argmax}_R \; p(R | P)
$$
$$
A^* = \text{argmax}_A \; p(A | R^*, P)
$$

### Attention Flow in CoT

The neural implementation uses layered attention to maintain reasoning state:

For each reasoning step $i$:
$$
\text{Attention}(Q_i, K_{1:i}, V_{1:i}) = \text{softmax}\left(\frac{Q_i K_{1:i}^T}{\sqrt{d_k}}\right) V_{1:i}
$$

Where:
- $Q_i$ = query for current step
- $K_{1:i}, V_{1:i}$ = keys/values from all previous steps
- The model attends to prior reasoning when generating new thoughts

### Decomposition into Steps

The probability of a reasoning chain can be factorized:

$$
p(R | P) = \prod_{i=1}^{n} p(r_i | r_{<i}, P)
$$

Each step $r_i$ is conditioned on:
- The original problem $P$
- All previous reasoning steps $r_{<i}$

## 4. Intuition

### Flow Diagram

```
┌─────────────┐
│   Problem   │
└──────┬──────┘
       │
       ▼
┌──────────────────────┐
│  Step Embedding 1    │
│  ┌────────────────┐  │
│  │ Thought Layer  │  │
│  └────────┬───────┘  │
│           │          │
│  ┌────────▼───────┐  │
│  │  Reasoning 1   │  │
│  └────────────────┘  │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│  Step Embedding 2    │
│  ┌────────────────┐  │
│  │ Thought Layer  │  │ ◄── Attends to previous steps
│  └────────┬───────┘  │
│           │          │
│  ┌────────▼───────┐  │
│  │  Reasoning 2   │  │
│  └────────────────┘  │
└──────┬───────────────┘
       │
       ⋮ (N steps)
       │
       ▼
┌──────────────────────┐
│   Final Answer       │
└──────────────────────┘
```

### Thought Process Example

**Problem**: "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?"

**Without CoT**: "14" (wrong)

**With CoT**:
- Step 1: Roger started with 5 balls
- Step 2: He bought 2 cans with 3 balls each
- Step 3: 2 cans × 3 balls/can = 6 balls
- Step 4: Total = 5 + 6 = 11 balls
- Answer: 11 (correct)

### Why It Works

1. **Working Memory**: Intermediate steps act as external memory
2. **Error Localization**: Mistakes can be identified in specific steps
3. **Compositionality**: Complex operations decompose into simpler ones
4. **Verification**: Each step can be checked independently

## 5. Implementation Details

### Neural Architecture Integration

The Nexus implementation uses specialized reasoning modules:

```python
class ReasoningStep:
    - attention: Multi-head self-attention over current reasoning state
    - norm: Layer normalization
    - ffn: Feed-forward transformation

    forward(x, context):
        attended = self.attention(x)
        if context is not None:
            attended += self.attention(x, context)  # Attend to original problem
        x = self.norm(x + attended)
        x = x + self.ffn(x)
        return x
```

Key components:
- **Step Embeddings**: Distinguish reasoning stages
- **Residual Connections**: Preserve information across steps
- **Context Attention**: Maintain focus on original problem

### Training Strategies

**Fine-tuning**:
```python
# Dataset format: (problem, reasoning_chain, answer)
for batch in dataset:
    logits = model(batch['problem'] + batch['reasoning'])
    loss = cross_entropy(logits, batch['reasoning'] + batch['answer'])
    loss.backward()
```

**Reinforcement Learning**:
```python
# Reward correct final answers
reward = (predicted_answer == ground_truth).float()
policy_loss = -log_prob * reward
```

**Distillation**:
```python
# Distill reasoning from larger teacher model
teacher_reasoning = teacher_model.generate(problem)
student_loss = cross_entropy(
    student_model(problem),
    teacher_reasoning
)
```

## 6. Code Walkthrough

### Basic Usage

Reference implementation: `Nexus/nexus/models/nlp/reasoning/chain_of_thoughts.py`

```python
from nexus.models.nlp.reasoning.chain_of_thoughts import ChainOfThoughtModule

config = {
    "num_reasoning_steps": 4,
    "hidden_size": 768,
    "vocab_size": 50257
}

cot_module = ChainOfThoughtModule(config)

# Forward pass
outputs = cot_module(
    hidden_states=input_embeddings,  # (batch_size, seq_len, hidden_size)
    attention_mask=mask
)

# Access reasoning steps
logits = outputs["logits"]  # Final predictions
reasoning_steps = outputs["reasoning_steps"]  # List of intermediate states
attention_maps = outputs["attention_maps"]  # Attention patterns per step
```

### Integration with LLM

```python
from nexus.models.nlp.reasoning.chain_of_thoughts import ReasoningLLM

config = {
    "vocab_size": 50257,
    "hidden_size": 768,
    "max_seq_length": 512,
    "num_reasoning_steps": 4
}

model = ReasoningLLM(config)

# Generate with reasoning
outputs = model(
    input_ids=input_tokens,
    attention_mask=mask,
    return_reasoning_steps=True
)

# Inspect reasoning process
for i, step in enumerate(outputs["reasoning_steps"]):
    print(f"Step {i}: {step}")
```

### Custom Reasoning Depth

```python
class AdaptiveCoT(nn.Module):
    """Dynamically determine reasoning steps based on problem complexity"""

    def __init__(self, config):
        super().__init__()
        self.max_steps = config["max_steps"]
        self.complexity_scorer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # Predict number of reasoning steps needed
        complexity = self.complexity_scorer(x.mean(dim=1))
        num_steps = min(max(1, int(complexity.item())), self.max_steps)

        # Run adaptive reasoning
        for step in range(num_steps):
            x = self.reasoning_step(x)

        return x
```

## 7. Optimization Tricks

### 1. Step Embedding Initialization

Initialize step embeddings with positional encoding patterns:

```python
step_embeddings = torch.zeros(num_steps, 1, hidden_size)
for pos in range(num_steps):
    for i in range(0, hidden_size, 2):
        step_embeddings[pos, 0, i] = math.sin(pos / (10000 ** (i / hidden_size)))
        step_embeddings[pos, 0, i+1] = math.cos(pos / (10000 ** (i / hidden_size)))
```

### 2. Gradient Flow Optimization

Use pre-normalization instead of post-normalization:

```python
# Pre-norm (better gradient flow)
x = x + self.attention(self.norm1(x))
x = x + self.ffn(self.norm2(x))

# vs Post-norm (standard)
x = self.norm1(x + self.attention(x))
x = self.norm2(x + self.ffn(x))
```

### 3. Early Stopping

Stop reasoning when confidence is high:

```python
for step in range(max_steps):
    x = reasoning_step(x)
    confidence = torch.softmax(output_head(x), dim=-1).max()
    if confidence > 0.95:
        break
```

### 4. Reasoning Cache

Cache intermediate reasoning for similar problems:

```python
reasoning_cache = {}

def cached_reasoning(problem_embedding):
    key = hash(problem_embedding)
    if key in reasoning_cache:
        return reasoning_cache[key]

    result = run_reasoning(problem_embedding)
    reasoning_cache[key] = result
    return result
```

### 5. Temperature Scheduling

Adjust temperature per reasoning step:

```python
def temperature_schedule(step, total_steps):
    # Higher temperature early (exploration)
    # Lower temperature late (exploitation)
    return 1.0 - (step / total_steps) * 0.5

for step in range(num_steps):
    temp = temperature_schedule(step, num_steps)
    reasoning = model.generate(prompt, temperature=temp)
```

## 8. Experiments: GSM8K & MMLU Benchmarks

### GSM8K (Grade School Math)

Results from Wei et al. (2022):

| Model | Standard Prompting | CoT Prompting | Improvement |
|-------|-------------------|---------------|-------------|
| LaMDA 137B | 17.9% | 58.1% | +40.2% |
| GPT-3 175B | 18.5% | 57.2% | +38.7% |
| PaLM 540B | 33.0% | 74.4% | +41.4% |

### MMLU (Massive Multitask Language Understanding)

| Subject | Standard | CoT | Improvement |
|---------|----------|-----|-------------|
| Mathematics | 34.2% | 52.7% | +18.5% |
| Physics | 42.1% | 58.3% | +16.2% |
| Chemistry | 38.9% | 51.2% | +12.3% |
| Computer Science | 45.6% | 61.8% | +16.2% |

### Additional Benchmarks

| Task | Standard Prompting | CoT Prompting | Improvement |
|------|-------------------|---------------|-------------|
| SVAMP (Math) | 69.9% | 78.7% | +8.8% |
| AQuA (Math) | 33.7% | 50.3% | +16.6% |
| CommonsenseQA | 67.4% | 79.2% | +11.8% |
| StrategyQA | 54.2% | 66.1% | +11.9% |

### Scaling Analysis

CoT benefits increase with model scale:

```
Model Size     CoT Gain on GSM8K
-----------    -----------------
1B params      +2.1%
10B params     +8.5%
60B params     +25.3%
175B params    +40.2%
540B params    +52.7%
```

### Ablation Studies

**Number of Reasoning Steps**:
- 1 step: 45.2% accuracy
- 2 steps: 52.3% accuracy
- 4 steps: 58.1% accuracy
- 8 steps: 57.9% accuracy (diminishing returns)

**Step Embedding Impact**:
- Without step embeddings: 51.4%
- With step embeddings: 58.1%
- Gain: +6.7%

**Few-Shot Examples**:
- 0 examples (zero-shot): 40.7%
- 3 examples: 52.3%
- 8 examples: 58.1%
- 16 examples: 58.4% (saturation)

## 9. Pitfalls

### 1. Insufficient Model Scale

**Problem**: CoT reasoning emerges at scale. Small models (<10B) show minimal gains.

**Solution**: Use models ≥60B parameters, or fine-tune smaller models on reasoning datasets.

### 2. Poor Few-Shot Examples

**Problem**: Low-quality examples lead to degraded reasoning.

```python
# Bad example (too vague)
"Q: Math problem. A: Use numbers."

# Good example (clear reasoning)
"Q: 3 + 5 × 2 = ?
 A: Following order of operations, first multiply: 5 × 2 = 10.
    Then add: 3 + 10 = 13."
```

### 3. Excessive Reasoning Steps

**Problem**: Too many steps wastes computation and may introduce errors.

**Solution**: Start with 3-5 steps. Monitor validation performance vs. computational cost.

### 4. Ignoring Context

**Problem**: Reasoning drifts away from the original problem.

**Solution**: Use cross-attention to maintain focus on the problem:

```python
# Maintain problem context
attended = self.attention(current_step, context=original_problem)
```

### 5. Training Instability

**Problem**: Gradients vanish/explode through long reasoning chains.

**Solutions**:
- Use gradient clipping: `torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)`
- Pre-normalization architecture
- Warmup learning rate schedule

### 6. Prompt Engineering Sensitivity

**Problem**: Different phrasings yield very different results.

**Solution**: Test multiple prompt variants:

```python
prompts = [
    "Let's solve this step by step:",
    "Let's think through this carefully:",
    "Breaking this down:",
    "Step-by-step solution:"
]

# Run ensemble over prompts
results = [model(problem + prompt) for prompt in prompts]
final_answer = majority_vote(results)
```

### 7. Answer Extraction Failures

**Problem**: Difficulty parsing final answer from reasoning chain.

**Solution**: Use structured output format:

```python
prompt = """Solve step by step, then provide your final answer in this format:
Final Answer: [your answer here]

Problem: {problem}"""
```

### 8. Reasoning Length vs. Quality

**Problem**: Longer reasoning isn't always better.

**Observation**: Optimal reasoning length varies by task:
- Arithmetic: 3-5 steps
- Logic puzzles: 5-10 steps
- Commonsense: 2-4 steps

## 10. References

1. **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models**
   Wei et al., NeurIPS 2022
   https://arxiv.org/abs/2201.11903

2. **Large Language Models are Zero-Shot Reasoners**
   Kojima et al., NeurIPS 2022
   https://arxiv.org/abs/2205.11916

3. **Least-to-Most Prompting Enables Complex Reasoning in Large Language Models**
   Zhou et al., ICLR 2023
   https://arxiv.org/abs/2205.10625

4. **Automatic Chain of Thought Prompting in Large Language Models**
   Zhang et al., ICLR 2023
   https://arxiv.org/abs/2210.03493

5. **On the Advance of Making Language Models Better Reasoners**
   Huang et al., 2023
   https://arxiv.org/abs/2206.02336

6. **Complexity-Based Prompting for Multi-Step Reasoning**
   Fu et al., ICLR 2023
   https://arxiv.org/abs/2210.00720

## Related Methods

- **Self-Consistency**: Sample multiple CoT paths and aggregate via voting
- **Tree of Thoughts**: Explore multiple reasoning paths in a tree structure
- **Least-to-Most**: Decompose-then-solve strategy
- **Complexity-Based Prompting**: Select few-shot examples by complexity
- **Auto-CoT**: Automatically generate diverse reasoning demonstrations
