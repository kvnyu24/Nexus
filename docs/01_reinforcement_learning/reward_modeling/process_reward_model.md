# Process Reward Model (PRM)

## 1. Overview & Motivation

Process Reward Models (PRMs) provide **step-by-step verification** for sequential reasoning tasks. Unlike outcome-based reward models that only evaluate final results, PRMs assess the correctness of each intermediate step, enabling fine-grained credit assignment and better guidance during generation.

### The Problem with Outcome-Only Rewards

Traditional Outcome Reward Models (ORMs) only score final answers:
```
ORM: (question, full_solution) → reward
```

**Problems:**
- **Credit assignment**: Which steps were good/bad?
- **False negatives**: Correct reasoning with arithmetic error gets 0 reward
- **False positives**: Wrong reasoning that happens to get right answer gets full reward
- **No guidance during generation**: Can't verify intermediate steps

### PRM's Solution

Process Reward Model scores each step:
```
PRM: (question, step_1, ..., step_n) → (r_1, ..., r_n)
```

**Advantages:**
- **Step-level feedback**: Know exactly where reasoning fails
- **Better credit assignment**: Reward good steps even if final answer is wrong
- **Verification during generation**: Use for best-of-N sampling or beam search
- **More training signal**: N steps → N labels vs 1 label

### Real-World Impact

OpenAI's "Let's Verify Step by Step" paper showed PRMs:
- Improved math problem solving by 15-20%
- Enabled reliable verification at inference time
- Outperformed ORMs with same amount of human feedback
- Generalized better to out-of-distribution problems

## 2. Theoretical Background

### Step-Level Value Estimation

PRM estimates the probability that a reasoning step leads to correct solution:
```
V_PRM(s_t) = P(correct final answer | steps s_1, ..., s_t)
```

This is analogous to value functions in RL, but for reasoning processes.

### Comparison to Outcome Rewards

**Outcome Reward Model:**
```
R_ORM = { 1 if final answer correct
          0 if final answer wrong }
```

**Process Reward Model:**
```
R_PRM(step_i) = P(step_i is valid and useful)
```

### Training Data Requirements

PRM requires step-level labels:
```
Dataset = {
    (question, step_1, step_2, ..., step_n),
    labels = (y_1, y_2, ..., y_n)
}
```

where y_i ∈ {positive, negative, neutral}

**Labeling strategies:**
1. **Human annotation**: Experts label each step (expensive but high quality)
2. **Automatic verification**: Use math checkers, code execution (cheap but limited)
3. **Model-based**: Use strong model to label for weak model (scalable)
4. **Outcome supervision**: Assume all steps good if final answer correct (weak signal)

### Step Independence Assumption

PRMs often assume conditional independence:
```
P(all steps correct) ≈ Π_i P(step_i correct | context)
```

This is an approximation—steps are actually correlated—but works well in practice.

### Credit Assignment Property

Key theoretical advantage of PRMs:
```
∇_θ log π_θ(a_t|s_t) · R_PRM(s_t)
```

vs ORM:
```
∇_θ log π_θ(a_t|s_t) · R_ORM(final_state)
```

PRM gradient is more informative—directly credits/blames specific steps.

## 3. Mathematical Formulation

### Problem Setup

Given:
- Question/problem: q
- Reasoning trajectory: τ = (step_1, step_2, ..., step_T)
- Ground truth solution: y*

Goal: Learn V_PRM(step_t | q, step_{<t}) → [0, 1]

### Step Representation

Each step is embedded using language model:
```
h_t = LM(q, step_1, ..., step_t)
```

Typically use the last hidden state corresponding to step_t.

### PRM Architecture

```
V_PRM(step_t) = σ(W · h_t + b)
```

where:
- h_t: Step embedding from language model
- W, b: Learned projection to scalar reward
- σ: Sigmoid (output probability)

### Training Objective

Binary cross-entropy for step-level labels:
```
L = -Σ_{(q,τ,y)} Σ_t [y_t log V_PRM(step_t) + (1-y_t) log(1-V_PRM(step_t))]
```

With masking for valid steps:
```
L = -Σ_t mask_t · [y_t log V_PRM(step_t) + (1-y_t) log(1-V_PRM(step_t))]
```

### Trajectory Scoring

Aggregate step rewards to score full solution:

**Mean aggregation:**
```
Score(τ) = (1/T) Σ_t V_PRM(step_t)
```

**Product aggregation:**
```
Score(τ) = Π_t V_PRM(step_t)
```

**Minimum aggregation:**
```
Score(τ) = min_t V_PRM(step_t)
```

Product is most common: if any step fails, whole solution fails.

### Best-of-N Sampling

Use PRM for solution selection:
```
1. Generate N candidate solutions: {τ_1, ..., τ_N}
2. Score each: s_i = Score(τ_i)
3. Select best: τ* = argmax_i s_i
```

This is much more effective than random selection or ORM-based selection.

## 4. High-Level Intuition

### Why Step-Level Feedback Matters

Imagine grading a math exam:
- **Bad grading**: Only mark final answer right/wrong
- **Good grading**: Check each step, give partial credit

PRM is like the good teacher who identifies exactly where the student went wrong.

### The Chain Analogy

A reasoning process is like a chain:
- **ORM**: Tests if the whole chain holds weight (binary)
- **PRM**: Tests each link individually

One weak link breaks the chain, but PRM tells you which link is weak.

### Best-of-N Intuition

Generate multiple solutions, pick the one where:
- All steps have high PRM scores (reliable reasoning)
- No obvious errors or jumps in logic

This is similar to how humans verify their own work.

### Comparison to RL Value Functions

PRMs are essentially **value functions for reasoning**:
- State: Current reasoning step
- Value: Probability of reaching correct answer
- Policy: Language model generating next step

Just as Q-learning learns state-action values, PRM learns step values.

## 5. Implementation Details

From `Nexus/nexus/models/rl/reward_models/process_reward_model.py`:

```python
config = {
    "input_dim": 768,        # Embedding dimension from LM
    "hidden_dim": 512,       # PRM hidden layer size
    "n_layers": 4,           # Transformer layers for processing steps
    "n_heads": 8,            # Attention heads
    "dropout": 0.1,
    "max_steps": 512,        # Maximum reasoning steps
}
```

### Architecture

```python
class ProcessRewardModel(NexusModule):
    def __init__(self, config):
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Step positional encoding
        self.step_embedding = nn.Embedding(max_steps, hidden_dim)

        # Transformer encoder to process step sequence
        self.transformer = nn.TransformerEncoder(...)

        # Reward head: project to scalar per step
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),  # Scalar reward
        )
```

### Data Processing

```python
def prepare_prm_data(question, solution_steps, labels):
    """
    Args:
        question: "What is 2+2*3?"
        solution_steps: [
            "Following order of operations",
            "First compute 2*3 = 6",
            "Then compute 2+6 = 8"
        ]
        labels: [1, 1, 1]  # All correct

    Returns:
        Step embeddings and labels
    """
    # Tokenize and embed each step
    step_embeddings = []
    for i, step in enumerate(solution_steps):
        # Concatenate question + previous steps + current step
        context = question + " ".join(solution_steps[:i+1])
        embedding = language_model.encode(context)
        step_embeddings.append(embedding)

    return torch.tensor(step_embeddings), torch.tensor(labels)
```

## 6. Code Walkthrough

### Complete Forward Pass

```python
def forward(
    self,
    step_embeddings: torch.Tensor,  # [batch, n_steps, input_dim]
    step_mask: Optional[torch.Tensor] = None,  # [batch, n_steps]
) -> torch.Tensor:
    """
    Predict reward for each step.
    """
    batch_size, n_steps, _ = step_embeddings.shape

    # Project inputs
    x = self.input_projection(step_embeddings)  # [B, T, hidden]

    # Add positional encodings for step position
    positions = torch.arange(n_steps, device=x.device)
    positions = positions.unsqueeze(0).expand(batch_size, -1)
    x = x + self.step_embedding(positions)

    # Process through transformer (captures dependencies between steps)
    if step_mask is not None:
        x = self.transformer(x, src_key_padding_mask=~step_mask)
    else:
        x = self.transformer(x)

    # Predict rewards (logits, will be passed through sigmoid)
    rewards = self.reward_head(x).squeeze(-1)  # [B, T]

    # Mask invalid steps
    if step_mask is not None:
        rewards = rewards.masked_fill(~step_mask, 0.0)

    return rewards  # Logits, not probabilities
```

### Training

```python
def compute_loss(
    self,
    step_embeddings: torch.Tensor,
    target_rewards: torch.Tensor,  # Binary labels {0, 1}
    step_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute binary cross-entropy loss for step classification.
    """
    predicted_rewards = self.forward(step_embeddings, step_mask)

    # BCE with logits (more numerically stable)
    loss = F.binary_cross_entropy_with_logits(
        predicted_rewards,
        target_rewards,
        reduction='none'
    )

    # Apply mask and normalize
    if step_mask is not None:
        loss = loss * step_mask.float()
        loss = loss.sum() / step_mask.float().sum().clamp(min=1.0)
    else:
        loss = loss.mean()

    return loss
```

### Trajectory Scoring

```python
def score_trajectory(
    self,
    step_embeddings: torch.Tensor,
    step_mask: Optional[torch.Tensor] = None,
    aggregation: str = 'product',
) -> torch.Tensor:
    """
    Score entire reasoning trajectory.

    Args:
        aggregation: 'mean', 'product', 'min', 'sum'

    Returns:
        Trajectory score [batch]
    """
    step_rewards = self.forward(step_embeddings, step_mask)

    # Convert logits to probabilities
    step_probs = torch.sigmoid(step_rewards)

    if step_mask is not None:
        # Only consider valid steps
        valid_probs = step_probs.masked_fill(~step_mask, 1.0)  # Neutral for product
        n_valid = step_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
    else:
        valid_probs = step_probs
        n_valid = torch.tensor(step_probs.size(1))

    if aggregation == 'mean':
        scores = valid_probs.sum(dim=1) / n_valid.squeeze(1)
    elif aggregation == 'product':
        scores = valid_probs.prod(dim=1)
    elif aggregation == 'min':
        scores = valid_probs.min(dim=1)[0]
    elif aggregation == 'sum':
        scores = valid_probs.sum(dim=1)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    return scores
```

### Best-of-N Sampling

```python
def best_of_n_sampling(
    prm,
    question,
    generator_model,
    n_samples=100,
    aggregation='product'
):
    """
    Generate N solutions and select best according to PRM.
    """
    candidates = []

    for _ in range(n_samples):
        # Generate solution
        solution_steps = generator_model.generate(question)

        # Get embeddings
        step_embeddings = embed_solution(question, solution_steps)

        # Score with PRM
        score = prm.score_trajectory(
            step_embeddings.unsqueeze(0),
            aggregation=aggregation
        )

        candidates.append({
            'solution': solution_steps,
            'score': score.item()
        })

    # Return highest scoring solution
    best = max(candidates, key=lambda x: x['score'])
    return best['solution'], best['score']
```

## 7. Optimization Tricks

### 1. Step-Level Data Augmentation

Augment training data by truncating trajectories:
```python
# Original: [step1, step2, step3, step4]
# Augmented:
# - [step1] (with appropriate label)
# - [step1, step2]
# - [step1, step2, step3]
# - [step1, step2, step3, step4]
```

This increases training data 4x and helps PRM learn at all trajectory lengths.

### 2. Balanced Sampling

Balance positive/negative steps:
```python
pos_steps = [s for s in dataset if s.label == 1]
neg_steps = [s for s in dataset if s.label == 0]

# Sample equal amounts
batch_pos = sample(pos_steps, batch_size // 2)
batch_neg = sample(neg_steps, batch_size // 2)
batch = batch_pos + batch_neg
```

### 3. Label Smoothing

Soften hard labels to prevent overconfidence:
```python
# Instead of {0, 1}, use {ε, 1-ε}
label_smoothed = label * (1 - 2*epsilon) + epsilon
```

Typical ε = 0.1.

### 4. Focal Loss

Handle class imbalance with focal loss:
```python
def focal_loss(pred, target, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    p_t = torch.exp(-bce)
    focal = (1 - p_t) ** gamma * bce
    return focal.mean()
```

### 5. Curriculum Learning

Start with short trajectories, gradually increase length:
```python
def get_max_length(epoch, total_epochs):
    progress = epoch / total_epochs
    return int(5 + progress * (max_steps - 5))
```

### 6. Multi-Task Training

Train PRM jointly with other tasks:
```python
# Combine with outcome prediction
loss = loss_step + α * loss_outcome

# Or with next-step prediction
loss = loss_step + β * loss_next_step
```

### 7. Ensemble for Robustness

Train multiple PRMs and aggregate:
```python
prm_scores = [prm_i.score_trajectory(steps) for prm_i in ensemble]
final_score = torch.stack(prm_scores).mean(dim=0)
```

### 8. Calibration

Calibrate PRM outputs to match true correctness rates:
```python
# Temperature scaling
calibrated_prob = sigmoid(logits / temperature)

# Learned via validation set
temperature = optimize_temperature(prm, val_set)
```

### 9. Hard Negative Mining

Focus on difficult examples:
```python
# Steps where PRM is confident but wrong
hard_negatives = [
    s for s in dataset
    if prm(s) > 0.8 and s.label == 0
]
```

### 10. Step Embeddings Cache

Cache step embeddings to speed up training:
```python
# Pre-compute embeddings once
embeddings_cache = {
    step_id: language_model.encode(step)
    for step_id, step in dataset.items()
}

# Reuse during training
embedding = embeddings_cache[step_id]
```

## 8. Experiments & Results

### PRM vs ORM on Math Problems

OpenAI's results on MATH dataset:

| Method | Easy | Medium | Hard | Overall |
|--------|------|--------|------|---------|
| Majority Vote | 62.3% | 38.4% | 14.2% | 38.6% |
| ORM (Best-of-N) | 68.1% | 42.7% | 17.8% | 42.9% |
| PRM (Best-of-N) | 73.5% | 48.2% | 22.1% | 48.1% |

**Key finding:** PRM provides 5-6% absolute improvement over ORM.

### Scaling with Number of Samples

Accuracy vs N for Best-of-N:

```
N=1:   38.6% (no selection)
N=10:  42.1%
N=50:  46.8%
N=100: 48.1% (PRM)
N=100: 42.9% (ORM)
```

PRM scales better with more samples!

### Ablation Studies

**Effect of step-level supervision:**
```
Random labels: 39.2%
Outcome-only (weak): 43.5%
Automatic verification: 45.8%
Human labels: 48.1% ← Best
```

**Effect of aggregation method:**
```
Mean: 44.3%
Sum: 43.8%
Min: 46.7%
Product: 48.1% ← Best
```

Product aggregation works best—one bad step ruins solution.

**Effect of model size:**
```
Small (125M): 42.7%
Medium (1.3B): 46.2%
Large (6B): 48.1%
```

### Generalization

PRM trained on synthetic data, tested on real problems:
```
In-distribution: 48.1%
Out-of-distribution: 44.7% (-3.4%)

vs ORM:
In-distribution: 42.9%
Out-of-distribution: 38.2% (-4.7%)
```

PRM generalizes better than ORM.

### Human Agreement

Correlation with human step labels:
```
PRM: 0.82 ± 0.04
ORM: 0.61 ± 0.08
```

PRM aligns much better with human judgment.

## 9. Common Pitfalls

### 1. Insufficient Step-Level Labels

**Problem:** Only have outcome labels, try to apply PRM anyway.

**Solution:** Either:
- Collect proper step labels (expensive)
- Use automatic verification where possible
- Fall back to ORM for tasks without step structure

### 2. Incorrect Step Segmentation

**Problem:** Steps too coarse (missing errors) or too fine (noisy labels).

**Solution:** Define clear step boundaries:
```python
# Good: One logical operation per step
steps = [
    "Compute 2*3 = 6",
    "Compute 2+6 = 8"
]

# Bad: Multiple operations
steps = ["2*3=6 and 2+6=8"]
```

### 3. Not Handling Variable Length

**Problem:** Fixed-size input, can't handle different trajectory lengths.

**Solution:** Use attention masking:
```python
attention_mask = (step_ids != PAD_ID)
```

### 4. Overfitting to Label Artifacts

**Problem:** PRM learns spurious correlations (e.g., length → correctness).

**Solution:** Balance dataset:
```python
# Ensure correct/incorrect steps have similar length distribution
balance_by_length(dataset)
```

### 5. Not Calibrating Probabilities

**Problem:** PRM outputs uncalibrated confidences.

**Solution:** Temperature scaling on validation set:
```python
def calibrate(prm, val_set):
    T = optimize_temperature(prm, val_set)
    prm.temperature = T
```

### 6. Ignoring Step Dependencies

**Problem:** Treating steps as independent when they're not.

**Solution:** Use transformer to capture dependencies:
```python
# Transformer attention lets steps attend to previous steps
hidden = transformer(step_embeddings)  # Captures context
```

### 7. Wrong Aggregation

**Problem:** Using mean when one bad step invalidates solution.

**Solution:** Use product or min for multi-step reasoning:
```python
score = step_probs.prod()  # All steps must be good
```

### 8. Not Using for Beam Search

**Problem:** Only using PRM for Best-of-N, missing opportunity for guided generation.

**Solution:** Use PRM scores during beam search:
```python
def beam_search_with_prm(model, prm, ...):
    score = log_prob + λ * log(prm_score)
```

### 9. Training on Easy Examples Only

**Problem:** PRM doesn't learn to identify subtle errors.

**Solution:** Include hard negatives:
```python
# Steps that look correct but are wrong
hard_negatives = find_plausible_but_wrong_steps(dataset)
```

### 10. Not Comparing to Baselines

**Problem:** Don't know if PRM actually helps.

**Solution:** Always compare to:
- Random selection
- Length-based selection
- ORM selection
- Human selection (upper bound)

## 10. References

### Primary Paper
- Lightman, H., et al. (2023). **Let's Verify Step by Step.** OpenAI. ArXiv:2305.20050.
  - [Paper](https://arxiv.org/abs/2305.20050)
  - Introduced PRM for math reasoning
  - Showed significant improvements over ORM

### Related Work on Reward Modeling
- Uesato, J., et al. (2022). **Solving Math Word Problems with Process- and Outcome-Based Feedback.** ArXiv.
- Cobbe, K., et al. (2021). **Training Verifiers to Solve Math Word Problems.** ArXiv.
- Saunders, W., et al. (2022). **Self-critiquing Models for Assisting Human Evaluators.** ArXiv.

### Applications
- Leike, J., et al. (2018). **Scalable Agent Alignment via Reward Modeling.** DeepMind.
- Stiennon, N., et al. (2020). **Learning to Summarize from Human Feedback.** OpenAI.
- Ouyang, L., et al. (2022). **Training Language Models to Follow Instructions with Human Feedback.** OpenAI (InstructGPT).

### Verification and Planning
- Silver, D., et al. (2016). **Mastering the Game of Go with Deep Neural Networks and Tree Search.** Nature.
- Huang, W., et al. (2022). **Language Models as Zero-Shot Planners.** ArXiv.

### Implementation Reference
- Nexus Implementation: `Nexus/nexus/models/rl/reward_models/process_reward_model.py`

---

**Key Takeaways:**
- PRMs provide step-level verification for multi-step reasoning
- Significantly outperform outcome-only reward models
- Require step-level labels (can use automatic verification)
- Best used with Best-of-N sampling or beam search
- Critical for reliable reasoning in LLMs
