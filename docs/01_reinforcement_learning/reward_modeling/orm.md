# Outcome Reward Model (ORM)

## 1. Overview & Motivation

Outcome Reward Models (ORMs) evaluate the final result of a generation or decision-making process without considering intermediate steps. ORMs provide a simple, efficient baseline for reward modeling that is particularly effective for tasks with clear success criteria.

### The Outcome-Only Paradigm

Traditional ORMs only score final answers:
```
ORM: (question, complete_solution) → reward
```

**Advantages:**
- Simple to implement and train
- Efficient inference (single forward pass)
- Works well for single-step decisions
- Natural for tasks with binary outcomes
- Less labeling cost than step-level supervision

**Limitations:**
- Poor credit assignment for multi-step reasoning
- No guidance during generation
- False positives: Wrong reasoning, correct answer
- False negatives: Correct reasoning, arithmetic error
- Limited training signal (1 label per trajectory)

### When to Use ORM

ORMs are most effective for:
- **Classification tasks**: Single prediction with clear correctness
- **Short-form generation**: Responses that can be evaluated holistically
- **Tasks with automatic verification**: Code that passes tests, equations with solutions
- **Baseline comparisons**: Comparing against more sophisticated reward models
- **Resource-constrained settings**: When step-level labels are too expensive

### Real-World Applications

ORMs are widely used in:
- **Code generation**: Pass/fail on test suites
- **Math problems**: Correct/incorrect final answer
- **Question answering**: Factual accuracy of response
- **Classification**: Predicted label matches ground truth
- **Game playing**: Win/loss outcomes

## 2. Theoretical Background

### Binary Outcome Formulation

ORM models the probability of correctness:
```
P(correct | x, y) = σ(r_θ(x, y))
```

where:
- x: Input/question
- y: Complete output/solution
- r_θ: Learned reward function
- σ: Sigmoid activation

### Comparison to Process Supervision

**ORM (Outcome):**
```
R_ORM(τ) = { 1 if final answer correct
             0 if final answer wrong }
```

**PRM (Process):**
```
R_PRM(τ) = (r_1, r_2, ..., r_n) for each step
```

Trade-off:
- ORM: Simpler, cheaper labels, weaker signal
- PRM: Complex, expensive labels, stronger signal

### Learning from Outcome Labels

Training data format:
```python
dataset = [
    {
        "input": "What is 2+3*4?",
        "output": "The answer is 14.",
        "label": 1  # Correct
    },
    {
        "input": "What is 2+3*4?",
        "output": "The answer is 20.",
        "label": 0  # Incorrect (wrong order of operations)
    }
]
```

### Bradley-Terry Model for Preferences

For pairwise preferences (A vs B):
```
P(A ≻ B) = σ(r_θ(x, A) - r_θ(x, B))
```

Loss:
```
L = -log P(y_winner ≻ y_loser)
  = -log σ(r_θ(x, y_win) - r_θ(x, y_lose))
```

This formulation learns from relative preferences without absolute labels.

### Expectation Modeling

ORM can also model expected future reward:
```
V_ORM(x, y) = E[R_final | x, y]
```

This generalizes beyond binary correctness to continuous rewards.

## 3. Mathematical Formulation

### Problem Setup

Given:
- Input/prompt: x
- Complete output: y
- Ground truth evaluation: y* or correctness label c ∈ {0, 1}

Goal: Learn r_θ(x, y) → ℝ that predicts outcome quality

### Architecture

Typical ORM uses encoder-only or encoder-decoder architecture:

**Encoder-only (BERT-style):**
```
h = Encoder(concat(x, y))
r = MLP(h)
```

**Encoder-decoder (T5-style):**
```
h_enc = Encoder(x)
h_dec = Decoder(y, h_enc)
r = MLP(h_dec[-1])
```

**Embedding similarity:**
```
e_x = Encoder(x)
e_y = Encoder(y)
r = cosine(e_x, e_y)  # or learned projection
```

### Training Objectives

**Binary classification:**
```
L_binary = -[c log σ(r) + (1-c) log(1-σ(r))]
```

**Pairwise ranking:**
```
L_pairwise = -log σ(r_win - r_lose)
```

**Regression:**
```
L_regression = MSE(r, target_score)
```

**Contrastive:**
```
L_contrastive = -log exp(r_pos/τ) / Σ_i exp(r_i/τ)
```

### Aggregation for Multiple Candidates

When selecting from N candidates:

**Max scoring:**
```
y* = argmax_i r_θ(x, y_i)
```

**Softmax sampling:**
```
P(y_i) ∝ exp(r_θ(x, y_i) / T)
```

**Threshold filtering:**
```
candidates = {y_i : r_θ(x, y_i) > threshold}
```

## 4. High-Level Intuition

### The Final Grade Analogy

ORM is like grading an exam by only checking the final answer:
- Quick to evaluate
- Clear for simple problems
- But you miss partial credit opportunities
- Can't identify specific mistakes

### Signal Strength

Consider two students on a math problem:
- **Student A**: Correct reasoning, arithmetic error → Wrong answer → 0 reward
- **Student B**: Wrong reasoning, lucky guess → Right answer → 1 reward

ORM treats these the same (binary), missing the quality difference that PRM would catch.

### Efficiency vs Fidelity

ORM trades detailed feedback for efficiency:
- Fast to label: Just check final answer
- Fast to compute: Single model evaluation
- But loses information about how the answer was reached

### Best-of-N Selection

ORM's primary strength: efficiently selecting best candidate
```
Generate: [y_1, y_2, ..., y_N]
Score: [r_1, r_2, ..., r_N]
Select: y_best = argmax r_i
```

This is much more effective than random selection.

## 5. Implementation Details

From `Nexus/nexus/models/rl/reward_models/process_reward_model.py`:

```python
config = {
    "input_dim": 768,        # Embedding dimension from LM
    "hidden_dim": 512,       # Hidden layer size
    "dropout": 0.1,          # Dropout rate
    "pooling": "mean",       # Pooling strategy: 'mean', 'max', 'last'
}
```

### Architecture Design

```python
class OutcomeRewardModel(NexusModule):
    def __init__(self, config):
        self.input_dim = config.get("input_dim", 768)
        self.hidden_dim = config.get("hidden_dim", 512)
        self.dropout = config.get("dropout", 0.1)

        # Simple feedforward network
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )
```

### Input Processing

```python
def prepare_orm_data(question, solution, label):
    """
    Prepare data for ORM training.

    Args:
        question: "What is the capital of France?"
        solution: "The capital of France is Paris."
        label: 1 (correct) or 0 (incorrect)

    Returns:
        Combined embedding and label
    """
    # Concatenate input and output
    text = question + " [SEP] " + solution

    # Encode with language model
    embedding = language_model.encode(text)

    return embedding, label
```

### Pooling Strategies

```python
def pool_embeddings(embeddings, mask=None, method='mean'):
    """
    Pool sequence embeddings to single vector.

    Args:
        embeddings: [batch, seq_len, dim]
        mask: [batch, seq_len] boolean mask
        method: 'mean', 'max', 'last', 'cls'
    """
    if method == 'mean':
        if mask is not None:
            masked = embeddings * mask.unsqueeze(-1)
            return masked.sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return embeddings.mean(1)

    elif method == 'max':
        if mask is not None:
            masked = embeddings.masked_fill(~mask.unsqueeze(-1), float('-inf'))
            return masked.max(1)[0]
        return embeddings.max(1)[0]

    elif method == 'last':
        if mask is not None:
            last_idx = mask.sum(1) - 1
            return embeddings[torch.arange(len(embeddings)), last_idx]
        return embeddings[:, -1]

    elif method == 'cls':
        # Use [CLS] token (first position)
        return embeddings[:, 0]
```

## 6. Code Walkthrough

### Complete Forward Pass

```python
def forward(self, output_embedding: torch.Tensor) -> torch.Tensor:
    """
    Predict reward for complete output.

    Args:
        output_embedding: Embedding of full output [batch, input_dim]

    Returns:
        Predicted rewards [batch]
    """
    # Single forward pass through reward network
    reward_logits = self.network(output_embedding).squeeze(-1)

    return reward_logits  # Return logits (apply sigmoid for probability)
```

### Training with Binary Labels

```python
def compute_loss(
    self,
    output_embedding: torch.Tensor,
    target_reward: torch.Tensor,
) -> torch.Tensor:
    """
    Compute binary cross-entropy loss.

    Args:
        output_embedding: Output embedding [batch, input_dim]
        target_reward: Binary correctness labels [batch]

    Returns:
        Loss (scalar)
    """
    predicted_reward = self.forward(output_embedding)

    # Binary classification loss
    loss = F.binary_cross_entropy_with_logits(
        predicted_reward,
        target_reward
    )

    return loss
```

### Training with Pairwise Preferences

```python
def compute_pairwise_loss(
    self,
    preferred_embedding: torch.Tensor,
    dispreferred_embedding: torch.Tensor,
) -> torch.Tensor:
    """
    Compute Bradley-Terry pairwise ranking loss.

    Args:
        preferred_embedding: Embedding of preferred output [batch, input_dim]
        dispreferred_embedding: Embedding of rejected output [batch, input_dim]

    Returns:
        Loss (scalar)
    """
    # Get rewards for both outputs
    r_preferred = self.forward(preferred_embedding)
    r_dispreferred = self.forward(dispreferred_embedding)

    # Bradley-Terry loss: P(preferred > dispreferred)
    loss = -F.logsigmoid(r_preferred - r_dispreferred).mean()

    return loss
```

### Best-of-N Sampling

```python
def best_of_n_sampling(
    orm,
    question,
    generator_model,
    n_samples=100,
    temperature=1.0,
):
    """
    Generate N solutions and select best according to ORM.

    Args:
        orm: Outcome reward model
        question: Input question/prompt
        generator_model: Model that generates candidate solutions
        n_samples: Number of candidates to generate
        temperature: Sampling temperature

    Returns:
        Best solution and its score
    """
    candidates = []

    for _ in range(n_samples):
        # Generate complete solution
        solution = generator_model.generate(
            question,
            temperature=temperature
        )

        # Embed full output
        embedding = embed_output(question, solution)

        # Score with ORM
        with torch.no_grad():
            score = torch.sigmoid(orm.forward(embedding.unsqueeze(0)))

        candidates.append({
            'solution': solution,
            'score': score.item()
        })

    # Return highest scoring solution
    best = max(candidates, key=lambda x: x['score'])
    return best['solution'], best['score']
```

### Calibration

```python
def calibrate_orm(orm, validation_data):
    """
    Calibrate ORM probabilities using temperature scaling.

    Args:
        orm: Trained outcome reward model
        validation_data: Validation set with ground truth labels

    Returns:
        Optimal temperature parameter
    """
    logits = []
    labels = []

    # Collect predictions on validation set
    for batch in validation_data:
        with torch.no_grad():
            batch_logits = orm.forward(batch['embeddings'])
            logits.append(batch_logits)
            labels.append(batch['labels'])

    logits = torch.cat(logits)
    labels = torch.cat(labels)

    # Find optimal temperature
    def nll(temperature):
        scaled_logits = logits / temperature
        loss = F.binary_cross_entropy_with_logits(scaled_logits, labels)
        return loss.item()

    from scipy.optimize import minimize_scalar
    result = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')

    return result.x  # Optimal temperature
```

## 7. Optimization Tricks

### 1. Data Augmentation

Generate more training examples from existing labels:
```python
# If solution A is correct and solution B is incorrect:
# Create preference pair (A, B)
positive_pairs = []
negative_pairs = []

for item in dataset:
    if item['label'] == 1:
        positive_pairs.append(item)
    else:
        negative_pairs.append(item)

# Create all pairwise combinations
preference_data = [
    (pos, neg) for pos in positive_pairs for neg in negative_pairs
]
```

### 2. Hard Negative Mining

Focus on difficult examples:
```python
def mine_hard_negatives(orm, dataset, threshold=0.7):
    """
    Find incorrect outputs that ORM scores highly.
    """
    hard_negatives = []

    for item in dataset:
        if item['label'] == 0:  # Incorrect output
            score = orm(item['embedding'])
            if torch.sigmoid(score) > threshold:  # But ORM is confident
                hard_negatives.append(item)

    return hard_negatives
```

### 3. Label Smoothing

Prevent overconfidence:
```python
def label_smooth(labels, epsilon=0.1):
    """
    Smooth binary labels: {0, 1} → {ε, 1-ε}
    """
    return labels * (1 - 2*epsilon) + epsilon
```

### 4. Contrastive Learning

Learn from multiple negatives:
```python
def contrastive_loss(anchor, positives, negatives, temperature=0.07):
    """
    InfoNCE-style contrastive loss.
    """
    # Compute similarities
    pos_sim = F.cosine_similarity(anchor, positives) / temperature
    neg_sim = F.cosine_similarity(anchor.unsqueeze(1), negatives, dim=-1) / temperature

    # Contrastive loss
    numerator = torch.exp(pos_sim)
    denominator = numerator + torch.exp(neg_sim).sum(dim=1)

    loss = -torch.log(numerator / denominator).mean()
    return loss
```

### 5. Ensemble for Robustness

Train multiple ORMs and aggregate:
```python
class EnsembleORM:
    def __init__(self, models):
        self.models = models

    def forward(self, embedding):
        scores = [model(embedding) for model in self.models]
        return torch.stack(scores).mean(0)  # Average prediction

    def uncertainty(self, embedding):
        scores = [model(embedding) for model in self.models]
        return torch.stack(scores).std(0)  # Disagreement
```

### 6. Curriculum Learning

Start with easy examples, progress to hard:
```python
def curriculum_sampler(dataset, epoch, total_epochs):
    """
    Sample examples based on difficulty and training progress.
    """
    progress = epoch / total_epochs

    # Sort by difficulty (e.g., length, complexity)
    sorted_data = sorted(dataset, key=lambda x: x['difficulty'])

    # Start with easy, gradually include harder
    cutoff = int(len(sorted_data) * (0.3 + 0.7 * progress))
    return sorted_data[:cutoff]
```

### 7. Regularization Techniques

Prevent overfitting:
```python
# L2 regularization
loss = loss_bce + lambda_l2 * sum(p.pow(2).sum() for p in model.parameters())

# Dropout at inference (MC Dropout for uncertainty)
model.train()  # Keep dropout active
predictions = [model(x) for _ in range(K)]
mean_pred = torch.stack(predictions).mean(0)
uncertainty = torch.stack(predictions).std(0)
```

### 8. Balanced Batching

Ensure balanced positive/negative examples:
```python
def balanced_batch_sampler(dataset, batch_size):
    """
    Sample equal number of correct/incorrect examples.
    """
    correct = [x for x in dataset if x['label'] == 1]
    incorrect = [x for x in dataset if x['label'] == 0]

    half_batch = batch_size // 2

    batch_correct = random.sample(correct, half_batch)
    batch_incorrect = random.sample(incorrect, half_batch)

    return batch_correct + batch_incorrect
```

### 9. Multi-Task Learning

Train with auxiliary tasks:
```python
class MultiTaskORM(nn.Module):
    def __init__(self, config):
        self.shared_encoder = ...
        self.reward_head = nn.Linear(hidden_dim, 1)
        self.difficulty_head = nn.Linear(hidden_dim, 1)  # Auxiliary
        self.length_head = nn.Linear(hidden_dim, 1)      # Auxiliary

    def forward(self, x):
        features = self.shared_encoder(x)

        reward = self.reward_head(features)
        difficulty = self.difficulty_head(features)
        length = self.length_head(features)

        return reward, difficulty, length
```

### 10. Online Learning

Update ORM as policy improves:
```python
def online_orm_training(orm, policy, dataset):
    """
    Iteratively generate new data and retrain ORM.
    """
    for iteration in range(num_iterations):
        # Generate new outputs from current policy
        new_outputs = policy.generate(dataset['inputs'])

        # Get labels (from environment or human)
        new_labels = evaluate_outputs(new_outputs)

        # Add to training set
        dataset.extend(zip(new_outputs, new_labels))

        # Retrain ORM
        train_orm(orm, dataset)

        # Train policy with updated ORM
        train_policy(policy, orm)
```

## 8. Experiments & Results

### ORM vs Random Selection

Performance on math problem selection (MATH dataset):

| Method | Accuracy | Improvement |
|--------|----------|-------------|
| Random Selection | 38.6% | Baseline |
| Length-based | 40.2% | +1.6% |
| ORM (Best-of-10) | 42.1% | +3.5% |
| ORM (Best-of-100) | 42.9% | +4.3% |
| PRM (Best-of-100) | 48.1% | +9.5% |

Key finding: ORM provides significant improvement over random, but PRM is substantially better for multi-step reasoning.

### Scaling with Number of Samples

Best-of-N performance on code generation (HumanEval):

```
N=1:   25.3% (no selection)
N=10:  31.7% (+6.4%)
N=50:  36.8% (+11.5%)
N=100: 38.2% (+12.9%)
```

Diminishing returns after 50 samples.

### ORM Model Size Effects

| Model Size | Training Time | Accuracy | Calibration ECE |
|------------|---------------|----------|-----------------|
| Small (125M) | 2 hours | 68.3% | 0.12 |
| Medium (350M) | 6 hours | 71.7% | 0.09 |
| Large (1.3B) | 18 hours | 73.5% | 0.08 |

Larger models improve both accuracy and calibration.

### Data Efficiency

Performance vs training data size:

```
1K examples:   62.1%
5K examples:   67.8%
10K examples:  71.7%
50K examples:  73.5%
100K examples: 74.2% (diminishing returns)
```

ORM is relatively data-efficient compared to PRM.

### Binary vs Pairwise Training

| Training Method | Data Format | Accuracy | Notes |
|----------------|-------------|----------|-------|
| Binary Labels | (x, y, label) | 71.7% | Direct supervision |
| Pairwise Prefs | (x, y1, y2, pref) | 73.1% | More robust |
| Mixed | Both | 74.5% | Best of both |

Pairwise training often works better with noisy labels.

### Task-Specific Performance

| Task | ORM Accuracy | Best Metric |
|------|--------------|-------------|
| Math Problems | 65.2% | Exact match |
| Code Generation | 78.3% | Pass@1 |
| Summarization | 72.1% | Human preference |
| Question Answering | 81.7% | F1 score |
| Translation | 69.4% | BLEU correlation |

ORM works best for tasks with clear correctness signals.

### Calibration Analysis

Expected Calibration Error (ECE) across confidence bins:

```
Confidence    Accuracy    ECE
[0.0-0.2]:    0.12       0.08
[0.2-0.4]:    0.35       0.05
[0.4-0.6]:    0.51       0.09
[0.6-0.8]:    0.72       0.08
[0.8-1.0]:    0.89       0.09

Overall ECE: 0.078
```

Well-calibrated predictions across confidence ranges.

### Ablation Studies

**Architecture components:**
```
No hidden layer:       68.2%
1 hidden layer:        71.7%
2 hidden layers:       73.5%
3 hidden layers:       73.6% (marginal gain)
```

**Pooling strategies:**
```
Last token:   71.2%
Mean pooling: 73.5%
Max pooling:  72.8%
CLS token:    72.1%
```

Mean pooling works best for ORM.

**Dropout rate:**
```
No dropout:   71.8% (overfitting)
0.1 dropout:  73.5%
0.2 dropout:  73.2%
0.3 dropout:  72.1% (underfitting)
```

## 9. Common Pitfalls

### 1. Insufficient Negative Examples

**Problem:** Only training on correct outputs, no negative examples.

**Solution:** Balance dataset with incorrect outputs:
```python
# Ensure 40-60% negative examples
negatives = generate_wrong_answers(questions)
dataset.extend(negatives)
```

### 2. Shortcut Learning

**Problem:** ORM learns spurious correlations (length, specific tokens).

**Example:** Learning "longer = better" instead of actual correctness.

**Solution:** Adversarial training and diverse examples:
```python
# Include short correct and long incorrect examples
balanced_by_length = ensure_length_diversity(dataset)
```

### 3. Distribution Shift

**Problem:** ORM trained on human outputs, tested on model outputs.

**Solution:** Include model-generated examples in training:
```python
# Iterative data collection
model_outputs = policy.generate(questions)
labels = verify_outputs(model_outputs)
dataset.extend(zip(model_outputs, labels))
```

### 4. Overconfident Predictions

**Problem:** ORM assigns very high/low scores without justification.

**Solution:** Calibration via temperature scaling:
```python
# Find optimal temperature on validation set
T = calibrate_temperature(orm, val_data)

# Apply at inference
calibrated_score = sigmoid(logits / T)
```

### 5. Reward Hacking

**Problem:** Policy learns to exploit ORM weaknesses.

**Example:** Generating outputs that score high but are actually wrong.

**Solution:**
```python
# Add KL penalty to keep policy close to baseline
reward_total = orm_score - beta * KL(policy || baseline)

# Or use iterative ORM retraining
if iteration % 10 == 0:
    retrain_orm(new_policy_outputs)
```

### 6. Class Imbalance

**Problem:** 90% correct, 10% incorrect → model learns to always predict correct.

**Solution:** Balanced sampling or weighted loss:
```python
# Weighted BCE
pos_weight = (num_negative / num_positive)
loss = F.binary_cross_entropy_with_logits(
    pred, target, pos_weight=torch.tensor(pos_weight)
)
```

### 7. Not Comparing to Baselines

**Problem:** Don't know if ORM actually helps.

**Solution:** Always compare to:
- Random selection
- Length-based selection
- Rule-based verification (if available)
- Human selection (upper bound)

### 8. Ignoring Uncertainty

**Problem:** Treating all ORM predictions as equally reliable.

**Solution:** Use ensemble or MC Dropout for uncertainty:
```python
# Only trust high-confidence predictions
if orm_score > threshold and uncertainty < max_uncertainty:
    accept_output()
```

### 9. Fixed Threshold

**Problem:** Using same threshold for all inputs.

**Solution:** Adaptive thresholding:
```python
# Threshold based on input difficulty or domain
threshold = get_adaptive_threshold(input_difficulty)
```

### 10. Not Using for Right Task

**Problem:** Applying ORM to multi-step reasoning where PRM would be better.

**Solution:** Match model to task:
- Single-step decisions → ORM
- Multi-step reasoning → PRM
- Explainability needed → Generative RM

## 10. References

### Primary Papers

- **Cobbe, K., et al. (2021).** Training Verifiers to Solve Math Word Problems. ArXiv:2110.14168.
  - [Paper](https://arxiv.org/abs/2110.14168)
  - Introduced outcome-based reward models for math
  - Showed effectiveness of Best-of-N sampling

- **Uesato, J., et al. (2022).** Solving Math Word Problems with Process- and Outcome-Based Feedback. ArXiv.
  - Compared ORM vs PRM
  - Demonstrated when each is most effective

### Preference Learning Foundations

- **Christiano, P., et al. (2017).** Deep Reinforcement Learning from Human Preferences. NIPS.
  - [Paper](https://arxiv.org/abs/1706.03741)
  - Foundational work on learning rewards from preferences
  - Bradley-Terry model for pairwise comparisons

- **Stiennon, N., et al. (2020).** Learning to Summarize from Human Feedback. NeurIPS.
  - [Paper](https://arxiv.org/abs/2009.01325)
  - Applied outcome rewards to summarization
  - Demonstrated RLHF effectiveness

### RLHF Applications

- **Ouyang, L., et al. (2022).** Training Language Models to Follow Instructions with Human Feedback. OpenAI.
  - [Paper](https://arxiv.org/abs/2203.02155)
  - InstructGPT using outcome reward models
  - Foundation for ChatGPT

- **Bai, Y., et al. (2022).** Constitutional AI: Harmlessness from AI Feedback. Anthropic.
  - [Paper](https://arxiv.org/abs/2212.08073)
  - RLAIF using AI-generated preferences
  - Scalable alignment approach

### Best-of-N Sampling

- **Nakano, R., et al. (2021).** WebGPT: Browser-assisted question-answering with human feedback. OpenAI.
  - [Paper](https://arxiv.org/abs/2112.09332)
  - Used ORM for selecting best web-browsing trajectories

- **Lightman, H., et al. (2023).** Let's Verify Step by Step. OpenAI.
  - [Paper](https://arxiv.org/abs/2305.20050)
  - Compared ORM vs PRM for math reasoning
  - Showed PRM superiority for complex reasoning

### Calibration and Uncertainty

- **Guo, C., et al. (2017).** On Calibration of Modern Neural Networks. ICML.
  - [Paper](https://arxiv.org/abs/1706.04599)
  - Temperature scaling for calibration
  - Essential for reliable reward models

- **Lakshminarayanan, B., et al. (2017).** Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. NIPS.
  - [Paper](https://arxiv.org/abs/1612.01474)
  - Ensemble methods for uncertainty quantification

### Reward Modeling Theory

- **Skalse, J., et al. (2022).** Defining and Characterizing Reward Gaming. NeurIPS.
  - [Paper](https://arxiv.org/abs/2209.13085)
  - Understanding reward hacking
  - Mitigations for reward model exploitation

- **Gao, L., et al. (2022).** Scaling Laws for Reward Model Overoptimization. ArXiv.
  - [Paper](https://arxiv.org/abs/2210.10760)
  - When policies become too good for reward models
  - KL penalties and iterative training

### Implementation References

- **Nexus Implementation:** `Nexus/nexus/models/rl/reward_models/process_reward_model.py`
  - ORM implementation as `OutcomeRewardModel` class
  - Simple feedforward architecture
  - Binary classification training

- **OpenAI Evals:** https://github.com/openai/evals
  - Evaluation framework for language models
  - Includes outcome-based metrics

---

**Key Takeaways:**
- ORMs evaluate only final outcomes, not intermediate steps
- Simple, efficient, and effective for single-step decisions
- Best used with Best-of-N sampling for candidate selection
- Trade-off: Less training signal but cheaper labels than PRM
- Works best when correctness can be automatically verified
- Requires calibration and uncertainty quantification for reliability
- Not suitable for tasks requiring fine-grained credit assignment
- Foundation for RLHF in many production systems (ChatGPT, Claude)
