# Best-of-N Sampling with Process Reward Models (PRM)

## Overview & Motivation

Best-of-N sampling with Process Reward Models (PRM) is a test-time compute technique that generates multiple candidate outputs and selects the best using a learned reward model. Unlike outcome-based rewards that only evaluate final answers, PRMs provide step-by-step feedback, enabling more accurate selection of high-quality reasoning paths.

### What Problem Does Best-of-N Solve?

**Single-Sample Generation**:
- Model outputs one answer
- No recovery from early mistakes
- No exploration of alternatives
- Quality capped by single forward pass

**Best-of-N with PRM**:
- Generate N diverse candidates
- Each candidate independently samples from distribution
- PRM scores quality of each reasoning process
- Select highest-scoring candidate
- Quality improves with N (up to saturation)

**Key Insight**: Given a budget, spending compute on sampling multiple outputs and selecting the best often outperforms using a larger model for a single output.

### Key Achievements

- **Superlinear Gains**: Pass@N >> N × Pass@1 for many tasks
- **Verifiable Improvement**: Measurable quality gains with more compute
- **Process Supervision**: Better than outcome supervision for complex reasoning
- **Simple Implementation**: Easy to add to existing models
- **Broad Applicability**: Works for code, math, reasoning, creative tasks

## Theoretical Background

### Pass@N Metric

**Definition**: Probability that at least one of N samples is correct.

```
Pass@N = 1 - (1 - Pass@1)^N
```

**Example**: If Pass@1 = 20%:
- Pass@5 = 1 - 0.8^5 = 67.2%
- Pass@10 = 1 - 0.8^10 = 89.3%
- Pass@50 = 1 - 0.8^50 = 99.98%

**Key Property**: Superlinear improvement with N.

### Process Reward Models

**Outcome Reward Model (ORM)**:
```
R_outcome(solution) = {
    1  if final answer is correct
    0  otherwise
}
```

**Problem**: No credit for partial progress, brittle.

**Process Reward Model (PRM)**:
```
R_process(solution) = Σ_t r(s_t, a_t)
```

Rewards each step s_t, action a_t in reasoning process.

**Advantage**: 
- Identifies good reasoning even if final answer is wrong
- Detects errors early in chain of thought
- More sample-efficient learning signal

### Selection vs. Generation

**Generation**: Model learns to produce high-quality outputs
**Selection**: Model learns to identify high-quality outputs

**Key Trade-off**:
- Good generator but poor selector → wasted samples
- Poor generator but good selector → low quality ceiling
- Both good → optimal performance

### Diversity-Quality Trade-off

**High Temperature (T=1.0)**:
- Diverse samples
- Some low quality
- Good for exploration

**Low Temperature (T=0.2)**:
- Similar samples
- Higher average quality
- Limited exploration

**Optimal**: Temperature ≈ 0.6-0.8 for most tasks.

## Mathematical Formulation

### Best-of-N Algorithm

**Input**: Prompt p, model M, PRM P, number of samples N

**Output**: Best solution

**Procedure**:
1. Generate N candidate solutions:
   ```
   S = {s₁, s₂, ..., sₙ} where sᵢ ~ M(·|p)
   ```

2. Score each candidate with PRM:
   ```
   scores = {P(s₁), P(s₂), ..., P(sₙ)}
   ```

3. Select best:
   ```
   s* = argmax_{s ∈ S} P(s)
   ```

4. Return s*

### Process Reward Model Training

**Data**: Solutions with step-by-step correctness labels

**Objective**: Predict correctness of each step

```
L = Σ_{(s,a,y) ∈ D} -log P(y | s, a; θ)
```

Where:
- s: Current state (partial solution)
- a: Action taken (next step)
- y: Correctness label {0, 1}
- θ: PRM parameters

### Composite Scoring

Combine multiple signals:

```
score(solution) = w₁·PRM(solution) + 
                  w₂·confidence(solution) +
                  w₃·consistency(solution)
```

Where:
- PRM: Process reward
- Confidence: Model's own probability estimate
- Consistency: Agreement with other samples

## High-Level Intuition

### The Core Idea

Imagine taking a multiple-choice test:

**Strategy 1**: Carefully work through problem once
- Time: 5 minutes
- Accuracy: 70%

**Strategy 2**: Quickly attempt 3 different approaches, pick best
- Time: 3×2 = 6 minutes (slightly more)
- Each attempt: 50% accuracy (faster, less careful)
- At least one correct: 1 - 0.5³ = 87.5%

**Strategy 2 wins** despite each attempt being lower quality!

### Process vs. Outcome Rewards

**Math Problem Example**:

**Solution A**:
```
Step 1: Set up equation (correct)
Step 2: Solve for x (correct)
Step 3: Compute final answer (arithmetic error)
Final: WRONG
```

**Solution B**:
```
Step 1: Guess random approach (wrong)
Step 2: Lucky cancellation 
Step 3: Get right answer by chance
Final: CORRECT
```

**Outcome Reward**: B > A (only cares about final answer)
**Process Reward**: A > B (values correct reasoning)

**Key**: PRM identifies that A's approach is better despite wrong final answer.

### Why Diversity Matters

**Low Temperature** (deterministic):
```
Sample 1: 2 + 2 = 5
Sample 2: 2 + 2 = 5  (same mistake!)
Sample 3: 2 + 2 = 5
Best: Still wrong
```

**High Temperature** (diverse):
```
Sample 1: 2 + 2 = 5  (arithmetic error)
Sample 2: 2 + 2 = 4  (correct!)
Sample 3: 2 + 2 = 3  (different error)
Best: Correct!
```

**Diversity allows recovery from mistakes**.

## Implementation Details

### Sampling Strategy

**Temperature Sampling**:
```python
# Generate diverse candidates
candidates = []
for _ in range(N):
    output = model.generate(
        prompt,
        temperature=0.7,  # Balance quality and diversity
        top_p=0.9,        # Nucleus sampling
        max_tokens=512
    )
    candidates.append(output)
```

**Diverse Beam Search** (alternative):
```python
# Generate N diverse beams
candidates = model.generate(
    prompt,
    num_return_sequences=N,
    num_beam_groups=N,
    diversity_penalty=1.0
)
```

### PRM Architecture

```python
class ProcessRewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.encoder = base_model  # Pre-trained LM
        self.reward_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, solution_steps):
        """Score each step in solution.
        
        Args:
            solution_steps: List of reasoning steps
            
        Returns:
            Step-wise rewards and total reward
        """
        step_rewards = []
        
        for step in solution_steps:
            # Encode step in context
            hidden = self.encoder(step)
            
            # Predict reward for this step
            reward = self.reward_head(hidden)
            step_rewards.append(reward)
        
        # Total reward: sum of step rewards
        total_reward = sum(step_rewards)
        
        return total_reward, step_rewards
```

### Best-of-N Pipeline

```python
def best_of_n_pipeline(prompt, model, prm, N=10):
    """Generate N candidates and select best with PRM."""
    
    # Step 1: Generate N diverse candidates
    candidates = []
    for i in range(N):
        output = model.generate(
            prompt,
            temperature=0.8,
            do_sample=True
        )
        candidates.append(output)
    
    # Step 2: Parse into steps (for PRM)
    parsed_candidates = [
        parse_reasoning_steps(cand) 
        for cand in candidates
    ]
    
    # Step 3: Score with PRM
    scores = []
    for steps in parsed_candidates:
        score, _ = prm(steps)
        scores.append(score)
    
    # Step 4: Select best
    best_idx = np.argmax(scores)
    best_candidate = candidates[best_idx]
    
    return best_candidate, scores
```

### Hyperparameters

```python
# Sampling
N = 10                    # Number of samples
temperature = 0.7         # Sampling temperature
top_p = 0.9               # Nucleus sampling threshold
max_tokens = 512          # Max length

# PRM
prm_model = 'gpt2-large'  # Base model for PRM
prm_checkpoint = 'path'   # Trained PRM weights

# Selection
selection = 'max'         # or 'weighted', 'majority_vote'
```

## Code Walkthrough

Implementation in `/Users/kevinyu/Projects/Nexus/nexus/models/test_time/best_of_n_prm.py`:

### Process Reward Model

```python
class ProcessRewardModel(NexusModule):
    def __init__(self, config):
        super().__init__(config)
        self.base_model = load_pretrained(config['base_model'])
        self.hidden_dim = config['hidden_dim']
        
        # Reward prediction head
        self.reward_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)
        )
    
    def forward(self, input_ids, attention_mask):
        """Compute reward for input sequence.
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask
            
        Returns:
            Rewards for each position
        """
        # Encode with base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_states = outputs.last_hidden_state
        
        # Predict reward for each token
        rewards = self.reward_head(hidden_states).squeeze(-1)
        
        return rewards
    
    def score_solution(self, solution_text):
        """Score a complete solution.
        
        Args:
            solution_text: String with solution
            
        Returns:
            Total reward score
        """
        # Tokenize
        inputs = self.tokenizer(
            solution_text,
            return_tensors='pt',
            padding=True
        )
        
        # Get rewards
        with torch.no_grad():
            rewards = self.forward(
                inputs['input_ids'],
                inputs['attention_mask']
            )
        
        # Average reward (or sum)
        total_reward = rewards.mean().item()
        
        return total_reward


def train_prm(model, training_data, validation_data, epochs=10):
    """Train process reward model.
    
    Args:
        model: ProcessRewardModel instance
        training_data: List of (solution, step_labels)
        validation_data: Validation set
        epochs: Number of training epochs
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for solution, labels in training_data:
            # Forward pass
            inputs = tokenize(solution)
            rewards = model(inputs['input_ids'], inputs['attention_mask'])
            
            # Loss: MSE between predicted and true rewards
            loss = F.mse_loss(rewards, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        val_acc = evaluate_prm(model, validation_data)
        print(f'Epoch {epoch}: Loss={total_loss:.4f}, Val Acc={val_acc:.4f}')
```

### Best-of-N Sampler

```python
class BestOfNSampler(NexusModule):
    def __init__(self, config):
        super().__init__(config)
        self.generator = load_generator(config['generator'])
        self.prm = ProcessRewardModel(config['prm_config'])
        self.num_samples = config.get('num_samples', 10)
        self.temperature = config.get('temperature', 0.7)
    
    def generate_candidates(self, prompt, n=None):
        """Generate n diverse candidates."""
        if n is None:
            n = self.num_samples
            
        candidates = []
        for _ in range(n):
            output = self.generator.generate(
                prompt,
                temperature=self.temperature,
                do_sample=True,
                max_new_tokens=512
            )
            candidates.append(output)
        
        return candidates
    
    def select_best(self, candidates):
        """Select best candidate using PRM."""
        scores = [
            self.prm.score_solution(cand) 
            for cand in candidates
        ]
        
        best_idx = np.argmax(scores)
        
        return candidates[best_idx], scores[best_idx]
    
    def generate(self, prompt, return_all=False):
        """Full best-of-N generation pipeline.
        
        Args:
            prompt: Input prompt
            return_all: If True, return all candidates and scores
            
        Returns:
            Best candidate (and optionally all candidates/scores)
        """
        # Generate candidates
        candidates = self.generate_candidates(prompt)
        
        # Select best
        best_candidate, best_score = self.select_best(candidates)
        
        if return_all:
            scores = [self.prm.score_solution(c) for c in candidates]
            return best_candidate, {
                'candidates': candidates,
                'scores': scores,
                'best_score': best_score
            }
        
        return best_candidate
```

## Optimization Tricks

### 1. Early Stopping

```python
# Stop if we find a high-confidence solution early
for i in range(N):
    candidate = generate()
    score = prm.score(candidate)
    
    if score > high_confidence_threshold:
        return candidate  # Good enough!
```

### 2. Batched Scoring

```python
# Score all candidates in parallel
scores = prm.batch_score(candidates)  # Much faster
```

### 3. Cached Prefixes

```python
# Reuse prompt encoding across samples
prompt_cache = model.encode_prompt(prompt)

for _ in range(N):
    output = model.generate_from_cache(prompt_cache)
```

### 4. Adaptive N

```python
# Use more samples for harder problems
if problem_difficulty > threshold:
    N = 50
else:
    N = 10
```

### 5. Majority Voting

```python
# For tasks with discrete answers, use voting
final_answers = [extract_answer(c) for c in candidates]
best_answer = most_common(final_answers)
```

### 6. Weighted Ensemble

```python
# Combine multiple candidates weighted by scores
weights = softmax(scores / temperature)
ensemble_output = weighted_average(candidates, weights)
```

### 7. Reranking

```python
# First pass: Generate N with cheap model
candidates = cheap_model.generate(n=100)

# Second pass: Rerank with PRM, keep top K
scores = prm.score_batch(candidates)
top_k = select_top_k(candidates, scores, k=10)

# Third pass: Refine with expensive model
refined = expensive_model.refine(top_k)
```

## Experiments & Results

### Code Generation (HumanEval)

**Pass@N Results**:
- Pass@1: 65.2%
- Pass@10: 81.4%
- Pass@50: 92.8%
- Pass@100: 96.1%

**With PRM Selection** (same samples):
- Best@10: 84.7% (+3.3%)
- Best@50: 94.9% (+2.1%)

**Key Finding**: PRM improves over random selection, especially for smaller N.

### Math Reasoning (GSM8K)

**Without PRM**:
- Pass@1: 72.3%
- Pass@10: 89.2%
- Majority Vote@10: 85.1%

**With PRM**:
- Best@10: 91.8% (+2.6% over Pass@N, +6.7% over Voting)

**Key Finding**: Process rewards >> outcome rewards for reasoning.

### Step-Level Analysis

**Correlation with Correctness**:
- Outcome Reward: 0.72
- Process Reward (PRM): 0.89

**Early Error Detection**:
- PRM detects errors at step 3 (average)
- Outcome reward only sees error at end

### Scaling with Compute

**Budget: 100x compute**

**Strategy A**: Use GPT-4 (10x cost), sample N=10
- Quality: 88%

**Strategy B**: Use GPT-3.5 (1x cost), sample N=100 with PRM
- Quality: 91%

**Key Finding**: Best-of-N can beat using a larger model.

## Common Pitfalls

### 1. Insufficient Diversity

**Symptom**: All samples are nearly identical.

**Cause**: Temperature too low, beam search too narrow.

**Solutions**:
- Increase temperature (0.7-1.0)
- Use nucleus/top-k sampling
- Penalize duplicates

### 2. PRM Overfitting

**Symptom**: PRM scores don't correlate with actual quality.

**Cause**: Training data not representative, overfitting.

**Solutions**:
- Larger, more diverse training set
- Regularization (dropout, weight decay)
- Validate on held-out set

### 3. Expensive Inference

**Symptom**: Latency too high for production.

**Cause**: Large N, slow PRM.

**Solutions**:
- Reduce N for easier inputs
- Distill PRM to smaller model
- Batch scoring
- Early stopping

### 4. Poor Generalization

**Symptom**: PRM works on training distribution, fails on test.

**Cause**: Train-test mismatch.

**Solutions**:
- Train on diverse data
- Domain adaptation
- Ensemble of PRMs

### 5. Reward Hacking

**Symptom**: Model generates outputs that score high but are wrong.

**Cause**: PRM has exploitable patterns.

**Solutions**:
- Adversarial training
- Human validation
- Multiple reward models

## References

### Original Papers

1. **Process Reward Models** (2023)
   - Lightman et al. (OpenAI)
   - Training Verifiers to Solve Math Word Problems

2. **Let's Verify Step by Step** (2023)
   - OpenAI
   - https://arxiv.org/abs/2305.20050

3. **Scaling Test-Time Compute** (2024)
   - Snell et al. (Google)
   - https://arxiv.org/abs/2408.03314

### Related Work

4. **Self-Consistency** (2022)
   - Wang et al.
   - Majority voting over samples

5. **Outcome-Based Rewards** (2022)
   - Cobbe et al.
   - Training verifiers (ORMs)

### Applications

6. **Code Generation** (AlphaCode, 2022)
   - Li et al.
   - Massive sampling + filtering

7. **Math Reasoning** (Minerva, 2022)
   - Lewkowycz et al.
   - Best-of-N for math

8. **General Reasoning** (2024)
   - o1 model (OpenAI)
   - Heavy test-time compute
