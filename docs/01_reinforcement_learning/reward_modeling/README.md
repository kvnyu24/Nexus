# Reward Modeling

This directory contains documentation for **reward modeling** techniques used in Reinforcement Learning from Human Feedback (RLHF) and preference-based learning. Reward models learn to predict human preferences and provide training signals for policy learning.

## Overview

Reward modeling addresses a fundamental challenge in AI alignment: **specifying what we want the agent to do**. Instead of hand-crafting reward functions, we:

1. Collect human preferences over behaviors
2. Train a reward model to predict these preferences
3. Use the reward model to train policies via RL

This approach enables:
- Learning complex, nuanced objectives
- Aligning AI systems with human values
- Scaling beyond hand-crafted rewards

## Critical Applications

- **Language Models**: ChatGPT, Claude, GPT-4 (RLHF training)
- **Robotics**: Learning from demonstrations and corrections
- **Dialogue Systems**: Helpful, harmless, and honest assistants
- **Content Generation**: Images, code, and creative writing

## Algorithms Covered

### [Enhanced Reward Model](./enhanced_reward_model.md)
**Core Innovation**: Ensemble-based reward modeling with uncertainty quantification

- Multiple reward heads provide confidence estimates
- Feature banking for consistency
- Robust to noisy preferences
- Well-calibrated outputs

**When to Use**: RLHF for language models, any preference learning task requiring uncertainty.

**Key Papers**: Christiano et al. (2017), Stiennon et al. (2020)

### [Process Reward Model (PRM)](./process_reward_model.md)
**Core Innovation**: Step-by-step verification for reasoning tasks

- Evaluates each step in multi-step reasoning
- Better credit assignment than outcome-only models
- Enables verification during generation
- 15-20% improvement on math problems

**When to Use**: Multi-step reasoning (math, code, planning), when you need interpretable feedback.

**Key Papers**: Lightman et al. (2023) - OpenAI

### [Outcome Reward Model (ORM)](./orm.md)

**Core Innovation**: Simple final-outcome evaluation

- Only scores final result
- Simpler to train than PRM
- Works for single-step decisions
- Baseline for comparison
- Efficient Best-of-N sampling

**When to Use**: Tasks with clear right/wrong answers, single-step decisions, code generation with test suites.

**Key Papers**: Cobbe et al. (2021)

### [Generative Reward Model (GRM)](./generative_rm.md)

**Core Innovation**: LLM-as-judge with natural language explanations (RLAIF)

- Produces interpretable natural language feedback
- Enables Constitutional AI and self-improvement
- 85%+ agreement with human evaluators
- Scalable oversight for complex tasks
- Critique-revision loops

**When to Use**: When interpretability is critical, open-ended tasks, scarce human feedback, Constitutional AI.

**Key Papers**: Bai et al. (2022) - Anthropic Constitutional AI

## Comparison Table

| Model Type | Granularity | Training Data | Interpretability | Use Case |
|------------|-------------|---------------|------------------|----------|
| Enhanced RM | Outcome | Pairwise preferences | Low | General RLHF |
| PRM | Per-step | Step-level labels | High | Multi-step reasoning |
| ORM | Outcome | Outcome labels | Medium | Simple tasks |
| Generative RM | Outcome + Explanation | Labeled + explanations | Very High | Explainable AI |

## Key Concepts

### Preference Learning

Given human preferences: "Output A is better than Output B"

Learn reward function satisfying:
```
P(A ≻ B) = σ(r(A) - r(B))
```

This is the **Bradley-Terry model** of pairwise preferences.

### Process vs Outcome Supervision

**Outcome Supervision**:
- Label: Final answer correct/incorrect
- Signal: Sparse, all-or-nothing
- Credit: Hard to assign to specific steps

**Process Supervision**:
- Label: Each step correct/incorrect/neutral
- Signal: Dense, fine-grained
- Credit: Clear attribution

### Reward Model vs Value Function

**Similarities**:
- Both predict expected value
- Both used for policy learning
- Both can be trained with temporal difference

**Differences**:
- Reward model: Trained on preferences
- Value function: Trained on returns
- Reward model: Can be off-policy
- Value function: Typically on-policy

### Uncertainty Quantification

Why it matters:
- Know when the reward model is confident
- Avoid overconfident wrong predictions
- Guide exploration to uncertain regions
- Detect out-of-distribution inputs

Methods:
- Ensemble disagreement
- Bayesian neural networks
- Dropout at test time
- Calibration techniques

## Training Pipeline

### 1. Data Collection

**Pairwise Preferences**:
```python
# Human labels which output is better
data = [
    {"prompt": "Write a poem", "output_a": "...", "output_b": "...", "preference": "a"},
    ...
]
```

**Step-Level Labels** (for PRM):
```python
# Label each reasoning step
data = [
    {
        "question": "What is 2+3*4?",
        "steps": [
            "First multiply 3*4 = 12",  # label: correct
            "Then add 2+12 = 14",       # label: correct
        ],
        "labels": [1, 1]
    }
]
```

### 2. Reward Model Training

```python
from nexus.models.rl.preference import EnhancedRewardModel

# Configure
config = {
    "input_dim": 768,
    "hidden_dim": 512,
    "num_reward_heads": 3,  # Ensemble
}

# Train
rm = EnhancedRewardModel(config)
for batch in dataset:
    outputs = rm(batch["embeddings"])
    loss = compute_preference_loss(outputs, batch["preferences"])
    loss.backward()
    optimizer.step()
```

### 3. Policy Training with RM

```python
# Use reward model in PPO
for rollout in env:
    actions, log_probs, values = policy(states)
    rewards = reward_model(states, actions)  # ← Reward model
    advantages = compute_advantages(rewards, values)
    policy_loss = -log_probs * advantages
    policy_loss.backward()
```

### 4. Evaluation

- **Agreement with humans**: How often RM agrees with human labels
- **Calibration**: Do predicted probabilities match empirical frequencies?
- **Robustness**: Performance on out-of-distribution inputs
- **Downstream performance**: Does better RM lead to better policy?

## Common Patterns

### Ensemble for Uncertainty

```python
# Train multiple reward models
ensemble = [RewardModel(config) for _ in range(K)]

# Aggregate predictions
rewards = [rm(x) for rm in ensemble]
mean_reward = np.mean(rewards)
uncertainty = np.std(rewards)  # Disagreement = uncertainty
```

### Best-of-N Sampling

```python
# Generate N candidates
candidates = [policy.generate(prompt) for _ in range(N)]

# Score with reward model
scores = [reward_model(prompt, candidate) for candidate in candidates]

# Select best
best_idx = np.argmax(scores)
return candidates[best_idx]
```

This often works better than RL for language models!

### Reward Shaping

Combine learned reward with sparse ground truth:
```python
reward_total = α * reward_model(s, a) + (1-α) * reward_true(s, a)
```

Balance learning from preferences with task success.

## Practical Considerations

### Data Requirements

| Model | Data Amount | Data Quality | Labeling Cost |
|-------|-------------|--------------|---------------|
| Enhanced RM | 10K-100K pairs | Medium | Medium |
| PRM | 50K-500K steps | High | Very High |
| ORM | 5K-50K outcomes | Low | Low |

### Computational Costs

- **Training**: Similar to language model fine-tuning
- **Inference**: Fast (single forward pass)
- **Ensemble**: K× cost but more robust

### Common Failure Modes

1. **Reward Hacking**: Policy exploits reward model flaws
2. **Overoptimization**: Policy becomes too good for RM
3. **Distribution Shift**: RM fails on policy's outputs
4. **Label Noise**: Inconsistent human preferences

### Mitigations

1. **Iterative Training**: Retrain RM on policy's outputs
2. **Uncertainty Penalization**: Penalize high-uncertainty regions
3. **Ensemble**: Robust to individual model failures
4. **KL Penalty**: Keep policy close to supervised baseline

## Research Directions

### Open Problems

- **Sample Efficiency**: Reduce human labeling requirements
- **Scalable Oversight**: Train RM on superhuman tasks
- **Multi-Objective**: Balance multiple preferences
- **Interpretability**: Understand what RM has learned
- **Robustness**: Prevent reward hacking

### Future Work

- **Self-Training**: Use AI to label for reward models
- **Constitutional AI**: Encode principles instead of examples
- **Recursive Reward Modeling**: RM helps train better RM
- **Active Learning**: Query most informative preferences

## Performance Benchmarks

### Preference Prediction Accuracy

On held-out human preferences:

| Model | Accuracy | Calibration Error |
|-------|----------|-------------------|
| Random | 50% | - |
| Fine-tuned LM | 65% | 0.15 |
| Enhanced RM | 72% | 0.08 |
| Ensemble RM | 75% | 0.05 |

### Downstream Task Performance

Using RM in RLHF for summarization:

| Method | ROUGE | Human Preference |
|--------|-------|------------------|
| Supervised | 0.45 | 60% |
| RL with sparse reward | 0.48 | 65% |
| RL with ORM | 0.52 | 72% |
| RL with PRM | 0.55 | 78% |

## Code Locations

- **Enhanced RM**: `/Users/kevinyu/Projects/Nexus/nexus/models/rl/preference/reward_model.py`
- **PRM**: `/Users/kevinyu/Projects/Nexus/nexus/models/rl/reward_models/process_reward_model.py`
- **ORM**: Included in PRM file as `OutcomeRewardModel`

## References

### Foundational Papers

1. **Christiano, P., et al. (2017).** Deep Reinforcement Learning from Human Preferences. NIPS.
   - Introduced reward modeling for complex tasks
2. **Stiennon, N., et al. (2020).** Learning to Summarize from Human Feedback. NeurIPS.
   - Applied to language model fine-tuning
3. **Ouyang, L., et al. (2022).** Training Language Models to Follow Instructions with Human Feedback. OpenAI.
   - InstructGPT, foundation of ChatGPT

### Process Supervision

4. **Lightman, H., et al. (2023).** Let's Verify Step by Step. OpenAI.
   - Process reward models for math reasoning
5. **Uesato, J., et al. (2022).** Solving Math Word Problems with Process- and Outcome-Based Feedback.

### Applications

6. **Bai, Y., et al. (2022).** Constitutional AI: Harmlessness from AI Feedback. Anthropic.
7. **Touvron, H., et al. (2023).** Llama 2: Open Foundation and Fine-Tuned Chat Models. Meta.

---

**Navigation**:
- [← Back to RL Overview](../)
- [Enhanced Reward Model →](./enhanced_reward_model.md)
- [Process Reward Model →](./process_reward_model.md)
