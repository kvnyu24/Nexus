# LLM Alignment Methods

LLM alignment aims to make language models behave according to human preferences and values. This directory covers modern **preference optimization** and **reinforcement learning from human feedback (RLHF)** methods.

## The Alignment Problem

Pre-trained LLMs are trained to predict next tokens, not to be helpful/harmless/honest. Key challenges:

1. **Misalignment**: LLMs don't inherently follow user intent
2. **Safety**: May generate harmful, biased, or toxic content
3. **Truthfulness**: Can confidently state falsehoods (hallucination)
4. **Instruction following**: Need fine-tuning to follow instructions

## Alignment Paradigm Evolution

### 1. Supervised Fine-Tuning (SFT)
```
Dataset: (prompt, high-quality response) pairs
Loss: -log P(response | prompt)
```
**Limitation**: Requires expensive human demonstrations.

### 2. RLHF (Classic)
```
1. Train reward model on preference data
2. Use PPO to optimize policy against reward model
```
**Limitation**: Complex, unstable, requires value network.

### 3. Direct Preference Optimization (Modern)
```
Skip reward model, optimize directly on preferences
```
**Advantage**: Simpler, more stable, no RM needed.

## Algorithm Taxonomy

### **Preference-Based Methods** (Pairwise Comparisons)

Use datasets of (prompt, chosen_response, rejected_response):

| Algorithm | Year | Needs Reference Model | Key Innovation |
|-----------|------|----------------------|----------------|
| **DPO** | 2023 | Yes | Direct optimization, no reward model |
| **IPO** | 2023 | Yes | Bounded loss prevents overfitting |
| **SimPO** | 2024 | No | Length-normalized implicit reward |
| **ORPO** | 2024 | No | Monolithic SFT + alignment |

### **Binary Feedback Methods**

Use individual examples labeled as good/bad:

| Algorithm | Year | Key Innovation |
|-----------|------|----------------|
| **KTO** | 2024 | Prospect theory, no pairwise needed |

### **Policy Gradient Methods**

Use reward model or rule-based rewards:

| Algorithm | Year | Critic Needed | Key Innovation |
|-----------|------|---------------|----------------|
| **GRPO** | 2024 | No | Group-relative advantages |
| **RLOO** | 2024 | No | Leave-one-out baseline |
| **ReMax** | 2024 | No | Greedy action baseline |

### **Advanced / Hybrid**

| Algorithm | Year | Type | Key Innovation |
|-----------|------|------|----------------|
| **SPIN** | 2024 | Self-play | Iterative self-improvement |
| **RLVR** | 2024 | Verification | Reward modeling + verification |

## Algorithm Comparison

### Memory Efficiency

**Most Efficient** (Single Model):
- SimPO, ORPO: No reference model needed
- GRPO, RLOO, ReMax: No critic needed

**Medium** (Reference Model):
- DPO, IPO, KTO: Reference model (frozen, no gradients)

**Least Efficient** (Multiple Models):
- Classic RLHF (PPO): Policy + Critic + Reference + Reward Model
- SPIN: Policy + Reference (updated iteratively)

### Data Requirements

**Preference Pairs** (chosen vs. rejected):
- DPO, IPO, SimPO, ORPO

**Binary Labels** (good vs. bad):
- KTO

**Rewards** (scalar scores):
- GRPO, RLOO, ReMax, RLVR

**No Labels** (self-play):
- SPIN

### Stability

**Most Stable**:
- DPO, SimPO: Closed-form, no sampling variance
- ORPO: Monolithic, simpler training

**Medium**:
- IPO, KTO: Bounded objectives
- GRPO, RLOO: Variance-reduced baselines

**Less Stable**:
- SPIN: Iterative, can collapse
- Classic PPO: High variance, complex

## When to Use Each Algorithm

### Have Preference Pairs

**Default choice**: DPO
- Simple, effective, well-studied
- Requires reference model but worth it

**Need extra stability**: IPO
- Bounded loss prevents DPO's overfitting
- Slightly better on some benchmarks

**Limited memory**: SimPO or ORPO
- SimPO: Reference-free, length normalization
- ORPO: Combines SFT + alignment, no reference needed

### Have Binary Feedback (Good/Bad)

**Use**: KTO
- More data-efficient than collecting pairs
- Based on prospect theory (loss aversion)

### Have Reward Model

**Memory-efficient**: GRPO, RLOO, or ReMax
- No critic network needed (unlike PPO)
- GRPO: Group-based advantages
- RLOO: Leave-one-out baseline (most theoretically sound)
- ReMax: Greedy baseline (simplest, fastest)

**Want classic approach**: PPO (not covered here, see `ppo.py`)

### Want Self-Improvement

**Use**: SPIN
- Iterative training against old self
- No human labels after initialization

## Common Hyperparameters

### All Preference Methods

```python
beta = 0.1  # KL penalty strength (higher = stay closer to reference)
```

### DPO-Family

```python
beta = 0.1           # Default for DPO
beta = 0.5           # Higher for IPO (more regularization)
gamma = 0.5          # SimPO reward margin
lambda_weight = 1.0  # ORPO odds-ratio weight
```

### Policy Gradient Methods

```python
group_size = 4-8       # GRPO: samples per prompt
num_samples = 4        # RLOO: K samples for leave-one-out
temperature = 1.0      # Sampling temperature
max_grad_norm = 1.0    # Gradient clipping
```

### KTO

```python
lambda_good = 1.0  # Weight for desirable examples
lambda_bad = 1.0   # Weight for undesirable (>1.0 for loss aversion)
```

## Implementation Patterns

All alignment methods in this directory follow similar interfaces:

```python
from nexus.models.rl.preference import DPOAgent, KTOAgent, SimPOAgent
from nexus.models.rl import GRPOAgent

# Preference-based (DPO, IPO, SimPO, ORPO)
agent = DPOAgent(config={
    "policy": model,
    "reference_policy": ref_model,
    "beta": 0.1,
})

batch = {
    "chosen_input_ids": ...,
    "rejected_input_ids": ...,
    "chosen_attention_mask": ...,
    "rejected_attention_mask": ...,
}
metrics = agent.update(batch)

# Binary feedback (KTO)
agent = KTOAgent(config={
    "policy": model,
    "reference_policy": ref_model,
    "beta": 0.1,
})

batch = {
    "input_ids": ...,
    "is_desirable": torch.tensor([1, 0, 1, 0]),  # Binary labels
    "attention_mask": ...,
}
metrics = agent.update(batch)

# Policy gradient (GRPO, RLOO, ReMax)
agent = GRPOAgent(config={
    "policy": model,
    "group_size": 8,
})

# Generate samples
samples, log_probs = agent.generate_samples(prompts, attention_mask)

# Get rewards (from reward model or environment)
rewards = reward_model(samples)

# Update
batch = {"input_ids": samples, "rewards": rewards, "old_log_probs": log_probs}
metrics = agent.update(batch)
```

## Datasets

### Preference Datasets

- **Anthropic HH-RLHF**: Helpful & Harmless conversations
- **OpenAssistant**: Multilingual preferences
- **UltraFeedback**: GPT-4 annotated preferences
- **PKU-SafeRLHF**: Safety-focused preferences

### Format

```json
{
  "prompt": "How do I bake a cake?",
  "chosen": "Here's a simple recipe: ...",
  "rejected": "Just buy one from the store."
}
```

## Evaluation

### Automated Metrics

1. **Win Rate**: GPT-4 as judge (chosen > rejected)
2. **Reward Model Score**: Trained preference model
3. **Perplexity**: Lower on high-quality held-out data

### Human Evaluation

1. **Helpfulness**: Does response address the query?
2. **Harmlessness**: Avoids toxic/biased content?
3. **Honesty**: Factually accurate?

### Benchmarks

- **AlpacaEval**: LLM-as-judge against GPT-4
- **MT-Bench**: Multi-turn conversations
- **TruthfulQA**: Factual accuracy
- **HHH Eval**: Helpful, Harmless, Honest

## Best Practices

### 1. Start with SFT

Always do supervised fine-tuning before alignment:
```
Pre-trained → SFT → Alignment
```

### 2. Use High-Quality Data

Quality > Quantity for preferences. Better to have 10K high-quality pairs than 100K noisy ones.

### 3. Monitor for Reward Hacking

Check if model exploits reward model flaws:
```python
# Validate on held-out data
# Check edge cases
# Monitor response length (common hack)
```

### 4. Regularize to Reference

Always use KL penalty to prevent:
- Mode collapse
- Reward hacking
- Catastrophic forgetting

### 5. Length Normalization

For preference methods, consider length bias:
```python
# Longer responses often preferred
# Use length normalization (SimPO)
# Or penalize length in reward model
```

## Common Pitfalls

### 1. Forgetting to Freeze Reference Model

```python
# WRONG
reference_model = copy(policy)
reference_model.requires_grad = True  # Still trains!

# RIGHT
reference_model = copy(policy)
for param in reference_model.parameters():
    param.requires_grad = False
```

### 2. Wrong Log-Prob Shift

```python
# WRONG: Predict t from t
log_probs = F.log_softmax(logits, dim=-1)

# RIGHT: Predict t+1 from t
log_probs = F.log_softmax(logits[:, :-1], dim=-1)
```

### 3. Not Masking Prompt Tokens

```python
# WRONG: Compute loss on prompt too
loss = -log_probs.mean()

# RIGHT: Only on response
loss = -(log_probs * response_mask).sum() / response_mask.sum()
```

### 4. Ignoring Length Bias

Longer responses often preferred (more information). Consider:
- Length normalization (SimPO)
- Length penalty in rewards
- Balanced dataset sampling

## References

### Foundational Papers

```bibtex
@article{ouyang2022training,
  title={Training language models to follow instructions with human feedback},
  author={Ouyang, Long and Wu, Jeff and Jiang, Xu and others},
  journal={NeurIPS},
  year={2022}
}

@article{rafailov2023dpo,
  title={Direct Preference Optimization: Your Language Model is Secretly a Reward Model},
  author={Rafailov, Rafael and Sharma, Archit and Mitchell, Eric and Ermon, Stefano and Manning, Christopher D and Finn, Chelsea},
  journal={NeurIPS},
  year={2023}
}
```

### Algorithm Documentation

- [DPO - Direct Preference Optimization](./dpo.md)
- [GRPO - Group Relative Policy Optimization](./grpo.md)
- [KTO - Kahneman-Tversky Optimization](./kto.md)
- [SimPO - Simple Preference Optimization](./simpo.md)
- [ORPO - Odds Ratio Preference Optimization](./orpo.md)
- [IPO - Identity Preference Optimization](./ipo.md)
- [SPIN - Self-Play Fine-Tuning](./spin.md)
- [RLOO - REINFORCE Leave-One-Out](./rloo.md)
- [ReMax - REINFORCE with Maximum Baseline](./remax.md)
- [RLVR - Reinforcement Learning from Verification Rewards](./rlvr.md)
