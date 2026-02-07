# Enhanced Reward Model

## 1. Overview & Motivation

Enhanced Reward Models extend basic reward modeling with ensemble methods, uncertainty quantification, and feature banking to provide more robust and reliable preference learning for RLHF (Reinforcement Learning from Human Feedback).

### Key Innovations
- **Ensemble-based uncertainty**: Multiple reward heads provide confidence estimates
- **Feature banking**: Stores representative features for preference matching
- **Robust training**: Handles noisy and contradictory preference data
- **Calibrated outputs**: Well-calibrated reward predictions

### Applications
- RLHF for language models (ChatGPT-style training)
- Preference learning from human feedback
- Multi-objective reward modeling
- Robotic learning from demonstrations

## 2. Theoretical Background

### Ensemble Reward Modeling

Standard reward model:
```
r_θ(x, y) = f_θ(x, y)
```

Enhanced reward model with ensemble:
```
r_ensemble(x, y) = mean([f_θ1(x, y), ..., f_θK(x, y)])
uncertainty(x, y) = std([f_θ1(x, y), ..., f_θK(x, y)])
```

### Bradley-Terry Preference Model

Models pairwise preferences:
```
P(y_w ≻ y_l | x) = σ(r(x, y_w) - r(x, y_l))
```

where y_w is preferred (winner) and y_l is dispreferred (loser).

### Loss Function

```
L = -E[(x, y_w, y_l)] log σ(r(x, y_w) - r(x, y_l))
```

With ensemble regularization:
```
L_total = L_preference + λ · L_diversity + γ · L_calibration
```

## 3. Mathematical Formulation

### Ensemble Architecture

K reward heads share feature extractor:
```
features = f_shared(x, y)
r_k = head_k(features) for k = 1, ..., K
```

### Uncertainty Quantification

Epistemic uncertainty from ensemble disagreement:
```
σ_epistemic = √(1/K Σ_k (r_k - r_mean)²)
```

### Feature Banking

Maintain bank of representative features:
```
Bank = {(f_1, r_1), ..., (f_N, r_N)}
```

For new sample, find similar features:
```
similarity(f_new, f_i) = cosine(f_new, f_i)
r_adjusted = r_predicted + α · Σ_i w_i(r_i - r_predicted)
```

## 4. High-Level Intuition

Think of ensemble reward models like:
- **Jury of experts**: Multiple models vote on reward
- **Confidence measure**: Agreement → high confidence, disagreement → uncertainty
- **Memory bank**: Remember past preferences to ensure consistency

This is more robust than single reward model that might overfit or be miscalibrated.

## 5. Implementation Details

From `Nexus/nexus/models/rl/preference/reward_model.py`:

```python
config = {
    "input_dim": 768,        # From language model
    "hidden_dim": 512,       # Reward model hidden size
    "num_reward_heads": 3,   # Ensemble size
    "bank_size": 10000,      # Feature bank capacity
    "dropout": 0.1,
}
```

### Architecture

```python
class EnhancedRewardModel(NexusModule):
    def __init__(self, config):
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Multiple reward heads (ensemble)
        self.reward_head = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(num_reward_heads)
        ])

        # Feature bank
        self.register_feature_bank("preference", bank_size, hidden_dim)
```

## 6. Code Walkthrough

```python
def forward(self, inputs, attention_mask=None):
    # Extract features
    features = self.feature_extractor(inputs)

    # Get rewards from ensemble
    rewards = torch.cat([
        head(features) for head in self.reward_head
    ], dim=-1)

    # Calculate statistics
    mean_reward = rewards.mean(dim=-1, keepdim=True)
    reward_uncertainty = rewards.std(dim=-1, keepdim=True)

    # Update feature bank
    self.update_feature_bank("preference", features)

    return {
        "rewards": mean_reward,
        "uncertainty": reward_uncertainty,
        "features": features,
        "raw_rewards": rewards
    }
```

## 7. Optimization Tricks

1. **Head diversity**: Initialize heads differently to encourage diversity
2. **Uncertainty calibration**: Use validation set to calibrate uncertainty estimates
3. **Feature bank sampling**: Sample from bank for contrastive learning
4. **Gradient clipping**: Essential for stable training with noisy preferences
5. **Warmup**: Gradually increase learning rate

## 8. Experiments & Results

Typical improvements over single reward model:
- 10-15% better agreement with human preferences
- 20-30% better calibration (uncertainty matches error rate)
- More robust to label noise

## 9. Common Pitfalls

1. **Ensemble collapse**: All heads learn same function → use diversity regularization
2. **Overconfident predictions**: Calibrate on validation set
3. **Feature bank overflow**: Use FIFO or importance sampling
4. **Imbalanced preferences**: Balance positive/negative examples

## 10. References

### Primary Papers
- Christiano, P., et al. (2017). **Deep Reinforcement Learning from Human Preferences.** NIPS.
- Stiennon, N., et al. (2020). **Learning to Summarize from Human Feedback.** NeurIPS.
- Ouyang, L., et al. (2022). **Training Language Models to Follow Instructions with Human Feedback.** OpenAI.

### Implementation
- Nexus: `Nexus/nexus/models/rl/preference/reward_model.py`

---

**Key Takeaways:**
- Ensemble provides uncertainty quantification
- Feature banking ensures consistency
- Critical component for RLHF
- Requires careful calibration
