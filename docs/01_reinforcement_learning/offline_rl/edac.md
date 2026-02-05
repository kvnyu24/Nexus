# EDAC: Ensemble Diversified Actor-Critic

**Paper**: [Uncertainty-Based Offline Reinforcement Learning with Diversified Q-Ensemble](https://arxiv.org/abs/2110.01548) (An et al., NeurIPS 2021)

**Status**: Reference documentation (implementation pending)

## Overview

EDAC uses ensemble diversity as a conservatism mechanism:
- High Q-value + High disagreement → Uncertain, penalize
- High Q-value + Low disagreement → Confident, allow

### Key Innovation

**CQL**: Explicitly penalizes all OOD actions
**EDAC**: Uses ensemble disagreement to identify OOD actions, then penalizes those

## Mathematical Formulation

### Ensemble Diversity Penalty

```
std(s,a) = std([Q₁(s,a), Q₂(s,a), ..., Q_N(s,a)])

L_diversity = E_{(s,a)~π}[ max(0, std(s,a) - threshold) ]

Total loss: L_CQL + η * L_diversity
```

**Intuition**: If critics disagree (high std), the action is OOD → penalize.

### Diversified Training

Train each critic on different data subsets:
```python
for i, critic in enumerate(critics):
    subset = random.sample(dataset, k=batch_size)
    loss_i = td_error(critic, subset)
```

**Why**: Increases diversity, improves uncertainty estimation.

## Algorithm

```python
class EDACAgent:
    def __init__(self, config):
        self.num_critics = config.get("num_critics", 10)
        self.critics = [QNetwork(...) for _ in range(self.num_critics)]
        self.eta = config.get("eta", 1.0)  # Diversity weight

    def compute_diversity_penalty(self, states, actions):
        """Penalize high variance across ensemble."""
        q_values = [critic(states, actions) for critic in self.critics]
        q_std = torch.std(torch.stack(q_values), dim=0)
        penalty = torch.relu(q_std - threshold).mean()
        return penalty

    def update(self, batch):
        # Standard CQL loss
        cql_loss = self.compute_cql_loss(batch)

        # Diversity penalty
        policy_actions = self.actor(batch["states"])
        diversity_loss = self.compute_diversity_penalty(
            batch["states"], policy_actions
        )

        # Total critic loss
        critic_loss = cql_loss + self.eta * diversity_loss

        # Update
        optimize(self.critics, critic_loss)
        optimize(self.actor, actor_loss)
```

## Key Hyperparameters

```python
config = {
    "num_critics": 10,           # Ensemble size
    "eta": 1.0,                  # Diversity penalty weight
    "diversity_threshold": 0.1,  # Disagreement threshold
    "cql_alpha": 5.0,            # Base CQL weight
}
```

## Advantages vs. CQL

1. **Adaptive**: Only penalizes truly uncertain actions
2. **Better uncertainty**: Ensemble captures epistemic uncertainty
3. **Stochastic environments**: Handles aleatoric uncertainty better

## Disadvantages

1. **Computational cost**: 10x forward passes
2. **More complex**: Ensemble + diversity penalty
3. **Hyperparameter tuning**: η, threshold, ensemble size

## When to Use EDAC

**Best for**:
- Stochastic environments (robot with noisy dynamics)
- Need uncertainty estimates
- Have computational budget

**Avoid when**:
- Deterministic tasks (IQL/TD3+BC sufficient)
- Limited compute
- Need simplicity

## References

```bibtex
@inproceedings{an2021edac,
  title={Uncertainty-Based Offline Reinforcement Learning with Diversified Q-Ensemble},
  author={An, Gaon and Moon, Seungyong and Kim, Jang-Hyun and Song, Hyun Oh},
  booktitle={NeurIPS},
  year={2021}
}
```
