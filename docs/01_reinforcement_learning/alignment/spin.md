# SPIN: Self-Play Fine-Tuning

**Paper**: Self-Play Fine-Tuning Converts Weak Language Models to Strong (Chen et al., 2024)

**Status**: Reference documentation (implementation pending)

## Overview

SPIN uses **self-play** to iteratively improve the model by training against its own previous generations.

### Key Innovation

No human preferences needed after initialization - the model plays against itself.

```
Iteration t:
1. Generate responses with π_t
2. Train π_{t+1} to prefer human responses over π_t responses
3. Repeat
```

## Algorithm

```
1. Start with SFT model π_0
2. For t = 0, 1, 2, ...:
   a. Generate responses: y_generated ~ π_t(·|x)
   b. Create synthetic preference pairs:
      - Chosen: human response y_human
      - Rejected: model response y_generated
   c. Train π_{t+1} using DPO on (y_human, y_generated) pairs
   d. If y_generated ≈ y_human, stop (Nash equilibrium)
```

## Mathematical Formulation

```
L_SPIN = E_{x, y_human, y_generated}[
    -log σ(β · (log π_{t+1}(y_human|x)/π_t(y_human|x)
              - log π_{t+1}(y_generated|x)/π_t(y_generated|x)))
]
```

Similar to DPO, but rejected responses come from previous policy iteration.

## Hyperparameters

```python
config = {
    "num_iterations": 3,      # Self-play iterations
    "beta": 0.1,              # DPO beta
    "generation_temp": 1.0,   # Sampling temperature
}
```

## Advantages

1. No additional human labels after SFT
2. Continuous improvement via self-play
3. Converges to Nash equilibrium (theoretically)

## Disadvantages

1. Can collapse if not careful
2. Requires strong initial SFT model
3. Expensive (generate new data each iteration)

## Implementation Sketch

```python
class SPINTrainer:
    def __init__(self, sft_model, human_data):
        self.policy = sft_model
        self.human_data = human_data

    def train(self, num_iterations=3):
        for t in range(num_iterations):
            # Generate synthetic rejected responses
            rejected_responses = self.generate_from_policy(
                self.human_data["prompts"]
            )

            # Create preference dataset
            preference_data = {
                "prompts": self.human_data["prompts"],
                "chosen": self.human_data["responses"],
                "rejected": rejected_responses
            }

            # Train with DPO
            self.policy = dpo_train(
                self.policy,
                preference_data,
                reference=copy.deepcopy(self.policy)
            )

            # Check convergence
            if responses_match(rejected_responses, self.human_data["responses"]):
                break
```

## When to Use SPIN

**Best for**:
- Limited preference data
- Want continual improvement
- Have strong SFT baseline

**Avoid when**:
- Weak initial model (will amplify errors)
- Limited compute (expensive iterations)

## References

```bibtex
@article{chen2024spin,
  title={Self-Play Fine-Tuning Converts Weak LMs to Strong},
  author={Chen, Zixiang and others},
  year={2024}
}
```
