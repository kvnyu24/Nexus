# ICM: Intrinsic Curiosity Module

## 1. Overview

ICM (Intrinsic Curiosity Module) provides exploration bonuses based on prediction error of a learned forward dynamics model. By rewarding the agent for encountering states where predictions fail, ICM encourages curiosity-driven exploration of novel states.

**Paper**: "Curiosity-driven Exploration by Self-Supervised Prediction" (Pathak et al., ICML 2017)

**Key Innovation**: Using prediction error in a learned feature space (not raw pixels) to avoid the "noisy TV" problem where unpredictable but irrelevant distractors dominate intrinsic rewards.

**Use Cases**:
- Sparse-reward environments
- Exploration in games (VizDoom, Super Mario)
- Robotic manipulation with sparse feedback
- Any task where discovering new states is valuable

## 2. Theory

### 2.1 Intrinsic Reward

ICM adds an intrinsic reward to the environment reward:
```
r_total = r_extrinsic + η · r_intrinsic
```

Where intrinsic reward is forward model prediction error:
```
r_intrinsic = ||φ(s_{t+1}) - φ̂(s_t, a_t)||^2
```

- φ: Feature encoder
- φ̂: Forward model (predicts next features)
- η: Scaling factor

### 2.2 Feature Learning

To avoid rewarding unpredictable but irrelevant states (noisy TV problem), ICM learns features φ using an inverse dynamics model:
```
â_t = g(φ(s_t), φ(s_{t+1}))
```

The inverse model predicts which action was taken given two consecutive states. This forces φ to capture only action-relevant features, ignoring distractors.

### 2.3 Complete Objective

```
L_ICM = (1-β) · L_inverse + β · L_forward

L_inverse = -log P(a_t | φ(s_t), φ(s_{t+1}))  # Cross-entropy
L_forward = ||φ(s_{t+1}) - f(φ(s_t), a_t)||^2  # MSE
```

β controls the balance (typically 0.2).

## 3. Mathematical Formulation

### Networks

**Feature Encoder**: φ(s) → R^d
```
For images: CNN → feature vector
For vectors: MLP → feature vector
```

**Forward Model**: f(φ(s), a) → R^d
```
Predicts: φ̂(s') = f(φ(s), a)
```

**Inverse Model**: g(φ(s), φ(s')) → a
```
Predicts: â = g(φ(s), φ(s'))
```

### Training

1. Sample transition (s, a, r, s')
2. Encode features: φ_t = φ(s), φ_{t+1} = φ(s')
3. Forward model loss: L_f = ||φ_{t+1} - f(φ_t, a)||^2
4. Inverse model loss: L_i = CE(a, g(φ_t, φ_{t+1}))
5. Total ICM loss: L = (1-β)L_i + β L_f
6. Intrinsic reward: r_int = η · L_f

## 4. Implementation Details

### Architecture

```python
class FeatureEncoder(nn.Module):
    """Encode states to feature space"""
    def __init__(self, state_dim, feature_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )

class ForwardModel(nn.Module):
    """Predict next features from current features and action"""
    def __init__(self, feature_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim)
        )

    def forward(self, features, action):
        # One-hot encode action if discrete
        x = torch.cat([features, action], dim=-1)
        return self.model(x)

class InverseModel(nn.Module):
    """Predict action from state transition"""
    def __init__(self, feature_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, features_t, features_t1):
        x = torch.cat([features_t, features_t1], dim=-1)
        return self.model(x)  # Logits for discrete, continuous for continuous
```

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| feature_dim | 256 | Feature space dimensionality |
| beta | 0.2 | Forward vs inverse loss weight |
| eta | 0.01 | Intrinsic reward scaling |
| learning_rate | 1e-3 | ICM learning rate |

## 5. Code Walkthrough (from `/nexus/models/rl/icm.py`)

### Intrinsic Reward Computation

```python
def compute_intrinsic_reward(self, state, action, next_state):
    with torch.no_grad():
        # Encode states
        state_features = self.encoder(state)
        next_state_features = self.encoder(next_state)

        # Predict next state features
        predicted_features = self.forward_model(state_features, action)

        # Intrinsic reward = prediction error
        intrinsic_reward = 0.5 * (predicted_features - next_state_features).pow(2).sum(dim=-1)

        return self.eta * intrinsic_reward
```

### ICM Update

```python
def update(self, batch):
    states = batch["states"]
    actions = batch["actions"]
    next_states = batch["next_states"]

    # Encode states
    state_features = self.encoder(states)
    next_state_features = self.encoder(next_states)

    # Forward model loss
    predicted_features = self.forward_model(state_features, actions)
    forward_loss = 0.5 * (predicted_features - next_state_features.detach()).pow(2).mean()

    # Inverse model loss
    predicted_actions = self.inverse_model(state_features, next_state_features)
    if self.discrete_actions:
        inverse_loss = F.cross_entropy(predicted_actions, actions.long())
    else:
        inverse_loss = F.mse_loss(predicted_actions, actions)

    # Total loss
    total_loss = (1 - self.beta) * inverse_loss + self.beta * forward_loss

    # Optimize
    self.optimizer.zero_grad()
    total_loss.backward()
    self.optimizer.step()

    return {
        "icm_loss": total_loss.item(),
        "inverse_loss": inverse_loss.item(),
        "forward_loss": forward_loss.item(),
    }
```

### Wrapper for RL Agent

```python
class ICMWrapper(NexusModule):
    """Wraps an RL agent with ICM for exploration"""
    def __init__(self, agent, icm_config):
        self.agent = agent
        self.icm = ICM(icm_config)

    def update(self, batch):
        # Update ICM
        icm_metrics = self.icm.update(batch)

        # Compute combined rewards
        intrinsic_rewards = self.icm.compute_intrinsic_reward(
            batch["states"], batch["actions"], batch["next_states"]
        )
        combined_rewards = batch["rewards"] + intrinsic_rewards

        # Update agent with augmented rewards
        batch_augmented = batch.copy()
        batch_augmented["rewards"] = combined_rewards
        agent_metrics = self.agent.update(batch_augmented)

        return {**icm_metrics, **agent_metrics}
```

## 6. Optimization Tricks

1. **Feature Normalization**: Normalize features to prevent scale issues
   ```python
   features = F.normalize(self.encoder(state), dim=-1)
   ```

2. **Gradient Clipping**: Prevent exploding gradients
   ```python
   nn.utils.clip_grad_norm_(self.icm.parameters(), max_norm=5.0)
   ```

3. **Reward Scaling**: Normalize intrinsic rewards
   ```python
   intrinsic_reward = (intrinsic_reward - mean) / (std + 1e-8)
   ```

4. **Detach Target Features**: Don't backprop through target in forward loss
   ```python
   forward_loss = ||predict(φ(s), a) - φ(s').detach()||^2
   ```

## 7. Experimental Results

### VizDoom Benchmark

| Environment | ICM | No Exploration | Improvement |
|-------------|-----|----------------|-------------|
| Sparse (baseline) | 450 | 50 | 9x |
| Dense (baseline) | 1200 | 1100 | 1.09x |
| Very Sparse | 230 | 0 | ∞ |

### Super Mario Bros

- **With ICM**: Completes 30% more levels
- **Exploration**: Discovers 40% more unique states
- **Sample Efficiency**: 2-3x faster learning

## 8. Common Pitfalls

### 8.1 Noisy TV Problem

**Problem**: Agent gets stuck watching unpredictable but irrelevant distractors

**Solutions**:
- Use inverse model to learn action-relevant features
- Increase beta (prioritize inverse model)
- Add auxiliary tasks (e.g., reconstruction)

### 8.2 Intrinsic Reward Explosion

**Problem**: Intrinsic rewards dominate, agent ignores extrinsic rewards

**Solutions**:
- Normalize intrinsic rewards
- Anneal eta over training
- Use separate value functions for intrinsic/extrinsic

### 8.3 Feature Collapse

**Problem**: Encoder learns degenerate features (all zeros)

**Solutions**:
- Stronger inverse model loss (lower beta)
- Add feature diversity regularization
- Use contrastive learning for encoder

## 9. Extensions

### 9.1 Disagreement-Based ICM

Use ensemble of forward models:
```python
predictions = [model_i(φ(s), a) for model_i in ensemble]
disagreement = std(predictions)
intrinsic_reward = disagreement.mean()
```

### 9.2 Episodic Curiosity

Track visited states per episode:
```python
if state in episode_memory:
    intrinsic_reward *= 0.1  # Reduce bonus for revisited states
episode_memory.add(state)
```

### 9.3 RND-ICM Hybrid

Combine ICM with RND:
```python
icm_reward = ||predict(φ(s), a) - φ(s')||^2
rnd_reward = ||predictor(s') - target(s')||^2
intrinsic_reward = icm_reward + rnd_reward
```

## 10. References

1. **ICM**: Pathak et al., "Curiosity-driven Exploration by Self-Supervised Prediction", ICML 2017 [arXiv:1705.05363](https://arxiv.org/abs/1705.05363)

2. **Prediction Error**: Schmidhuber, "Formal Theory of Creativity, Fun, and Intrinsic Motivation", 2010

3. **Noisy TV Problem**: Burda et al., "Large-Scale Study of Curiosity-Driven Learning", ICLR 2019

4. **Visual Feature Learning**: Pathak et al., "Self-Supervised Visual Planning with Temporal Skip Connections", CoRL 2017

### Related Work

5. **RND**: Burda et al., "Exploration by Random Network Distillation", ICLR 2019
6. **NGU**: Badia et al., "Never Give Up: Learning Directed Exploration Strategies", ICLR 2020
7. **RIDE**: Raileanu & Rocktäschel, "RIDE: Rewarding Impact-Driven Exploration", ICLR 2020
