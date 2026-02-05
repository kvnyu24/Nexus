# RND: Random Network Distillation

## 1. Overview

RND (Random Network Distillation) is an exploration method that uses prediction error on a fixed random network as an intrinsic reward signal. Unlike ICM, RND doesn't require an inverse model and naturally avoids the "noisy TV" problem because the target network is deterministic and state-only.

**Paper**: "Exploration by Random Network Distillation" (Burda et al., ICLR 2019)

**Key Innovation**: Using a fixed random network as the prediction target provides a consistent exploration signal that decreases as states become familiar, without needing to learn environment dynamics.

**Use Cases**:
- Hard exploration problems (Montezuma's Revenge, Pitfall)
- Deterministic environments
- When simplicity and computational efficiency are priorities
- Avoiding dynamic prediction (no noisy TV problem)

## 2. Theory

### 2.1 Core Idea

RND maintains two networks:
1. **Fixed target network** f: S → R^k (randomly initialized, never trained)
2. **Predictor network** f̂: S → R^k (trained to match target)

Intrinsic reward is prediction error:
```
r_intrinsic(s) = ||f̂(s) - f(s)||^2
```

### 2.2 Why This Works

**Novel states**: Predictor hasn't seen them → high error → high reward
**Familiar states**: Predictor trained on them → low error → low reward

The fixed network provides consistent targets - unlike environment dynamics, which may be stochastic.

### 2.3 Avoiding Noisy TV

Unlike ICM:
- **No action dependence**: Target is f(s), not f(s,a)
- **Deterministic target**: Random network is fixed
- **No dynamics learning**: Doesn't try to predict next state

This naturally ignores unpredictable distractors (they're just more input dimensions to the fixed function).

### 2.4 Running Statistics

RND normalizes observations and intrinsic rewards using running statistics:
```
s_normalized = (s - μ_s) / (σ_s + ε)
r_normalized = r_intrinsic / (σ_r + ε)
```

This handles varying scales across environments.

## 3. Mathematical Formulation

### Networks

**Target Network** (fixed):
```
f_target(s; θ_target): S → R^k
θ_target ~ random initialization (never updated)
```

**Predictor Network** (trainable):
```
f̂_pred(s; θ_pred): S → R^k
```

### Loss Function

```
L_RND = E_s[||f̂_pred(s) - f_target(s)||^2]
```

Only θ_pred is updated (θ_target frozen).

### Complete Algorithm

```
1. Initialize:
   - f_target with random weights (freeze)
   - f̂_pred with random weights
   - Running mean/std for observations and rewards

2. For each environment step:
   a) Collect transition (s, a, r_ext, s')

   b) Compute intrinsic reward:
      r_int = ||f̂_pred(normalize(s')) - f_target(normalize(s'))||^2
      r_int_normalized = r_int / (σ_r + ε)

   c) Store (s, a, r_ext + β·r_int_normalized, s')

3. Update predictor:
   - Sample batch from buffer
   - Minimize L_RND on batch
   - Update running statistics

4. Update policy:
   - Use standard RL algorithm (PPO, etc.)
   - On data with augmented rewards
```

## 4. Implementation Details

### Network Architecture

```python
class FixedRandomNetwork(nn.Module):
    """Target network (never trained)"""
    def __init__(self, state_dim, feature_dim=256, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, state):
        return self.network(state)

class PredictorNetwork(nn.Module):
    """Trainable predictor"""
    def __init__(self, state_dim, feature_dim=256, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, state):
        return self.network(state)
```

**Key Difference**: Predictor has one more hidden layer (more capacity to learn the fixed mapping).

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| feature_dim | 256 | Output dimensionality |
| hidden_dim | 256 | Hidden layer size |
| learning_rate | 1e-3 | Predictor learning rate |
| intrinsic_scale | 1.0 | Intrinsic reward weight |
| extrinsic_scale | 1.0 | Extrinsic reward weight |

### Running Statistics

```python
# Observation normalization
obs_mean = running_mean(observations)
obs_std = running_std(observations)
obs_normalized = (obs - obs_mean) / (obs_std + 1e-8)

# Reward normalization
reward_std = running_std(intrinsic_rewards)
reward_normalized = intrinsic_reward / (reward_std + 1e-8)
```

## 5. Code Walkthrough (from `/nexus/models/rl/rnd.py`)

### Intrinsic Reward Computation

```python
def compute_intrinsic_reward(self, state, normalize=True):
    with torch.no_grad():
        # Normalize observation
        normalized_state = self._normalize_obs(state)

        # Compute features
        target_features = self.target(normalized_state)
        predicted_features = self.predictor(normalized_state)

        # MSE as intrinsic reward
        intrinsic_reward = (target_features - predicted_features).pow(2).mean(dim=-1)

        if normalize:
            self._update_reward_stats(intrinsic_reward)
            intrinsic_reward = self._normalize_reward(intrinsic_reward)

        return intrinsic_reward
```

### Predictor Update

```python
def update(self, batch):
    states = batch["states"]

    # Update observation statistics
    self._update_obs_stats(states)
    normalized_states = self._normalize_obs(states)

    # Compute features
    target_features = self.target(normalized_states).detach()  # No gradient
    predicted_features = self.predictor(normalized_states)

    # MSE loss
    loss = F.mse_loss(predicted_features, target_features)

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    return {
        "rnd_loss": loss.item(),
        "mean_intrinsic_reward": intrinsic_reward.mean().item(),
    }
```

### RND Wrapper

```python
class RNDWrapper(NexusModule):
    def __init__(self, agent, rnd_config):
        self.agent = agent
        self.rnd = RNDModule(rnd_config)
        self.intrinsic_scale = rnd_config.get("intrinsic_reward_scale", 1.0)
        self.extrinsic_scale = rnd_config.get("extrinsic_reward_scale", 1.0)

    def compute_combined_reward(self, state, extrinsic_reward):
        intrinsic_reward = self.rnd.compute_intrinsic_reward(state)
        combined = (
            self.extrinsic_scale * extrinsic_reward
            + self.intrinsic_scale * intrinsic_reward
        )
        return combined

    def update(self, batch):
        # Update RND predictor
        rnd_metrics = self.rnd.update(batch)

        # Compute combined rewards
        combined_rewards = self.compute_combined_reward(
            batch["states"], batch["rewards"]
        )

        # Update agent with augmented rewards
        batch_augmented = batch.copy()
        batch_augmented["rewards"] = combined_rewards
        agent_metrics = self.agent.update(batch_augmented)

        return {**rnd_metrics, **agent_metrics}
```

## 6. Optimization Tricks

### 6.1 Separate Value Functions

Use two value functions:
- V_extrinsic: For environment rewards only
- V_intrinsic: For intrinsic rewards only

```python
V_total = γ_ext · V_extrinsic + γ_int · V_intrinsic
```

This allows different discount factors (γ_int = 0.99, γ_ext = 0.999).

### 6.2 Observation Whitening

Whiten observations before feeding to networks:
```python
obs_whitened = (obs - mean) @ whitening_matrix
```

### 6.3 Non-Episodic Returns

Don't reset intrinsic returns at episode boundaries:
```python
# Extrinsic: reset at done
R_ext = 0 if done else γ * V_ext(s')

# Intrinsic: never reset
R_int = γ * V_int(s')  # Always continue
```

This encourages exploration across episode boundaries.

### 6.4 Predictor Capacity

Make predictor slightly larger than target:
```python
target_layers = [256, 256]
predictor_layers = [256, 256, 256]  # Extra layer
```

## 7. Experimental Results

### Montezuma's Revenge

| Algorithm | Mean Score | Max Score |
|-----------|------------|-----------|
| RND + PPO | 8,152 | 10,070 |
| ICM | 3,340 | 4,800 |
| A3C | 0 | 142 |
| Bootstrapped DQN | 950 | - |

RND achieves **superhuman performance** (human average: 4,753).

### Pitfall

- **RND**: Solves 2 (out of 255) rooms
- **ICM**: Solves 0 rooms
- **Random**: Solves 0 rooms

### Atari Suite (54 Games)

- **Median human-normalized score**: 0.98 (near-human across all games)
- **Games with progress**: 48 / 54
- **Games solved**: 12 / 54

## 8. Common Pitfalls

### 8.1 Observation Normalization Critical

**Problem**: Without normalization, different state dimensions dominate

**Solution**: Always normalize observations using running mean/std

### 8.2 Reward Scale Mismatch

**Problem**: Intrinsic rewards much larger/smaller than extrinsic

**Solutions**:
- Normalize intrinsic rewards
- Tune intrinsic_scale hyperparameter
- Use separate value functions

### 8.3 Predictor Overfitting

**Problem**: Predictor perfectly matches target everywhere, no exploration

**Solutions**:
- Ensure continuous new data (keep exploring)
- Check that prediction error decreases for visited states
- Monitor intrinsic reward distribution

### 8.4 Non-Stationarity

**Problem**: Running statistics change during training

**Solutions**:
- Use large batch sizes for stable statistics
- Warmup period before using intrinsic rewards
- Exponential moving average for statistics

## 9. Extensions

### 9.1 Next-State RND

Predict next state instead of current:
```python
intrinsic_reward = ||f̂(s') - f(s')||^2
```

This encourages exploring consequences of actions.

### 9.2 Ensemble RND

Use ensemble of fixed networks:
```python
targets = [f_1(s), ..., f_K(s)]
predictors = [f̂_1(s), ..., f̂_K(s)]
intrinsic_reward = mean(||f̂_i(s) - f_i(s)||^2)
```

### 9.3 Episodic RND

Track visited states per episode:
```python
if state_similar_to(episode_memory):
    intrinsic_reward *= decay_factor
```

### 9.4 Combined RND + ICM

Use both methods:
```python
rnd_reward = ||f̂(s) - f(s)||^2  # Novelty
icm_reward = ||pred(φ(s), a) - φ(s')||^2  # Prediction error
intrinsic_reward = α · rnd_reward + (1-α) · icm_reward
```

## 10. References

### Original Papers

1. **RND**: Burda et al., "Exploration by Random Network Distillation", ICLR 2019 [arXiv:1810.12894](https://arxiv.org/abs/1810.12894)

2. **Large-Scale Study**: Burda et al., "Large-Scale Study of Curiosity-Driven Learning", ICLR 2019

### Related Work

3. **ICM**: Pathak et al., "Curiosity-driven Exploration by Self-Supervised Prediction", ICML 2017

4. **NGU**: Badia et al., "Never Give Up: Learning Directed Exploration Strategies", ICLR 2020

5. **Pseudocounts**: Bellemare et al., "Count-Based Exploration with Neural Density Models", ICML 2016

### Implementation References

6. [OpenAI Baselines](https://github.com/openai/random-network-distillation): Original implementation

7. [Stable-Baselines3 Contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib): Clean PyTorch implementation

### Analysis

8. **RND Analysis**: Raileanu & Rocktäschel, "RIDE: Rewarding Impact-Driven Exploration for Procedurally-Generated Environments", ICLR 2020

9. **Exploration Survey**: Aubret et al., "A Survey on Intrinsic Motivation in Reinforcement Learning", 2022
