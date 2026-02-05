# TD-MPC2: Scalable, Robust World Models for Continuous Control

## 1. Overview

TD-MPC2 is a model-based reinforcement learning algorithm that combines learned latent dynamics models with Model Predictive Control (MPC). It achieves state-of-the-art results on continuous control benchmarks through scalable world models, robust planning, and efficient online learning.

**Paper**: "TD-MPC2: Scalable, Robust World Models for Continuous Control" (Hansen et al., ICLR 2024)

**Status**: ⚠️ **NOT YET IMPLEMENTED** - Documentation prepared for future implementation

**Key Innovations**:
- Scalable architecture (1M+ parameters)
- Single-step temporal difference learning for world model
- Robust MPC with learned Q-function as objective
- Multi-task pretraining

**Use Cases**:
- Continuous control robotics
- Real-world robot manipulation
- Sim-to-real transfer
- Multi-task learning

## 2. Theory and Background

### 2.1 Latent Dynamics Model

TD-MPC2 learns dynamics in latent space:
```
z_t = h_θ(o_t)  # Encoder
z_{t+1} = f_θ(z_t, a_t)  # Dynamics
r_t = g_θ(z_t, a_t)  # Reward
```

Unlike DreamerV3's stochastic RSSM, TD-MPC2 uses deterministic dynamics for simplicity and speed.

### 2.2 Model Predictive Control

At each timestep:
1. Encode current observation: z = h(o)
2. Plan action sequence [a_t, ..., a_{t+H}] by optimizing:
   ```
   max_{a_t:t+H} Σ γ^i Q(f^i(z, a_t:t+i), a_{t+i})
   ```
3. Execute first action a_t
4. Replan at next step

Uses Cross-Entropy Method (CEM) or gradient-based optimization for planning.

### 2.3 TD Learning for World Models

TD-MPC2 trains the world model using temporal difference learning:
```
L_model = E[(Q(z_t, a_t) - (r_t + γ Q(z_{t+1}, a_{t+1})))^2]
```

This directly optimizes for planning performance rather than pure prediction accuracy.

## 3. Mathematical Formulation

### Model Architecture

```
Encoder: o_t → MLP → z_t  # [obs_dim] → [latent_dim]
Dynamics: [z_t, a_t] → MLP → z_{t+1}  # [latent_dim + action_dim] → [latent_dim]
Reward: [z_t, a_t] → MLP → r_t  # [latent_dim + action_dim] → [1]
Q-function: [z_t, a_t] → MLP → Q_t  # [latent_dim + action_dim] → [1]
Policy: z_t → MLP → π(a|z)  # [latent_dim] → [action_dim]
```

### Loss Function

```
L_total = L_consistency + L_reward + L_value + L_policy

L_consistency = ||z_{t+1} - f(h(o_t), a_t)||^2
L_reward = ||r_t - g(h(o_t), a_t)||^2
L_value = ||Q(z_t, a_t) - (r_t + γ Q(z_{t+1}, π(z_{t+1})))||^2
L_policy = -Q(z_t, π(z_t))
```

### Planning Objective

CEM optimization:
```
1. Sample N action sequences from Gaussian
2. Evaluate: score = Σ γ^i Q(z_i, a_i) for each sequence
3. Keep top K sequences
4. Refit Gaussian to elite set
5. Repeat for M iterations
6. Return mean of final Gaussian
```

## 4. Implementation Sketch

### Network Architecture (Proposed)

```python
class TDMPC2WorldModel(nn.Module):
    def __init__(self, obs_dim, action_dim, latent_dim=512, hidden_dim=512):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )

        # Dynamics
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )

        # Reward
        self.reward = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Q-function
        self.q_function = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Policy (for implicit planning)
        self.policy = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

    def encode(self, obs):
        return self.encoder(obs)

    def predict(self, z, a):
        za = torch.cat([z, a], dim=-1)
        next_z = self.dynamics(za)
        reward = self.reward(za)
        q_value = self.q_function(za)
        return next_z, reward, q_value

    def imagine(self, z, actions):
        """Rollout dynamics for planning"""
        states, rewards, values = [z], [], []
        for a in actions:
            z, r, q = self.predict(z, a)
            states.append(z)
            rewards.append(r)
            values.append(q)
        return states, rewards, values
```

### CEM Planning

```python
def plan_cem(world_model, z, horizon=10, n_samples=512, n_elites=64, n_iterations=6):
    """Cross-Entropy Method planning"""
    action_dim = world_model.action_dim
    device = z.device

    # Initialize Gaussian
    mean = torch.zeros(horizon, action_dim, device=device)
    std = torch.ones(horizon, action_dim, device=device)

    for _ in range(n_iterations):
        # Sample action sequences
        noise = torch.randn(n_samples, horizon, action_dim, device=device)
        actions = mean + std * noise  # [n_samples, horizon, action_dim]
        actions = torch.clamp(actions, -1, 1)

        # Evaluate with world model
        scores = []
        for i in range(n_samples):
            states, rewards, _ = world_model.imagine(z, actions[i])
            # Return = discounted sum of Q-values
            gamma = 0.99
            score = sum(gamma**t * world_model.q_function(torch.cat([s, a], -1))
                       for t, (s, a) in enumerate(zip(states[:-1], actions[i])))
            scores.append(score)

        scores = torch.stack(scores)

        # Select elites
        elite_indices = scores.topk(n_elites).indices
        elite_actions = actions[elite_indices]

        # Refit Gaussian
        mean = elite_actions.mean(dim=0)
        std = elite_actions.std(dim=0).clamp(min=0.1)

    # Return first action
    return mean[0]
```

## 5. Expected Performance

Based on paper results:

### DMControl Benchmark

| Task | TD-MPC2 | DreamerV3 | SAC |
|------|---------|-----------|-----|
| Humanoid Walk | 920 | 850 | 650 |
| Quadruped Run | 850 | 820 | 600 |
| Dog Walk | 750 | 700 | 500 |
| Average | 832 | 800 | 630 |

### Key Advantages

- Faster planning than DreamerV3 (deterministic dynamics)
- More robust to observation noise
- Better sim-to-real transfer
- Scales to 1M+ parameters

## 6. Implementation Roadmap

### Phase 1: Core Components
- [ ] Latent encoder (MLP)
- [ ] Deterministic dynamics model
- [ ] Reward predictor
- [ ] Q-function network
- [ ] Policy network

### Phase 2: Planning
- [ ] CEM optimizer
- [ ] MPPI optimizer (alternative)
- [ ] Gradient-based planning
- [ ] Action repeat handling

### Phase 3: Training
- [ ] TD loss for world model
- [ ] Policy learning (implicit planning)
- [ ] Data augmentation
- [ ] Multi-step consistency

### Phase 4: Scaling
- [ ] Large-scale architecture (1M params)
- [ ] Multi-task pretraining
- [ ] Distributed training
- [ ] Real-world deployment tools

## 7. Anticipated Challenges

1. **Planning Speed**: CEM requires many model queries (can parallelize)
2. **Latent Collapse**: Encoder may learn degenerate representations
3. **TD Bias**: Q-function errors propagate through planning
4. **Action Smoothness**: Need action repeat or temporal coherence

## 8. Comparison with Alternatives

| Feature | TD-MPC2 | DreamerV3 | MBPO |
|---------|---------|-----------|------|
| Latent Dynamics | Deterministic | Stochastic (RSSM) | State-space |
| Planning | MPC (CEM) | Actor-Critic | Model-free (SAC) |
| Learning | TD | Maximum Likelihood + RL | Dyna-style |
| Speed | Fast | Moderate | Fast |
| Sample Efficiency | High | Very High | High |

## 9. Related Work

1. **TD-MPC**: Hansen et al., "Temporal Difference Learning for Model Predictive Control", ICML 2022
2. **DreamerV3**: Hafner et al., "Mastering Diverse Domains through World Models", 2023
3. **MPPI**: Williams et al., "Model Predictive Path Integral Control", 2017

## 10. References

1. **TD-MPC2**: Hansen et al., "TD-MPC2: Scalable, Robust World Models for Continuous Control", ICLR 2024 [arXiv:2310.16828](https://arxiv.org/abs/2310.16828)

2. **TD-MPC**: Hansen et al., "Temporal Difference Learning for Model Predictive Control", ICML 2022

3. **Cross-Entropy Method**: Rubinstein & Kroese, "The Cross-Entropy Method: A Unified Approach to Combinatorial Optimization", 2004

**Implementation Status**: This algorithm is documented but not yet implemented. Contributions welcome!
