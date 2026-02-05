# MBPO: Model-Based Policy Optimization

## 1. Overview

MBPO (Model-Based Policy Optimization) bridges model-based and model-free RL by learning an ensemble of dynamics models and using them to generate short synthetic rollouts that augment real data. This enables sample-efficient learning while mitigating compounding model errors through short branched rollouts.

**Paper**: "When to Trust Your Model: Model-Based Policy Optimization" (Janner et al., NeurIPS 2019)

**Key Innovation**: Theoretical guarantee that short model rollouts under bounded model error lead to monotonic policy improvement, justifying the approach and providing principled rollout length selection.

**Use Cases**:
- Sample-limited robotics tasks
- Continuous control with expensive simulations
- Transfer learning (model generalizes across tasks)
- Any domain where sample efficiency is critical

## 2. Theory and Background

### 2.1 Model Error Accumulation

Model errors compound over rollout length k:
```
Total error ≈ ε_model · k
```

Where ε_model is single-step model error. Long rollouts (k large) lead to unrealistic trajectories that hurt policy learning.

### 2.2 Monotonic Improvement Guarantee

MBPO proves that under bounded model error, using rollouts of length k ≤ k* guarantees:
```
J_π_new ≥ J_π_old
```

Where k* depends on model error, policy improvement, and discount factor:
```
k* ∝ 1 / (ε_model · ε_policy)
```

**Intuition**: If model error is small and policy changes are gradual, we can safely use model rollouts for training.

### 2.3 Branched Rollouts

MBPO performs **branched rollouts** starting from real states:
1. Collect real transition (s, a, r, s')
2. Starting from s', use model to generate k-step trajectory
3. Add synthetic transitions to replay buffer
4. Train model-free algorithm (SAC) on mixed real+synthetic data

This anchors rollouts to real states, reducing error accumulation.

## 3. Mathematical Formulation

### Ensemble Dynamics Model

MBPO uses an ensemble of N probabilistic models:
```
M = {M_1, ..., M_N}
M_i(s, a) → N(μ_i, Σ_i)  # Gaussian over (Δs, r)
```

Each model predicts:
- Next state delta: Δs = s' - s
- Reward: r

**Why ensemble?**
- Uncertainty estimation (disagreement = high uncertainty)
- Robust predictions (use elite subset)
- Diverse models improve coverage

### Elite Model Selection

After training on real data:
1. Evaluate each model on validation set
2. Select top-m models (elite set) with lowest validation loss
3. Use only elite models for rollouts

### Policy Training

Train SAC (or any off-policy algorithm) on batch:
```
B = B_real ∪ B_model

Where:
- B_real: Real environment transitions (small)
- B_model: Model-generated transitions (large)
- |B_real| / |B_total| = real_ratio (e.g., 0.05)
```

This massive augmentation improves sample efficiency.

## 4. Implementation Details

### Probabilistic Dynamics Model

```python
class ProbabilisticDynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        self.mean_head = nn.Linear(hidden_dim, state_dim + 1)  # Δs + r
        self.logvar_head = nn.Linear(hidden_dim, state_dim + 1)

        # Learnable bounds for logvar
        self.max_logvar = nn.Parameter(torch.ones(1, state_dim + 1) * 0.5)
        self.min_logvar = nn.Parameter(torch.ones(1, state_dim + 1) * -10.0)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        features = self.network(x)

        mean = self.mean_head(features)
        logvar = self.logvar_head(features)

        # Soft-clamp logvar
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        return mean, logvar

    def predict(self, state, action, deterministic=False):
        mean, logvar = self(state, action)

        if deterministic:
            prediction = mean
        else:
            std = (0.5 * logvar).exp()
            prediction = mean + std * torch.randn_like(std)

        # Split into next_state_delta and reward
        next_state_delta = prediction[:, :-1]
        reward = prediction[:, -1:]

        next_state = state + next_state_delta
        return next_state, reward
```

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| ensemble_size | 7 | Number of dynamics models |
| elite_size | 5 | Number of elite models |
| rollout_length | 1-5 | Model rollout length (starts at 1, increases) |
| rollout_batch_size | 256 | Batch size for rollouts |
| real_ratio | 0.05 | Fraction of real data in training |
| model_lr | 3e-4 | Dynamics model learning rate |
| hidden_dim | 256 | Model hidden layer size |

### Training Loop

```python
for step in range(total_steps):
    # 1. Collect real data
    action = agent.select_action(state)
    next_state, reward, done, _ = env.step(action)
    real_buffer.add(state, action, reward, next_state, done)

    # 2. Train dynamics models
    if step % model_train_freq == 0:
        batch = real_buffer.sample(model_batch_size)
        dynamics.update(batch)

    # 3. Generate synthetic rollouts
    if step % rollout_freq == 0:
        # Sample real states
        start_states = real_buffer.sample_states(rollout_batch_size)

        # Generate k-step rollouts
        for _ in range(rollout_length):
            action = agent.select_action(start_states)
            next_states, rewards = dynamics.predict(start_states, action)
            # Add to model buffer
            model_buffer.add(start_states, action, rewards, next_states, dones=0)
            start_states = next_states

    # 4. Train policy on mixed data
    if step % agent_train_freq == 0:
        # Mix real and synthetic data
        real_batch = real_buffer.sample(int(batch_size * real_ratio))
        model_batch = model_buffer.sample(int(batch_size * (1 - real_ratio)))
        combined_batch = combine(real_batch, model_batch)

        # Update SAC
        agent.update(combined_batch)

    # 5. Increase rollout length gradually
    if step % rollout_schedule_freq == 0:
        rollout_length = min(rollout_length + 1, max_rollout_length)
```

## 5. Code Walkthrough (from `/nexus/models/rl/mbpo.py`)

### Ensemble Training

```python
def update(self, batch):
    states = batch["states"]
    actions = batch["actions"]
    next_states = batch["next_states"]
    rewards = batch["rewards"]

    # Target: (Δs, r)
    targets = torch.cat([next_states - states, rewards.unsqueeze(-1)], dim=-1)

    total_loss = 0.0
    individual_losses = []

    for i, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):
        mean, logvar = model(states, actions)

        # Gaussian negative log-likelihood
        inv_var = (-logvar).exp()
        mse_loss = ((mean - targets) ** 2 * inv_var).mean()
        var_loss = logvar.mean()

        # Regularize logvar bounds
        bound_loss = 0.01 * (model.max_logvar.sum() - model.min_logvar.sum())

        loss = mse_loss + var_loss + bound_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        individual_losses.append(loss.item())

    # Update elite indices
    sorted_indices = torch.tensor(individual_losses).argsort()
    self.elite_indices = sorted_indices[:self.elite_size]

    return {"model_loss": total_loss / self.ensemble_size}
```

### Rollout Generation

```python
def generate_rollouts(self, start_states):
    all_states, all_actions, all_rewards, all_next_states, all_dones = [], [], [], [], []

    states = start_states

    with torch.no_grad():
        for t in range(self.rollout_length):
            # Select actions using current policy
            actions = self.agent.select_action(states)

            # Predict next state with random elite model
            elite_idx = self.elite_indices[torch.randint(0, self.elite_size, (1,)).item()]
            model = self.models[elite_idx]
            next_states, rewards = model.predict(states, actions)

            # Simple done prediction (task-specific)
            dones = torch.zeros(states.size(0), device=states.device)

            # Store
            all_states.append(states)
            all_actions.append(actions)
            all_rewards.append(rewards)
            all_next_states.append(next_states)
            all_dones.append(dones)

            states = next_states

    return {
        "states": torch.cat(all_states, dim=0),
        "actions": torch.cat(all_actions, dim=0),
        "rewards": torch.cat(all_rewards, dim=0),
        "next_states": torch.cat(all_next_states, dim=0),
        "dones": torch.cat(all_dones, dim=0),
    }
```

## 6. Optimization Tricks

1. **Rollout Length Schedule**: Start with k=1, gradually increase
   ```python
   k = min(1 + floor(step / schedule_freq), max_k)
   ```

2. **Elite Model Selection**: Only use best models for rollouts

3. **Uncertainty-Aware Rollouts**: Penalize high-uncertainty states
   ```python
   ensemble_predictions = [model.predict(s, a) for model in elite_models]
   uncertainty = std(ensemble_predictions)
   reward_adjusted = reward - beta * uncertainty
   ```

4. **Terminal Function Learning**: Learn termination prediction
   ```python
   done_pred = sigmoid(done_predictor(s'))
   ```

5. **Real Data Prioritization**: Always include some real data (real_ratio > 0)

## 7. Experimental Results

### Continuous Control Benchmarks

| Environment | MBPO (1M) | SAC (1M) | Speedup |
|-------------|-----------|----------|---------|
| HalfCheetah | 12,000 | 10,000 | 1.2x |
| Hopper | 3,500 | 2,800 | 1.25x |
| Walker2d | 5,200 | 3,500 | 1.5x |
| Ant | 6,000 | 4,000 | 1.5x |

**Key Results**:
- 2-10x more sample efficient than model-free SAC
- Competitive asymptotic performance
- Robust to model errors with short rollouts

## 8. Common Pitfalls

1. **Rollout Length Too Long**: Causes model errors to accumulate
   - **Solution**: Start with k=1, increase slowly

2. **Poor Model Training**: Overfitting or underfitting dynamics
   - **Solution**: Early stopping, ensemble diversity, validation set

3. **Exploration Insufficient**: Model only accurate in visited regions
   - **Solution**: Continue real environment exploration, don't rely only on model

4. **Memory Overflow**: Model buffer grows indefinitely
   - **Solution**: Limit buffer size, evict old synthetic data

## 9. Extensions

### 9.1 Uncertainty Penalization

Add intrinsic penalty for uncertain regions:
```python
predictions = [model(s, a) for model in elite_models]
uncertainty = std(predictions)
reward_modified = reward - lambda_uncertainty * uncertainty
```

### 9.2 Learned Termination

Predict episode termination:
```python
done_pred = TerminationNetwork(s')
```

### 9.3 Multi-Step Models

Predict k steps ahead directly:
```python
s_k, r_sum = MultiStepModel(s, [a_0, ..., a_{k-1}])
```

## 10. References

1. **MBPO**: Janner et al., "When to Trust Your Model: Model-Based Policy Optimization", NeurIPS 2019 [arXiv:1906.08253](https://arxiv.org/abs/1906.08253)

2. **PETS**: Chua et al., "Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models", NeurIPS 2018

3. **SAC**: Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor", ICML 2018

4. **Model-Based RL Survey**: Moerland et al., "Model-based Reinforcement Learning: A Survey", 2023
