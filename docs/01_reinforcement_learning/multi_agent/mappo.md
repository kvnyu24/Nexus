# MAPPO: Multi-Agent Proximal Policy Optimization

## 1. Overview

Multi-Agent Proximal Policy Optimization (MAPPO) extends the highly successful PPO algorithm to multi-agent cooperative settings using the Centralized Training with Decentralized Execution (CTDE) paradigm. It combines independent actor networks for each agent with a shared centralized critic that has access to global state information during training.

**Paper**: "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" (Yu et al., NeurIPS 2022)

**Key Innovation**: Demonstrating that PPO, with proper engineering and a shared centralized critic, can match or exceed the performance of specialized multi-agent algorithms while maintaining simplicity and stability.

**Use Cases**:
- Cooperative robot control (multi-robot coordination, swarms)
- Multi-agent navigation
- Team-based game AI
- Autonomous vehicle platoons
- Distributed control systems

## 2. Theory and Background

### 2.1 Multi-Agent Problem Formulation

In cooperative multi-agent settings, we have:
- **N agents** indexed by i ∈ {1, ..., N}
- **Local observations** o_i for each agent
- **Global state** s (may not be observable during execution)
- **Joint action** a = (a_1, ..., a_N)
- **Shared reward** r (team reward)

The goal is to learn policies π_i(a_i | o_i) that maximize the expected team return:

```
J = E[∑_{t=0}^∞ γ^t r_t]
```

### 2.2 Centralized Training with Decentralized Execution

**Training Phase**:
- Centralized critic V(s) has access to global state s
- Used to compute advantages for all agents
- Enables better credit assignment

**Execution Phase**:
- Each agent uses only its actor π_i(a_i | o_i)
- No communication or global information needed
- Fully decentralized and scalable

### 2.3 PPO Objective

For each agent i, MAPPO optimizes the clipped surrogate objective:

```
L^CLIP(θ_i) = E[min(r_i(θ_i) Â, clip(r_i(θ_i), 1-ε, 1+ε) Â)]
```

Where:
- r_i(θ_i) = π_θ_i(a_i | o_i) / π_θ_i_old(a_i | o_i) is the importance ratio
- Â is the advantage from the shared critic
- ε is the clip range (typically 0.2)

### 2.4 Generalized Advantage Estimation

Advantages are computed using GAE with the shared critic:

```
Â_t = ∑_{l=0}^∞ (γλ)^l δ_{t+l}
δ_t = r_t + γV(s_{t+1}) - V(s_t)
```

The shared critic provides a common baseline for all agents, improving credit assignment.

## 3. Mathematical Formulation

### Complete Loss Function

For each agent i:

```
L_total^i = L_policy^i + c_1 L_value + c_2 H(π_θ_i)
```

Where:
- **Policy loss**: L_policy^i = -E[min(r_i Â, clip(r_i, 1-ε, 1+ε) Â)]
- **Value loss**: L_value = E[(V(s) - V_target)^2]
- **Entropy bonus**: H(π_θ_i) = -E[π_θ_i log π_θ_i]
- c_1 = 0.5, c_2 = 0.01 (typical values)

### Value Target

```
V_target = Â + V(s)
```

Where advantages are normalized:
```
Â = (Â - mean(Â)) / (std(Â) + ε)
```

### Gradient Flow

1. **Actor gradients**: Flow through the policy network and clipping operation
2. **Critic gradients**: MSE between V(s) and returns, no gradient from actors
3. **Separate optimizers**: Each actor and the shared critic have independent optimizers

## 4. Intuitive Explanation

Think of MAPPO as a team of students (agents) working on a group project:

1. **Individual Actions**: Each student makes their own decisions based on what they can see
2. **Shared Feedback**: A teacher (centralized critic) evaluates the overall project quality, seeing everyone's contributions
3. **Credit Assignment**: The teacher's feedback helps each student understand how their actions contributed to the team's success
4. **Independent Learning**: Each student improves their individual skills, but benefits from the team-level feedback

### Why It Works

1. **Stability**: PPO's clipped objective prevents destructively large policy updates
2. **Credit Assignment**: Shared critic provides consistent value estimates across agents
3. **Scalability**: Decentralized execution scales to many agents
4. **Simplicity**: No complex mixing networks or coordination mechanisms
5. **Generalization**: Parameter sharing across agents (optional) improves sample efficiency

## 5. Implementation Details

### Network Architecture

```python
# Actor (per-agent, uses local observation)
obs -> [Linear(256) + LayerNorm + ReLU] x2 -> mean, log_std

# Shared Critic (uses global state)
state -> [Linear(256) + LayerNorm + ReLU] x2 -> V(s)
```

**Key Design Choices**:
- LayerNorm for training stability
- Separate actor for each agent (or parameter-shared)
- Single shared critic for all agents
- Gaussian policy for continuous actions

### Hyperparameters

| Parameter | Typical Value | Description |
|-----------|---------------|-------------|
| clip_range | 0.2 | PPO clipping parameter |
| gamma | 0.99 | Discount factor |
| gae_lambda | 0.95 | GAE lambda parameter |
| value_coef | 0.5 | Value loss coefficient |
| entropy_coef | 0.01 | Entropy bonus coefficient |
| max_grad_norm | 0.5 | Gradient clipping threshold |
| learning_rate | 3e-4 | Learning rate for all networks |
| num_epochs | 10 | PPO epochs per batch |
| batch_size | 2048 | Timesteps per update |

### Training Loop

```python
for episode in range(num_episodes):
    # Collect experience
    observations, states, actions, rewards, dones = [], [], [], [], []

    for t in range(horizon):
        # Each agent selects action
        actions_t = [actor_i.select(obs_i) for i, obs_i in enumerate(obs)]

        # Execute in environment
        next_obs, reward, done, state = env.step(actions_t)

        # Store transition
        observations.append(obs)
        states.append(state)
        actions.append(actions_t)
        rewards.append(reward)
        dones.append(done)

        obs = next_obs

    # Compute advantages with shared critic
    values = critic(states)
    advantages, returns = compute_gae(values, rewards, dones)

    # Update each actor
    for agent_i in range(n_agents):
        for epoch in range(ppo_epochs):
            # PPO clipped surrogate objective
            update_actor(agent_i, observations, actions, advantages)

    # Update shared critic
    for epoch in range(ppo_epochs):
        update_critic(states, returns)
```

## 6. Code Walkthrough

### Core Components (from `/nexus/models/rl/mappo.py`)

#### 1. Shared Critic

```python
class SharedCritic(NexusModule):
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # Stability
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
```

**Key Points**:
- Takes global state as input
- LayerNorm layers for training stability
- Single output: V(s)

#### 2. Actor Network

```python
class MAPPOActor(NexusModule):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # Learnable std
```

**Key Points**:
- Uses only local observation
- Gaussian policy: outputs mean and std
- Learnable log_std parameter (state-independent)

#### 3. GAE Computation

```python
def compute_gae(self, values, rewards, dones, next_values):
    advantages = torch.zeros_like(rewards)
    gae = 0.0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = next_values[t]
        else:
            next_value = values[t + 1]

        # TD error
        delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]

        # GAE
        gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
        advantages[t] = gae

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    returns = advantages + values

    return advantages, returns
```

**Key Points**:
- Backward pass through time
- Normalizes advantages for stability
- Returns = advantages + values (for critic target)

#### 4. PPO Update

```python
def update(self, batch):
    # Update shared critic
    values = self.critic(batch["states"]).squeeze(-1)
    value_loss = F.mse_loss(values, batch["returns"])

    self.critic_optimizer.zero_grad()
    value_loss.backward()
    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
    self.critic_optimizer.step()

    # Update each agent's actor
    for i in range(self.num_agents):
        # Get current policy distribution
        dist = self.actors[i].get_distribution(batch["observations"][i])
        new_log_probs = dist.log_prob(batch["actions"][i]).sum(dim=-1, keepdim=True)

        # Importance ratio
        ratio = (new_log_probs - batch["old_log_probs"][i]).exp()

        # PPO clipped objective
        surr1 = ratio * batch["advantages"]
        surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * batch["advantages"]
        policy_loss = -torch.min(surr1, surr2).mean()

        # Add entropy bonus
        entropy = dist.entropy().sum(dim=-1).mean()
        actor_loss = policy_loss - self.entropy_coef * entropy

        # Optimize
        self.actor_optimizers[i].zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actors[i].parameters(), self.max_grad_norm)
        self.actor_optimizers[i].step()
```

## 7. Optimization Tricks

### 7.1 Architecture Tricks

1. **LayerNorm over BatchNorm**: More stable for RL where batch statistics vary
2. **Separate optimizers**: Allows different learning rates for actors/critic
3. **Parameter sharing**: Share actor weights across agents to reduce parameters:
   ```python
   self.actor = MAPPOActor(obs_dim, action_dim)  # Single actor
   # Use with agent_id as input
   ```

### 7.2 Training Tricks

1. **Advantage Normalization**: Critical for multi-agent stability
   ```python
   advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
   ```

2. **Gradient Clipping**: Use aggressive clipping (0.5) to prevent instability
   ```python
   nn.utils.clip_grad_norm_(parameters, max_grad_norm=0.5)
   ```

3. **Value Function Clipping**: Optionally clip value updates:
   ```python
   values_clipped = old_values + torch.clamp(
       values - old_values, -clip_range, clip_range
   )
   value_loss = torch.max(
       F.mse_loss(values, returns),
       F.mse_loss(values_clipped, returns)
   )
   ```

4. **Learning Rate Annealing**: Linearly decay LR over training
   ```python
   lr = initial_lr * (1 - step / total_steps)
   ```

5. **Episode-based Training**: Always train on full episodes for proper advantage computation

### 7.3 Exploration Tricks

1. **Action Noise Decay**: Start with high exploration, decay over time
   ```python
   std = initial_std * (decay_factor ** episode)
   ```

2. **Entropy Bonus Annealing**: Decay entropy coefficient over training
   ```python
   entropy_coef = initial_coef * (1 - step / total_steps)
   ```

## 8. Experimental Results

### 8.1 Benchmark Environments

**SMAC (StarCraft Multi-Agent Challenge)**:
- 3m (3 Marines): Win rate 95%+ in 2M steps
- 8m (8 Marines): Win rate 90%+ in 5M steps
- 2s3z (2 Stalkers, 3 Zealots): Win rate 85%+ in 10M steps

**MPE (Multi-Particle Environments)**:
- Simple Spread: -130 reward (coverage task)
- Simple Tag: -40 reward (predator-prey)
- Simple Adversary: -100 reward (mixed cooperative-competitive)

**Google Research Football**:
- 3 vs 3: 60%+ win rate against built-in AI
- Full game: Competitive with handcrafted bots

### 8.2 Ablation Studies

| Configuration | Performance | Notes |
|---------------|-------------|-------|
| MAPPO (full) | 100% | Baseline |
| Without shared critic | 65% | Individual critics hurt coordination |
| Without LayerNorm | 75% | Training instability |
| Without advantage norm | 70% | Poor credit assignment |
| Larger clip range (0.3) | 85% | Less stable |
| Smaller clip range (0.1) | 90% | Slower learning |
| Parameter sharing | 105% | Better generalization |

### 8.3 Hyperparameter Sensitivity

**Most Sensitive**:
- `clip_range`: 0.2 is optimal for most tasks
- `gae_lambda`: 0.95 works well, lower values (0.9) help in sparse reward settings
- `learning_rate`: 3e-4 to 1e-3 depending on task complexity

**Less Sensitive**:
- `value_coef`: 0.5-1.0 all work reasonably
- `entropy_coef`: 0.01-0.001 (anneal over training)
- `max_grad_norm`: 0.5-10.0 (lower is safer)

## 9. Common Pitfalls and Solutions

### 9.1 Training Instability

**Problem**: Policy collapses, values explode, or performance degrades suddenly

**Solutions**:
- Reduce learning rate (try 1e-4)
- Decrease clip_range to 0.1
- Increase max_grad_norm clipping
- Use smaller batch sizes
- Check for NaN/Inf values in advantages

### 9.2 Poor Credit Assignment

**Problem**: Agents don't coordinate effectively, individual behaviors dominate

**Solutions**:
- Ensure shared critic receives global state (not concatenated observations)
- Normalize advantages per-batch, not per-agent
- Increase value_coef to improve critic training
- Use longer episodes for better temporal credit assignment

### 9.3 Slow Convergence

**Problem**: Training takes too long or plateaus early

**Solutions**:
- Parameter sharing across agents (huge speedup)
- Increase batch size (more stable gradients)
- Use curriculum learning (start with easier scenarios)
- Pretrain critic on expert trajectories
- Check exploration (increase initial entropy)

### 9.4 Overfitting to Training Scenarios

**Problem**: Good training performance, poor generalization

**Solutions**:
- Randomize initial conditions
- Use domain randomization
- Add noise to observations
- Train on diverse scenarios simultaneously
- Regularize with larger entropy bonus

### 9.5 Communication Overhead

**Problem**: Centralized critic slows down training

**Solutions**:
- Use smaller critic network
- Update critic less frequently (every N actor updates)
- Compress state representation
- Use distributed training (data parallelism)

## 10. Extensions and Variants

### 10.1 Heterogeneous MAPPO

Different agent types with different observation/action spaces:
```python
self.actors = nn.ModuleList([
    MAPPOActor(obs_dim_i, action_dim_i)
    for obs_dim_i, action_dim_i in zip(obs_dims, action_dims)
])
```

### 10.2 Communication MAPPO

Add learnable communication:
```python
class CommMAPPOActor(MAPPOActor):
    def forward(self, obs, messages):
        features = self.network(torch.cat([obs, messages], dim=-1))
        return self.mean_head(features), self.log_std.exp()
```

### 10.3 Hierarchical MAPPO

Multi-level hierarchy with high-level and low-level policies:
- High-level: Sets goals for sub-teams
- Low-level: Executes primitive actions
- Use temporal abstraction (options/skills)

### 10.4 MAPPO with Attention

Add attention mechanism for agent interactions:
```python
class AttentionCritic(nn.Module):
    def forward(self, state, agent_features):
        # Self-attention over agent features
        attended = self.attention(agent_features, agent_features, agent_features)
        # Combine with global state
        return self.value_head(torch.cat([state, attended], dim=-1))
```

### 10.5 Curriculum MAPPO

Progressive difficulty increase:
```python
if performance > threshold:
    task_difficulty += 1
    num_agents += 1  # or increase map size, add obstacles, etc.
```

## 11. References

### Original Papers

1. **MAPPO**: Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games", NeurIPS 2022 [arXiv:2103.01955](https://arxiv.org/abs/2103.01955)

2. **PPO**: Schulman et al., "Proximal Policy Optimization Algorithms", 2017 [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)

3. **GAE**: Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation", ICLR 2016 [arXiv:1506.02438](https://arxiv.org/abs/1506.02438)

### Related Work

4. **IPPO**: De Witt et al., "Is Independent Learning All You Need in the StarCraft Multi-Agent Challenge?", 2020

5. **MADDPG**: Lowe et al., "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments", NIPS 2017

6. **QMIX**: Rashid et al., "QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning", ICML 2018

### Implementation References

7. [PyMARL2](https://github.com/oxwhirl/pymarl2): Benchmark MARL implementations
8. [SMAC](https://github.com/oxwhirl/smac): StarCraft Multi-Agent Challenge
9. [EPyMARL](https://github.com/uoe-agents/epymarl): Extended PyMARL

### Surveys and Tutorials

10. Zhang et al., "Multi-Agent Reinforcement Learning: A Selective Overview of Theories and Algorithms", 2021

11. Gronauer & Diepold, "Multi-Agent Deep Reinforcement Learning: A Survey", Artificial Intelligence Review, 2022
