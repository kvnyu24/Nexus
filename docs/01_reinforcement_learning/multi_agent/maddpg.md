# MADDPG: Multi-Agent Deep Deterministic Policy Gradient

## 1. Overview

MADDPG extends DDPG to multi-agent settings with centralized training and decentralized execution. Each agent maintains its own actor (policy) network that only uses local observations, while critics have access to all agents' observations and actions during training, enabling stable learning in mixed cooperative-competitive environments.

**Paper**: "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments" (Lowe et al., NIPS 2017)

**Key Innovation**: Per-agent critics with global information during training, allowing actors to learn in non-stationary environments caused by other evolving agents.

**Use Cases**:
- Multi-robot coordination with continuous control
- Autonomous vehicle interactions
- Multi-agent physical simulations
- Mixed cooperative-competitive games
- Multi-agent manipulation tasks

## 2. Theory and Background

### 2.1 Centralized Training, Decentralized Execution

**Training**: Each agent i has a centralized critic Q_i(o_1, ..., o_n, a_1, ..., a_n) that observes:
- All agents' observations o_1, ..., o_n
- All agents' actions a_1, ..., a_n

**Execution**: Agent i's actor π_i(a_i | o_i) uses only local observation o_i

This addresses the non-stationarity problem: from agent i's perspective, the environment appears non-stationary because other agents' policies are changing. The centralized critic provides stable training signals by considering other agents' behaviors.

### 2.2 Actor-Critic Objective

For agent i, MADDPG optimizes:

**Critic Loss** (centralized):
```
L_i(θ_i^Q) = E[(Q_i(o, a | θ_i^Q) - y_i)^2]
y_i = r_i + γ Q_i'(o', a' | θ_i^{Q'})
```

Where a' = (μ_1'(o_1'), ..., μ_n'(o_n')) are actions from target actors.

**Actor Loss** (decentralized):
```
∇_θ_i^μ J(θ_i^μ) = E[∇_θ_i^μ μ_i(o_i | θ_i^μ) · ∇_a_i Q_i(o, a_1,...,a_i,...,a_n | θ_i^Q)]
                                                                       ↑
                                                              a_i = μ_i(o_i)
```

The actor gradient flows through the critic to maximize Q-value by changing agent i's action.

### 2.3 Policy Ensembles for Robustness

To handle non-stationarity, MADDPG can use policy ensembles:
- Train K sub-policies for each agent
- Sample one at episode start
- Improves robustness to other agents' behavior changes

## 3. Mathematical Formulation

### Complete Algorithm

**For each episode:**
1. Initialize random process N for exploration
2. For t = 1 to T:
   - Each agent selects action: a_i = μ_i(o_i) + N_t
   - Execute actions, observe rewards r = (r_1, ..., r_n) and next observations o'
   - Store (o, a, r, o') in replay buffer D

**For each update step:**
1. Sample minibatch of S samples from D
2. For each agent i:

   **Update critic:**
   ```
   y_i = r_i + γ Q_i'(o', μ_1'(o_1'), ..., μ_n'(o_n'))
   L_i = (1/S) Σ (Q_i(o, a) - y_i)^2
   Update θ_i^Q by minimizing L_i
   ```

   **Update actor:**
   ```
   Sample gradient: ∇_θ_i^μ (1/S) Σ Q_i(o, a_1,...,a_i,...,a_n) where a_i = μ_i(o_i)
   Update θ_i^μ by ascending this gradient
   ```

3. Soft update target networks:
   ```
   θ_i^{Q'} ← τ θ_i^Q + (1-τ) θ_i^{Q'}
   θ_i^{μ'} ← τ θ_i^μ + (1-τ) θ_i^{μ'}
   ```

## 4. Implementation Details

### Network Architecture

```python
# Actor (per agent, decentralized)
obs_i -> [Linear(64) + ReLU] x2 -> Linear(action_dim) -> Tanh -> * max_action

# Critic (per agent, centralized)
concat(o_1,...,o_n, a_1,...,a_n) -> [Linear(64) + ReLU] x2 -> Linear(1)
```

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| gamma | 0.99 | Discount factor |
| tau | 0.01 | Soft target update rate |
| actor_lr | 1e-3 | Actor learning rate |
| critic_lr | 1e-3 | Critic learning rate |
| noise_std | 0.1 | Exploration noise std |
| batch_size | 1024 | Replay buffer batch size |
| buffer_size | 1e6 | Replay buffer capacity |

### Key Implementation Points

```python
class MADDPGAgent:
    def __init__(self, n_agents, obs_dims, action_dims):
        # Per-agent actors (decentralized)
        self.actors = [Actor(obs_dims[i], action_dims[i]) for i in range(n_agents)]
        self.actor_targets = [deepcopy(actor) for actor in self.actors]

        # Per-agent critics (centralized)
        total_obs = sum(obs_dims)
        total_act = sum(action_dims)
        self.critics = [Critic(total_obs, total_act) for i in range(n_agents)]
        self.critic_targets = [deepcopy(critic) for critic in self.critics]

        # Per-agent optimizers
        self.actor_optimizers = [Adam(actor.parameters(), lr=actor_lr)
                                 for actor in self.actors]
        self.critic_optimizers = [Adam(critic.parameters(), lr=critic_lr)
                                  for critic in self.critics]
```

## 5. Code Walkthrough (from `/nexus/models/rl/maddpg.py`)

### Actor Network

```python
class MADDPGActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64, max_action=1.0):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        self.max_action = max_action

    def forward(self, obs):
        return self.max_action * self.network(obs)
```

### Centralized Critic

```python
class MADDPGCritic(nn.Module):
    def __init__(self, total_obs_dim, total_action_dim, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(total_obs_dim + total_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, all_obs, all_actions):
        x = torch.cat([all_obs, all_actions], dim=-1)
        return self.network(x)
```

### Update Function

```python
def update(self, batch):
    all_obs = torch.cat(batch["observations"], dim=-1)
    all_actions = torch.cat(batch["actions"], dim=-1)
    all_next_obs = torch.cat(batch["next_observations"], dim=-1)

    for agent_i in range(self.n_agents):
        # --- Update Critic ---
        with torch.no_grad():
            # Get target actions from all agents
            next_actions = [self.actor_targets[i](batch["next_observations"][i])
                           for i in range(self.n_agents)]
            all_next_actions = torch.cat(next_actions, dim=-1)

            # Target Q-value
            target_q = self.critic_targets[agent_i](all_next_obs, all_next_actions)
            targets = batch["rewards"][agent_i] + self.gamma * (1 - batch["dones"]) * target_q.squeeze(-1)

        # Current Q-value
        current_q = self.critics[agent_i](all_obs, all_actions).squeeze(-1)

        # Critic loss
        critic_loss = F.mse_loss(current_q, targets)

        self.critic_optimizers[agent_i].zero_grad()
        critic_loss.backward()
        self.critic_optimizers[agent_i].step()

        # --- Update Actor ---
        # Get current actions from all agents (agent_i uses trainable actor)
        current_actions = []
        for i in range(self.n_agents):
            if i == agent_i:
                current_actions.append(self.actors[i](batch["observations"][i]))
            else:
                current_actions.append(self.actors[i](batch["observations"][i]).detach())
        all_current_actions = torch.cat(current_actions, dim=-1)

        # Actor loss: negative Q-value
        actor_loss = -self.critics[agent_i](all_obs, all_current_actions).mean()

        self.actor_optimizers[agent_i].zero_grad()
        actor_loss.backward()
        self.actor_optimizers[agent_i].step()

    # Soft update targets
    self._soft_update()
```

## 6. Optimization Tricks

1. **Exploration Noise Annealing**: Decay noise over time
   ```python
   noise_std = initial_std * (decay ** episode)
   ```

2. **Gradient Clipping**: Prevent exploding gradients
   ```python
   nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm=10.0)
   ```

3. **Batch Normalization**: Normalize observations in critic
   ```python
   self.obs_bn = nn.BatchNorm1d(obs_dim)
   ```

4. **Prioritized Experience Replay**: Sample important transitions more
   ```python
   priorities = td_error.abs() + epsilon
   ```

## 7. Experimental Results

### MPE Benchmark

| Environment | MADDPG Win % | DDPG Win % | IQL Win % |
|-------------|--------------|------------|-----------|
| Cooperative Navigation | 85% | 60% | 45% |
| Predator-Prey | 92% | 70% | 55% |
| Physical Deception | 78% | 50% | 40% |

**Key Findings**:
- MADDPG excels in mixed cooperative-competitive settings
- Centralized critics provide stable learning despite non-stationarity
- Scales to 10+ agents with careful tuning

## 8. Common Pitfalls

1. **Exploration Insufficient**: Use Ornstein-Uhlenbeck noise or epsilon-greedy
2. **Critic Divergence**: Lower learning rates, gradient clipping
3. **Reward Scaling**: Normalize rewards per agent
4. **Memory Issues**: Concatenating all observations/actions can be large

## 9. Extensions

### 9.1 Attention Mechanism

Replace concatenation with attention:
```python
class AttentionCritic(nn.Module):
    def forward(self, agent_obs, agent_actions):
        # Attention over agents
        embeddings = [self.embed(torch.cat([obs_i, act_i], -1))
                     for obs_i, act_i in zip(agent_obs, agent_actions)]
        attended = self.attention(embeddings)
        return self.value_head(attended)
```

### 9.2 Communication

Add communication channel:
```python
# Each agent broadcasts message
messages = [self.comm_encoder(obs_i) for obs_i in observations]
# Aggregate messages
agg_message = sum(messages) / len(messages)
# Actor uses aggregated message
action = self.actor(torch.cat([obs_i, agg_message], -1))
```

## 10. References

1. **MADDPG**: Lowe et al., "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments", NIPS 2017 [arXiv:1706.02275](https://arxiv.org/abs/1706.02275)

2. **DDPG**: Lillicrap et al., "Continuous Control with Deep Reinforcement Learning", ICLR 2016

3. **Multi-Agent Particle Envs**: [GitHub](https://github.com/openai/multiagent-particle-envs)

4. **QMIX**: Rashid et al., ICML 2018 (for comparison with value-based methods)
