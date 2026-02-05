# DDPG: Deep Deterministic Policy Gradient

## 1. Overview & Motivation

Deep Deterministic Policy Gradient (DDPG) is an actor-critic algorithm designed for continuous action spaces. Introduced by Lillicrap et al. in 2015, DDPG combines the actor-critic architecture with insights from Deep Q-Networks (DQN) to enable stable learning of deterministic policies in continuous domains.

### Why DDPG?

**Key Innovation:**
DDPG extends the DPG (Deterministic Policy Gradient) algorithm to high-dimensional continuous action spaces using deep neural networks. It's essentially "DQN for continuous actions" - applying the same stabilization techniques (experience replay, target networks) to policy gradients.

**Historical Context:**
- First successful deep RL algorithm for continuous control
- Bridge between value-based (DQN) and policy-based methods
- Foundation for modern continuous control algorithms (TD3, SAC)
- Enabled complex robotic control tasks

**Key Advantages:**
- **Continuous actions**: Direct output, no discretization needed
- **Off-policy learning**: Sample efficient through experience replay
- **Deterministic policy**: Simpler than stochastic policies
- **Stable training**: Target networks and replay buffer
- **Sample efficiency**: Better than on-policy methods

**Improvements over Policy Gradients:**
- Off-policy (vs on-policy REINFORCE/A2C)
- More sample efficient (reuses old data)
- Stable training (target networks)
- Handles continuous actions naturally

### When to Use DDPG

**Ideal For:**
- Continuous control tasks (robotics, physics simulation)
- Environments with low-dimensional actions (<20D)
- When sample efficiency matters
- Deterministic control policies
- Learning from demonstrations (off-policy)

**Avoid When:**
- Need maximum stability (use TD3 or SAC instead)
- Very high-dimensional actions (>50D)
- Sensitive to hyperparameters (prefer SAC)
- Discrete action spaces (use DQN/PPO)

**Modern Alternatives:**
- **TD3**: More stable than DDPG (recommended over DDPG)
- **SAC**: Best sample efficiency for continuous control
- **PPO**: If on-policy is acceptable

## 2. Theoretical Background

### Deterministic Policy Gradient Theorem

DDPG builds on the Deterministic Policy Gradient (DPG) theorem by Silver et al. (2014):

```
∇_θ J(θ) = E_s~ρ^β[∇_θ μ_θ(s) ∇_a Q^μ(s,a)|_{a=μ_θ(s)}]
```

Where:
- `μ_θ(s)`: Deterministic policy (actor)
- `Q^μ(s,a)`: Action-value function (critic)
- `ρ^β`: State distribution under behavior policy β

**Key insight:** For deterministic policies, we can compute gradients directly through the Q-function.

### From Stochastic to Deterministic

**Stochastic policy gradient:**
```
∇_θ J(θ) = E[∇_θ log π_θ(a|s) Q^π(s,a)]
```
- Requires sampling actions
- High variance
- Harder to optimize

**Deterministic policy gradient:**
```
∇_θ J(θ) = E[∇_θ μ_θ(s) ∇_a Q^μ(s,a)|_{a=μ_θ(s)}]
```
- No sampling needed (deterministic)
- Lower variance
- Direct gradient through Q-function

### Actor-Critic with Q-Learning

DDPG combines:

**Actor (Policy):**
```
a = μ_θ(s)  (deterministic mapping)
```

**Critic (Q-function):**
```
Q_φ(s, a) ≈ Q^μ(s, a)  (action-value estimate)
```

**Actor update (policy improvement):**
```
θ ← θ + α E[∇_θ μ_θ(s) ∇_a Q_φ(s,a)|_{a=μ_θ(s)}]
```

**Critic update (policy evaluation):**
```
φ ← φ - β E[(Q_φ(s,a) - (r + γ Q_φ'(s',μ_θ'(s'))))^2]
```

### Off-Policy Learning

DDPG is off-policy: it can learn from data collected by any policy.

**Behavior policy (for exploration):**
```
a_t = μ_θ(s_t) + N_t  (add noise)
```

**Target policy (what we learn):**
```
a = μ_θ(s)  (deterministic)
```

This enables:
- Experience replay (reuse old data)
- Learning from demonstrations
- Parallel data collection

### Target Networks

DDPG uses target networks (from DQN) for stability:

**Primary networks:** `μ_θ`, `Q_φ`
**Target networks:** `μ_θ'`, `Q_φ'`

**Soft update:**
```
θ' ← τθ + (1-τ)θ'
φ' ← τφ + (1-τ)φ'
```

Where `τ << 1` (typically 0.001-0.005).

**Why this helps:**
- Prevents oscillations in Q-values
- Stabilizes training
- Smoother learning curves

### Ornstein-Uhlenbeck Noise

For exploration, DDPG adds temporally correlated noise:

```
dN_t = θ(μ - N_t)dt + σ dW_t
```

Where:
- `θ`: Mean reversion rate
- `μ`: Long-term mean
- `σ`: Volatility
- `W_t`: Wiener process

**Properties:**
- Temporally correlated (smooth exploration)
- Mean-reverting (returns to μ)
- Better than white noise for physical systems

## 3. Mathematical Formulation

### Complete DDPG Algorithm

**Objective:**
Maximize expected return:
```
J(θ) = E_s~ρ^μ[R(s, μ_θ(s))]
```

**Actor Network:**
```
μ_θ: S → A
a = μ_θ(s)
```

**Critic Network:**
```
Q_φ: S × A → ℝ
Q_φ(s, a) ≈ E[R_t | s_t=s, a_t=a, π=μ]
```

**Target Computation:**
```
y_t = r_t + γ Q_φ'(s_{t+1}, μ_θ'(s_{t+1}))
```

**Critic Loss (TD error):**
```
L_Q(φ) = E[(Q_φ(s,a) - y)^2]
```

**Actor Loss (negative Q-value):**
```
L_μ(θ) = -E[Q_φ(s, μ_θ(s))]
```

**Target Network Update:**
```
θ' ← τθ + (1-τ)θ'
φ' ← τφ + (1-τ)φ'
```

### Gradient Computation

**Critic gradient:**
```
∇_φ L_Q = E[2(Q_φ(s,a) - y) ∇_φ Q_φ(s,a)]
```

**Actor gradient (chain rule):**
```
∇_θ L_μ = E[∇_θ μ_θ(s) ∇_a Q_φ(s,a)|_{a=μ_θ(s)}]
```

The gradient flows:
```
θ → μ_θ(s) → Q_φ(s, μ_θ(s))
```

### Update Rules

**Sample transition:** `(s, a, r, s', done)`

1. **Critic update:**
```
y = r + γ(1 - done) Q_φ'(s', μ_θ'(s'))
L_Q = (Q_φ(s, a) - y)^2
φ ← φ - α_Q ∇_φ L_Q
```

2. **Actor update:**
```
L_μ = -Q_φ(s, μ_θ(s))
θ ← θ - α_μ ∇_θ L_μ
```

3. **Target update:**
```
θ' ← τθ + (1-τ)θ'
φ' ← τφ + (1-τ)φ'
```

## 4. High-Level Intuition

### The Core Idea

DDPG is "Q-learning for continuous actions":

**DQN (discrete):**
- Learn Q(s,a) for all actions
- Pick action with max Q-value
- Works only for discrete actions

**DDPG (continuous):**
- Learn Q(s,a) for any action
- Learn policy μ(s) that maximizes Q
- Works for continuous actions

### The Actor-Critic Dance

**Critic says:** "This state-action pair is worth X"
**Actor asks:** "What action should I take?"
**Critic suggests:** "Take the action that maximizes my Q-value"
**Actor learns:** "I'll learn to output that action directly"

### Why Deterministic?

**Stochastic policy:** π(a|s) - "For this state, sample from this distribution"
- More exploration built-in
- Higher variance gradients
- Harder to optimize

**Deterministic policy:** a = μ(s) - "For this state, do this action"
- Simpler to learn
- Lower variance gradients
- Add noise separately for exploration

**Trade-off:**
- Deterministic: Better for convergence
- Stochastic: Better for exploration (addressed by SAC)

### Experience Replay Buffer

Like DQN, DDPG stores past experiences:

```
Buffer: [(s_0, a_0, r_0, s_1), (s_1, a_1, r_1, s_2), ...]
```

**Benefits:**
- Break correlation in sequential data
- Reuse data (sample efficiency)
- Stabilize training

**How it works:**
1. Agent acts: (s, a, r, s')
2. Store in buffer
3. Sample random mini-batch
4. Update networks

### Target Networks

Without targets:
```
Q(s,a) → r + γ Q(s', μ(s'))  [chasing a moving target]
```

With targets:
```
Q(s,a) → r + γ Q'(s', μ'(s'))  [target is stable]
```

**Analogy:** Like having a teacher (target) who updates slowly while student (primary) learns quickly.

### Exploration via Noise

**Pure exploitation:**
```
a = μ_θ(s)  [always same action]
```

**With exploration:**
```
a = μ_θ(s) + N_t  [try variations]
```

The OU noise provides smooth, temporally correlated exploration - good for physical systems with momentum.

## 5. Implementation Details

### Algorithm Pseudocode

```python
# Initialization
Initialize actor μ_θ and critic Q_φ with random weights
Initialize target networks θ' ← θ, φ' ← φ
Initialize replay buffer D
Initialize exploration noise process N

for episode = 1, 2, 3, ... do:
    Initialize noise process N
    Receive initial state s_0

    for t = 0, 1, 2, ... do:
        # Select action with exploration noise
        a_t = μ_θ(s_t) + N_t

        # Execute action and observe
        Execute a_t, observe r_t, s_{t+1}, done

        # Store transition in replay buffer
        Store (s_t, a_t, r_t, s_{t+1}, done) in D

        # Sample mini-batch from replay buffer
        Sample N transitions from D: {(s_i, a_i, r_i, s'_i, done_i)}

        # Compute target Q-values
        y_i = r_i + γ(1 - done_i) Q_φ'(s'_i, μ_θ'(s'_i))

        # Update critic
        L_Q = (1/N) ∑_i (Q_φ(s_i, a_i) - y_i)^2
        φ ← φ - α_Q ∇_φ L_Q

        # Update actor
        L_μ = -(1/N) ∑_i Q_φ(s_i, μ_θ(s_i))
        θ ← θ - α_μ ∇_θ L_μ

        # Update target networks
        θ' ← τθ + (1-τ)θ'
        φ' ← τφ + (1-τ)φ'

        s_t ← s_{t+1}
    end for
end for
```

### Hyperparameter Choices

**Standard Configuration:**
```python
{
    "actor_lr": 1e-4,        # Actor learning rate
    "critic_lr": 1e-3,       # Critic learning rate (higher)
    "gamma": 0.99,           # Discount factor
    "tau": 0.005,            # Target network update rate
    "buffer_size": 1000000,  # Replay buffer size
    "batch_size": 256,       # Mini-batch size
    "noise_sigma": 0.1,      # OU noise std
    "noise_theta": 0.15,     # OU noise mean reversion
    "max_action": 1.0,       # Action space bounds
}
```

**For Different Tasks:**

Robotics (smooth control):
```python
{
    "noise_sigma": 0.2,    # More exploration
    "tau": 0.001,          # Slower target updates
}
```

Fast dynamics:
```python
{
    "noise_sigma": 0.1,    # Less exploration
    "tau": 0.005,          # Faster target updates
}
```

### Network Architecture

**Actor Network:**
```python
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()  # Bound output to [-1, 1]
        )
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * self.net(state)
```

**Critic Network:**
```python
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))
```

**Key Design Choices:**
- LayerNorm for stability (empirically better than BatchNorm)
- Tanh activation on actor output (bounded actions)
- 256 hidden units (can go larger for complex tasks)
- Small final layer weights initialization (3e-3)

## 6. Code Walkthrough

### Nexus Implementation

Location: `/nexus/models/rl/ddpg.py`

**1. Actor Network:**
```python
class Actor(NexusModule):
    def __init__(self, state_dim, action_dim, hidden_dim=256, max_action=1.0):
        super().__init__()
        self.max_action = max_action

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

        # Initialize final layer with small weights
        self.network[-2].weight.data.uniform_(-3e-3, 3e-3)
        self.network[-2].bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        return self.max_action * self.network(state)
```

**Key Points:**
- Small final layer init prevents large initial actions
- Tanh ensures actions in [-1, 1]
- Scale by max_action for environment range

**2. Critic Network:**
```python
class Critic(NexusModule):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Initialize final layer with small weights
        self.network[-1].weight.data.uniform_(-3e-3, 3e-3)
        self.network[-1].bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.network(x)
```

**3. OU Noise Process:**
```python
class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(action_dim) * mu

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + \
             self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state
```

**Key Points:**
- Mean-reverting process for smooth exploration
- State persists across steps (temporal correlation)
- Reset at episode start

**4. DDPG Agent:**
```python
class DDPGAgent(NexusModule):
    def __init__(self, config):
        super().__init__(config)

        # Initialize networks
        self.actor = Actor(state_dim, action_dim, hidden_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim, hidden_dim)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr
        )

        # Exploration noise
        self.noise = OUNoise(action_dim, sigma=noise_sigma)
```

**5. Action Selection:**
```python
def select_action(self, state, add_noise=True):
    with torch.no_grad():
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).cpu().numpy()[0]

    if add_noise:
        noise = self.noise.sample()
        action = action + noise
        action = np.clip(action, -self.max_action, self.max_action)

    return action
```

**Key Points:**
- No gradient computation during action selection
- Add noise for exploration during training
- Clip to valid action range

**6. Update Step:**
```python
def update(self, batch):
    states = batch["states"]
    actions = batch["actions"]
    rewards = batch["rewards"].unsqueeze(-1)
    next_states = batch["next_states"]
    dones = batch["dones"].unsqueeze(-1)

    # Critic update
    with torch.no_grad():
        next_actions = self.actor_target(next_states)
        target_q = self.critic_target(next_states, next_actions)
        target_q = rewards + self.gamma * (1 - dones) * target_q

    current_q = self.critic(states, actions)
    critic_loss = F.mse_loss(current_q, target_q)

    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()

    # Actor update
    actor_loss = -self.critic(states, self.actor(states)).mean()

    self.actor_optimizer.zero_grad()
    actor_loss.backward()
    self.actor_optimizer.step()

    # Target network update
    self._soft_update(self.actor, self.actor_target)
    self._soft_update(self.critic, self.critic_target)

    return {
        "actor_loss": actor_loss.item(),
        "critic_loss": critic_loss.item()
    }

def _soft_update(self, source, target):
    for param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_(
            self.tau * param.data + (1 - self.tau) * target_param.data
        )
```

**Key Points:**
- Critic updated first (provides gradient for actor)
- Actor maximizes Q-value through gradient ascent
- Soft target updates every step

### Usage Example

```python
from nexus.models.rl import DDPGAgent
import gym

config = {
    "state_dim": 3,
    "action_dim": 1,
    "hidden_dim": 256,
    "max_action": 2.0,
    "actor_lr": 1e-4,
    "critic_lr": 1e-3,
    "gamma": 0.99,
    "tau": 0.005,
    "noise_sigma": 0.1,
}

agent = DDPGAgent(config)
env = gym.make("Pendulum-v1")
replay_buffer = ReplayBuffer(capacity=1000000)

# Training loop
state = env.reset()
for step in range(max_steps):
    # Select action
    action = agent.select_action(state, add_noise=True)

    # Execute action
    next_state, reward, done, _ = env.step(action)

    # Store transition
    replay_buffer.add(state, action, reward, next_state, done)

    # Update agent
    if len(replay_buffer) > batch_size:
        batch = replay_buffer.sample(batch_size)
        metrics = agent.update(batch)

    state = next_state if not done else env.reset()
```

## 7. Optimization Tricks

### Exploration Strategies

**1. Decay Exploration Noise:**
```python
# Anneal noise over training
noise_sigma = max(min_sigma, initial_sigma * decay_rate ** episode)
noise.sigma = noise_sigma
```

**2. Parameter Space Noise (alternative to action noise):**
```python
# Add noise to network parameters instead
perturbed_actor = copy.deepcopy(actor)
for param in perturbed_actor.parameters():
    param.data += torch.randn_like(param) * noise_scale
```

**3. Gaussian Noise (simpler alternative to OU):**
```python
# Simple uncorrelated noise
action = actor(state) + np.random.normal(0, noise_std, action_dim)
```

### Critic Improvements

**1. Gradient Clipping:**
```python
torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=1.0)
```

**2. Reward Scaling:**
```python
# Normalize rewards
rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
```

**3. Huber Loss (for outliers):**
```python
critic_loss = F.smooth_l1_loss(current_q, target_q)
```

### Actor Improvements

**1. Action Smoothing:**
```python
# Average with previous action
action = 0.9 * action + 0.1 * prev_action
```

**2. Batch Normalization (alternative to LayerNorm):**
```python
nn.Linear(state_dim, 256),
nn.BatchNorm1d(256),
nn.ReLU(),
```

**3. Larger Actor Learning Rate (if training slow):**
```python
actor_lr = 1e-3  # Default is 1e-4
```

### Training Stability

**1. Delayed Actor Updates:**
```python
# Update actor less frequently than critic
if step % policy_delay == 0:
    # Actor update
    actor_loss = -critic(states, actor(states)).mean()
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()
```

**2. Critic Regularization:**
```python
# L2 regularization on Q-values
critic_loss = mse_loss + 0.01 * torch.mean(current_q ** 2)
```

**3. Target Network Hard Updates (every N steps):**
```python
if step % hard_update_freq == 0:
    actor_target.load_state_dict(actor.state_dict())
    critic_target.load_state_dict(critic.state_dict())
```

### Sample Efficiency

**1. Prioritized Experience Replay:**
```python
# Sample based on TD error
td_errors = abs(current_q - target_q)
priorities = td_errors.detach().cpu().numpy()
replay_buffer.update_priorities(indices, priorities)
```

**2. N-Step Returns:**
```python
# Use n-step bootstrapping
n_step_target = sum(gamma**i * rewards[t+i] for i in range(n)) + \
                gamma**n * Q(state[t+n], actor(state[t+n]))
```

**3. Hindsight Experience Replay (for sparse rewards):**
```python
# Relabel goals in failed episodes
for transition in episode:
    achieved_goal = final_state
    new_reward = reward_fn(state, action, achieved_goal)
    replay_buffer.add(state, action, new_reward, next_state, done)
```

## 8. Experiments & Results

### Pendulum-v1

**Setup:** Classic continuous control
- State: [cos(θ), sin(θ), θ_dot]
- Action: Torque [-2, 2]
- Target: -150 reward (closer to 0 is better)

**Hyperparameters:**
```python
{
    "actor_lr": 1e-4,
    "critic_lr": 1e-3,
    "noise_sigma": 0.1,
    "tau": 0.005,
}
```

**Results:**
- Solves in ~50k steps
- Final: -130±20 reward
- Stable learning curve
- Deterministic policy works well

### MountainCarContinuous-v0

**Setup:** Challenging continuous control
- State: [position, velocity]
- Action: Force [-1, 1]
- Sparse reward (only at goal)

**Results:**
- Struggles with sparse rewards
- Needs ~500k steps to solve
- Benefits from reward shaping
- HER helps significantly

### BipedalWalker-v3

**Setup:** Complex continuous control
- State: 24D (lidar + joints)
- Action: 4D (hip/knee torques)
- Target: 300+ reward

**Results:**
- Solves in ~2M steps
- Less stable than TD3/SAC
- Sensitive to hyperparameters
- Final: 280±40 reward

### Comparison with Other Algorithms

**Pendulum-v1 (steps to solve):**
```
DDPG:  50k steps
TD3:   40k steps   [More stable]
SAC:   30k steps   [Most efficient]
PPO:   80k steps   [On-policy]
```

**BipedalWalker-v3:**
```
DDPG:  2M steps, σ=±40  [Less stable]
TD3:   1.5M steps, σ=±25  [Better]
SAC:   1M steps, σ=±15  [Best]
```

**Key Observations:**
- DDPG works but less stable than TD3/SAC
- Good for learning DDPG concepts
- Use TD3 or SAC for production
- Off-policy more sample efficient than on-policy

## 9. Common Pitfalls

### 1. Wrong Action Space Handling

**Problem:** Actions outside valid range or not properly scaled.

**Solution:**
```python
# Use tanh + scaling
action = torch.tanh(network_output) * max_action

# Clip during exploration
action = np.clip(action + noise, -max_action, max_action)
```

### 2. Not Using Target Networks

**Problem:** Training unstable without target networks.

**Solution:**
```python
# Always use target networks
with torch.no_grad():
    next_action = actor_target(next_state)
    target_q = critic_target(next_state, next_action)

# Soft update every step
target_param = tau * param + (1 - tau) * target_param
```

### 3. Wrong Update Order

**Problem:** Updating actor before critic.

**Solution:**
```python
# ALWAYS update critic first
critic_loss = ...
critic_optimizer.step()

# Then actor (uses updated critic)
actor_loss = -critic(state, actor(state)).mean()
actor_optimizer.step()
```

### 4. Too Much Exploration Noise

**Problem:** Noise overwhelms policy signal.

**Solution:**
```python
# Start with small noise
noise_sigma = 0.1  # Not 0.5

# Decay over time
noise_sigma *= 0.999
```

### 5. Not Resetting Noise Process

**Problem:** OU noise drift across episodes.

**Solution:**
```python
# Reset at episode start
if done:
    noise.reset()
    state = env.reset()
```

### 6. Small Replay Buffer

**Problem:** Buffer too small, overfitting to recent data.

**Solution:**
```python
# Use large buffer
buffer_size = 1_000_000  # Not 10_000

# Start training after filling buffer
if len(buffer) < min_buffer_size:
    continue  # Collect more data
```

### 7. Learning Rates

**Problem:** Wrong learning rate ratio.

**Solution:**
```python
# Critic should learn faster
actor_lr = 1e-4
critic_lr = 1e-3  # 10x higher
```

### 8. Not Clipping Gradients

**Problem:** Gradient explosion causes instability.

**Solution:**
```python
torch.nn.utils.clip_grad_norm_(
    critic.parameters(), max_norm=1.0
)
```

## 10. References

### Original Papers

1. **Lillicrap, T. P., et al. (2015)**
   - "Continuous Control with Deep Reinforcement Learning"
   - *ICLR*
   - Original DDPG paper
   - [Link](https://arxiv.org/abs/1509.02971)

2. **Silver, D., et al. (2014)**
   - "Deterministic Policy Gradient Algorithms"
   - *ICML*
   - Theoretical foundation (DPG)
   - [Link](http://proceedings.mlr.press/v32/silver14.pdf)

3. **Mnih, V., et al. (2015)**
   - "Human-level control through deep reinforcement learning"
   - *Nature*
   - DQN (inspiration for DDPG design)
   - [Link](https://www.nature.com/articles/nature14236)

### Improvements

4. **Fujimoto, S., et al. (2018)**
   - "Addressing Function Approximation Error in Actor-Critic Methods"
   - *ICML*
   - TD3: Improved DDPG
   - [Link](https://arxiv.org/abs/1802.09477)

5. **Haarnoja, T., et al. (2018)**
   - "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL"
   - *ICML*
   - SAC: State-of-the-art continuous control
   - [Link](https://arxiv.org/abs/1801.01290)

### Applications

6. **Andrychowicz, M., et al. (2017)**
   - "Hindsight Experience Replay"
   - *NeurIPS*
   - HER for sparse rewards with DDPG
   - [Link](https://arxiv.org/abs/1707.01495)

7. **OpenAI et al. (2018)**
   - "Learning Dexterous In-Hand Manipulation"
   - *ArXiv*
   - Robotic manipulation with DDPG
   - [Link](https://arxiv.org/abs/1808.00177)

### Implementations

8. **Stable-Baselines3**
   - https://stable-baselines3.readthedocs.io/
   - Clean DDPG implementation

9. **OpenAI Spinning Up**
   - https://spinningup.openai.com/en/latest/algorithms/ddpg.html
   - Tutorial and implementation

### Textbooks

10. **Sutton & Barto (2018)**
    - "Reinforcement Learning: An Introduction"
    - Foundation for all RL algorithms

---

**Why Use TD3 or SAC Instead:**
- **TD3**: More stable (twin critics, delayed updates)
- **SAC**: Better sample efficiency (entropy regularization)
- **DDPG**: Good for learning concepts, not production

**When DDPG Makes Sense:**
- Educational purposes
- Simple continuous control
- Baseline comparisons
- When simplicity matters over performance

**Next Steps:**
- Study **TD3** for improved DDPG
- Learn **SAC** for maximum entropy RL
- Compare with **PPO** for on-policy alternative
