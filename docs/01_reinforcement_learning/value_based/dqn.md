# Deep Q-Network (DQN)

## Overview & Motivation

Deep Q-Network (DQN) revolutionized reinforcement learning by demonstrating that deep neural networks could successfully learn control policies directly from high-dimensional sensory inputs. Before DQN, Q-learning was limited to small, discrete state spaces using tabular representations.

### What Problem Does DQN Solve?

Traditional Q-learning maintains a table Q(s,a) for every state-action pair. This becomes infeasible when:
- State space is continuous (e.g., robot positions)
- State space is high-dimensional (e.g., images with millions of pixels)
- State space is too large to enumerate (e.g., chess with ~10^43 positions)

DQN solves this by using a neural network as a function approximator to estimate Q-values for any state, enabling RL to scale to complex domains.

### Key Achievements

- First deep RL algorithm to learn directly from pixels
- Achieved human-level performance on 29 Atari games
- Single architecture worked across diverse games without hand-crafted features
- Published in Nature (2015), marking a watershed moment for deep RL

## Theoretical Background

### Q-Learning Foundation

Q-learning is an off-policy temporal difference (TD) learning algorithm that learns the optimal action-value function Q*(s,a), representing the expected return starting from state s, taking action a, and following the optimal policy thereafter.

The optimal Q-function satisfies the Bellman optimality equation:

```
Q*(s,a) = E[r + γ max_a' Q*(s',a') | s,a]
```

Traditional Q-learning uses the following update rule:

```
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
```

### From Tabular to Deep Q-Learning

DQN parameterizes Q(s,a) with a neural network Q(s,a;θ), where θ represents the network weights. The network is trained by minimizing the loss:

```
L(θ) = E[(r + γ max_a' Q(s',a';θ^-) - Q(s,a;θ))^2]
```

Where θ^- represents the parameters of a separate target network.

### Historical Context

**1989**: Watkins introduces Q-learning in his PhD thesis
**1992**: Watkins & Dayan prove convergence for tabular Q-learning
**1993**: Tesauro's TD-Gammon uses neural networks for backgammon
**2013**: Mnih et al. publish DQN preprint on arXiv
**2015**: DQN Nature paper demonstrates human-level Atari performance

## Mathematical Formulation

### Q-Function Approximation

The Q-network is a function approximator:

```
Q(s,a;θ) : S × A → R
```

Where:
- S is the state space
- A is the action space
- θ are the neural network parameters

### Loss Function

DQN minimizes the temporal difference (TD) error using the following loss:

```
L(θ) = E_{(s,a,r,s')~D}[(y_DQN - Q(s,a;θ))^2]
```

Where the target is:

```
y_DQN = r + γ max_a' Q(s',a';θ^-)
```

Key components:
- **D**: Replay buffer containing past transitions
- **θ**: Current network parameters (online network)
- **θ^-**: Target network parameters (updated periodically)
- **γ**: Discount factor (typically 0.99)

### Gradient Update

The gradient of the loss with respect to θ is:

```
∇_θ L(θ) = E[(r + γ max_a' Q(s',a';θ^-) - Q(s,a;θ)) ∇_θ Q(s,a;θ)]
```

This gradient is used with standard optimizers (Adam, RMSprop) to update the network.

## High-Level Intuition

### The Core Idea

Think of Q-values as "quality scores" for actions. If you know the quality of every action in every situation, you can act optimally by always choosing the highest-quality action.

### How DQN Works (Simple Analogy)

Imagine learning to play video games:

1. **Try Actions**: Press buttons (explore) to see what happens
2. **Remember Experiences**: Store what happened (state, action, reward, next state)
3. **Learn Patterns**: Notice that certain actions in certain situations lead to good outcomes
4. **Update Strategy**: Adjust your understanding of which actions are good
5. **Repeat**: Keep playing, learning, and improving

### Key Insights

1. **Function Approximation**: Instead of storing Q-values for every state-action pair (impossible for images), use a neural network to predict Q-values for any state

2. **Experience Replay**: Store past experiences and randomly sample them for training. This breaks temporal correlations and improves sample efficiency

3. **Target Network**: Use a slowly-changing copy of the Q-network to compute targets. This stabilizes training by preventing a moving target problem

4. **ε-Greedy Exploration**: Balance exploration (trying random actions) and exploitation (using current knowledge) by occasionally taking random actions

## Implementation Details

### Network Architecture

The standard DQN architecture for Atari:

```
Input (84x84x4 frame stack)
    ↓
Conv Layer 1: 32 filters, 8x8, stride 4, ReLU
    ↓
Conv Layer 2: 64 filters, 4x4, stride 2, ReLU
    ↓
Conv Layer 3: 64 filters, 3x3, stride 1, ReLU
    ↓
Fully Connected: 512 units, ReLU
    ↓
Output: |A| units (one per action)
```

For simple state spaces (e.g., CartPole), a simple MLP suffices:

```
Input (state_dim)
    ↓
FC Layer 1: hidden_dim units, ReLU
    ↓
FC Layer 2: hidden_dim units, ReLU
    ↓
Output: action_dim units
```

### Hyperparameters

Standard hyperparameters for Atari games:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning rate | 0.00025 | Adam optimizer |
| Discount (γ) | 0.99 | Future reward discount |
| Replay buffer size | 1,000,000 | Transitions stored |
| Batch size | 32 | Mini-batch size |
| Target update freq | 10,000 | Steps between target updates |
| ε start | 1.0 | Initial exploration |
| ε end | 0.1 | Final exploration |
| ε decay steps | 1,000,000 | Annealing schedule |
| Frame skip | 4 | Actions repeated |
| Frame stack | 4 | Frames concatenated |

For simpler environments (CartPole, LunarLander):

| Parameter | Value |
|-----------|-------|
| Learning rate | 0.001 |
| Hidden dim | 128 |
| Replay buffer | 10,000 |
| Batch size | 64 |
| Target update | 10 episodes |
| ε decay | 0.995 per episode |

### Training Loop

The DQN training procedure:

```
1. Initialize replay buffer D
2. Initialize Q-network with random weights θ
3. Initialize target network θ^- = θ

For episode = 1 to M:
    Initialize state s_0

    For t = 0 to T:
        # Select action
        With probability ε: select random action a_t
        Otherwise: a_t = argmax_a Q(s_t, a; θ)

        # Execute action
        Execute a_t, observe reward r_t and next state s_{t+1}

        # Store transition
        Store (s_t, a_t, r_t, s_{t+1}) in D

        # Train
        Sample mini-batch from D
        Compute targets: y_i = r_i + γ max_a' Q(s'_i, a'; θ^-)
        Update θ by minimizing (y_i - Q(s_i, a_i; θ))^2

        # Update target network
        Every C steps: θ^- ← θ

        # Decay exploration
        ε ← max(ε_min, ε * decay)
```

## Code Walkthrough

### Nexus Implementation

Location: `/Users/kevinyu/Projects/Nexus/nexus/models/rl/dqn/dqn.py`

#### Network Definition (Lines 8-20)

```python
class DQNNetwork(NexusModule):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
```

Simple 3-layer MLP that outputs Q-values for all actions. The output layer has no activation, allowing Q-values to be any real number.

#### Agent Initialization (Lines 22-40)

```python
class DQNAgent(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Hyperparameters
        self.gamma = config.get("gamma", 0.99)
        self.epsilon = config.get("epsilon", 1.0)
        self.epsilon_decay = config.get("epsilon_decay", 0.995)

        # Networks
        self.q_network = DQNNetwork(...)
        self.target_network = DQNNetwork(...)
        self.target_network.load_state_dict(self.q_network.state_dict())
```

Creates two identical networks:
- **q_network**: The online network, updated every step
- **target_network**: The target network, updated periodically

#### Action Selection (Lines 42-49)

```python
def select_action(self, state: np.ndarray, training: bool = True) -> int:
    if training and np.random.random() < self.epsilon:
        return np.random.randint(self.action_dim)  # Explore

    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()  # Exploit
```

ε-greedy policy:
- With probability ε: random action (exploration)
- With probability 1-ε: action with highest Q-value (exploitation)

#### Update Function (Lines 51-76)

```python
def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
    states = batch["states"]
    actions = batch["actions"]
    rewards = batch["rewards"]
    next_states = batch["next_states"]
    dones = batch["dones"]

    # Compute current Q values (predictions)
    current_q = self.q_network(states).gather(1, actions.unsqueeze(1))

    # Compute target Q values
    with torch.no_grad():
        next_q = self.target_network(next_states).max(1)[0].unsqueeze(1)
        target_q = rewards.unsqueeze(1) + self.gamma * next_q * (1 - dones.unsqueeze(1))

    # Compute loss and update
    loss = F.smooth_l1_loss(current_q, target_q)

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
```

Key steps:
1. **Current Q-values**: Forward pass through online network
2. **Target Q-values**: Use target network with Bellman equation
3. **Loss**: Smooth L1 loss (Huber loss) for robustness
4. **Update**: Standard gradient descent

Note: `gather(1, actions.unsqueeze(1))` selects the Q-value for the action that was actually taken.

#### Target Network Update (Lines 78-79)

```python
def update_target_network(self):
    self.target_network.load_state_dict(self.q_network.state_dict())
```

Hard update: Completely copy weights from online to target network. Called every N episodes or steps.

### Complete Training Example

Location: `/Users/kevinyu/Projects/Nexus/examples/rl/train_dqn.py`

```python
import gym
from nexus.models.rl.dqn import DQNAgent
from nexus.data.replay_buffer import ReplayBuffer

# Setup
env = gym.make('CartPole-v1')
agent = DQNAgent(config)
replay_buffer = ReplayBuffer(capacity=10000)

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0

    while not done:
        # Act
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)

        # Store
        replay_buffer.push(state, action, reward, next_state, done)

        # Learn
        if len(replay_buffer) >= batch_size:
            batch = replay_buffer.sample(batch_size)
            metrics = agent.update(batch)

    # Update target
    if episode % target_update_frequency == 0:
        agent.update_target_network()
```

## Optimization Tricks

### 1. Smooth L1 Loss (Huber Loss)

Use Huber loss instead of MSE for more robust training:

```python
loss = F.smooth_l1_loss(current_q, target_q)
```

This is less sensitive to outliers than MSE.

### 2. Gradient Clipping

Clip gradients to prevent exploding gradients:

```python
torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
```

### 3. Reward Clipping

For Atari, clip rewards to [-1, 1] to normalize across games:

```python
reward = np.clip(reward, -1, 1)
```

### 4. Frame Stacking

Stack 4 consecutive frames to provide temporal information:

```python
# Stack last 4 frames along channel dimension
state = np.concatenate([frame_t-3, frame_t-2, frame_t-1, frame_t], axis=0)
```

### 5. Frame Skipping

Repeat each action for 4 frames to reduce computation:

```python
total_reward = 0
for _ in range(frame_skip):
    next_state, reward, done, info = env.step(action)
    total_reward += reward
    if done:
        break
```

### 6. Learning Rate Scheduling

Use a learning rate scheduler for better convergence:

```python
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100000, gamma=0.5)
```

### 7. Replay Buffer Size

Larger buffers improve stability but use more memory:
- Atari: 1M transitions
- Simple tasks: 10K-100K transitions

### 8. Target Network Update Frequency

Balance stability and learning speed:
- Too frequent: unstable (moving target)
- Too rare: slow learning
- Typical: Every 1K-10K steps

## Experiments & Results

### CartPole-v1

**Task**: Balance a pole on a cart by moving left or right
**State**: 4D continuous (position, velocity, angle, angular velocity)
**Actions**: 2 discrete (left, right)
**Success**: Average reward > 195 over 100 episodes

**Expected Performance**:
- Convergence: ~100-200 episodes
- Final performance: 195-500 reward
- Training time: ~5-10 minutes on CPU

**Typical Learning Curve**:
```
Episodes 0-50:    Random performance (~20-30 reward)
Episodes 50-100:  Rapid improvement (30-150 reward)
Episodes 100-200: Near-optimal (150-500 reward)
```

### LunarLander-v2

**Task**: Land a lunar module safely between flags
**State**: 8D continuous
**Actions**: 4 discrete (nothing, fire left, fire main, fire right)
**Success**: Average reward > 200

**Expected Performance**:
- Convergence: ~300-500 episodes
- Final performance: 200-250 reward
- Training time: ~20-30 minutes on CPU

### Atari Games

**DQN Performance** (from Nature paper):

| Game | Random | Human | DQN |
|------|--------|-------|-----|
| Breakout | 1.7 | 30.5 | 401.2 |
| Pong | -20.7 | 14.6 | 18.9 |
| Space Invaders | 148.0 | 1652.0 | 1976.0 |
| Q*bert | 163.9 | 13455.0 | 10596.0 |
| Seaquest | 68.4 | 42055.0 | 5286.0 |

**Training Requirements**:
- Steps: 50M frames (200M with frame skip)
- Time: ~1 week on GPU
- Performance: Human-level on ~half of games

### Hyperparameter Sensitivity

**Most Important** (in order):
1. Learning rate (most critical)
2. Replay buffer size
3. Target network update frequency
4. Discount factor (γ)
5. Network architecture

**Less Important**:
- Batch size (32-64 works well)
- Epsilon decay schedule (as long as it's gradual)

## Common Pitfalls

### 1. Overestimation Bias

**Problem**: DQN consistently overestimates Q-values due to max operator in the target.

**Symptom**: Q-values grow unboundedly, performance degrades

**Solution**: Use Double DQN (see [double_dqn.md](double_dqn.md))

### 2. Catastrophic Forgetting

**Problem**: Network forgets previously learned policies when learning new experiences.

**Symptom**: Performance suddenly drops after good results

**Solutions**:
- Increase replay buffer size
- Lower learning rate
- Use more stable architectures (Dueling DQN)

### 3. Deadly Triad

**Problem**: The combination of function approximation + bootstrapping + off-policy learning can diverge.

**Symptoms**: NaN losses, exploding Q-values, training instability

**Solutions**:
- Gradient clipping
- Target networks (already in DQN)
- Smaller learning rates
- Regularization

### 4. Insufficient Exploration

**Problem**: Agent converges to suboptimal policy too quickly.

**Symptoms**: Performance plateaus early, never improves

**Solutions**:
- Longer epsilon decay (1M steps)
- Higher final epsilon (0.1 instead of 0.01)
- Use noisy networks (Rainbow)

### 5. Sparse Rewards

**Problem**: Reward signal is too infrequent for learning.

**Symptoms**: Random behavior persists, no learning signal

**Solutions**:
- Reward shaping (carefully!)
- Curriculum learning
- Hindsight experience replay
- Intrinsic motivation (ICM, RND)

### 6. Correlated Samples

**Problem**: Training on sequential experiences violates i.i.d. assumption.

**Symptoms**: High variance, unstable training

**Solutions**:
- Experience replay (already in DQN)
- Larger batch sizes
- Multiple parallel environments

### 7. Moving Target

**Problem**: Target values change as Q-network is updated.

**Symptoms**: Oscillating Q-values, unstable training

**Solutions**:
- Target network (already in DQN)
- Longer update intervals
- Lower learning rate

### 8. Wrong Loss Function

**Problem**: Using MSE with large outliers causes instability.

**Solution**: Use Huber loss (smooth_l1_loss) - already default in Nexus

## Debugging Tips

### Check Q-values

```python
with torch.no_grad():
    q_values = agent.q_network(state)
    print(f"Q-values: {q_values}")
    print(f"Mean Q: {q_values.mean():.2f}, Max Q: {q_values.max():.2f}")
```

Good signs:
- Q-values in reasonable range (not exploding)
- Q-values increase over training
- Spread between max and mean Q-values

### Monitor Epsilon

```python
print(f"Epsilon: {agent.epsilon:.4f}")
```

Ensure epsilon is decaying gradually, not too fast.

### Check Loss

```python
metrics = agent.update(batch)
print(f"Loss: {metrics['loss']:.4f}")
```

Loss should decrease over time. If increasing or NaN, you have a problem.

### Visualize Replay Buffer

```python
states, _, rewards, _, _ = zip(*replay_buffer.buffer)
print(f"Rewards: min={min(rewards):.2f}, max={max(rewards):.2f}, mean={np.mean(rewards):.2f}")
```

Ensure diverse experiences in the buffer.

### Log to TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
writer.add_scalar('Reward/episode', total_reward, episode)
writer.add_scalar('Loss/train', loss, step)
writer.add_scalar('Q_value/mean', mean_q, step)
```

## References

### Core Papers

1. **DQN (arXiv)**: [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
   Mnih et al., 2013
   Original DQN paper introducing experience replay and target networks

2. **DQN (Nature)**: [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
   Mnih et al., 2015
   Nature publication with full experimental details and results

3. **Q-Learning**: [Q-Learning](https://link.springer.com/article/10.1007/BF00992698)
   Watkins & Dayan, 1992
   Original Q-learning algorithm and convergence proof

### Blog Posts & Tutorials

- [Deep RL Course - DQN](https://huggingface.co/deep-rl-course/unit3/introduction): Comprehensive tutorial with code
- [Lil'Log - DQN](https://lilianweng.github.io/posts/2018-02-19-rl-overview/#deep-q-network): Excellent conceptual overview
- [PyTorch DQN Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html): Official PyTorch tutorial
- [OpenAI Spinning Up - DQN](https://spinningup.openai.com/en/latest/algorithms/dqn.html): Theory and implementation

### Implementations

- **Dopamine** (Google): https://github.com/google/dopamine
- **Stable-Baselines3**: https://github.com/DLR-RM/stable-baselines3
- **CleanRL**: https://github.com/vwxyzjn/cleanrl
- **RLlib** (Ray): https://docs.ray.io/en/latest/rllib/index.html

### Related Algorithms

- [Double DQN](double_dqn.md): Fixes overestimation bias
- [Dueling DQN](dueling_dqn.md): Improved architecture
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952): Better sampling
- [Rainbow](rainbow.md): Combines all improvements

### Videos

- [DeepMind's DQN Explained](https://www.youtube.com/watch?v=rFwQDDbYTm4): Visual explanation
- [Stanford CS234](https://www.youtube.com/playlist?list=PLoROMvodv4rOSOPzutgyCTapiGlY2Nd8u): Full RL course
- [David Silver's RL Course](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ): Classic lectures

## Next Steps

After mastering DQN, explore:
1. [Double DQN](double_dqn.md) - Learn to fix overestimation bias
2. [Dueling DQN](dueling_dqn.md) - Understand better architectures
3. [Rainbow](rainbow.md) - See how everything combines

For different problem types:
- Continuous actions: SAC, TD3, DDPG
- On-policy learning: PPO, A2C
- Model-based RL: Dreamer, MuZero
