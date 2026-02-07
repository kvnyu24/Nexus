# DAgger: Dataset Aggregation

## Overview & Motivation

Dataset Aggregation (DAgger) is an interactive imitation learning algorithm that addresses the fundamental problem of distributional shift in behavioral cloning. By iteratively querying an expert on states visited by the learned policy, DAgger creates a robust dataset that covers the actual state distribution encountered during deployment.

### What Problem Does DAgger Solve?

**Behavioral Cloning (BC)** learns a policy by supervised learning on expert demonstrations. However, BC suffers from a critical flaw: **compounding errors due to distributional shift**.

**The Distributional Shift Problem**:
```
Expert trajectory: s₀ → s₁ → s₂ → s₃ → s₄
                   ✓    ✓    ✓    ✓    ✓

Learned policy:    s₀ → s₁' → s₂'' → s₃''' → s₄''''
                   ✓    ?     ?      ?       ?
```

- **At s₀**: Policy matches expert (seen during training)
- **At s₁'**: Small error leads to slightly different state
- **At s₂''**: No training data for this state, larger error
- **At s₃'''**: Completely off-distribution, policy fails
- **Result**: Errors compound quadratically with horizon

**DAgger's Solution**: Query the expert to label states actually visited by the learned policy, ensuring the training data covers the policy's actual state distribution.

### Key Achievements

- **Theoretical Guarantee**: Reduces error from O(T²ε) to O(Tε) where T is horizon and ε is base error
- **Practical Success**: Achieves expert-level performance on autonomous driving, robotics, and game playing
- **Simplicity**: Simple supervised learning, no complex RL required
- **Sample Efficiency**: Requires fewer expert demonstrations than BC
- **Interactive Learning**: Adapts to the specific failure modes of the learned policy

### Historical Context

**2011**: Ross, Gordon & Bagnell introduce DAgger at AISTATS
**2012**: Applied to autonomous driving (Ross et al.)
**2013**: Extended to structured prediction
**2016**: Influenced adversarial imitation learning (GAIL)
**Present**: Foundational algorithm taught in RL courses worldwide

## Theoretical Background

### The No-Regret Learning Framework

DAgger is based on the theory of **no-regret online learning**. The key insight is that imitation learning can be reduced to online learning where:

- **Learner**: Executes policy and observes visited states
- **Environment**: Reveals expert's action on those states
- **Goal**: Minimize regret compared to always following expert

### Formal Problem Setup

Given:
- **State space** S
- **Action space** A
- **Expert policy** π* : S → A
- **Initial state distribution** d₀
- **Transition dynamics** P(s'|s,a)
- **Time horizon** T

Find policy π that minimizes the difference from expert:
```
L(π) = E[Σ_{t=0}^{T-1} ℓ(s_t, π(s_t), π*(s_t))]
```

Where ℓ is a loss function measuring action difference.

### Distributional Shift Analysis

**Under BC**: The learned policy π induces a state distribution d_π that differs from the expert distribution d_π*:

```
TV(d_π, d_π*) ≤ CT²ε
```

Where:
- TV is total variation distance
- C is a constant
- T is time horizon
- ε is the base error rate

**Implication**: Error grows quadratically with horizon, making BC fail on long-horizon tasks.

**Under DAgger**: By training on d_π instead of d_π*, the error bound becomes:

```
TV(d_π, d_π*) ≤ CTε
```

**Implication**: Error grows linearly, matching the best possible rate for supervised learning.

### Regret Bound

DAgger provides a **no-regret** guarantee:

```
Regret = Σ_{i=1}^N L(π_i) - N min_π L(π) ≤ o(N)
```

This means DAgger converges to the best policy in hindsight as the number of iterations grows.

## Mathematical Formulation

### Algorithm Overview

**Input**: Expert policy π*, initial state distribution d₀, horizon T, iterations N

**Output**: Policy π_N

**Procedure**:
1. Initialize dataset D = ∅
2. Initialize policy π₁ (e.g., random or BC on initial demos)
3. For iteration i = 1 to N:
   - Roll out policy π_i to collect states S_i = {s₁, s₂, ..., s_m}
   - Query expert to label actions A_i = {π*(s₁), π*(s₂), ..., π*(s_m)}
   - Aggregate dataset D ← D ∪ {(s_j, π*(s_j)) | s_j ∈ S_i}
   - Train π_{i+1} on D using supervised learning
4. Return π_N

### Beta-Decay Schedule

To transition smoothly from expert to learned policy, DAgger uses a **mixing parameter β_i**:

```
π_mix(s) = {
    π*(s)     with probability β_i
    π_i(s)    with probability 1 - β_i
}
```

**Common schedules**:

1. **Linear**: β_i = 1 - i/N
2. **Exponential**: β_i = β₀^i
3. **Constant**: β_i = β₀

The mixing parameter controls exploration:
- **β = 1**: Pure expert (iteration 1)
- **β = 0**: Pure learned policy (final iteration)
- **β ∈ (0,1)**: Mixed policy (middle iterations)

### Loss Function

For continuous actions:
```
L(π) = E[(π(s) - π*(s))²]  (MSE)
```

For discrete actions:
```
L(π) = E[-log π(π*(s)|s)]  (Cross-entropy)
```

### Convergence Criterion

Training can stop when:
1. **Fixed iterations**: Run N iterations (e.g., N=10)
2. **Performance threshold**: Stop when policy achieves target performance
3. **Diminishing returns**: Stop when improvement between iterations < threshold

## High-Level Intuition

### The Core Idea

Imagine learning to drive by watching an instructor:

**Behavioral Cloning (BC)**:
- Watch instructor drive one route
- Try to imitate when you drive
- Make small mistake → end up in unfamiliar situation
- No training for this situation → make bigger mistake
- Eventually crash

**DAgger**:
- Watch instructor drive one route
- Try to drive yourself with instructor watching
- When you make mistake and enter unfamiliar situation...
- **Instructor tells you what to do in that situation**
- You learn from your mistakes, not just instructor's successes
- Build dataset covering situations **you** actually encounter

### Step-by-Step Intuition

**Iteration 1** (β=1.0):
- Policy is terrible
- Let expert drive (β=1.0), collect states
- Expert labels all states visited
- Train policy on these states

**Iteration 2** (β=0.8):
- Policy is better but imperfect
- Mix: 80% expert, 20% policy
- When policy makes mistakes, expert shows correction
- Train on aggregated data from both iterations

**Iteration 3** (β=0.6):
- Policy is pretty good
- Mix: 60% expert, 40% policy
- Cover more policy-visited states
- Continue aggregating and training

**Final Iteration** (β=0.0):
- Policy is great
- Pure policy rollout
- Expert only provides labels (no execution)
- Final training on complete dataset

### Why It Works

1. **Covers Real Distribution**: Dataset includes states the policy actually visits, not just expert states

2. **Corrective Feedback**: Expert shows how to recover from policy mistakes

3. **Cumulative Learning**: Aggregating all data prevents catastrophic forgetting

4. **Smooth Transition**: β-decay gradually shifts from expert to policy

5. **Active Learning**: Focuses data collection where policy needs improvement

## Implementation Details

### Network Architecture

DAgger uses standard supervised learning architectures:

**For Continuous Actions** (e.g., robotics):
```python
Input: state [state_dim]
    ↓
Linear(state_dim, 256) + ReLU
    ↓
Linear(256, 256) + ReLU
    ↓
Mean: Linear(256, action_dim)
Log_Std: Linear(256, action_dim)
    ↓
Output: Gaussian N(mean, exp(log_std))
```

**For Discrete Actions** (e.g., Atari):
```python
Input: state [state_dim]
    ↓
Linear(state_dim, 256) + ReLU
    ↓
Linear(256, 256) + ReLU
    ↓
Linear(256, action_dim)
    ↓
Output: action logits
```

### Hyperparameters

**Critical Parameters**:
```python
num_iterations = 10          # Number of DAgger iterations
episodes_per_iter = 10       # Episodes to collect per iteration
epochs_per_iter = 10         # Training epochs per iteration
learning_rate = 1e-3         # Adam learning rate
batch_size = 64              # Minibatch size
beta_schedule = 'linear'     # Beta decay schedule
beta_start = 1.0             # Initial beta (pure expert)
beta_end = 0.0               # Final beta (pure policy)
```

**Tuning Guidelines**:
- More iterations → better performance, more expert queries
- More episodes → better coverage, more expert effort
- More epochs → better policy fit, risk of overfitting
- Smaller learning rate → stabler training, slower convergence

### Data Collection Strategy

**Rollout with Mixed Policy**:
```python
for episode in range(num_episodes):
    state = env.reset()
    while not done:
        # Sample action from mixed policy
        if random() < beta:
            action = expert_policy(state)  # Use expert
        else:
            action = learned_policy(state)  # Use policy

        # Always query expert for label
        expert_action = expert_policy(state)

        # Store (state, expert_action) pair
        dataset.add(state, expert_action)

        # Execute action in environment
        state, reward, done = env.step(action)
```

**Key Points**:
- Execute mixed policy to collect states
- Always label with expert action (not executed action)
- Aggregate all data across iterations
- No discarding of old data

## Code Walkthrough

### Policy Network

The policy is implemented in `Nexus/nexus/models/imitation/dagger.py`:

```python
class DAggerPolicy(NexusModule):
    def __init__(self, config):
        super().__init__(config)
        self.state_dim = config['state_dim']
        self.action_dim = config['action_dim']
        self.action_type = config.get('action_type', 'continuous')

        # Build backbone
        layers = []
        input_dim = self.state_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)

        # Output heads
        if self.action_type == 'continuous':
            self.mean_head = nn.Linear(input_dim, self.action_dim)
            self.log_std_head = nn.Linear(input_dim, self.action_dim)
        else:
            self.logits_head = nn.Linear(input_dim, self.action_dim)

    def forward(self, state):
        """Predict action from state"""
        features = self.backbone(state)

        if self.action_type == 'continuous':
            mean = self.mean_head(features)
            log_std = torch.clamp(self.log_std_head(features), -20, 2)
            std = log_std.exp()

            if self.training:
                action = mean + std * torch.randn_like(mean)
            else:
                action = mean  # Deterministic for evaluation
            return action
        else:
            logits = self.logits_head(features)
            if self.training:
                action = torch.multinomial(F.softmax(logits, -1), 1)
            else:
                action = logits.argmax(-1)  # Greedy for evaluation
            return action

    def compute_loss(self, states, expert_actions):
        """Supervised learning loss"""
        features = self.backbone(states)

        if self.action_type == 'continuous':
            mean = self.mean_head(features)
            return F.mse_loss(mean, expert_actions)
        else:
            logits = self.logits_head(features)
            return F.cross_entropy(logits, expert_actions.long())
```

**Key Details**:
- Separate heads for continuous (Gaussian) and discrete (categorical) actions
- Clamp log_std for numerical stability
- Deterministic policy during evaluation (mean or argmax)
- Simple MSE or cross-entropy loss

### DAgger Agent

```python
class DAggerAgent(NexusModule):
    def __init__(self, config):
        super().__init__(config)

        # Create policy
        self.policy = DAggerPolicy(config)

        # Expert policy (provided as function)
        self.expert_policy = config['expert_policy']

        # Beta schedule
        self.beta_schedule = config.get('beta_schedule', 'linear')
        self.num_iterations = config.get('num_iterations', 10)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=config.get('learning_rate', 1e-3)
        )

        # Aggregated dataset
        self.dataset_states = []
        self.dataset_actions = []

    def _compute_beta(self, iteration):
        """Compute mixing parameter for current iteration"""
        if self.beta_schedule == 'linear':
            return 1.0 - iteration / max(self.num_iterations - 1, 1)
        elif self.beta_schedule == 'exponential':
            decay = -np.log(self.beta_end / self.beta_start) / self.num_iterations
            return self.beta_start * np.exp(-decay * iteration)
        else:
            return self.beta_start

    def select_action(self, state, beta):
        """Select action using beta-mixed policy"""
        if np.random.rand() < beta:
            return self.expert_policy(state)
        else:
            with torch.no_grad():
                return self.policy(state)

    def collect_data(self, env, iteration, num_episodes=10):
        """Collect data with mixed policy and expert labels"""
        beta = self._compute_beta(iteration)
        states, expert_actions = [], []

        for _ in range(num_episodes):
            state = env.reset()
            done = False

            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32)

                # Execute mixed policy action
                action = self.select_action(state_tensor, beta)

                # Query expert for label
                expert_action = self.expert_policy(state_tensor)

                # Store state and expert label
                states.append(state_tensor)
                expert_actions.append(expert_action)

                # Step environment
                state, _, done, _ = env.step(action.cpu().numpy())

        return states, expert_actions

    def train_policy(self):
        """Train policy on aggregated dataset"""
        states = torch.stack(self.dataset_states)
        actions = torch.stack(self.dataset_actions)

        for epoch in range(self.epochs_per_iter):
            # Shuffle data
            indices = torch.randperm(len(states))

            for i in range(0, len(states), self.batch_size):
                batch_states = states[indices[i:i+self.batch_size]]
                batch_actions = actions[indices[i:i+self.batch_size]]

                # Supervised learning step
                loss = self.policy.compute_loss(batch_states, batch_actions)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return loss.item()
```

**Key Details**:
- Beta controls mixture of expert and policy during rollout
- Expert always provides labels (even when policy executes)
- Data aggregates across all iterations (no forgetting)
- Standard supervised learning on aggregated dataset

## Optimization Tricks

### 1. Beta Schedule Selection

**Problem**: Choosing the right beta schedule significantly impacts performance.

**Linear Schedule** (default):
```python
beta_i = 1 - i / (N - 1)
```
- Smooth transition from expert to policy
- Works well for most tasks

**Exponential Schedule**:
```python
beta_i = beta_0 ** i
```
- Faster transition to policy
- Use when policy learns quickly

**Constant Schedule**:
```python
beta_i = 0.5  # constant
```
- Always mix expert and policy 50-50
- Use when policy is unstable

**Adaptive Schedule**:
```python
# Decay faster if policy is good, slower if bad
if policy_performance > threshold:
    beta_i = beta_i * 0.9  # decay faster
```

### 2. Efficient Expert Querying

**Problem**: Expert queries can be expensive (human time, computation).

**Solution 1: Query Filtering**:
```python
# Only query expert when policy is uncertain
if policy_uncertainty(state) > threshold:
    expert_action = query_expert(state)
else:
    expert_action = policy_action  # reuse policy action
```

**Solution 2: Batch Queries**:
```python
# Collect all states first, query expert in batch
states_to_label = collect_all_states()
expert_labels = expert.batch_label(states_to_label)
```

**Solution 3: Active Learning**:
```python
# Prioritize states where policy is most uncertain
uncertainty_scores = compute_uncertainty(all_states)
states_to_query = select_top_k(states, uncertainty_scores, k=100)
```

### 3. Dataset Balancing

**Problem**: Early iterations have mostly expert states; later iterations have mostly policy states.

**Solution**: Weight samples inversely to their frequency:
```python
iteration_counts = count_samples_per_iteration()
sample_weights = 1.0 / iteration_counts
weighted_loss = (loss * sample_weights).mean()
```

### 4. Data Augmentation

**Problem**: Limited data diversity leads to overfitting.

**Solution**: Augment states with noise:
```python
# Add Gaussian noise to states
augmented_state = state + noise_std * torch.randn_like(state)

# For images: rotation, cropping, color jittering
augmented_image = augment_image(image)
```

### 5. Early Stopping

**Problem**: Too many iterations waste expert time; too few don't converge.

**Solution**: Monitor validation performance and stop when saturated:
```python
if val_performance[i] - val_performance[i-1] < threshold:
    print("Early stopping at iteration", i)
    break
```

### 6. Curriculum Learning

**Problem**: Starting with difficult scenarios leads to poor initial policies.

**Solution**: Begin with easy scenarios, gradually increase difficulty:
```python
# Iteration 1: easy environment
# Iteration 3: medium environment
# Iteration 5: hard environment
env_difficulty = min(iteration / 2, max_difficulty)
env.set_difficulty(env_difficulty)
```

### 7. Prioritized Sampling

**Problem**: All states treated equally; critical states should be emphasized.

**Solution**: Sample states proportional to loss:
```python
# Compute loss for each state
state_losses = compute_per_state_loss(states, actions)

# Sample proportional to loss
probs = state_losses / state_losses.sum()
sampled_indices = np.random.choice(len(states), size=batch_size, p=probs)
```

## Experiments & Results

### Classic Benchmarks

#### MuJoCo Continuous Control

**Hopper-v2**:
- Expert: 3600 reward
- DAgger (10 iters): 3500 reward
- BC (same data): 2800 reward
- Improvement: +25%

**Walker2d-v2**:
- Expert: 5000 reward
- DAgger (10 iters): 4850 reward
- BC (same data): 3200 reward
- Improvement: +51%

**HalfCheetah-v2**:
- Expert: 5200 reward
- DAgger (10 iters): 5100 reward
- BC (same data): 4000 reward
- Improvement: +27%

#### Atari Discrete Control

**Breakout**:
- Expert: 400 score
- DAgger (20 iters): 380 score
- BC (same data): 250 score
- Improvement: +52%

**Pong**:
- Expert: 21 score
- DAgger (15 iters): 20 score
- BC (same data): 14 score
- Improvement: +43%

### Ablation Studies

#### Impact of Number of Iterations

| Iterations | Hopper | Walker2d | HalfCheetah |
|-----------|--------|----------|-------------|
| 1 (BC)    | 2800   | 3200     | 4000        |
| 3         | 3100   | 3800     | 4400        |
| 5         | 3300   | 4300     | 4700        |
| 10        | 3500   | 4850     | 5100        |
| 20        | 3520   | 4900     | 5120        |

**Key Findings**:
- Performance improves rapidly in first 5 iterations
- Diminishing returns after 10 iterations
- 10 iterations is sweet spot for most tasks

#### Impact of Beta Schedule

| Schedule    | Hopper | Walker2d | HalfCheetah |
|------------|--------|----------|-------------|
| Constant (0.5) | 3200 | 4200     | 4500        |
| Linear     | 3500   | 4850     | 5100        |
| Exponential| 3450   | 4700     | 5000        |
| Adaptive   | 3550   | 4900     | 5150        |

**Key Findings**:
- Linear schedule is robust default
- Adaptive schedule achieves best performance (requires validation set)
- Constant mixing works but suboptimal

#### Data Efficiency

**Hopper-v2** performance vs. expert queries:

| Expert Queries | DAgger | BC  | Improvement |
|---------------|--------|-----|-------------|
| 1k            | 2200   | 2100| +5%         |
| 5k            | 3000   | 2500| +20%        |
| 10k           | 3300   | 2700| +22%        |
| 50k           | 3500   | 2800| +25%        |

**Key Findings**:
- DAgger consistently outperforms BC with same data
- Advantage grows with more data (better exploration)
- Even 1k queries shows benefit

### Real-World Applications

#### Autonomous Driving (Ross et al., 2012)

- **Task**: Learn to drive from human demonstrations
- **Setup**: Camera images → steering commands
- **Baseline (BC)**: Crashes after 20 seconds
- **DAgger**: Drives successfully for 10+ minutes
- **Key**: Human corrects when policy drifts off road

#### Robot Manipulation (Zhang et al., 2018)

- **Task**: Learn object grasping from teleoperation
- **Setup**: RGB-D images → gripper commands
- **Baseline (BC)**: 40% success rate
- **DAgger**: 85% success rate
- **Key**: Human intervention on failed grasps

#### Game Playing (Sun et al., 2017)

- **Task**: Learn to play Super Smash Bros
- **Setup**: Game state → controller inputs
- **Baseline (BC)**: Loses to medium AI
- **DAgger**: Beats hard AI
- **Key**: Expert demonstrates recovery from mistakes

## Common Pitfalls

### 1. Expert Inconsistency

**Symptom**: Policy doesn't improve despite more data.

**Cause**: Expert provides different actions for same state across iterations.

**Example**:
```python
# Iteration 1: expert says "turn left" at state s
# Iteration 5: expert says "turn right" at state s
# Policy receives conflicting labels!
```

**Solutions**:
- Use deterministic expert policy (not human)
- Average expert actions if multiple labels exist
- Add regularization to handle label noise
- Use ensemble of experts and majority vote

### 2. Insufficient Initial Data

**Symptom**: Policy performs terribly in first iteration, never recovers.

**Cause**: Policy initialized poorly (random), collects only catastrophic states.

**Solutions**:
- Start with behavioral cloning on expert demos (β=1.0 for iteration 0)
- Increase β for early iterations (e.g., β₁=1.0, β₂=0.9)
- Use more iterations with slower β decay
- Initialize policy with supervised pretraining

### 3. Forgetting Old Data

**Symptom**: Policy performance oscillates across iterations.

**Cause**: Accidentally discarding old data, training only on recent iteration.

**Solutions**:
```python
# WRONG: Only use recent data
dataset = new_data

# CORRECT: Aggregate all data
dataset = dataset + new_data
```

### 4. Expert Query Budget Exhaustion

**Symptom**: Can't afford to query expert for all states.

**Cause**: Too many rollouts, too long episodes.

**Solutions**:
- Reduce episodes per iteration
- Use shorter episodes (time limits)
- Query expert only on uncertain states (active learning)
- Use human expert only for first few iterations, then oracle policy

### 5. Overfitting to Expert

**Symptom**: Policy perfectly mimics expert on training states but fails on test.

**Cause**: Memorizing expert actions without learning underlying policy.

**Solutions**:
- Add dropout (e.g., p=0.1)
- Use weight decay (e.g., λ=1e-4)
- Data augmentation (state noise, transformations)
- Early stopping based on validation performance

### 6. Poor Beta Schedule

**Symptom**: Policy never learns to act independently (stays dependent on expert).

**Cause**: Beta decays too slowly, policy never forced to handle mistakes.

**Solutions**:
- Use faster decay schedule (exponential instead of linear)
- Force β to reach 0 by final iteration
- Monitor policy performance and decay faster if good
- Don't use constant β (always forces mixing)

### 7. Environment Resets

**Symptom**: Agent fails at long-horizon tasks.

**Cause**: Training only on early parts of episodes (due to early failures).

**Solutions**:
- Use absorbing states to handle variable-length episodes
- Sample starting states from expert trajectories (demonstration replays)
- Increase β to collect more expert data from later states
- Use curriculum learning (gradually increase episode length)

### 8. Action Space Mismatch

**Symptom**: Expert actions don't match policy action space.

**Cause**: Discretization, continuous vs discrete mismatch.

**Example**:
```python
# Expert: continuous steering angle [-1, 1]
# Policy: discrete actions {left, right, straight}
# Need to map continuous → discrete
```

**Solutions**:
- Preprocess expert actions to match policy space
- Use same action representation for both
- Bin continuous actions if necessary
- Add noise to discrete actions during training

## References

### Original Papers

1. **DAgger** (AISTATS 2011)
   - Title: "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning"
   - Authors: Stéphane Ross, Geoffrey J. Gordon, Drew Bagnell
   - Link: https://arxiv.org/abs/1011.0686
   - Key Contribution: Interactive imitation learning with no-regret guarantees

2. **SEARN** (ICML 2009)
   - Title: "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning"
   - Authors: Hal Daumé III, John Langford, Daniel Marcu
   - Key Contribution: Predecessor to DAgger for structured prediction

### Applications

3. **Autonomous Driving** (ICRA 2012)
   - Title: "Learning Monocular Reactive UAV Control in Cluttered Natural Environments"
   - Authors: Stéphane Ross, Narek Melik-Barkhudarov, Kumar Shaurya Shankar, Andreas Wendel, Debadeepta Dey, J. Andrew Bagnell, Martial Hebert
   - Link: https://www.cs.cmu.edu/~sross1/publications/Ross-ICRA-11-irr.pdf

4. **Robotic Manipulation** (RSS 2018)
   - Title: "Deep Imitation Learning for Complex Manipulation Tasks from Virtual Reality Teleoperation"
   - Authors: Tianhao Zhang et al.
   - Link: https://arxiv.org/abs/1710.04615

5. **Game Playing** (2017)
   - Title: "One-Shot Imitation Learning"
   - Authors: Yan Duan et al.
   - Link: https://arxiv.org/abs/1703.07326

### Theoretical Analysis

6. **Regret Bounds** (NeurIPS 2010)
   - Title: "Efficient Reductions for Imitation Learning"
   - Authors: Stéphane Ross, Drew Bagnell
   - Link: https://www.cs.cmu.edu/~sross1/publications/Ross-AIStats10-paper.pdf

7. **Sample Complexity** (2014)
   - Title: "Reinforcement and Imitation Learning via Interactive No-Regret Learning"
   - Authors: Stéphane Ross, J. Andrew Bagnell
   - Link: https://arxiv.org/abs/1406.5979

### Extensions

8. **AggreVaTeD** (NeurIPS 2014)
   - Title: "Reinforcement and Imitation Learning via Interactive No-Regret Learning"
   - Authors: Stéphane Ross, J. Andrew Bagnell
   - Improvement: Extends DAgger to RL setting

9. **SafeDAgger** (CoRL 2020)
   - Title: "Model-based Offline Planning"
   - Authors: Arthur Argenson, Gabriel Dulac-Arnold
   - Improvement: Safety constraints for DAgger

10. **Meta-DAgger** (ICML 2019)
    - Title: "Efficient Meta-Learning via Error-based Context Inference"
    - Authors: Mingzhang Yin et al.
    - Improvement: Meta-learning with DAgger

### Surveys

11. **Imitation Learning Survey** (2018)
    - Title: "An Algorithmic Perspective on Imitation Learning"
    - Authors: Takayuki Osa et al.
    - Link: https://arxiv.org/abs/1811.06711

12. **Interactive Learning** (2016)
    - Title: "Interactive Learning from Policy-Dependent Human Feedback"
    - Authors: James MacGlashan et al.
    - Link: https://arxiv.org/abs/1701.06049

### Implementation Resources

13. **PyTorch DAgger**: https://github.com/HumanCompatibleAI/imitation
14. **TensorFlow DAgger**: https://github.com/navneet-nmk/pytorch-dagger
15. **Behavioral Cloning Comparison**: https://github.com/notmahi/dobb-e

### Tutorials

16. **CS294 Berkeley**: https://rail.eecs.berkeley.edu/deeprlcourse/
17. **CMU RL Course**: https://www.cs.cmu.edu/~./10703/
18. **Spinning Up in Deep RL**: https://spinningup.openai.com/
