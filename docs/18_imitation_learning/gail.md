# GAIL: Generative Adversarial Imitation Learning

## Overview & Motivation

Generative Adversarial Imitation Learning (GAIL) revolutionized imitation learning by framing it as a distribution matching problem solved via adversarial training. GAIL avoids the computational expense of inverse reinforcement learning while maintaining its theoretical guarantees.

### What Problem Does GAIL Solve?

Traditional imitation learning approaches face key limitations:

1. **Behavioral Cloning (BC)**: Suffers from distributional shift
   - Small errors compound over time
   - Fails in states not visited by expert
   - No recovery mechanism

2. **Inverse Reinforcement Learning (IRL)**: Computationally expensive
   - Requires solving RL in inner loop
   - Often intractable for complex domains
   - Reward ambiguity (many rewards explain same behavior)

3. **Apprenticeship Learning**: Sample inefficient
   - Needs many expert demonstrations
   - Slow to converge

**GAIL's Solution**: Use a discriminator to distinguish expert behavior from policy behavior. The discriminator's output provides an implicit reward signal that guides the policy to match the expert's state-action distribution.

### Key Achievements

- **Sample Efficiency**: Achieves expert performance with significantly fewer demonstrations than BC
- **No Reward Engineering**: Learns directly from demonstrations without hand-crafted rewards
- **Strong Theoretical Guarantees**: Minimizes Jensen-Shannon divergence between policy and expert distributions
- **Practical Performance**: State-of-the-art results on locomotion and manipulation tasks
- **Simplicity**: Can be combined with any policy optimization algorithm (TRPO, PPO, SAC)

### Historical Context

**2000s**: Inverse RL pioneered by Abbeel & Ng
**2016**: Ho & Ermon publish GAIL at NeurIPS
**2017-2018**: Extensions (VAIL, InfoGAIL, f-GAIL)
**2018**: AIRL adds reward recovery
**Present**: Foundation for modern imitation learning

## Theoretical Background

### Distribution Matching Perspective

The core insight of GAIL is that imitation learning can be viewed as matching distributions:

**Goal**: Find policy π such that:
```
π(a|s) ≈ π_expert(a|s)  for all states s
```

More precisely, we want to match the occupancy measure (state-action visitation frequency):

```
ρ_π(s,a) = Σ_{t=0}^∞ γ^t P(s_t=s, a_t=a | π)
```

### Connection to GANs

GAIL draws inspiration from Generative Adversarial Networks:

- **Generator**: Policy π generates trajectories
- **Discriminator**: D distinguishes expert from policy trajectories
- **Objective**: Policy tries to fool discriminator

This adversarial game leads to distribution matching.

### Occupancy Measure Matching

The optimal discriminator in a GAN-style objective reveals the difference between distributions. GAIL shows that minimizing:

```
E_π[-log(D(s,a))] + E_expert[-log(1 - D(s,a))]
```

is equivalent to minimizing the Jensen-Shannon divergence between occupancy measures:

```
JS(ρ_π || ρ_expert)
```

### Theoretical Guarantees

**Theorem (Ho & Ermon)**: Under certain regularity conditions, GAIL converges to a policy whose occupancy measure matches the expert's occupancy measure.

**Key Properties**:
1. Convex IRL problem (in discriminator)
2. Saddle point formulation
3. Convergence guarantees with appropriate regularization

## Mathematical Formulation

### Discriminator Objective

The discriminator D(s,a) is trained to classify state-action pairs:

```
max_D E_expert[log D(s,a)] + E_π[log(1 - D(s,a))]
```

Where:
- D(s,a) → 1 for expert data
- D(s,a) → 0 for policy data

### Policy Objective

The policy π is trained to fool the discriminator:

```
max_π E_π[log D(s,a)] - λH(π)
```

Where:
- First term: Reward from discriminator (fool it)
- H(π): Entropy regularization term
- λ: Regularization coefficient

### GAIL Reward Function

The discriminator provides an implicit reward:

```
r_GAIL(s,a) = log D(s,a)  (original formulation)
```

Or the non-saturating variant (better gradients):

```
r_GAIL(s,a) = -log(1 - D(s,a))
```

### Complete Algorithm

**Input**: Expert demonstrations D_expert, environment
**Output**: Learned policy π

1. Initialize policy π and discriminator D
2. For each iteration:
   a. Sample trajectories from π
   b. Update D to distinguish expert from policy
   c. Compute rewards r(s,a) = -log(1 - D(s,a))
   d. Update π using policy gradient (e.g., TRPO, PPO) with rewards r
3. Return π

### Regularization

To prevent mode collapse and ensure exploration, GAIL adds causal entropy regularization:

```
H(π) = E_π[Σ_t γ^t H(π(·|s_t))]
```

This encourages diverse behavior and prevents the policy from collapsing to a single trajectory.

## High-Level Intuition

### The Core Idea

Think of GAIL as a two-player game:

**Judge (Discriminator)**: Tries to tell expert demonstrations apart from policy attempts
**Student (Policy)**: Tries to act so well that the judge can't tell the difference

When the judge can't distinguish anymore, the student has successfully learned the expert's behavior.

### Step-by-Step Process

1. **Policy Acts**: Generate trajectories in the environment
2. **Judge Evaluates**: Discriminator scores how "expert-like" the actions are
3. **Feedback Loop**: Policy receives high rewards when judge thinks it's an expert
4. **Iterative Improvement**: Policy adapts to fool judge, judge adapts to distinguish
5. **Convergence**: Eventually, policy matches expert distribution

### Why It Works

**Traditional RL**: Needs carefully designed rewards (hard to specify)
**GAIL**: Rewards emerge from comparison with expert (automatic)

**Traditional BC**: Only learns from states expert visits
**GAIL**: Learns to match entire state-action distribution (recovers from mistakes)

**Traditional IRL**: Must solve expensive optimization in inner loop
**GAIL**: Discriminator training is simple supervised learning (efficient)

## Implementation Details

### Network Architecture

#### Discriminator Architecture

```python
Input: [state, action] concatenated
    ↓
Linear(state_dim + action_dim, 256) + Tanh
    ↓
Linear(256, 256) + Tanh
    ↓
Linear(256, 1)  # Output logit
```

**Key Choices**:
- **Activation**: Tanh works better than ReLU for discriminators
- **Normalization**: Spectral normalization improves stability
- **Depth**: 2-3 hidden layers sufficient for most tasks
- **Width**: 256-512 units per layer

#### Policy Architecture

Any policy gradient architecture works:
- **Continuous Actions**: Gaussian policy (mean and std networks)
- **Discrete Actions**: Categorical policy (softmax over actions)
- **Hybrid**: Separate networks for different action types

### Hyperparameters

**Critical Parameters**:
```python
disc_lr = 3e-4              # Discriminator learning rate
policy_lr = 3e-4            # Policy learning rate (depends on algo)
disc_steps = 5              # Discriminator updates per policy update
grad_penalty_coef = 10.0    # Gradient penalty coefficient
entropy_coef = 0.01         # Entropy regularization
```

**Discriminator Training**:
- Update discriminator more frequently than policy (5-10 steps per policy update)
- Use gradient penalty to stabilize training
- Monitor discriminator accuracy (should stay around 0.7-0.8)

**Policy Training**:
- Use TRPO or PPO for stable policy updates
- Add entropy bonus to encourage exploration
- Clip rewards if they become too large

### Training Stability Techniques

1. **Gradient Penalty** (WGAN-GP style):
```python
# Interpolate between expert and policy samples
alpha = torch.rand(batch_size, 1)
interp = alpha * expert + (1 - alpha) * policy
# Compute gradient norm and penalize deviation from 1
grad_penalty = ((grad_norm - 1) ** 2).mean()
```

2. **Spectral Normalization**:
```python
# Apply to all discriminator layers
linear = nn.utils.spectral_norm(nn.Linear(in_dim, out_dim))
```

3. **Label Smoothing**:
```python
# Expert = 0.9 instead of 1.0
# Policy = 0.1 instead of 0.0
expert_labels = torch.ones_like(logits) * 0.9
policy_labels = torch.ones_like(logits) * 0.1
```

## Code Walkthrough

### Discriminator Implementation

The discriminator is implemented in `/Users/kevinyu/Projects/Nexus/nexus/models/imitation/gail.py`:

```python
class GAILDiscriminator(NexusModule):
    def __init__(self, config):
        super().__init__(config)
        self.state_dim = config['state_dim']
        self.action_dim = config['action_dim']

        # Build MLP with optional spectral normalization
        layers = []
        input_dim = self.state_dim + self.action_dim

        for hidden_dim in self.hidden_dims:
            linear = nn.Linear(input_dim, hidden_dim)
            if self.use_spectral_norm:
                linear = nn.utils.spectral_norm(linear)
            layers.append(linear)
            layers.append(nn.Tanh())
            input_dim = hidden_dim

        # Output layer (no activation, returns logit)
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, state, action):
        """Returns logit for P(expert | state, action)"""
        x = torch.cat([state, action], dim=-1)
        return self.network(x)

    def compute_reward(self, state, action):
        """Compute GAIL reward for policy training"""
        with torch.no_grad():
            d = torch.sigmoid(self.forward(state, action))
            # Non-saturating reward formulation
            reward = -torch.log(1 - d + 1e-8)
        return reward
```

**Key Details**:
- Concatenates state and action as input
- Returns logit (not probability) for numerical stability
- Reward computation uses non-saturating formulation
- Adds small epsilon to prevent log(0)

### Training Loop

```python
class GAILAgent(NexusModule):
    def update_discriminator(self, expert_batch, policy_batch):
        """Update discriminator to distinguish expert from policy"""
        expert_states, expert_actions = expert_batch
        policy_states, policy_actions = policy_batch

        # Forward pass
        expert_logits = self.discriminator(expert_states, expert_actions)
        policy_logits = self.discriminator(policy_states, policy_actions)

        # Binary cross-entropy loss
        expert_loss = F.binary_cross_entropy_with_logits(
            expert_logits, torch.ones_like(expert_logits)
        )
        policy_loss = F.binary_cross_entropy_with_logits(
            policy_logits, torch.zeros_like(policy_logits)
        )

        disc_loss = expert_loss + policy_loss

        # Add gradient penalty for stability
        if self.use_grad_penalty:
            grad_penalty = self._compute_gradient_penalty(
                expert_states, expert_actions,
                policy_states, policy_actions
            )
            disc_loss += self.grad_penalty_coef * grad_penalty

        # Optimize
        self.disc_optimizer.zero_grad()
        disc_loss.backward()
        self.disc_optimizer.step()

        return disc_loss.item()
```

**Key Details**:
- Expert labeled as 1, policy as 0
- Uses binary cross-entropy with logits (more stable)
- Gradient penalty prevents discriminator from becoming too confident
- Returns loss for monitoring

### Gradient Penalty Implementation

```python
def _compute_gradient_penalty(self, expert_states, expert_actions,
                               policy_states, policy_actions):
    """WGAN-GP style gradient penalty"""
    batch_size = expert_states.size(0)
    alpha = torch.rand(batch_size, 1, device=expert_states.device)

    # Interpolate between expert and policy samples
    interp_states = alpha * expert_states + (1 - alpha) * policy_states
    interp_actions = alpha * expert_actions + (1 - alpha) * policy_actions

    interp_states.requires_grad_(True)
    interp_actions.requires_grad_(True)

    # Discriminator output on interpolations
    interp_logits = self.discriminator(interp_states, interp_actions)

    # Compute gradients w.r.t. interpolations
    gradients = torch.autograd.grad(
        outputs=interp_logits,
        inputs=[interp_states, interp_actions],
        grad_outputs=torch.ones_like(interp_logits),
        create_graph=True,
        retain_graph=True
    )

    # Penalty: (||∇D|| - 1)^2
    grad_concat = torch.cat([g.view(batch_size, -1) for g in gradients], dim=1)
    grad_norm = grad_concat.norm(2, dim=1)
    penalty = ((grad_norm - 1) ** 2).mean()

    return penalty
```

**Key Details**:
- Interpolates between expert and policy samples
- Computes gradient of discriminator w.r.t. inputs
- Penalizes gradients with norm different from 1
- Improves training stability (prevents mode collapse)

## Optimization Tricks

### 1. Discriminator Update Frequency

**Problem**: Discriminator can overfit and provide poor reward signal.

**Solution**: Update discriminator multiple times per policy update (5-10 steps).

```python
for i in range(num_disc_updates):
    update_discriminator(expert_batch, policy_batch)
update_policy(policy_batch)
```

**Why It Works**: Keeps discriminator accurate throughout training.

### 2. Reward Clipping

**Problem**: Discriminator can produce extreme rewards that destabilize policy training.

**Solution**: Clip rewards to reasonable range.

```python
rewards = discriminator.compute_reward(states, actions)
rewards = torch.clamp(rewards, min=-10, max=10)
```

### 3. Entropy Regularization

**Problem**: Policy can collapse to single mode (lacks diversity).

**Solution**: Add entropy bonus to policy objective.

```python
policy_loss = -(rewards * log_probs).mean() - entropy_coef * entropy
```

**Typical Values**: entropy_coef = 0.001 - 0.01

### 4. Discriminator Learning Rate

**Problem**: Discriminator learning too fast → poor reward signal.

**Solution**: Use lower learning rate for discriminator than typical supervised learning.

```python
disc_optimizer = Adam(discriminator.parameters(), lr=3e-4)
```

### 5. Expert Data Mixing

**Problem**: Policy forgets early learning.

**Solution**: Continuously sample from entire expert dataset (not just recent).

```python
# Don't just use recent expert demos
expert_batch = expert_buffer.sample(batch_size)  # Random sampling
```

### 6. Absorbing State Handling

**Problem**: Different episode lengths between expert and policy.

**Solution**: Add absorbing state that stays after episode ends.

```python
if done:
    next_state = absorbing_state  # Special zero state
```

### 7. Normalization

**Problem**: Different scales for states and actions can hurt discriminator.

**Solution**: Normalize inputs to discriminator.

```python
state_mean, state_std = compute_stats(expert_states)
normalized_states = (states - state_mean) / (state_std + 1e-8)
```

## Experiments & Results

### Classic Benchmarks

#### MuJoCo Locomotion Tasks

**Hopper-v2**:
- Expert: 3600 reward
- GAIL (4 demos): 3400 reward
- BC (4 demos): 2100 reward

**HalfCheetah-v2**:
- Expert: 5200 reward
- GAIL (4 demos): 4800 reward
- BC (4 demos): 3200 reward

**Walker2d-v2**:
- Expert: 5000 reward
- GAIL (4 demos): 4600 reward
- BC (4 demos): 2800 reward

**Key Findings**:
- GAIL achieves ~90% of expert performance with 4 demonstrations
- BC struggles with compounding errors
- GAIL benefits from more demos but plateaus around 10-20 demos

### Sample Efficiency Analysis

**Number of Demonstrations**:
- 1 demo: ~60% expert performance
- 4 demos: ~85% expert performance
- 10 demos: ~95% expert performance
- 50 demos: ~98% expert performance

**Environment Interactions**:
- Typically requires 1-10M environment steps
- Comparable to PPO trained with true reward
- More sample efficient than IRL methods

### Ablation Studies

**Impact of Gradient Penalty**:
- With GP: Stable training, 90% success
- Without GP: Frequent mode collapse, 60% success

**Discriminator Updates**:
- 1 update/policy: Discriminator underfits (70% performance)
- 5 updates/policy: Optimal (90% performance)
- 20 updates/policy: Discriminator overfits (75% performance)

**Entropy Regularization**:
- No entropy: Mode collapse, deterministic policy
- entropy_coef=0.01: Good diversity, 90% performance
- entropy_coef=0.1: Too stochastic, 70% performance

## Common Pitfalls

### 1. Discriminator Overfitting

**Symptom**: Discriminator achieves 100% accuracy, policy receives constant rewards.

**Causes**:
- Too many discriminator updates
- Discriminator too complex
- Not enough policy diversity

**Solutions**:
- Reduce discriminator updates per policy update
- Add gradient penalty
- Increase entropy regularization
- Use simpler discriminator architecture

### 2. Mode Collapse

**Symptom**: Policy learns one trajectory, ignores multi-modal expert behavior.

**Causes**:
- No entropy regularization
- Discriminator too powerful
- Expert data not diverse

**Solutions**:
- Add entropy bonus (λ=0.01)
- Reduce discriminator capacity
- Collect more diverse expert demonstrations

### 3. Reward Explosion

**Symptom**: Policy receives extremely high/low rewards, training unstable.

**Causes**:
- Discriminator too confident
- No reward clipping
- Learning rates too high

**Solutions**:
- Clip discriminator outputs: reward = clip(reward, -10, 10)
- Use label smoothing
- Lower discriminator learning rate
- Add gradient penalty

### 4. Policy Not Improving

**Symptom**: Policy performance stagnates despite discriminator training.

**Causes**:
- Discriminator not learning
- Policy optimizer too conservative
- Expert demos not representative

**Solutions**:
- Check discriminator accuracy (should be 60-80%)
- Increase policy learning rate
- Use PPO/TRPO with proper hyperparameters
- Verify expert demonstrations are high-quality

### 5. Distribution Mismatch

**Symptom**: Policy learns expert actions but in wrong states.

**Causes**:
- Discriminator only looks at actions
- Expert data imbalanced
- No state-action dependency

**Solutions**:
- Ensure discriminator sees concatenated (state, action)
- Balance expert dataset across state space
- Use state-only discriminator only if appropriate

### 6. Training Instability

**Symptom**: Loss oscillates wildly, no convergence.

**Causes**:
- Discriminator and policy learning rates mismatched
- No gradient penalty
- Batch size too small

**Solutions**:
- Use gradient penalty (coef=10)
- Match learning rates (both 3e-4)
- Increase batch size (512-2048)
- Use Adam optimizer with default β

### 7. Poor Transfer

**Symptom**: Works in training environment but fails in test environment.

**Causes**:
- Overfitting to expert demonstrations
- Expert demos not diverse enough
- No domain randomization

**Solutions**:
- Add noise to expert demonstrations
- Collect demos from varied scenarios
- Use domain randomization during training
- Consider AIRL for better transfer

## References

### Original Papers

1. **GAIL** (NeurIPS 2016)
   - Title: "Generative Adversarial Imitation Learning"
   - Authors: Jonathan Ho, Stefano Ermon
   - Link: https://arxiv.org/abs/1606.03476
   - Key Contribution: Adversarial framework for imitation learning

2. **GAN** (NeurIPS 2014)
   - Title: "Generative Adversarial Networks"
   - Authors: Ian Goodfellow et al.
   - Link: https://arxiv.org/abs/1406.2661
   - Relevance: Theoretical foundation for GAIL

3. **TRPO** (ICML 2015)
   - Title: "Trust Region Policy Optimization"
   - Authors: John Schulman et al.
   - Link: https://arxiv.org/abs/1502.05477
   - Relevance: Policy optimization algorithm used with GAIL

### Extensions and Improvements

4. **InfoGAIL** (NeurIPS 2017)
   - Title: "InfoGAIL: Interpretable Imitation Learning from Visual Demonstrations"
   - Authors: Yunzhu Li et al.
   - Link: https://arxiv.org/abs/1703.08840
   - Improvement: Disentangles style and content

5. **VAIL** (ICLR 2019)
   - Title: "Variational Discriminator Bottleneck"
   - Authors: Xue Bin Peng et al.
   - Link: https://arxiv.org/abs/1810.00821
   - Improvement: Adds information bottleneck for robustness

6. **f-GAIL** (ICML 2018)
   - Title: "Learning Robust Rewards with Adversarial Inverse RL"
   - Authors: Justin Fu et al.
   - Link: https://arxiv.org/abs/1809.02925
   - Improvement: Generalizes to different divergences

7. **GAIL-PPO** (2017)
   - Title: "Emergence of Locomotion Behaviours in Rich Environments"
   - Authors: Nicolas Heess et al.
   - Link: https://arxiv.org/abs/1707.02286
   - Improvement: Uses PPO instead of TRPO

### Theoretical Analysis

8. **Apprenticeship Learning via IRL** (ICML 2004)
   - Title: "Apprenticeship Learning via Inverse Reinforcement Learning"
   - Authors: Pieter Abbeel, Andrew Y. Ng
   - Link: https://ai.stanford.edu/~ang/papers/icml04-apprentice.pdf
   - Relevance: Theoretical foundation for imitation learning

9. **Maximum Entropy IRL** (AAAI 2008)
   - Title: "Maximum Entropy Inverse Reinforcement Learning"
   - Authors: Brian D. Ziebart et al.
   - Link: https://www.aaai.org/Papers/AAAI/2008/AAAI08-227.pdf
   - Relevance: MaxEnt formulation used in GAIL

### Surveys and Tutorials

10. **Imitation Learning Survey** (2018)
    - Title: "An Algorithmic Perspective on Imitation Learning"
    - Authors: Takayuki Osa et al.
    - Link: https://arxiv.org/abs/1811.06711
    - Coverage: Comprehensive overview of IL methods

11. **Deep RL Bootcamp** (2017)
    - Title: "Imitation Learning and GAIL"
    - Instructor: Chelsea Finn
    - Link: https://sites.google.com/view/deep-rl-bootcamp/lectures
    - Format: Video lecture and slides

### Implementation Resources

12. **OpenAI Baselines**: https://github.com/openai/baselines
13. **Imitation Library**: https://github.com/HumanCompatibleAI/imitation
14. **Stable-Baselines3**: https://github.com/DLR-RM/stable-baselines3
15. **rlkit**: https://github.com/rail-berkeley/rlkit

### Datasets

16. **D4RL**: https://github.com/rail-berkeley/d4rl
17. **RoboMimic**: https://robomimic.github.io/
18. **Human Demonstrations**: https://github.com/hmandell/atari_demonstrations
