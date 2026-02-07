# Imitation Learning Methods

This directory contains comprehensive documentation for imitation learning algorithms implemented in Nexus. Imitation learning enables agents to learn behaviors by observing expert demonstrations, bypassing the need for explicit reward engineering.

## Table of Contents

1. [GAIL (Generative Adversarial Imitation Learning)](#gail)
2. [DAgger (Dataset Aggregation)](#dagger)
3. [MEGA-DAgger](#mega-dagger)
4. [AIRL (Adversarial Inverse Reinforcement Learning)](#airl)

## Overview

Imitation learning addresses a fundamental question: **How can an agent learn to perform a task by watching an expert, without knowing the underlying reward function?**

This is crucial because:
- Reward engineering is often difficult and time-consuming
- Expert demonstrations are frequently available (human demonstrations, recorded trajectories)
- Many tasks are easier to demonstrate than to specify formally

### Core Paradigms

**Behavioral Cloning (BC)**: Directly supervised learning from expert state-action pairs
- Simple but suffers from distributional shift
- No exploration of states not visited by expert

**Inverse Reinforcement Learning (IRL)**: Learn the reward function that explains expert behavior
- Recovers underlying objectives
- Computationally expensive (requires solving RL in inner loop)

**Adversarial Imitation**: Use discriminators to distinguish expert from policy behavior
- Avoids explicit reward learning
- More sample efficient than IRL
- Combines benefits of BC and IRL

## Algorithm Progression

We recommend studying these algorithms in the following order:

### 1. DAgger (Start Here)
- **File**: [dagger.md](dagger.md)
- **Difficulty**: Beginner-Intermediate
- **Key Concepts**: Interactive learning, covariate shift, expert queries
- **Use Case**: Learning from imperfect demonstrations with expert feedback

### 2. GAIL
- **File**: [gail.md](gail.md)
- **Difficulty**: Intermediate
- **Key Concepts**: Adversarial training, discriminator rewards, GAN stability
- **Use Case**: Learning complex behaviors from expert demonstrations

### 3. AIRL
- **File**: [airl.md](airl.md)
- **Difficulty**: Advanced
- **Key Concepts**: Reward function recovery, disentangling dynamics, transfer learning
- **Use Case**: When you need interpretable rewards or transfer to new environments

### 4. MEGA-DAgger (Advanced)
- **File**: [mega_dagger.md](mega_dagger.md)
- **Difficulty**: Advanced
- **Key Concepts**: Model-based learning, world models, safety-aware exploration
- **Use Case**: Safety-critical domains with limited expert interaction

## Comparison Table

| Algorithm | Paradigm | Expert Queries | Reward Recovery | Sample Efficiency | Complexity |
|-----------|----------|----------------|-----------------|-------------------|------------|
| DAgger | Interactive BC | Required | ❌ | High (with expert) | Low |
| GAIL | Adversarial | Not Required | ❌ | Medium | Medium |
| AIRL | Adversarial IRL | Not Required | ✅ | Medium | High |
| MEGA-DAgger | Model-Based | Minimal | ❌ | Very High | Very High |

## Detailed Comparisons

### Sample Efficiency

**Most to Least Efficient:**
1. **MEGA-DAgger**: Uses learned world model for planning, minimizes expert queries
2. **DAgger**: Direct expert queries reduce compound errors
3. **GAIL/AIRL**: Require many environment interactions to train policy and discriminator

### Computational Cost

**Least to Most Expensive:**
1. **DAgger**: Simple supervised learning with occasional expert queries
2. **GAIL**: Policy optimization + discriminator training
3. **AIRL**: Additional reward function learning and disentanglement
4. **MEGA-DAgger**: World model learning + planning + expert queries

### When Expert Access is Limited

**Best to Worst:**
1. **GAIL**: Works with fixed dataset of demonstrations
2. **AIRL**: Also works with fixed dataset
3. **MEGA-DAgger**: Uses world model to minimize queries
4. **DAgger**: Requires frequent expert access during training

## When to Use Each Algorithm

### Use DAgger when:
- You have access to an expert that can provide labels during training
- Covariate shift is a major concern
- You want a simple, interpretable approach
- Computational resources are limited
- Real-time expert feedback is available

**Typical Applications:**
- Autonomous driving with human supervisor
- Robot manipulation with human corrections
- Game playing with expert annotations

### Use GAIL when:
- You have a fixed dataset of expert demonstrations
- Expert access during training is not available
- You want to learn complex, multi-modal behaviors
- Sample efficiency during training is not critical
- You don't need an interpretable reward function

**Typical Applications:**
- Learning from human gameplay recordings
- Robotics with demonstration datasets
- Character animation from motion capture

### Use AIRL when:
- You need to recover interpretable reward functions
- Transfer to new environment dynamics is required
- You want to understand the expert's objectives
- Computational cost is acceptable
- Domain knowledge can inform reward structure

**Typical Applications:**
- Learning human preferences for alignment
- Transfer learning across robot morphologies
- Understanding expert decision-making
- Multi-task learning with shared rewards

### Use MEGA-DAgger when:
- Expert access is expensive or dangerous
- Safety is critical (avoid bad states)
- Sample efficiency is paramount
- You can learn accurate world models
- Planning in model space is feasible

**Typical Applications:**
- High-stakes medical procedures
- Autonomous vehicles (minimize dangerous situations)
- Expensive robotic systems
- Space exploration

## Core Concepts

### Distributional Shift

The fundamental challenge in imitation learning: the learner's state distribution differs from the expert's.

```
Expert: s₀ → s₁ → s₂ → s₃ (expert states)
Learner: s₀ → s₁' → s₂'' → s₃''' (different states due to errors)
```

**Solutions:**
- **DAgger**: Query expert on learner's states
- **GAIL/AIRL**: Train policy to match expert distribution
- **MEGA-DAgger**: Use world model to avoid bad states

### Adversarial Training

GAIL and AIRL use a discriminator D(s,a) that classifies state-action pairs:
- D(s,a) = 1 for expert demonstrations
- D(s,a) = 0 for policy rollouts

The discriminator's output provides a reward signal:
```
r(s,a) = -log(1 - D(s,a))  # GAIL
r(s,a) = log(D(s,a)) - log(1 - D(s,a))  # AIRL
```

### Model-Based Acceleration

MEGA-DAgger learns a world model M(s,a) → s' to:
- Simulate trajectories without environment interaction
- Plan using learned dynamics
- Identify states where expert input is needed
- Train policy in imagination

## Best Practices

### Data Collection

1. **Expert Quality**: Ensure demonstrations are truly expert-level
   - Suboptimal demonstrations hurt all methods
   - GAIL/AIRL particularly sensitive to noisy data

2. **Diversity**: Collect demonstrations from diverse scenarios
   - Cover edge cases and rare events
   - Multiple experts can improve robustness

3. **Labeling**: For DAgger, expert must label policy-visited states
   - Make labeling interface efficient
   - Consider active learning for query selection

### Training Stability

1. **GAIL/AIRL**: Use techniques from GAN training
   - Gradient penalty for discriminator
   - Spectral normalization
   - Batch normalization
   - Careful learning rate tuning

2. **DAgger**: Balance dataset mixing
   - β-decay schedule for expert data weighting
   - Don't discard early expert demonstrations

3. **MEGA-DAgger**: World model accuracy is critical
   - Use uncertainty estimates
   - Fall back to expert when model is uncertain
   - Iteratively improve model with real data

### Evaluation

1. **Performance Metrics**:
   - Task success rate
   - Distance to expert trajectory
   - Environment-specific rewards

2. **Distributional Metrics**:
   - State visitation frequency
   - Action distribution similarity
   - Trajectory diversity

3. **Sample Efficiency**:
   - Number of expert demonstrations needed
   - Number of environment interactions
   - Number of expert queries (DAgger, MEGA-DAgger)

## Implementation Overview

All implementations in Nexus follow a consistent API:

```python
from nexus.models.imitation import GAILAgent, DAggerAgent, AIRLAgent, MEGADAggerAgent

# GAIL Example
config = {
    "state_dim": 17,
    "action_dim": 6,
    "hidden_dims": [256, 256],
    "policy_lr": 3e-4,
    "discriminator_lr": 3e-4,
    "use_spectral_norm": True
}

agent = GAILAgent(config)

# Training loop
for epoch in range(num_epochs):
    # Collect policy rollouts
    policy_batch = collect_rollouts(agent, env)

    # Sample expert demonstrations
    expert_batch = expert_buffer.sample(batch_size)

    # Update discriminator and policy
    metrics = agent.update(policy_batch, expert_batch)

# DAgger Example
config = {
    "state_dim": 17,
    "action_dim": 6,
    "hidden_dims": [256, 256],
    "learning_rate": 3e-4,
    "beta_decay": 0.95
}

agent = DAggerAgent(config)

# Training loop
for epoch in range(num_epochs):
    # Collect policy rollouts
    states, _ = collect_rollouts(agent, env)

    # Query expert for labels on policy-visited states
    expert_actions = expert.label(states)

    # Update policy
    metrics = agent.update(states, expert_actions)
```

## Common Challenges and Solutions

### Challenge 1: Compounding Errors (Distributional Shift)

**Problem**: Small errors accumulate over time, leading to state distributions unseen during training.

**Solutions**:
- **DAgger**: Query expert on learner-visited states
- **GAIL/AIRL**: Match state-action distributions via adversarial training
- **MEGA-DAgger**: Use world model to plan and avoid error-prone states
- **All**: Add noise to expert demonstrations during training

### Challenge 2: Insufficient Expert Data

**Problem**: Limited demonstrations lead to overfitting and poor generalization.

**Solutions**:
- Data augmentation (trajectory perturbations)
- Regularization (dropout, weight decay)
- Ensemble methods
- Active learning to request more data where needed

### Challenge 3: GAN Training Instability (GAIL/AIRL)

**Problem**: Discriminator and policy training can be unstable, leading to mode collapse or divergence.

**Solutions**:
- Gradient penalty (WGAN-GP style)
- Spectral normalization on discriminator
- Lower discriminator learning rate
- Multiple discriminator updates per policy update
- Batch normalization

### Challenge 4: Reward Ambiguity (GAIL)

**Problem**: Many reward functions can explain the same behavior.

**Solutions**:
- Use AIRL if reward recovery is important
- Add reward shaping or prior knowledge
- Constrain reward function space
- Multi-task learning with shared rewards

### Challenge 5: World Model Errors (MEGA-DAgger)

**Problem**: Inaccurate world model leads to poor planning and policy learning.

**Solutions**:
- Ensemble world models for uncertainty
- Use model only where confident
- Mix model-based and model-free updates
- Continuously update model with real data
- Conservative planning with pessimistic models

## Key Papers

### Foundational

1. **Behavioral Cloning Basics**:
   - [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://arxiv.org/abs/1011.0686) (Ross et al., AISTATS 2011)

2. **DAgger**:
   - [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://arxiv.org/abs/1011.0686) (Ross et al., AISTATS 2011)

3. **GAIL**:
   - [Generative Adversarial Imitation Learning](https://arxiv.org/abs/1606.03476) (Ho & Ermon, NeurIPS 2016)

4. **AIRL**:
   - [Learning Robust Rewards with Adversarial Inverse Reinforcement Learning](https://arxiv.org/abs/1710.11248) (Fu et al., ICLR 2018)

5. **MEGA-DAgger**:
   - [Model-based Generative Adversarial Imitation Learning](https://arxiv.org/abs/2004.03763) (Vuong et al., CoRL 2020)

### Surveys and Tutorials

1. **Imitation Learning Survey**: [An Algorithmic Perspective on Imitation Learning](https://arxiv.org/abs/1811.06711) (Osa et al., 2018)
2. **Inverse RL Survey**: [A Survey of Inverse Reinforcement Learning: Techniques, Applications, and Open Problems](https://arxiv.org/abs/1806.06877) (Arora & Doshi, 2021)

## Additional Resources

### Tutorials
- [Stanford CS237A: Imitation Learning](https://web.stanford.edu/class/cs237a/)
- [Berkeley CS287: Advanced Robotics - Imitation Learning Module](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa19/)
- [Imitation Learning Tutorial by Sergey Levine](https://sites.google.com/view/icml2018-imitation-learning/)

### Implementations
- [imitation](https://github.com/HumanCompatibleAI/imitation): Clean implementations of IL algorithms
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3): Includes GAIL
- [rlkit](https://github.com/rail-berkeley/rlkit): Research codebase with AIRL

### Datasets
- [D4RL](https://github.com/rail-berkeley/d4rl): Offline RL and IL benchmark datasets
- [RoboMimic](https://robomimic.github.io/): Robot manipulation demonstrations
- [Atari demonstrations](https://github.com/hmandell/atari_demonstrations)

## File Structure

```
18_imitation_learning/
├── README.md              # This file
├── gail.md               # Generative Adversarial Imitation Learning
├── dagger.md             # Dataset Aggregation
├── mega_dagger.md        # Model-based DAgger
└── airl.md               # Adversarial Inverse RL
```

## Getting Started

1. **New to Imitation Learning?** Start with [DAgger](dagger.md) to understand the distributional shift problem and interactive learning
2. **Have expert demonstrations?** Jump to [GAIL](gail.md) for adversarial imitation learning
3. **Need interpretable rewards?** Study [AIRL](airl.md) for reward recovery
4. **Limited expert access?** Explore [MEGA-DAgger](mega_dagger.md) for model-based efficiency

Each algorithm documentation includes:
- Theoretical foundations
- Mathematical formulations
- Implementation details
- Code walkthroughs
- Optimization tricks
- Common pitfalls
- Experimental results

## Deep Dive: Core Challenges in Imitation Learning

### 1. The Distributional Shift Problem

The fundamental challenge in imitation learning is **distributional shift**: the learner's visited state distribution differs from the expert's.

**Why This Happens**:
```
Time t=0: Learner starts at same state as expert ✓
Time t=1: Small error → slightly different state
Time t=2: No training data for this state → larger error
Time t=3: Completely off-distribution → catastrophic failure
```

**Quantitative Analysis**:

For a policy π with per-step error ε:
- **Behavioral Cloning**: Expected error ~ O(T²ε)
- **DAgger**: Expected error ~ O(Tε)
- **Perfect Oracle**: Expected error ~ O(ε)

Where T is the time horizon.

**Example: Autonomous Driving**

Expert trajectory:
```
Lane center → Lane center → Lane center → Lane center
```

BC-trained policy:
```
Lane center → 10cm right → 25cm right → 50cm right → OFF ROAD
```

The 10cm error at t=1 compounds geometrically because:
1. Training data only covers "lane center" states
2. No examples of "how to recover from 10cm right"
3. Policy extrapolates poorly to novel states
4. Errors accumulate multiplicatively

**Solution Approaches**:

| Method | Approach | Error Bound |
|--------|----------|-------------|
| BC | Train on expert demos only | O(T²ε) |
| BC + Noise | Add state noise to demos | O(T^1.5ε) |
| DAgger | Query expert on learner states | O(Tε) |
| GAIL | Match state-action distributions | O(Tε) |
| SafeDAgger | DAgger with safety constraints | O(Tε) |

### 2. Expert Data Quality and Availability

**Challenge**: Expert demonstrations are expensive, potentially inconsistent, and limited in quantity.

**Expert Quality Spectrum**:

1. **Optimal Expert** (Rare):
   - Perfectly solves the task
   - Consistent across demonstrations
   - Examples: Optimal game-playing AI, physics simulators

2. **Near-Optimal Expert** (Common):
   - Very high performance (>95% optimal)
   - Mostly consistent
   - Examples: Professional human demonstrators, strong RL policies

3. **Imperfect Expert** (Most Common):
   - Good but suboptimal (70-90% optimal)
   - Some inconsistencies
   - Examples: Average human demonstrators, heuristic policies

4. **Mixed Quality Experts** (Realistic):
   - Multiple experts with varying quality
   - Disagreement on edge cases
   - Examples: Crowdsourced demonstrations, multiple humans

**Algorithm Robustness to Expert Quality**:

| Algorithm | Optimal Expert | Noisy Expert | Multiple Experts |
|-----------|----------------|--------------|------------------|
| BC | ✓✓✓ | ✗ | ✓ (averaging) |
| DAgger | ✓✓✓ | ✓ | ✓ (single query) |
| GAIL | ✓✓✓ | ✓✓ | ✓✓ |
| AIRL | ✓✓✓ | ✓✓ | ✓✓ |
| MEGA-DAgger | ✓✓ | ✓✓✓ | ✓✓✓ |

### 3. Sample Efficiency

**The Trade-off**: Expert demonstrations are expensive; environment interactions are cheap (usually).

**Data Requirements by Method**:

**Behavioral Cloning**:
- Expert demos needed: 100-1000 trajectories
- Environment interactions: 0 (offline)
- Expert queries during training: 0
- Total cost: High (many expert demos)

**DAgger**:
- Expert demos needed: 10-50 trajectories (initial)
- Environment interactions: 10K-100K steps
- Expert queries during training: 10K-100K labels
- Total cost: Very high (continuous expert access)

**GAIL**:
- Expert demos needed: 4-50 trajectories
- Environment interactions: 1M-10M steps
- Expert queries during training: 0
- Total cost: Medium (demos only, but many env steps)

**AIRL**:
- Expert demos needed: 4-50 trajectories
- Environment interactions: 1M-10M steps
- Expert queries during training: 0
- Total cost: Medium-high (like GAIL but more computation)

**MEGA-DAgger**:
- Expert demos needed: 10-50 trajectories
- Environment interactions: 50K-500K steps
- Expert queries during training: 1K-10K labels
- Total cost: Medium (less expert access than DAgger)

### 4. Reward Function Recovery

**Question**: Should we learn the reward function or just the policy?

**Policy-Only Approaches** (BC, DAgger, GAIL):
- **Pros**: Simpler, faster training
- **Cons**: No interpretability, no transfer, task-specific

**Reward Recovery Approaches** (IRL, AIRL):
- **Pros**: Interpretable, transferable, reusable
- **Cons**: More complex, slower training, identifiability issues

**When You Need Reward Recovery**:

1. **Transfer Learning**: Apply learned behavior to new environment dynamics
   - Example: Robot policy trained on one morphology, deployed on another
   - Solution: AIRL learns dynamics-independent reward

2. **Multi-Task Learning**: Share learned objectives across tasks
   - Example: "Reach goal" reward for multiple navigation tasks
   - Solution: Learn reward once, reuse for new goals

3. **Interpretability**: Understand what the expert is optimizing
   - Example: Human preference learning for AI alignment
   - Solution: Recovered reward function shows human values

4. **Debugging**: Diagnose policy failures
   - Example: Policy fails in corner case
   - Solution: Check reward function to understand intended behavior

**When Policy-Only is Sufficient**:

1. Fixed environment (no transfer needed)
2. Single task (no multi-task learning)
3. Black-box acceptable (no interpretability needed)
4. Speed critical (reward learning too slow)

## Advanced Topics

### Hybrid Approaches

Modern imitation learning often combines multiple paradigms:

**1. DAgger + GAIL (DAC)**:
- Use DAgger for initial policy learning
- Fine-tune with GAIL to smooth out distributional mismatch
- Best of both: DAgger's sample efficiency + GAIL's robustness

**2. BC Pretraining + RL Fine-tuning**:
- Pretrain policy with behavioral cloning
- Fine-tune with RL (if reward available)
- Accelerates RL training significantly

**3. GAIL + Reward Shaping**:
- Use GAIL discriminator as learned reward
- Add hand-crafted reward shaping for known objectives
- Combines learned and engineered rewards

**4. Multi-Modal GAIL**:
- Learn multiple skills from diverse demonstrations
- Use latent variable models (VAE, InfoGAIL)
- Captures multi-modal expert behavior

### Theoretical Connections

**Imitation Learning ↔ Reinforcement Learning**:

IL can be viewed as RL with specific reward structures:

| IL Method | Equivalent RL Reward |
|-----------|---------------------|
| BC | r(s,a) = -||a - π*(s)||² |
| GAIL | r(s,a) = log D(s,a) |
| AIRL | r(s,a) = learned reward function |
| DAgger | r(s,a) = expert agreement |

**Imitation Learning ↔ Optimal Control**:

For linear-quadratic systems:
- BC is supervised learning of linear controller
- IRL recovers quadratic cost function
- GAIL matches state-action distributions

**Imitation Learning ↔ Generative Modeling**:

GAIL connections to GANs:
- Expert data = real data distribution
- Policy rollouts = generated data
- Discriminator = distribution matcher
- Training = adversarial game

### Active Research Directions

**1. Sample-Efficient Imitation**:
- Few-shot imitation learning
- One-shot imitation from single demo
- Meta-learning for quick adaptation

**2. Interactive Imitation**:
- Learning from human feedback (RLHF)
- Active querying strategies
- Uncertainty-guided expert queries

**3. Safe Imitation Learning**:
- Safety constraints during learning
- Avoiding catastrophic states
- Conservative policy updates

**4. Hierarchical Imitation**:
- Learning skills and composition
- Temporal abstraction
- Options and primitives

**5. Imitation from Observations**:
- Learn from state-only demonstrations (no actions)
- Third-person imitation
- Video demonstrations

**6. Multi-Modal Imitation**:
- Learning from diverse demonstrations
- Handling multi-modal expert behavior
- Skill discovery and clustering

## Practical Guidelines

### Choosing the Right Algorithm

**Decision Tree**:

```
Do you have expert access during training?
├─ YES
│  └─ Is expert consistent and high-quality?
│     ├─ YES → DAgger
│     └─ NO → MEGA-DAgger (handles multiple/imperfect experts)
│
└─ NO (fixed demonstrations only)
   └─ Do you need interpretable rewards or transfer?
      ├─ YES → AIRL
      └─ NO → GAIL
```

**Budget Considerations**:

| Budget Constraint | Recommended Algorithm |
|-------------------|----------------------|
| Limited expert demos (<10) | GAIL or AIRL (maximize from few demos) |
| Limited environment interactions | DAgger (sample efficient with expert) |
| Limited computation | DAgger or BC (avoid adversarial training) |
| Limited expert availability | GAIL or AIRL (offline, no queries) |

### Implementation Checklist

Before deploying imitation learning:

**Data Collection**:
- [ ] Expert demonstrations are high-quality (>90% success rate)
- [ ] Demonstrations cover diverse scenarios
- [ ] State-action pairs are properly recorded
- [ ] Episode termination is handled correctly
- [ ] Data is normalized/preprocessed consistently

**Model Selection**:
- [ ] Algorithm matches your expert access pattern
- [ ] Network architecture appropriate for state/action spaces
- [ ] Hyperparameters validated on similar tasks
- [ ] Baseline comparison available (BC minimum)

**Training**:
- [ ] Validation set for early stopping
- [ ] Metrics tracked (loss, performance, distributional distance)
- [ ] Checkpointing for best model recovery
- [ ] Stability techniques applied (grad clip, normalization)

**Evaluation**:
- [ ] Test on held-out scenarios
- [ ] Measure distributional similarity to expert
- [ ] Long-horizon rollout evaluation
- [ ] Edge case testing
- [ ] Robustness to perturbations

### Common Failure Modes

**1. Mode Collapse** (GAIL/AIRL):
- **Symptom**: Policy learns only one trajectory, ignores diversity
- **Diagnosis**: Check if expert demos are multi-modal
- **Fix**: Increase entropy regularization, use InfoGAIL

**2. Catastrophic Forgetting** (DAgger):
- **Symptom**: Performance degrades in later iterations
- **Diagnosis**: Check if old data is being discarded
- **Fix**: Ensure data aggregation, not replacement

**3. Expert Mismatch** (All):
- **Symptom**: Policy performs differently than expert despite low loss
- **Diagnosis**: State/action spaces don't align
- **Fix**: Verify observation and action preprocessing

**4. Overfitting** (BC):
- **Symptom**: Perfect training loss, poor test performance
- **Diagnosis**: Too much model capacity, too little data
- **Fix**: Regularization, data augmentation, early stopping

**5. Training Instability** (GAIL/AIRL):
- **Symptom**: Discriminator loss oscillates wildly
- **Diagnosis**: Adversarial training instability
- **Fix**: Gradient penalty, spectral norm, careful LR tuning

## Resources for Further Learning

### Online Courses

1. **CS 285: Deep Reinforcement Learning** (Berkeley)
   - Instructor: Sergey Levine
   - URL: http://rail.eecs.berkeley.edu/deeprlcourse/
   - Module on Imitation Learning (Lectures 2-3)

2. **CS 330: Deep Multi-Task and Meta Learning** (Stanford)
   - Instructor: Chelsea Finn
   - URL: https://cs330.stanford.edu/
   - Covers few-shot imitation learning

3. **DeepMind x UCL: Deep Learning Lecture Series**
   - Guest lecture on Imitation Learning
   - URL: https://www.youtube.com/deepmind

### Research Groups

1. **Berkeley Robot Learning Lab**: https://rll.berkeley.edu/
2. **Stanford Vision and Learning Lab**: https://svl.stanford.edu/
3. **CMU Robot Learning Lab**: https://rll.ri.cmu.edu/
4. **Google Brain Robotics**: https://research.google/teams/brain/robotics/
5. **DeepMind Control**: https://deepmind.com/research/highlighted-research/agents

### Benchmarking

**Standard Environments**:
1. **MuJoCo**: Continuous control (locomotion, manipulation)
2. **Atari**: Discrete control (game playing)
3. **RoboSuite**: Robot manipulation
4. **Meta-World**: Multi-task manipulation
5. **D4RL**: Offline RL and IL datasets

**Evaluation Metrics**:
1. **Task Success Rate**: Binary success/failure
2. **Cumulative Reward**: Sum of rewards over episode
3. **Expert Performance Gap**: (Expert - Policy) / Expert
4. **Distribution Divergence**: KL or JS divergence from expert
5. **Sample Efficiency**: Performance vs. data used

### Software Libraries

**Implementations**:
1. **imitation**: https://github.com/HumanCompatibleAI/imitation
   - Clean, maintained implementations of BC, DAgger, GAIL, AIRL
   - Works with Gym environments
   - Good documentation

2. **Stable-Baselines3**: https://github.com/DLR-RM/stable-baselines3
   - Includes GAIL
   - Integrated with PPO/SAC
   - Production-ready

3. **rlkit**: https://github.com/rail-berkeley/rlkit
   - Research codebase from Berkeley
   - GAIL, AIRL implementations
   - Advanced features

4. **Spinning Up**: https://github.com/openai/spinningup
   - Educational implementations
   - Clear code, good tutorials
   - Focus on understanding

## Summary

Imitation learning provides powerful tools for learning from demonstrations. The key trade-offs are:

| Dimension | BC | DAgger | GAIL | AIRL | MEGA-DAgger |
|-----------|----|----|------|------|-------------|
| Sample Efficiency | Low | High | Medium | Medium | Very High |
| Expert Access Needed | Offline | Online | Offline | Offline | Minimal Online |
| Computational Cost | Low | Low | Medium | High | Very High |
| Distributional Shift | Poor | Good | Good | Good | Good |
| Reward Recovery | No | No | No | Yes | No |
| Transfer Learning | No | No | Poor | Good | No |
| Implementation Complexity | Low | Low | Medium | High | Very High |

**Recommendations**:
- **Start simple**: Try BC baseline first
- **Expert access**: Use DAgger if you have it
- **Offline demos**: Use GAIL for most tasks
- **Transfer/interpret**: Use AIRL when needed
- **Multiple experts**: Use MEGA-DAgger for imperfect experts

The field continues to evolve rapidly, with new methods combining the best aspects of these foundational approaches.
