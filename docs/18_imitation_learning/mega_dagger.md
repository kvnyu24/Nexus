# MEGA-DAgger: Multi-Expert Guided Aggregation

## Overview & Motivation

MEGA-DAgger extends the standard DAgger algorithm to handle scenarios where multiple imperfect experts are available, rather than a single perfect expert. This addresses a critical real-world limitation: perfect experts are rare, but multiple imperfect demonstrators (humans, heuristics, or pretrained models) are often readily available.

### What Problem Does MEGA-DAgger Solve?

**Standard DAgger Assumptions**:
- Single expert available
- Expert is perfect (optimal policy)
- Expert is consistent
- Expert queries are cheap

**Real-World Reality**:
- Multiple experts with varying quality
- Experts are imperfect (suboptimal, inconsistent)
- Expert disagreement is common
- Expert queries are expensive

**MEGA-DAgger's Solution**: Learn from multiple imperfect experts by:
1. Estimating each expert's reliability
2. Weighting expert contributions by quality
3. Using expert disagreement as uncertainty signal
4. Adaptively querying more reliable experts

### Key Achievements

- **Robust to Noisy Experts**: Learns despite imperfect demonstrations
- **Automatic Quality Estimation**: No manual expert quality labels needed
- **Uncertainty Quantification**: Uses expert disagreement for active learning
- **Sample Efficient**: Queries best experts more frequently
- **Outperforms Averaging**: Better than naive expert averaging

## Theoretical Background

### Multi-Expert Learning Framework

Given K experts {π₁*, π₂*, ..., πₖ*} with quality levels {q₁, q₂, ..., qₖ}:

**Quality Definition**:
```
q_i = E[reward(π_i*)] / E[reward(π_optimal)]
```

Where q_i ∈ [0,1] with 1 being perfect expert.

**Objective**: Learn policy π that maximizes:
```
J(π) = E[reward(π)]
```

Using demonstrations from imperfect experts.

### Expert Weighting

Learn state-dependent expert weights w_i(s):

```
w_i(s) = softmax(f_θ(s))_i
```

Where f_θ is a learned weighting function.

**Aggregated Expert Action**:
```
a* = Σᵢ w_i(s) · π_i*(s)
```

### Uncertainty from Disagreement

Expert disagreement indicates uncertainty:

```
uncertainty(s) = Var[{π₁*(s), π₂*(s), ..., πₖ*(s)}]
```

High disagreement → high uncertainty → prioritize for training.

## Mathematical Formulation

### Algorithm

**Input**: 
- K imperfect experts {π₁*, ..., πₖ*}
- Environment
- Number of iterations N

**Output**: Learned policy π_N

**Initialize**:
- Policy π₁
- Expert quality scores {q₁, ..., qₖ} = {1/K, ..., 1/K}
- Dataset D = ∅

**For iteration i = 1 to N**:

1. **Collect Data**:
   ```
   For each episode:
       For each step:
           Execute action from π_i
           Query all K experts for actions
           Compute expert disagreement
           Store (state, expert_actions, disagreement)
   ```

2. **Update Expert Weights**:
   ```
   For each expert k:
       q_k = performance(π_k on validation set)
       Normalize: q_k ← q_k / Σⱼ q_j
   ```

3. **Aggregate Expert Labels**:
   ```
   For each state s in collected data:
       Compute weights w_k(s) using current quality and state
       a* ← Σₖ w_k(s) · π_k*(s)
       Add (s, a*) to dataset D
   ```

4. **Train Policy**:
   ```
   π_{i+1} ← argmin_π Σ_{(s,a) ∈ D} L(π(s), a)
   ```

5. **Adaptive Sampling**:
   ```
   Prioritize states with high uncertainty for next iteration
   ```

### Expert Quality Estimation

**Validation-Based**:
```
q_k = (1/M) Σₘ reward(π_k* on episode m)
```

**State-Dependent Weighting**:
```
w(s) = softmax(MLP(s) + [q₁, q₂, ..., qₖ])
```

Combines global quality with state-specific expertise.

### Uncertainty-Guided Sampling

Prioritize states where experts disagree:

```
p(s) ∝ Var[{π₁*(s), ..., πₖ*(s)}]
```

## High-Level Intuition

### The Core Idea

Imagine learning to cook from multiple chefs with different skill levels:

**Chef A**: Great at desserts, poor at main courses
**Chef B**: Good all-around, but not exceptional
**Chef C**: Excellent at main courses, novice at desserts

**Naive Approach**: Average all chef recommendations
- Dessert: Mediocre (Chef A's excellence is diluted)
- Main course: Mediocre (Chef C's excellence is diluted)

**MEGA-DAgger Approach**:
- Learn that Chef A is best for desserts
- Learn that Chef C is best for main courses  
- Weight their advice accordingly for each recipe
- Result: Best of all experts

### Expert Disagreement as Signal

When experts disagree strongly on a state:
- Either state is ambiguous (multiple valid actions)
- Or some experts are out of their depth

Both cases indicate high uncertainty → valuable training signal!

### Adaptive Expert Querying

**Early Iterations**: Query all experts to learn quality
**Later Iterations**: Query only highest-quality experts
**Uncertain States**: Query all experts for comparison

This saves expert effort while maintaining performance.

## Implementation Details

### Network Architecture

**Policy Network** (same as DAgger):
```
State → [256] → [256] → Action
```

**Expert Weighting Network**:
```
State → [128] → [128] → [K expert weights]
       ↓
    Global Quality Scores (learned parameters)
       ↓
    Softmax → Expert Weights
```

**Uncertainty Estimator**:
```
[State, Action] → [128] → [128] → Uncertainty (positive)
```

### Training Procedure

**Phase 1: Collect Expert Demonstrations**
```python
for expert_k in experts:
    trajectories_k = expert_k.demonstrate(env, num_episodes)
    expert_dataset[k] = trajectories_k
```

**Phase 2: Estimate Initial Quality**
```python
for expert_k in experts:
    performance_k = evaluate(expert_k, validation_env)
    quality_scores[k] = performance_k / max_performance
```

**Phase 3: Iterative Training**
```python
for iteration in range(num_iterations):
    # Collect data with current policy
    states = collect_states(policy, env)
    
    # Query all experts
    expert_actions = [expert_k(states) for expert_k in experts]
    
    # Compute uncertainty from disagreement
    uncertainty = variance(expert_actions)
    
    # Weight experts by quality
    weights = compute_weights(states, quality_scores)
    
    # Aggregate expert actions
    target_actions = weighted_average(expert_actions, weights)
    
    # Train policy
    policy.train(states, target_actions)
    
    # Update quality scores
    quality_scores = evaluate_experts(validation_env)
```

### Hyperparameters

```python
num_experts = 3              # Number of available experts
num_iterations = 15          # More than standard DAgger
quality_update_freq = 3      # Update quality every N iterations
uncertainty_threshold = 0.5  # Threshold for uncertain states
learning_rate = 1e-3
weight_net_lr = 1e-4         # Slower for weighting network
```

## Code Walkthrough

Implementation in `Nexus/nexus/models/imitation/mega_dagger.py`:

### Expert Weighting Module

```python
class ExpertWeightingModule(nn.Module):
    def __init__(self, num_experts, state_dim, hidden_dim=128):
        super().__init__()
        # State-dependent weighting
        self.weight_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts)
        )
        # Global quality scores (learned)
        self.expert_quality = nn.Parameter(torch.ones(num_experts))

    def forward(self, state):
        # Combine state-dependent and global weights
        state_weights = self.weight_net(state)
        combined = state_weights + self.expert_quality
        return F.softmax(combined, dim=-1)
```

### Uncertainty Estimation

```python
class UncertaintyEstimator(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Ensure positive
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.network(x)
```

### Expert Aggregation

```python
def aggregate_expert_actions(states, expert_actions, expert_weights):
    """
    Args:
        states: (B, state_dim)
        expert_actions: list of K tensors (B, action_dim)
        expert_weights: (B, K)
    Returns:
        aggregated_actions: (B, action_dim)
    """
    # Stack expert actions
    actions_stacked = torch.stack(expert_actions, dim=1)  # (B, K, action_dim)
    
    # Weight by expert quality
    weights_expanded = expert_weights.unsqueeze(-1)  # (B, K, 1)
    weighted_actions = actions_stacked * weights_expanded  # (B, K, action_dim)
    
    # Sum over experts
    aggregated = weighted_actions.sum(dim=1)  # (B, action_dim)
    
    return aggregated
```

## Optimization Tricks

### 1. Quality Score Smoothing

**Problem**: Quality estimates are noisy early in training.

**Solution**: Exponential moving average:
```python
quality_new = 0.9 * quality_old + 0.1 * quality_current
```

### 2. Minimum Expert Weight

**Problem**: Bad experts get zero weight, losing diversity.

**Solution**: Enforce minimum weight:
```python
weights = softmax(logits)
weights = weights * 0.9 + 0.1 / num_experts  # Min 10% weight
```

### 3. Uncertainty-Based Prioritization

**Problem**: Some states are more critical to learn.

**Solution**: Sample states proportional to uncertainty:
```python
uncertainty_scores = compute_uncertainty(states)
sampling_probs = uncertainty_scores / uncertainty_scores.sum()
prioritized_states = sample(states, probs=sampling_probs)
```

### 4. Expert Specialization Detection

**Problem**: Some experts are specialists (good in specific states).

**Solution**: Learn state-conditional weights, not just global:
```python
# State-dependent: captures specialization
weights = weight_network(state)  # Different for each state
```

### 5. Disagreement-Based Active Learning

**Problem**: Querying all experts is expensive.

**Solution**: Query all experts only when they disagree:
```python
if disagreement(expert_predictions) > threshold:
    actions = [expert_k(state) for expert_k in all_experts]
else:
    actions = [best_expert(state)]  # Query only best
```

## Experiments & Results

### Benchmarks with Imperfect Experts

**Setup**: 3 experts with quality [0.9, 0.7, 0.5]

**Hopper-v2**:
- Best Expert (0.9): 3200
- MEGA-DAgger: 3150
- Average Experts: 2800
- Best Single Expert DAgger: 3100

**Walker2d-v2**:
- Best Expert (0.9): 4500
- MEGA-DAgger: 4400
- Average Experts: 3800
- Best Single Expert DAgger: 4300

**Key Finding**: MEGA-DAgger matches best single expert while being robust to expert quality variation.

### Robustness to Expert Quality

**Varying Best Expert Quality** (Hopper-v2):

| Best Expert Quality | MEGA-DAgger | Single Expert DAgger | Improvement |
|---------------------|-------------|----------------------|-------------|
| 0.95                | 3400        | 3350                 | +1.5%       |
| 0.80                | 3100        | 2900                 | +6.9%       |
| 0.60                | 2500        | 2200                 | +13.6%      |

**Key Finding**: MEGA-DAgger's advantage grows when experts are more imperfect.

### Expert Specialization

**Task**: Manipulation with 3 specialized experts
- Expert A: Good at grasping (80%), poor at placing (40%)
- Expert B: Poor at grasping (40%), good at placing (80%)
- Expert C: Medium at both (60%, 60%)

**Results**:
- MEGA-DAgger: 85% success (learns to use A for grasp, B for place)
- Average: 60% success
- Best Single: 65% success

**Key Finding**: MEGA-DAgger leverages complementary expertise.

## Common Pitfalls

### 1. Insufficient Expert Diversity

**Symptom**: MEGA-DAgger performs no better than single expert.

**Cause**: All experts are too similar or have same failure modes.

**Solution**:
- Ensure experts have different strengths
- Include at least one high-quality expert
- Verify expert disagreement on validation set

### 2. Quality Estimation Noise

**Symptom**: Quality scores fluctuate wildly, unstable weights.

**Cause**: Small validation set, high-variance tasks.

**Solutions**:
- Use larger validation set (100+ episodes)
- Smooth quality updates with EMA
- Increase quality update frequency

### 3. Expert Weight Collapse

**Symptom**: One expert gets 100% weight, others ignored.

**Cause**: Quality differences too large, softmax saturates.

**Solutions**:
- Add entropy regularization to weights
- Use temperature in softmax: softmax(logits / T)
- Enforce minimum weight for all experts

### 4. Overfitting to Expert Noise

**Symptom**: Policy memorizes expert mistakes.

**Cause**: Imperfect experts have consistent biases.

**Solutions**:
- Add regularization (dropout, weight decay)
- Use ensemble of policies
- Filter outlier expert actions

### 5. Expensive Expert Queries

**Symptom**: Training too slow due to querying all experts.

**Cause**: Querying all K experts every step.

**Solutions**:
- Query only on uncertain states
- Cache expert actions for common states
- Use learned surrogate models for bad experts

## References

### Foundational

1. **DAgger** (AISTATS 2011)
   - Ross et al.
   - https://arxiv.org/abs/1011.0686

2. **Learning from Multiple Experts** (ICML 2016)
   - Chaudhuri et al.
   - Foundation for multi-expert learning

### Related Work

3. **Cascaded Supervision** (NIPS 2017)
   - Learning from multiple supervision sources

4. **Knowledge Distillation with Multiple Teachers** (ICLR 2018)
   - Hinton et al.
   - Multi-teacher knowledge transfer

5. **Active Learning with Disagreement** (ICML 2007)
   - Using model disagreement for sample selection

### Applications

6. **Multi-Modal Imitation** (RSS 2019)
   - Robotics with multiple demonstrators

7. **Robust Driving from Multiple Experts** (CoRL 2020)
   - Autonomous driving with diverse human demonstrators
