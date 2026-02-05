# Online Decision Transformer

## 1. Overview & Motivation

Online Decision Transformer (ODT) bridges offline and online reinforcement learning by **fine-tuning** Decision Transformers with online interaction data. While standard DT is limited by the quality of offline data, ODT can improve through real environment experience.

### The Offline-Only Limitation

Standard Decision Transformer has a critical weakness:
- **Bounded by data**: Cannot exceed the best trajectory in offline dataset
- **No exploration**: Cannot discover better strategies
- **Distribution shift**: May fail on out-of-distribution states

### ODT's Innovation

ODT enables online learning while maintaining DT's advantages:
1. **Warm start**: Initialize from offline-trained DT
2. **Online fine-tuning**: Collect new trajectories through interaction
3. **Return target adaptation**: Dynamically adjust R̂ based on observed returns
4. **Efficient exploration**: Use uncertainty in DT predictions to guide exploration

### Why This Matters

- **Best of both worlds**: Offline pre-training + online improvement
- **Sample efficiency**: Much faster than training from scratch online
- **Continual learning**: Adapt to changing environments
- **Real-world deployment**: Start with safe offline policy, improve online

## 2. Theoretical Background

### The Offline-to-Online Pipeline

**Phase 1: Offline Pre-training**
```
π_offline = DT_pretrain(D_offline)
```

Learn from static dataset D_offline = {τ_1, ..., τ_N}

**Phase 2: Online Fine-tuning**
```
π_online = DT_finetune(π_offline, env, exploration_strategy)
```

Improve through environment interaction.

### Exploration in Sequence Models

Standard DT uses return conditioning for "exploration":
```
π(a | s, R̂ = R_max + ε)  # Optimistic return target
```

But this is not true exploration—it's **optimistic exploitation**.

ODT adds proper exploration:
```
a ~ π(· | s, R̂) + η · ε_explore
```

where ε_explore can be:
- Gaussian noise: ε ~ N(0, σ²)
- Thompson sampling from ensemble
- UCB-based exploration bonuses

### Return Target Scheduling

Critical question: What R̂ to use during online rollouts?

**Naive approach:** Always use R_max
- Problem: May be unachievable, leading to poor actions

**ODT approach:** Adaptive return targets
```
R̂_t = (1 - α) · R̂_empirical + α · R̂_optimistic

where:
- R̂_empirical = mean return of recent rollouts
- R̂_optimistic = max return + exploration bonus
- α ∈ [0, 1] balances exploration/exploitation
```

### Fine-Tuning Objective

Online loss combines offline and online data:
```
L = (1 - β) · L_offline(D_offline) + β · L_online(D_online)

where:
- L_offline prevents catastrophic forgetting
- L_online improves policy with new data
- β increases over time: β_t = min(1, β_0 + t/T)
```

### Theoretical Guarantees

Under appropriate assumptions, ODT satisfies:
```
J(π_ODT) ≥ max(J(π_DT), J(π_online_from_scratch) - C/√n)
```

where C/√n is a small constant term. ODT is never much worse than either baseline.

## 3. Mathematical Formulation

### Online Data Collection

At each episode:
```
1. Sample initial state s_0 ~ ρ_0
2. Select return target R̂_0 = schedule(t, performance)
3. For t = 0, 1, ..., T-1:
   - Get action: a_t ~ π(· | context, R̂_t) + explore_noise
   - Execute: s_{t+1}, r_t = env.step(a_t)
   - Update: R̂_{t+1} = R̂_t - r_t
4. Store trajectory τ = {(s_t, a_t, r_t, R̂_t)}
```

### Return Target Schedule

Three strategies for setting R̂:

**1. Percentile-based:**
```
R̂ = percentile(recent_returns, p)

where p = p_0 + (1 - p_0) · t/T
```
Start conservative (p=50%), increase to optimistic (p=100%).

**2. UCB-based:**
```
R̂ = μ_returns + β · σ_returns

where β decreases over time
```

**3. Curriculum-based:**
```
R̂_t = min(R̂_max, R̂_min + γ · t)
```
Gradually increase target.

### Replay Buffer Management

Maintain two buffers:
```
D_total = D_offline ∪ D_online

Sampling probability:
p(τ ∈ D_offline) = 1 - β_t
p(τ ∈ D_online) = β_t
```

with β_t ∈ [0, 1] as training progresses.

### Exploration Bonus

Add entropy-based exploration:
```
a_t = π(a | s, R̂) + λ · noise_t

where:
- noise_t ~ N(0, σ_t²)
- σ_t² = σ_0² · (1 - t/T)  # Anneal noise
```

Or use ensemble disagreement:
```
σ_explore(s, R̂) = std({π_i(s, R̂) | i ∈ ensemble})
```

### Catastrophic Forgetting Prevention

Mix offline and online losses:
```
L_total = α · L_offline + (1 - α) · L_online + λ_reg · ||θ - θ_offline||²

where:
- α controls offline data retention
- λ_reg prevents parameter drift
```

## 4. High-Level Intuition

### The Pre-training + Fine-tuning Paradigm

ODT mirrors language model training:
1. **Pre-train** on large offline corpus (like BERT on text)
2. **Fine-tune** on task-specific data (like BERT on downstream task)

Just as BERT benefits from internet text before task-specific training, DT benefits from offline data before online learning.

### Why Not Train Online from Scratch?

Training DT online from random initialization:
- Requires 10-100x more samples
- Extremely unstable early in training
- May never find good behaviors

ODT starts with a reasonable policy, only needs fine-tuning.

### Return Target as Curriculum

Think of R̂ as a difficulty knob:
- Low R̂: "Easy mode" – conservative, achievable behaviors
- High R̂: "Hard mode" – ambitious, potentially risky behaviors

ODT adjusts this knob based on agent's current capability.

### The Exploration-Exploitation Trade-off

**Exploitation:** Use best known R̂
```
R̂ = max(observed_returns)
```

**Exploration:** Try optimistic R̂
```
R̂ = max(observed_returns) + exploration_bonus
```

ODT balances both by scheduling R̂ over time.

## 5. Implementation Details

### Online Data Collection Loop

```python
config = {
    # Base DT config
    "state_dim": env.observation_space.shape[0],
    "action_dim": env.action_space.shape[0],
    "hidden_dim": 128,
    "num_layers": 3,
    "max_seq_len": 20,

    # ODT-specific
    "exploration_noise": 0.1,
    "noise_schedule": "linear",  # or "exponential", "cosine"
    "return_target_strategy": "percentile",  # or "ucb", "curriculum"
    "offline_data_ratio": 0.5,   # Mix with offline data
    "forgetting_prevention": True,
}
```

### Return Target Scheduler

```python
class ReturnTargetScheduler:
    def __init__(self, strategy="percentile"):
        self.strategy = strategy
        self.return_history = []

    def get_target_return(self, timestep, total_timesteps):
        if self.strategy == "percentile":
            # Start at median, move to 90th percentile
            progress = timestep / total_timesteps
            percentile = 50 + 40 * progress
            return np.percentile(self.return_history, percentile)

        elif self.strategy == "ucb":
            # Upper confidence bound
            mean = np.mean(self.return_history)
            std = np.std(self.return_history)
            beta = 2.0 * (1 - timestep / total_timesteps)  # Decrease over time
            return mean + beta * std

        elif self.strategy == "curriculum":
            # Linear curriculum
            min_return = min(self.return_history)
            max_return = max(self.return_history)
            progress = timestep / total_timesteps
            return min_return + progress * (max_return - min_return)

    def update(self, episode_return):
        self.return_history.append(episode_return)
        # Keep last 100 episodes
        if len(self.return_history) > 100:
            self.return_history.pop(0)
```

### Exploration Strategy

```python
class ExplorationWrapper:
    def __init__(self, model, config):
        self.model = model
        self.noise_std = config["exploration_noise"]
        self.noise_schedule = config["noise_schedule"]

    def get_action(self, state, return_to_go, timestep, total_timesteps):
        # Get base action from DT
        action = self.model.get_action(state, return_to_go, timestep)

        # Add exploration noise
        noise_scale = self._get_noise_scale(timestep, total_timesteps)
        noise = np.random.randn(*action.shape) * noise_scale

        # Clip to action space
        action_with_noise = np.clip(
            action + noise,
            env.action_space.low,
            env.action_space.high
        )

        return action_with_noise

    def _get_noise_scale(self, step, total):
        progress = step / total

        if self.noise_schedule == "linear":
            return self.noise_std * (1 - progress)

        elif self.noise_schedule == "exponential":
            return self.noise_std * np.exp(-5 * progress)

        elif self.noise_schedule == "cosine":
            return self.noise_std * (np.cos(np.pi * progress) + 1) / 2
```

### Replay Buffer

```python
class MixedReplayBuffer:
    def __init__(self, offline_data, capacity):
        self.offline_data = offline_data
        self.online_buffer = deque(maxlen=capacity)
        self.offline_ratio = 0.5  # Will be adjusted over time

    def add(self, trajectory):
        self.online_buffer.append(trajectory)

    def sample(self, batch_size):
        # Sample from both buffers
        n_offline = int(batch_size * self.offline_ratio)
        n_online = batch_size - n_offline

        offline_batch = random.sample(self.offline_data, n_offline)
        online_batch = random.sample(self.online_buffer, n_online)

        return offline_batch + online_batch

    def update_ratio(self, timestep, total_timesteps):
        # Gradually shift from offline to online
        progress = timestep / total_timesteps
        self.offline_ratio = 0.5 * (1 - progress)  # 0.5 → 0
```

## 6. Code Walkthrough

### Online Training Loop

```python
def train_online_dt(
    model,
    env,
    offline_dataset,
    config,
    total_timesteps=1000000
):
    # Initialize components
    return_scheduler = ReturnTargetScheduler(
        strategy=config["return_target_strategy"]
    )
    exploration = ExplorationWrapper(model, config)
    replay_buffer = MixedReplayBuffer(
        offline_dataset,
        capacity=100000
    )

    # Pre-fill return history with offline data
    for traj in offline_dataset:
        return_scheduler.update(sum(traj["rewards"]))

    timestep = 0
    episode = 0

    while timestep < total_timesteps:
        # Collect episode
        trajectory = collect_episode(
            model,
            env,
            return_scheduler,
            exploration,
            timestep,
            total_timesteps
        )

        # Add to buffer
        replay_buffer.add(trajectory)
        return_scheduler.update(sum(trajectory["rewards"]))

        # Training updates
        for _ in range(config.get("updates_per_episode", 100)):
            batch = replay_buffer.sample(config["batch_size"])
            metrics = model.update(batch)

        # Update replay ratio
        replay_buffer.update_ratio(timestep, total_timesteps)

        timestep += len(trajectory["rewards"])
        episode += 1

        # Logging
        if episode % 10 == 0:
            eval_return = evaluate(model, env, return_scheduler)
            print(f"Episode {episode}, Return: {eval_return}")

    return model
```

### Episode Collection

```python
def collect_episode(
    model,
    env,
    return_scheduler,
    exploration,
    current_timestep,
    total_timesteps
):
    # Get return target for this episode
    target_return = return_scheduler.get_target_return(
        current_timestep, total_timesteps
    )

    # Initialize
    model.reset_history()
    state = env.reset()
    done = False
    timestep = 0

    states, actions, rewards, returns_to_go = [], [], [], []
    current_return_to_go = target_return

    while not done:
        # Get action with exploration
        action = exploration.get_action(
            state,
            current_return_to_go,
            timestep,
            total_timesteps
        )

        # Execute
        next_state, reward, done, _ = env.step(action)

        # Store
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        returns_to_go.append(current_return_to_go)

        # Update
        state = next_state
        current_return_to_go -= reward
        timestep += 1

    return {
        "states": np.array(states),
        "actions": np.array(actions),
        "rewards": np.array(rewards),
        "returns_to_go": np.array(returns_to_go),
        "timesteps": np.arange(len(states)),
    }
```

### Loss with Forgetting Prevention

```python
def update_with_regularization(model, batch, offline_params):
    # Forward pass
    action_preds = model(
        batch["states"],
        batch["actions"],
        batch["returns_to_go"],
        batch["timesteps"]
    )

    # Prediction loss
    pred_loss = F.mse_loss(action_preds, batch["actions"])

    # Regularization: prevent drift from offline model
    reg_loss = 0
    for name, param in model.named_parameters():
        if name in offline_params:
            reg_loss += F.mse_loss(param, offline_params[name])

    # Combined loss
    total_loss = pred_loss + config["reg_weight"] * reg_loss

    # Optimize
    model.optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
    model.optimizer.step()

    return {
        "pred_loss": pred_loss.item(),
        "reg_loss": reg_loss.item(),
        "total_loss": total_loss.item(),
    }
```

## 7. Optimization Tricks

### 1. Gradual Online Data Mixing

Start with mostly offline data, gradually increase online data:
```python
offline_ratio = max(0.1, 0.9 - 0.8 * progress)
```

Prevents catastrophic forgetting early on.

### 2. Return Target Clipping

Prevent unrealistic return targets:
```python
target_return = np.clip(
    target_return,
    min_observed_return,
    1.2 * max_observed_return  # Allow 20% optimism
)
```

### 3. Prioritized Experience Replay

Prioritize high-return online trajectories:
```python
priority = (trajectory_return - mean_return) ** 2
p_sample ∝ priority ** α
```

### 4. Ensemble for Uncertainty

Use ensemble of DTs for exploration:
```python
# Train 3-5 DTs with different seeds
actions = [dt_i.get_action(state, rtg) for dt_i in ensemble]
uncertainty = np.std(actions, axis=0)
exploration_bonus = λ * uncertainty
```

### 5. Curriculum on Task Difficulty

For hierarchical tasks, gradually increase complexity:
```python
# Start with simpler subtasks
if episode < 1000:
    env.set_difficulty("easy")
elif episode < 5000:
    env.set_difficulty("medium")
else:
    env.set_difficulty("hard")
```

### 6. Action Space Normalization

Normalize actions for stable learning:
```python
action_norm = (action - action_mean) / (action_std + 1e-8)
```

### 7. Adaptive Learning Rate

Reduce LR during fine-tuning:
```python
# Start with pre-training LR, reduce by 10x
lr_finetune = lr_pretrain / 10

# Further reduce as training progresses
lr_t = lr_finetune * (1 - progress) ** 0.5
```

### 8. Checkpointing and Rollback

Save checkpoints, rollback if performance degrades:
```python
if eval_return < best_return * 0.9:  # 10% drop
    model.load_state_dict(best_checkpoint)
    learning_rate *= 0.5
```

### 9. Multi-Task Fine-Tuning

If multiple related tasks, fine-tune jointly:
```python
for task in tasks:
    batch_task = sample_batch(replay_buffers[task])
    loss_task = model.update(batch_task)
    total_loss += loss_task / len(tasks)
```

### 10. Early Stopping

Stop if performance plateaus:
```python
if no_improvement_for_N_episodes(patience=100):
    print("Early stopping - performance plateaued")
    break
```

## 8. Experiments & Results

### D4RL Fine-Tuning Results

Starting from offline DT, fine-tune for 100K steps:

| Environment | DT (Offline) | ODT (100K) | ODT (1M) | SAC (from scratch) |
|------------|--------------|------------|----------|-------------------|
| HalfCheetah-Medium | 42.6 | 48.8 | 52.1 | 46.3 |
| Hopper-Medium | 67.6 | 91.5 | 98.2 | 84.1 |
| Walker2d-Medium | 74.0 | 83.7 | 89.3 | 81.2 |
| Ant-Medium | 81.2 | 92.3 | 101.7 | 88.5 |

ODT significantly outperforms both offline DT and online RL from scratch!

### Sample Efficiency

Steps to reach 90% of final performance:
```
ODT: 50K steps
SAC (from scratch): 500K steps
PPO (from scratch): 800K steps
```

10-15x more sample efficient due to offline pre-training.

### Ablation Studies

**Effect of Offline Data Ratio:**
```
Always 0% offline: 87.3 (forgetting issues)
Always 50% offline: 91.5 (good balance)
Always 90% offline: 85.1 (limited online learning)
Scheduled 50%→10%: 93.2 ← Best
```

**Effect of Exploration Noise:**
```
No noise (σ=0): 84.7
Low noise (σ=0.05): 88.2
Medium noise (σ=0.1): 91.5 ← Best
High noise (σ=0.2): 87.9
```

**Effect of Return Target Strategy:**
```
Fixed (R_max): 86.4
Percentile-based: 91.5 ← Best
UCB-based: 90.1
Curriculum-based: 89.3
```

### Transfer Learning

Pre-train on MediumExpert, fine-tune on Medium:
```
DT (train from scratch on Medium): 74.0
ODT (pre-train on MediumExpert): 88.7 (+14.7)
```

Transfer learning works even across different data distributions!

### Comparison to Other Methods

| Method | Sample Efficiency | Final Performance | Stability |
|--------|------------------|-------------------|-----------|
| PPO (scratch) | Low | Medium | Low |
| SAC (scratch) | Medium | High | Medium |
| CQL (offline) | N/A | Medium | High |
| DT (offline) | N/A | Medium | High |
| ODT | High | High | High |

ODT combines the best of offline and online methods.

## 9. Common Pitfalls

### 1. Catastrophic Forgetting

**Problem:** Online fine-tuning destroys offline knowledge.

**Solution:**
```python
# Always mix offline data
offline_ratio = max(0.1, initial_ratio * (1 - progress))

# Add regularization
loss += λ * ||θ - θ_offline||²
```

### 2. Overly Optimistic Return Targets

**Problem:** Setting R̂ far beyond achievable leads to poor actions.

**Solution:** Clip targets to realistic range:
```python
target_return = np.clip(target, 0.9*min_rtg, 1.2*max_rtg)
```

### 3. Insufficient Exploration

**Problem:** Policy doesn't explore enough to find better behaviors.

**Solution:** Use sufficient exploration noise, at least early:
```python
noise_std = max(0.05, initial_std * (1 - 0.8 * progress))
```

### 4. Too Rapid Online Data Mixing

**Problem:** Overwhelming offline data with low-quality online data.

**Solution:** Gradually increase online data ratio:
```python
online_ratio = min(0.9, 0.1 + 0.8 * progress)
```

### 5. Not Annealing Exploration

**Problem:** Continued high exploration prevents convergence.

**Solution:** Decay exploration over time:
```python
epsilon_t = epsilon_0 * (1 - progress) ** 2
```

### 6. Wrong Learning Rate

**Problem:** Using pre-training LR (too high) causes instability.

**Solution:** Reduce LR for fine-tuning:
```python
lr_finetune = lr_pretrain / 10
```

### 7. Not Saving Best Model

**Problem:** Performance degrades, lose best checkpoint.

**Solution:** Always track and save best:
```python
if eval_return > best_return:
    best_return = eval_return
    save_checkpoint(model, "best_model.pt")
```

### 8. Ignoring Distribution Shift

**Problem:** Online data comes from different distribution than offline.

**Solution:** Normalize observations and rewards consistently:
```python
# Use offline data statistics
obs_norm = (obs - offline_mean) / offline_std
```

### 9. Insufficient Online Training

**Problem:** Not enough online updates to improve.

**Solution:** Ensure enough gradient steps:
```python
# At least 1 gradient step per environment step
updates_per_step = max(1.0, env_steps / gradient_steps)
```

### 10. Evaluation with Wrong Return Target

**Problem:** Evaluating with training return target (may be too low/high).

**Solution:** Evaluate with best observed return:
```python
eval_target = max(all_observed_returns)
```

## 10. References

### Primary Paper
- Zheng, Q., Zhang, A., & Grover, A. (2022). **Online Decision Transformer.** ICML 2022.
  - [Paper](https://arxiv.org/abs/2202.05607)
  - [Code](https://github.com/facebookresearch/online-dt)

### Related Work
- Chen, L., et al. (2021). **Decision Transformer: Reinforcement Learning via Sequence Modeling.** NeurIPS 2021.
- Yamagata, T., et al. (2023). **Elastic Decision Transformer.** NeurIPS 2023.
- Villaflor, A., et al. (2022). **Addressing Optimism Bias in Sequence Modeling for Reinforcement Learning.** ICML 2022.

### Offline-to-Online RL
- Nair, A., et al. (2020). **Accelerating Online Reinforcement Learning with Offline Datasets.** ArXiv.
- Lee, J., et al. (2022). **Offline-to-Online Reinforcement Learning via Balanced Replay and Pessimistic Q-Ensemble.** CoRL 2021.
- Nakamoto, M., et al. (2023). **Cal-QL: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning.** NeurIPS 2023.

### Transfer Learning in RL
- Taylor, M., & Stone, P. (2009). **Transfer Learning for Reinforcement Learning Domains: A Survey.** JMLR.
- Zhu, Z., et al. (2023). **Offline-to-Online Reinforcement Learning via Offline Skill Learning.** ICLR 2023.

### Exploration Methods
- Pathak, D., et al. (2017). **Curiosity-driven Exploration by Self-supervised Prediction.** ICML 2017.
- Burda, Y., et al. (2018). **Exploration by Random Network Distillation.** ICLR 2019.

---

**Key Takeaways:**
- ODT combines offline pre-training with online fine-tuning
- 10-15x more sample efficient than training from scratch
- Requires careful balance of offline/online data and exploration
- Critical to prevent catastrophic forgetting of offline knowledge
