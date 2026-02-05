# DreamerV3: Mastering Diverse Domains through World Models

## 1. Overview

DreamerV3 is a general-purpose reinforcement learning algorithm that learns a world model in latent space and trains behaviors purely through imagination. It achieves human-level performance across 150+ tasks spanning Atari, DMControl, Minecraft, and more, using a single set of hyperparameters.

**Paper**: "Mastering Diverse Domains through World Models" (Hafner et al., 2023)

**Key Innovations**:
- Discrete categorical latent representations (RSSM)
- Symlog predictions for scale-invariant learning
- Fixed hyperparameters across all domains
- Training entirely in imagination (no real env steps during policy updates)

**Use Cases**:
- Vision-based continuous control (robotics)
- Video game AI (Atari, Minecraft)
- Sim-to-real transfer
- Multi-task learning
- Sample-efficient RL

## 2. Theory and Background

### 2.1 Recurrent State Space Model (RSSM)

DreamerV3's world model uses RSSM with two types of latent states:
- **Deterministic state** h_t: GRU hidden state capturing history
- **Stochastic state** z_t: Discrete categorical distribution

```
Sequence model: h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
Prior:          p(z_t | h_t)
Posterior:      q(z_t | h_t, o_t)
```

The stochastic state uses K categorical distributions with N categories each, giving N^K possible values.

### 2.2 Learning in Imagination

DreamerV3 never uses real environment interactions during policy optimization:

1. **Collect data**: Interact with env using current policy
2. **Train world model**: Learn dynamics, reward, continuation predictors
3. **Imagine trajectories**: Starting from real states, imagine H steps ahead
4. **Train actor-critic**: Optimize policy and value function on imagined data

This separates environment interaction (exploration) from policy improvement (exploitation).

### 2.3 Symlog Predictions

Symlog transform handles varying reward/value scales:

```
symlog(x) = sign(x) * ln(|x| + 1)
symexp(x) = sign(x) * (exp(|x|) - 1)
```

All predictions (rewards, values) are made in symlog space, enabling scale-invariant learning across domains.

## 3. Mathematical Formulation

### World Model Losses

**Reconstruction Loss** (observation decoder):
```
L_rec = E_q[(o_t - decode(h_t, z_t))^2]
```

**Dynamics Loss** (KL divergence):
```
L_dyn = E_q[KL(posterior || prior)] = E_q[KL(q(z_t|h_t,o_t) || p(z_t|h_t))]
```

With free nats and KL balancing:
```
L_dyn = α · max(KL(q||p), free_nats) + (1-α) · max(KL(p||q), free_nats)
```

**Reward Loss**:
```
L_rew = E_q[(symlog(r_t) - predict_reward(h_t, z_t))^2]
```

**Continuation Loss** (episode termination):
```
L_cont = E_q[BCE(1 - done_t, predict_continue(h_t, z_t))]
```

### Actor-Critic Losses

**Critic Loss** (λ-returns in symlog space):
```
V_λ^t = reward_t + γ·cont_t·((1-λ)·V(s_{t+1}) + λ·V_λ^{t+1})
L_critic = E[(symlog(V_λ) - V(s))^2]
```

**Actor Loss** (maximize returns with discount weighting):
```
w_t = Π_{τ=0}^{t-1} cont_τ  # Geometric discount
L_actor = -E[w_t · symlog(V_λ^t)]
```

The actor loss flows gradients through:
actor → actions → imagined dynamics → rewards → λ-returns

## 4. Intuitive Explanation

DreamerV3 is like an architect who:

1. **Builds a mental model** (world model): Learns how buildings work by observing many construction projects
2. **Plans in imagination** (imagined rollouts): Designs buildings mentally before actual construction
3. **Learns from simulations** (actor-critic): Improves design skills by imagining outcomes of different choices
4. **Never learns from real construction** (pure imagination): Only updates design skills from mental simulations, not from feedback during actual building

### Why This Works

- **Sample Efficiency**: One real experience trains both world model AND generates many imagined experiences for the policy
- **Stable Learning**: World model provides consistent training signal (unlike changing real environment)
- **Generalization**: Latent space captures abstract concepts that transfer across tasks
- **Credit Assignment**: Imagined trajectories provide clear causal chains

## 5. Implementation Details

### Network Architecture

```python
# RSSM
obs -> Encoder(CNN/MLP) -> embedding
h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])
z_t ~ Categorical(MLP(h_t, obs_embedding))  # Posterior
z_t ~ Categorical(MLP(h_t))                  # Prior

# World Model Predictors
features = [h_t, z_t]  # Concatenated
reward = MLP(features)  # Symlog space
continue = MLP(features)  # Binary logit
obs_rec = Decoder(features)  # Reconstruction

# Actor-Critic
action_dist = MLP(features) -> Normal(mean, std)
value = MLP(features)  # Symlog space
target_value = EMA(value)  # Slow-moving target
```

### Hyperparameters (Fixed Across All Domains)

| Parameter | Value | Description |
|-----------|-------|-------------|
| stoch_dim | 32 | Number of categorical distributions |
| num_categories | 32 | Categories per distribution |
| hidden_dim | 512 | Deterministic state size |
| imagination_horizon | 15 | Steps to imagine ahead |
| gamma | 0.997 | Discount factor (long-term planning) |
| lambda | 0.95 | GAE/λ-return parameter |
| free_nats | 1.0 | KL divergence threshold |
| kl_balance | 0.8 | Balance prior/posterior KL |
| model_lr | 1e-4 | World model learning rate |
| actor_lr | 3e-5 | Actor learning rate |
| critic_lr | 3e-5 | Critic learning rate |
| ema_decay | 0.98 | Target critic decay rate |

### Training Loop

```python
# Phase 1: Collect data
for step in range(collect_steps):
    action = actor(encode(obs))
    next_obs, reward, done = env.step(action)
    buffer.add(obs, action, reward, done)

# Phase 2: Train world model
for _ in range(model_train_iterations):
    batch = buffer.sample()
    # Encode observations
    obs_embed = encoder(batch.obs)
    # Roll out RSSM
    h, z, prior, posterior = rssm.observe_sequence(obs_embed, batch.actions)
    # Predict
    reward_pred = reward_predictor([h, z])
    continue_pred = continue_predictor([h, z])
    obs_rec = decoder([h, z])
    # Losses
    L_rec = mse(symlog(batch.obs), obs_rec)
    L_rew = mse(symlog(batch.rewards), reward_pred)
    L_cont = bce(1 - batch.dones, continue_pred)
    L_dyn = kl_balanced(posterior, prior)
    # Update
    (L_rec + L_rew + L_cont + L_dyn).backward()

# Phase 3: Train actor-critic in imagination
for _ in range(actor_critic_iterations):
    # Sample starting states from real data
    start_states = buffer.sample_states()

    # Imagine trajectories
    imagined_states, rewards, continues = [], [], []
    state = start_states
    for _ in range(imagination_horizon):
        action = actor(state)
        next_state = rssm.imagine_step(state, action)
        reward = reward_predictor(next_state)
        continue = continue_predictor(next_state)

        imagined_states.append(next_state)
        rewards.append(symexp(reward))  # Back to normal space
        continues.append(sigmoid(continue))
        state = next_state

    # Compute λ-returns
    target_values = target_critic(imagined_states)
    lambda_returns = compute_lambda_returns(rewards, continues, target_values)

    # Update critic
    values = critic(imagined_states[:-1])
    L_critic = mse(values, symlog(lambda_returns))
    L_critic.backward()

    # Update actor (gradients through dynamics!)
    L_actor = -discount_weighted_mean(symlog(lambda_returns))
    L_actor.backward()  # Flows through: actor -> dynamics -> rewards

    # Update target critic
    ema_update(critic, target_critic, ema_decay)
```

## 6. Code Walkthrough (from `/nexus/models/rl/dreamer.py`)

### RSSM Core

```python
def observe_step(self, prev_state, prev_action, obs):
    # Sequence model: h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])
    x = self.sequence_input(torch.cat([prev_state["stoch"], prev_action], dim=-1))
    x = F.silu(x)
    deter = self.gru(x, prev_state["deter"])

    # Prior: p(z_t | h_t)
    prior_logits = self.prior_net(deter).view(-1, stoch_dim, num_categories)

    # Posterior: q(z_t | h_t, o_t)
    posterior_logits = self.posterior_net(torch.cat([deter, obs], dim=-1))
    posterior_logits = posterior_logits.view(-1, stoch_dim, num_categories)

    # Sample stochastic state (straight-through estimator)
    stoch = self._sample_categorical(posterior_logits)

    new_state = {"deter": deter, "stoch": stoch.view(-1, stoch_state_size)}
    return new_state, prior_logits, posterior_logits
```

### Imagination Trajectory

```python
def _imagine_trajectory(self, start_state):
    state = start_state
    features_list, reward_preds, continue_preds = [], [], []

    for _ in range(self.imagination_horizon):
        features = self.rssm.get_features(state)
        features_list.append(features)

        # Actor selects action
        dist = self.actor_critic.get_action_distribution(features)
        action = dist.rsample()  # Reparameterization trick

        # Imagine next state (prior only, no observation)
        state, _ = self.rssm.imagine_step(state, action)

        # Predict reward and continuation
        next_features = self.rssm.get_features(state)
        reward_pred = self.world_model.reward_predictor(next_features)
        continue_pred = torch.sigmoid(self.world_model.continue_predictor(next_features))

        reward_preds.append(reward_pred)
        continue_preds.append(continue_pred)

    # Bootstrap value
    features_list.append(self.rssm.get_features(state))

    return {
        "features": torch.stack(features_list, dim=1),
        "rewards": torch.stack(reward_preds, dim=1),
        "continues": torch.stack(continue_preds, dim=1),
    }
```

### Actor Loss (with gradient flow through dynamics)

```python
def _compute_actor_loss(self, start_state):
    # Imagine trajectory (actor gradients flow through dynamics)
    imagined = self._imagine_trajectory(start_state)

    # Convert from symlog space
    rewards = symexp(imagined["rewards"].squeeze(-1))
    continues = imagined["continues"].squeeze(-1)

    # Get target values (no gradient)
    with torch.no_grad():
        target_values = self.actor_critic.get_target_value(imagined["features"])

    # Compute λ-returns (gradients flow through rewards!)
    lambda_returns = compute_lambda_returns(rewards, continues, target_values, self.gamma, self.lambda_)

    # Discount weighting
    weights = torch.cumprod(torch.cat([torch.ones_like(continues[:, :1]), continues[:, :-1]], dim=1), dim=1)

    # Actor loss: maximize discounted λ-returns
    actor_loss = -(weights * symlog(lambda_returns)).mean()

    return actor_loss
```

## 7. Optimization Tricks

### 7.1 Symlog Predictions

Always use symlog for predictions:
```python
# Predict
reward_pred_symlog = reward_net(features)
# Loss
loss = F.mse_loss(reward_pred_symlog, symlog(true_reward))
# Use
reward = symexp(reward_pred_symlog)
```

### 7.2 Free Nats for KL

Prevent posterior collapse:
```python
kl_post = kl_divergence(posterior, prior).sum(dim=-1)
kl_post = torch.clamp(kl_post, min=free_nats).mean()
```

### 7.3 KL Balancing

Balance prior and posterior:
```python
kl_loss = kl_balance * kl(q||p) + (1 - kl_balance) * kl(p||q)
```

### 7.4 Gradient Through Dynamics

Critical for actor learning:
```python
# WRONG: No gradient through dynamics
with torch.no_grad():
    imagined_trajectory = imagine(actor, dynamics)
actor_loss = -lambda_returns(imagined_trajectory)

# CORRECT: Gradient flows through dynamics
imagined_trajectory = imagine(actor, dynamics)  # No detach!
actor_loss = -lambda_returns(imagined_trajectory)
```

### 7.5 Target Critic EMA

Slow-moving target for stability:
```python
# Update target critic
for param, target_param in zip(critic.parameters(), target_critic.parameters()):
    target_param.data.copy_(ema_decay * target_param.data + (1 - ema_decay) * param.data)
```

## 8. Experimental Results

### 8.1 Atari 100k

| Algorithm | Median Human-Normalized Score |
|-----------|-------------------------------|
| DreamerV3 | 1.48 (148% human) |
| DreamerV2 | 1.08 |
| Rainbow | 0.52 |
| SimPLe | 0.12 |

### 8.2 DMControl

| Task Suite | DreamerV3 | SAC | DrQ |
|------------|-----------|-----|-----|
| Proprio | 832 / 1000 | 742 / 1000 | 658 / 1000 |
| Vision | 714 / 1000 | - | 591 / 1000 |

### 8.3 Minecraft

- **Diamond Collection**: 67% success rate (vs 0% for model-free)
- **Sample Efficiency**: 10x fewer samples than PPO
- **Transfer**: Single world model learns 20+ Minecraft tasks

### 8.4 Key Findings

- Single set of hyperparameters works across ALL domains
- Scales from Atari (discrete, visual) to DMC (continuous, proprio) to Minecraft (open-ended)
- Sample efficiency: 10-100x better than model-free methods
- Imagination quality crucial: Better world model → Better policy

## 9. Common Pitfalls and Solutions

### 9.1 Posterior Collapse

**Problem**: Posterior matches prior exactly, stochastic state becomes useless

**Solutions**:
- Use free_nats (1.0-3.0) to prevent over-compression
- KL balancing (kl_balance=0.8)
- Monitor KL divergence (should be > free_nats)

### 9.2 Reward Prediction Errors

**Problem**: Reward predictor fails on extreme values

**Solutions**:
- Always use symlog transform
- Clip predicted rewards in symlog space
- Check reward distribution (symlog should center around 0)

### 9.3 Actor Doesn't Learn

**Problem**: Actor loss doesn't decrease, policy doesn't improve

**Solutions**:
- Verify gradients flow through dynamics (no .detach())
- Check imagination horizon (try 10-20 steps)
- Ensure world model is trained first (>10k steps)
- Lower actor learning rate (3e-5 → 1e-5)

### 9.4 World Model Overfitting

**Problem**: Perfect reconstruction but poor generalization

**Solutions**:
- Use KL regularization (free_nats > 0)
- Don't overtrain world model (stop when validation loss plateaus)
- Increase stochastic capacity (more categoricals)
- Add augmentation to observations

### 9.5 Memory Issues

**Problem**: Imagination requires large batches, OOM errors

**Solutions**:
- Gradient checkpointing for RSSM
- Smaller imagination horizon
- Batch size < 50
- Reduce sequence length during world model training

## 10. Extensions and Variants

### 10.1 DreamerV3 with Hindsight

Add goal-conditioned learning:
```python
class GoalConditionedDreamer(DreamerAgent):
    def imagine_trajectory(self, start_state, goal):
        # Condition actor on goal
        action = self.actor(torch.cat([state_features, goal], -1))
        # Rest same...
```

### 10.2 Multi-Task DreamerV3

Share world model across tasks:
```python
# Task-specific reward/continue predictors
self.reward_predictors = nn.ModuleDict({
    task_id: RewardPredictor(feature_dim)
    for task_id in task_ids
})

# Shared dynamics
self.rssm = RSSM(...)  # One for all tasks
```

### 10.3 Hierarchical DreamerV3

Two-level hierarchy:
```python
# High-level policy outputs subgoals
subgoal = high_level_actor(state)
# Low-level policy conditioned on subgoal
action = low_level_actor(torch.cat([state, subgoal], -1))
```

### 10.4 Model Ensemble

Multiple world models for uncertainty:
```python
self.world_models = nn.ModuleList([WorldModel(...) for _ in range(n_models)])
# Randomly select one per imagination
model = random.choice(self.world_models)
trajectory = model.imagine(...)
```

## 11. References

1. **DreamerV3**: Hafner et al., "Mastering Diverse Domains through World Models", 2023 [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)

2. **DreamerV2**: Hafner et al., "Mastering Atari with Discrete World Models", ICLR 2021

3. **DreamerV1**: Hafner et al., "Dream to Control: Learning Behaviors by Latent Imagination", ICLR 2020

4. **RSSM**: Hafner et al., "Learning Latent Dynamics for Planning from Pixels", ICML 2019

5. **PlaNet**: Hafner et al., "Learning Latent Dynamics for Planning from Pixels", ICML 2019

### Implementation References

6. [Official DreamerV3](https://github.com/danijar/dreamerv3): JAX implementation
7. [DreamerV3 PyTorch](https://github.com/NM512/dreamerv3-torch): Community PyTorch port

### Related Topics

8. Model-Based RL Survey: Moerland et al., "Model-based Reinforcement Learning: A Survey", 2023
9. World Models: Ha & Schmidhuber, "World Models", NeurIPS 2018
