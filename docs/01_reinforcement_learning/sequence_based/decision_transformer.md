# Decision Transformer

## 1. Overview & Motivation

Decision Transformer (DT) revolutionizes offline reinforcement learning by reframing it as a **sequence modeling problem** rather than a traditional RL problem. This paradigm shift eliminates the need for Bellman backups, bootstrapping, and dynamic programming, instead leveraging the power of transformer architectures to model trajectories.

### Key Insight
Traditional RL learns Q(s,a) or π(a|s). Decision Transformer learns π(a|s, R̂), where R̂ is the **desired return-to-go**. This conditioning on future returns enables:
- **Goal-conditioned behavior**: Specify desired performance at inference time
- **Trajectory stitching**: Combine suboptimal trajectories to produce optimal behavior
- **Simplicity**: No value function bootstrapping or policy gradients needed

### Why It Matters
- Achieves state-of-the-art on D4RL offline RL benchmarks
- More stable than value-based offline RL (no extrapolation error)
- Naturally handles credit assignment through attention
- Enables in-context learning and few-shot adaptation

## 2. Theoretical Background

### Problem Formulation

In standard RL, we maximize expected return:
```
π* = argmax_π E[Σ γ^t r_t | π]
```

Decision Transformer reframes this as **conditional sequence modeling**:
```
π(a_t | s_t, R̂_t) where R̂_t = Σ_{t'=t}^T r_{t'}
```

The key is conditioning on **returns-to-go** (R̂), which represent desired future cumulative reward.

### Autoregressive Factorization

The model autoregressively predicts actions given the history:
```
p(τ) = Π_t p(a_t | R̂_t, s_t, a_{t-1}, R̂_{t-1}, s_{t-1}, ..., R̂_1, s_1)
```

where τ = (R̂_1, s_1, a_1, R̂_2, s_2, a_2, ..., R̂_T, s_T, a_T)

### Connection to Imitation Learning

When R̂ = expert return, DT performs **goal-conditioned imitation learning**. The model learns to mimic behaviors that achieve specific returns, enabling performance interpolation and extrapolation.

### Trajectory Stitching Property

Unlike behavior cloning, DT can stitch together suboptimal trajectories:
- If trajectory A reaches high-value state S but fails afterward
- And trajectory B starts from similar state S and succeeds
- DT can combine them by conditioning on high returns throughout

This is because attention allows the model to look at all high-return subsequences in the dataset.

## 3. Mathematical Formulation

### Sequence Representation

Each timestep has three tokens:
```
τ = (R̂_1, s_1, a_1, R̂_2, s_2, a_2, ..., R̂_T, s_T, a_T)
```

where:
- R̂_t ∈ ℝ: return-to-go at time t
- s_t ∈ ℝ^d_s: state at time t
- a_t ∈ ℝ^d_a: action at time t

### Embedding Function

Each modality is embedded separately:
```
e_R(R̂_t) = W_R · R̂_t + E_time(t)
e_s(s_t) = W_s · s_t + E_time(t)
e_a(a_t) = W_a · a_t + E_time(t)
```

where E_time is a learned timestep embedding.

### Transformer Backbone

The stacked embeddings are processed by a GPT-style transformer:
```
h^(0) = [e_R(R̂_1), e_s(s_1), e_a(a_1), ..., e_R(R̂_T), e_s(s_T), e_a(a_T)]
h^(l+1) = TransformerBlock(h^(l))
```

### Prediction Heads

Action prediction uses the state token's hidden representation:
```
â_t = tanh(W_out · h_{s_t}^(L))
```

The loss is mean squared error:
```
L = E_τ [ Σ_t ||â_t - a_t||^2 ]
```

### Causal Masking

Crucial detail: The attention mask ensures:
- R̂_t can attend to: R̂_{≤t}, s_{<t}, a_{<t}
- s_t can attend to: R̂_{≤t}, s_{≤t}, a_{<t}
- a_t can attend to: R̂_{≤t}, s_{≤t}, a_{<t}

This maintains the autoregressive property.

## 4. High-Level Intuition

### The Core Idea: "Hindsight is 20/20"

Imagine you're learning to play a video game by watching replays. Traditional RL asks: "What should I do in this state?" Decision Transformer asks: "What did players do when they were heading toward a score of X?"

### Why Conditioning on Returns Works

The key insight: **actions are determined by intentions**. A chess player heading toward a draw plays differently than one aiming for checkmate. By conditioning on desired outcomes, the model learns the relationship between goals and behaviors.

### The Power of Attention

The transformer can attend to any relevant past experience:
- "When I was in a similar state heading toward high reward, what did I do?"
- "What sequence of actions led to the desired outcome in the dataset?"

This enables trajectory stitching without explicit planning or value estimation.

### Comparison to Traditional RL

**Traditional RL:**
```
Q(s, a) ← r + γ max_a' Q(s', a')  [Bootstrap from estimates]
```

**Decision Transformer:**
```
a ~ π(·|s, R̂)  [Generate action conditioned on desired return]
```

No bootstrapping = no error propagation through Bellman backups!

### Inference-Time Control

At test time, you can:
- Set R̂ = max observed return → best behavior
- Set R̂ = median return → conservative behavior
- Set R̂ > max return → attempt extrapolation (risky but possible)

## 5. Implementation Details

### Architecture Specifications

From `Nexus/nexus/models/rl/decision_transformer.py`:

```python
config = {
    "state_dim": env.observation_space.shape[0],
    "action_dim": env.action_space.shape[0],
    "hidden_dim": 128,          # Transformer hidden size
    "num_layers": 3,            # Transformer layers
    "num_heads": 1,             # Attention heads (1 works well!)
    "max_ep_len": 1000,        # Maximum episode length
    "max_seq_len": 20,         # Context window (K in paper)
    "dropout": 0.1,
    "action_tanh": True,       # For continuous actions
}
```

### Token Stacking Strategy

The implementation stacks tokens as:
```python
# Shape: (batch, seq_len, 3, hidden_dim)
stacked = torch.stack([
    return_embeddings,  # R̂_t
    state_embeddings,   # s_t
    action_embeddings   # a_t
], dim=2)

# Flatten to: (batch, 3*seq_len, hidden_dim)
stacked = stacked.reshape(batch, 3*seq_len, hidden_dim)
```

### Positional Encoding

Uses timestep embeddings added to each token:
```python
time_embeddings = self.embed_timestep(timesteps)
state_embeddings = self.embed_state(states) + time_embeddings
```

This is crucial for temporal reasoning.

### Context Window Management

The model uses a sliding window of the last K timesteps:
```python
context_len = min(len(history), max_seq_len)
states = states[-context_len:]  # Last K states
```

Typical K values: 10-20 (more doesn't always help due to attention dilution)

### Training Hyperparameters

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)

# Crucial: Learning rate warmup
scheduler = LambdaLR(optimizer, lambda step: min(1.0, step / 1000))

# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
```

## 6. Code Walkthrough

### Model Architecture

```python
class DecisionTransformer(NexusModule):
    def __init__(self, config):
        # Separate embeddings for each modality
        self.embed_return = nn.Linear(1, hidden_dim)
        self.embed_state = nn.Linear(state_dim, hidden_dim)
        self.embed_action = nn.Linear(action_dim, hidden_dim)
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_dim)

        # GPT-style transformer with causal masking
        self.transformer = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Only predict actions (not states or returns)
        self.predict_action = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # For continuous control
        )
```

### Forward Pass

```python
def forward(self, states, actions, returns_to_go, timesteps):
    # Embed each modality + add time
    state_emb = self.embed_state(states) + self.embed_timestep(timesteps)
    action_emb = self.embed_action(actions) + self.embed_timestep(timesteps)
    return_emb = self.embed_return(returns_to_go) + self.embed_timestep(timesteps)

    # Stack as (R, s, a, R, s, a, ...)
    stacked = torch.stack([return_emb, state_emb, action_emb], dim=2)
    stacked = stacked.reshape(batch, 3*seq_len, hidden_dim)
    stacked = self.embed_ln(stacked)

    # Process through transformer
    x = stacked
    for block in self.transformer:
        x = block(x)  # Includes causal attention

    # Extract action predictions from state positions
    x = x.reshape(batch, seq_len, 3, hidden_dim)
    action_preds = self.predict_action(x[:, :, 1])  # From state token

    return action_preds
```

### Causal Self-Attention

```python
class CausalSelfAttention(nn.Module):
    def forward(self, x):
        # Compute Q, K, V
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply causal mask (crucial!)
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        return (attn @ v)
```

### Training Loop

```python
def update(self, batch):
    states = batch["states"]              # (B, K, state_dim)
    actions = batch["actions"]            # (B, K, action_dim)
    returns_to_go = batch["returns_to_go"]  # (B, K, 1)
    timesteps = batch["timesteps"]        # (B, K)

    # Forward pass
    action_preds = self.model(states, actions, returns_to_go, timesteps)

    # MSE loss
    loss = F.mse_loss(action_preds, actions)

    # Optimize
    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.parameters(), 0.25)
    self.optimizer.step()
    self.scheduler.step()

    return {"loss": loss.item()}
```

### Inference

```python
def select_action(self, state, target_return, timestep):
    # Add to history
    self.state_history.append(state)
    self.return_history.append(target_return)

    # Get last K timesteps
    context = self.max_seq_len
    states = torch.cat(self.state_history[-context:])
    returns = torch.cat(self.return_history[-context:])
    actions = torch.cat(self.action_history[-context:])

    # Predict action
    with torch.no_grad():
        action = self.model.get_action(states, actions, returns, timesteps)

    # Add to history
    self.action_history.append(action)

    return action
```

## 7. Optimization Tricks

### 1. Learning Rate Warmup

Critical for stable training:
```python
# Warmup over first 1000 steps
scheduler = LambdaLR(optimizer, lambda step: min(1.0, step / 1000))
```

Without warmup, gradients can be unstable early in training.

### 2. Gradient Clipping

Prevents exploding gradients in transformer:
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
```

### 3. Layer Normalization

Pre-norm formulation is crucial:
```python
x = x + self.attn(self.ln1(x))  # LayerNorm before attention
x = x + self.mlp(self.ln2(x))   # LayerNorm before MLP
```

### 4. Context Length Selection

Tuning K (context length):
- Too small: Can't capture long-term dependencies
- Too large: Attention dilution, slower training
- Sweet spot: 10-20 timesteps for most tasks

### 5. Return Normalization

Normalize returns to [-1, 1] or [0, 1]:
```python
returns_to_go = (returns_to_go - mean) / (std + 1e-6)
```

Improves stability and convergence.

### 6. Action Space Handling

For discrete actions:
```python
self.predict_action = nn.Linear(hidden_dim, action_dim)  # No tanh
loss = F.cross_entropy(action_preds, actions)
```

For continuous actions:
```python
self.predict_action = nn.Sequential(
    nn.Linear(hidden_dim, action_dim),
    nn.Tanh()  # Bound to [-1, 1]
)
loss = F.mse_loss(action_preds, actions)
```

### 7. Timestep Embedding

Use learned embeddings rather than sinusoidal:
```python
self.embed_timestep = nn.Embedding(max_ep_len, hidden_dim)
```

Works better for RL where position = time has semantic meaning.

### 8. Weight Initialization

Use GPT-2 style initialization:
```python
def _init_weights(self, module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
```

### 9. Batch Construction

Sample trajectories with diverse returns:
```python
# Stratified sampling by return
high_return_trajs = sample(trajs[trajs.return > p90])
medium_return_trajs = sample(trajs[(trajs.return > p50) & (trajs.return < p90)])
low_return_trajs = sample(trajs[trajs.return < p50])
```

This ensures the model sees the full spectrum of behaviors.

### 10. Evaluation Protocol

Use percentile-based target returns:
```python
# Test with different return targets
for target_return in [p50, p75, p90, p100]:
    eval_performance = evaluate(model, env, target_return)
```

## 8. Experiments & Results

### D4RL Benchmark Performance

Decision Transformer achieves competitive or superior performance to conservative offline RL methods:

| Environment | BC | CQL | DT | DT (Online DT) |
|------------|-----|-----|-----|----------------|
| HalfCheetah-Medium | 42.6 | 44.0 | 42.6 | 48.8 |
| Hopper-Medium | 52.9 | 58.5 | 67.6 | 91.5 |
| Walker2d-Medium | 75.3 | 72.5 | 74.0 | 83.7 |
| HalfCheetah-Medium-Replay | 36.6 | 45.5 | 36.6 | 47.7 |
| Hopper-Medium-Replay | 18.1 | 95.0 | 82.7 | 100.9 |

### Key Observations

1. **Trajectory Stitching Works**: DT outperforms BC especially on medium-replay datasets, which contain diverse suboptimal trajectories.

2. **Context Length Matters**: Performance plateaus around K=20. Longer context doesn't help and slows training.

3. **Return Conditioning is Crucial**: Ablating return conditioning reduces DT to behavior cloning (much worse).

4. **Timestep Embeddings Matter**: Removing them drops performance by 10-20%, showing temporal information is critical.

### Ablation Studies

**Effect of Context Length:**
```
K=5:  72.3
K=10: 76.8
K=20: 82.7 ← Sweet spot
K=40: 81.2 (attention dilution)
```

**Effect of Model Size:**
```
1 layer:  68.4
3 layers: 82.7 ← Default
6 layers: 84.1 (diminishing returns)
```

**Effect of Hidden Dimension:**
```
64:  77.2
128: 82.7 ← Default
256: 83.5
```

### Inference-Time Behavior

Testing different target returns:
```python
target = min_return  →  Conservative behavior, low variance
target = median      →  Average performance
target = max_return  →  Best observed behavior
target = 1.2 * max   →  Extrapolation (sometimes works!)
```

### Computational Efficiency

Training time on single V100 GPU:
- Medium dataset (~1M transitions): ~2-3 hours
- Large dataset (~10M transitions): ~12-15 hours

Much faster than iterative offline RL methods (CQL, IQL).

## 9. Common Pitfalls

### 1. Not Normalizing Returns

**Problem:** Returns can span orders of magnitude, making learning unstable.

**Solution:**
```python
returns = (returns - dataset.mean_return) / dataset.std_return
```

### 2. Incorrect Token Ordering

**Problem:** Mismatch between embedding order and prediction order.

**Solution:** Always predict action from state token, not return token:
```python
# Correct
action_preds = self.predict_action(hidden_states[:, :, 1])  # State position

# Wrong
action_preds = self.predict_action(hidden_states[:, :, 0])  # Return position
```

### 3. Forgetting Causal Masking

**Problem:** Without causal mask, model can "cheat" by looking at future tokens.

**Solution:** Always use lower-triangular attention mask:
```python
mask = torch.tril(torch.ones(seq_len, seq_len))
attn = attn.masked_fill(mask == 0, float('-inf'))
```

### 4. Context Window Too Long

**Problem:** K > 50 often hurts performance due to attention dilution.

**Solution:** Start with K=20 and tune if needed. More isn't always better.

### 5. Wrong Loss Function

**Problem:** Using cross-entropy for continuous actions or MSE for discrete.

**Solution:** Match loss to action space:
- Continuous: MSE loss
- Discrete: Cross-entropy loss

### 6. Insufficient Training Data

**Problem:** DT needs diverse trajectories to learn trajectory stitching.

**Solution:** Ensure dataset has:
- Multiple return levels
- Various trajectory lengths
- Diverse initial states

### 7. Not Using Learning Rate Warmup

**Problem:** Transformer training can be unstable without warmup.

**Solution:** Always use warmup:
```python
scheduler = LambdaLR(optimizer, lambda s: min(1.0, s / warmup_steps))
```

### 8. Overfitting to High Returns

**Problem:** If dataset is imbalanced toward low returns, model may not learn high-return behavior.

**Solution:** Stratified sampling or upweight high-return trajectories:
```python
weights = (trajectory_returns - min_return) / (max_return - min_return)
sample_probs = weights / weights.sum()
```

### 9. Timestep Overflow

**Problem:** Episode length exceeds max_ep_len, causing embedding index error.

**Solution:** Use modulo or clipping:
```python
timesteps = torch.clamp(timesteps, 0, max_ep_len - 1)
```

### 10. Not Handling Episode Boundaries

**Problem:** Concatenating across episodes breaks temporal structure.

**Solution:** Either:
- Reset context at episode boundaries
- Use attention mask to prevent cross-episode attention

## 10. References

### Primary Paper
- Chen, L., Lu, K., Rajeswaran, A., Lee, K., Grover, A., Laskin, M., ... & Mordatch, I. (2021). **Decision Transformer: Reinforcement Learning via Sequence Modeling.** NeurIPS 2021.
  - [Paper](https://arxiv.org/abs/2106.01345)
  - [Official Code](https://github.com/kzl/decision-transformer)

### Follow-Up Works
- Yamagata, T., et al. (2023). **Elastic Decision Transformer.** NeurIPS 2023.
- Zheng, Q., et al. (2022). **Online Decision Transformer.** ICML 2022.
- Furuta, H., et al. (2022). **Generalized Decision Transformer for Offline Hindsight Information Matching.** ICLR 2022.

### Related Approaches
- Janner, M., Li, Q., & Levine, S. (2021). **Offline Reinforcement Learning as One Big Sequence Modeling Problem.** NeurIPS 2021.
- Lee, K., et al. (2022). **Multi-Game Decision Transformers.** NeurIPS 2022.

### Transformer Background
- Vaswani, A., et al. (2017). **Attention Is All You Need.** NeurIPS 2017.
- Radford, A., et al. (2019). **Language Models are Unsupervised Multitask Learners.** (GPT-2)

### Offline RL Baselines
- Kumar, A., Zhou, A., Tucker, G., & Levine, S. (2020). **Conservative Q-Learning for Offline Reinforcement Learning.** NeurIPS 2020.
- Kostrikov, I., et al. (2021). **Offline Reinforcement Learning with Implicit Q-Learning.** ICLR 2022.

### Implementation Reference
- Nexus Implementation: `Nexus/nexus/models/rl/decision_transformer.py`

---

**Next Steps:**
- Explore **Elastic Decision Transformer** for adaptive context lengths
- Try **Online Decision Transformer** for fine-tuning with online data
- Investigate **Q-Learning Decision Transformer** for value-based improvements
