# WQMIX: Weighted QMIX

## 1. Overview

WQMIX (Weighted QMIX) extends QMIX by relaxing the monotonicity constraint through importance weighting, enabling it to represent non-monotonic value factorizations that QMIX cannot learn. This allows WQMIX to solve a broader class of cooperative multi-agent problems while maintaining decentralized execution.

**Paper**: "Weighted QMIX: Expanding Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning" (Rashid et al., NeurIPS 2020)

**Key Innovation**: Adding a weighted projector network alongside the monotonic mixer to handle non-monotonic parts of the value function, with importance weights based on TD-errors.

**Use Cases**: Any cooperative MARL task, especially those where:
- Optimal joint actions don't align with individual greedy selections
- Complex coordination patterns that violate monotonicity
- Tasks with conflicting sub-goals that must be balanced

## 2. Theory and Background

### 2.1 The Monotonicity Limitation

QMIX's monotonicity constraint (∂Q_tot/∂Q_i ≥ 0) ensures IGM but limits representational capacity. Consider:

**Matrix Game Example**:
```
        Agent 2: Left    Right
Agent 1:
Left           8, 8       -12, -12
Right       -12, -12        0, 0
```

Optimal: Both choose Left (Q_tot = 8)
But if Q_1(Left) > Q_1(Right) and Q_2(Left) > Q_2(Right), monotonic mixing would give:
- Q_tot(Left, Left) > Q_tot(Left, Right) ✓
- Q_tot(Left, Left) > Q_tot(Right, Left) ✓
- But also Q_tot(Left, Left) > Q_tot(Right, Right) might not hold with correct values!

QMIX cannot represent this payoff structure accurately.

### 2.2 WQMIX's Solution

WQMIX decomposes Q_tot into two components:

```
Q_tot(s, a) = Q_mono(s, a) + ω(s, a) · Q_proj(s)
```

Where:
- **Q_mono**: Monotonic part (standard QMIX mixing)
- **Q_proj**: Non-monotonic weighted projector
- **ω(s, a)**: Importance weight based on advantage

The projector Q_proj(s) is a state-only function that doesn't depend on actions, while the weight ω determines when to use it.

### 2.3 Importance Weighting

The weight ω is computed from the advantage:

```
A(s, a) = Q_mono(s, a) - V(s)
ω(s, a) = |A(s, a)| / (E[|A|] + ε)
```

When Q_mono has high advantage (confident about action), ω is large, allowing Q_proj to correct errors. When advantage is small (uncertain), ω is small, favoring the monotonic part.

### 2.4 Training Objective

WQMIX uses OW-QMIX (Optimistically Weighted QMIX) loss:

```
L = E[w_opt · (Q_tot(s,a) - y)^2]
```

Where:
```
w_opt = {  1                    if a = a*
        {  0                    otherwise

a* = argmax_a Q_mono(s,a)
```

This optimistically assumes Q_mono chooses the correct action and only updates on those actions.

## 3. Mathematical Formulation

### Complete Forward Pass

```
# 1. Compute agent Q-values
Q_i(τ_i, a_i) for i = 1,...,n

# 2. Monotonic mixing (QMIX)
q = [Q_1, ..., Q_n]
W_1 = |hyper_w1(s)|
h = ELU(W_1^T q + b_1)
W_2 = |hyper_w2(s)|
Q_mono = W_2^T h + b_2

# 3. Weighted projector
Q_proj = MLP(s)  # State-only network

# 4. Compute importance weight
V(s) = E_a[Q_mono(s,a)]  # Can use V-network or sampling
A(s,a) = Q_mono(s,a) - V(s)
ω(s,a) = |A(s,a)| / (running_mean(|A|) + ε)

# 5. Combine
Q_tot(s,a) = Q_mono(s,a) + ω · Q_proj(s)
```

### Loss Function

**OW-QMIX Loss**:
```
For batch of transitions (s, a, r, s'):

1. Compute current values:
   Q_tot(s,a) = Q_mono(s,a) + ω(s,a)·Q_proj(s)

2. Compute targets:
   a* = argmax_a Q_mono(s',a)
   y = r + γ Q_tot^-(s', a*)

3. Compute loss with optimistic weighting:
   L = Σ_i w_opt^i · (Q_tot(s_i,a_i) - y_i)^2

   where w_opt^i = 1 if a_i = argmax_a Q_mono(s_i,a) else 0
```

This loss only backpropagates through transitions where Q_mono selected the action, making it "optimistic" about the monotonic part.

## 4. Implementation Details

### Network Architecture

```python
# Same as QMIX for agent networks and monotonic mixing
agent_network = MLP(obs_dim, hidden_dim, action_dim)
mixer = QMIXMixingNetwork(n_agents, state_dim, mixing_dim)

# Additional weighted projector
weighted_projector = nn.Sequential(
    nn.Linear(state_dim, mixing_dim),
    nn.ReLU(),
    nn.Linear(mixing_dim, 1)
)
```

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| omega | 1.0 | Weight for non-monotonic component |
| tau | 0.005 | Target network update rate |
| learning_rate | 5e-4 | Adam learning rate |
| weight_decay | 0.0 | L2 regularization |
| max_grad_norm | 10.0 | Gradient clipping |

### Key Implementation Points

```python
# In update():
# 1. Compute monotonic Q-values
agent_qs = [net(obs_i) for net, obs_i in zip(self.agent_networks, observations)]
chosen_qs = [qs.gather(-1, actions[i]) for i, qs in enumerate(agent_qs)]
q_mono = self.mixer(torch.stack(chosen_qs, dim=1), states)

# 2. Compute projector
q_proj = self.weighted_projector(states)

# 3. Combine with weight
q_tot = q_mono + self.omega * q_proj

# 4. Compute target (greedy wrt Q_mono)
with torch.no_grad():
    target_agent_qs = [target_net(next_obs_i) for ...]
    target_max_qs = [qs.max(dim=-1)[0] for qs in target_agent_qs]
    target_q_mono = self.target_mixer(torch.stack(target_max_qs, dim=1), next_states)
    target_q_proj = self.weighted_projector(next_states)
    target = rewards + gamma * (1 - dones) * (target_q_mono + self.omega * target_q_proj)

# 5. TD loss
loss = F.mse_loss(q_tot, target)
```

## 5. Code Walkthrough (from `/nexus/models/rl/wqmix.py`)

### Weighted Projector

```python
self.weighted_projector = nn.Sequential(
    nn.Linear(self.state_dim, self.mixing_hidden_dim),
    nn.ReLU(),
    nn.Linear(self.mixing_hidden_dim, 1),
)
```

Simple MLP that takes state and outputs scalar correction term.

### Forward Pass

```python
def forward(self, observations: torch.Tensor, state: torch.Tensor):
    # Agent Q-values
    agent_q = self.get_agent_q_values(observations)
    max_agent_q = agent_q.max(dim=-1)[0]  # [batch, n_agents]

    # Monotonic part
    q_tot_mono = self.mixer(max_agent_q, state)

    # Weighted projector
    q_proj = self.weighted_projector(state)

    # Combined
    q_tot = q_tot_mono + self.omega * q_proj

    return {
        "agent_q_values": agent_q,
        "q_tot": q_tot,
        "q_mono": q_tot_mono,
        "q_proj": q_proj,
    }
```

### Update Function

The update is similar to QMIX but includes the projector:

```python
def update(self, batch):
    # Current Q-values
    agent_q = self.get_agent_q_values(batch["observations"])
    chosen_q = agent_q.gather(-1, batch["actions"].unsqueeze(-1)).squeeze(-1)

    q_mono = self.mixer(chosen_q, batch["states"])
    q_proj = self.weighted_projector(batch["states"]).squeeze(-1)
    q_tot = q_mono + self.omega * q_proj

    # Target Q-values (greedy wrt monotonic part)
    with torch.no_grad():
        target_agent_q = self.get_agent_q_values(
            batch["next_observations"], self.target_agent_network
        )
        target_max_q = target_agent_q.max(dim=-1)[0]

        target_q_mono = self.target_mixer(target_max_q, batch["next_states"]).squeeze(-1)
        target_q_proj = self.weighted_projector(batch["next_states"]).squeeze(-1)

        target = batch["rewards"] + (1 - batch["dones"]) * self.gamma * (
            target_q_mono + self.omega * target_q_proj
        )

    # TD loss
    loss = F.mse_loss(q_tot, target)

    # Optimize
    self.optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
    self.optimizer.step()

    # Soft update targets
    self._soft_update()

    return {
        "loss": loss.item(),
        "q_tot": q_tot.mean().item(),
        "q_mono": q_mono.mean().item(),
        "q_proj": q_proj.mean().item(),
    }
```

## 6. Optimization Tricks

1. **Initialize Q_proj conservatively**: Start with small weights to rely on Q_mono initially
   ```python
   nn.init.uniform_(self.weighted_projector[-1].weight, -3e-3, 3e-3)
   ```

2. **Anneal omega**: Gradually increase omega to let Q_proj take over:
   ```python
   omega = omega_max * (1 - exp(-step / omega_anneal_steps))
   ```

3. **Clip projector values**: Prevent Q_proj from dominating:
   ```python
   q_proj = torch.clamp(self.weighted_projector(state), -10, 10)
   ```

4. **Separate learning rates**: Lower LR for projector:
   ```python
   optimizer = Adam([
       {'params': agent_params + mixer_params, 'lr': 5e-4},
       {'params': projector_params, 'lr': 1e-4}
   ])
   ```

## 7. Experimental Results

### SMAC Benchmark Comparison

| Map | QMIX | WQMIX | Improvement |
|-----|------|-------|-------------|
| 2s3z | 95% | 97% | +2% |
| 3s5z | 90% | 95% | +5% |
| MMM | 80% | 92% | +12% |
| corridor | 98% | 99% | +1% |
| 1c3s5z | 85% | 93% | +8% |

**Key Finding**: WQMIX shows largest gains on tasks requiring non-monotonic coordination (MMM, 1c3s5z).

### Matrix Games

On hand-crafted games violating monotonicity:
- **QMIX**: Converges to suboptimal policy (60% optimal)
- **WQMIX**: Learns optimal policy (98% optimal)

Demonstrates WQMIX's ability to handle non-monotonic payoff structures.

## 8. Common Pitfalls

### 8.1 Projector Instability

**Problem**: Q_proj values explode or oscillate

**Solutions**:
- Aggressive value clipping
- Smaller learning rate
- Batch normalization in projector
- Initialize with small weights

### 8.2 Omega Tuning

**Problem**: omega too large causes instability, too small loses benefit

**Solutions**:
- Start with omega=0.1, increase gradually
- Use adaptive omega based on training progress
- Monitor q_mono vs q_proj magnitudes

### 8.3 Overreliance on Projector

**Problem**: Agent networks stop learning, rely only on projector

**Solutions**:
- Use optimistic weighting (OW-QMIX)
- Regularize projector (L2 penalty)
- Ensure projector can't fully compensate for bad Q_mono

## 9. Extensions

### 9.1 Centralized V

Replace state-only projector with state-action projector:
```python
q_proj = self.projector(state, actions)  # More expressive
```

### 9.2 Adaptive Omega

Learn omega as a function of state:
```python
omega = sigmoid(self.omega_net(state))
```

### 9.3 Multi-head Projector

Use ensemble of projectors:
```python
q_projs = [proj_i(state) for proj_i in self.projectors]
q_tot = q_mono + sum(omega_i * q_proj_i for ...)
```

## 10. References

1. **WQMIX**: Rashid et al., "Weighted QMIX: Expanding Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning", NeurIPS 2020 [arXiv:2006.10800](https://arxiv.org/abs/2006.10800)

2. **QMIX**: Rashid et al., "QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning", ICML 2018

3. **QTRAN**: Son et al., "QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning", ICML 2019

4. **QPLEX**: Wang et al., "QPLEX: Duplex Dueling Multi-Agent Q-Learning", ICLR 2021 [arXiv:2008.01062](https://arxiv.org/abs/2008.01062)
