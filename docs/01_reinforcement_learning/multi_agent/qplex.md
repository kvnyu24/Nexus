# QPLEX: Duplex Dueling Multi-Agent Q-Learning

## 1. Overview

QPLEX (Q-value Partial Factorization with Duplex Dueling) achieves complete expressiveness for value function factorization through a duplex dueling architecture that separates state value and advantage functions. Unlike QMIX's monotonicity constraint, QPLEX can represent any joint Q-function while maintaining the Individual-Global-Max (IGM) property.

**Paper**: "QPLEX: Duplex Dueling Multi-Agent Q-Learning" (Wang et al., ICLR 2021)

**Key Innovation**: Transformation function that converts individual advantages into joint advantages through a duplex dueling structure, achieving complete IGM factorization without monotonicity constraints.

**Status**: ⚠️ **NOT YET IMPLEMENTED** - Documentation prepared for future implementation

**Use Cases**:
- Complex cooperative tasks requiring non-monotonic coordination
- Tasks where QMIX fails due to monotonicity limitations
- Scenarios needing maximum representational capacity
- StarCraft micromanagement with intricate coordination patterns

## 2. Theory and Background

### 2.1 The Factorization Problem Revisited

Previous methods have limitations:
- **VDN**: Additive factorization Q_tot = Σ Q_i (too restrictive)
- **QMIX**: Monotonic mixing Q_tot = f_mono(Q_1,...,Q_n) (cannot represent all functions)
- **QTRAN**: Complex architecture with factorization constraints (training instability)
- **WQMIX**: Weighted relaxation (still limited expressiveness)

**QPLEX's Goal**: Achieve complete factorization - represent any joint Q-function while maintaining IGM.

### 2.2 Duplex Dueling Architecture

QPLEX decomposes Q-values using dueling networks:

```
Q_i(τ_i, a_i) = V_i(τ_i) + A_i(τ_i, a_i)
Q_tot(s, a) = V_tot(s) + A_tot(s, a)
```

The key insight: factorize advantage functions instead of Q-functions directly.

### 2.3 Transformation Function

QPLEX introduces a transformation function T that converts individual advantages to joint advantage:

```
A_tot(s, a) = T(A_1(τ_1, a_1), ..., A_n(τ_n, a_n); s)
```

The transformation satisfies:
1. **IGM Condition**: argmax_a A_tot(s,a) = (argmax_a1 A_1,..., argmax_an A_n)
2. **Zero Constraint**: T(0,...,0; s) = 0 (when all agents choose greedy actions)

### 2.4 Complete Factorization

QPLEX's full factorization:

```
Q_tot(s, a) = V_tot(s) + T(
    A_1(τ_1, a_1) - A_1(τ_1, a_1*),
    ...,
    A_n(τ_n, a_n) - A_n(τ_n, a_n*);
    s
)
```

Where a_i* = argmax A_i(τ_i, a_i).

This guarantees:
- When all agents pick greedy actions: Q_tot(s, a*) = V_tot(s)
- Otherwise: Q_tot reflects deviation from greedy choices
- Can represent any factorizable joint Q-function

## 3. Mathematical Formulation

### Agent Networks

Each agent has:
1. **Value network**: V_i(τ_i)
2. **Advantage network**: A_i(τ_i, a_i)
3. **Q-function**: Q_i(τ_i, a_i) = V_i(τ_i) + A_i(τ_i, a_i) - mean_a A_i(τ_i, a)

### Transformation Network

The transformation T uses attention mechanism:

```
# 1. Compute advantage differences
Δ_i = A_i(τ_i, a_i) - A_i(τ_i, a_i*)

# 2. Attention-weighted aggregation
query = W_q(s)
keys = [W_k(Δ_i) for i in 1..n]
values = [W_v(Δ_i) for i in 1..n]

attention_weights = softmax(query @ keys^T / √d)
A_tot(s,a) = attention_weights @ values
```

### Loss Function

QPLEX optimizes the standard TD error:

```
L = E[(Q_tot(s,a) - y)^2]
y = r + γ max_a' Q_tot(s', a')
```

With consistency regularization:

```
L_total = L_TD + λ L_consistency

L_consistency = E[(Q_tot(s,a*) - V_tot(s))^2]
```

Where a* are greedy actions from individual agents. This enforces the zero-constraint.

## 4. Implementation Sketch

### Network Architecture (Proposed)

```python
class QPLEXAgentNetwork(nn.Module):
    """Per-agent dueling network"""
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super().__init__()
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Value stream
        self.value_head = nn.Linear(hidden_dim, 1)
        # Advantage stream
        self.advantage_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs):
        features = self.encoder(obs)
        value = self.value_head(features)  # [batch, 1]
        advantage = self.advantage_head(features)  # [batch, action_dim]

        # Dueling aggregation
        advantage = advantage - advantage.mean(dim=-1, keepdim=True)
        q_values = value + advantage

        return q_values, value, advantage

class QPLEXTransformation(nn.Module):
    """Transformation network for advantage factorization"""
    def __init__(self, n_agents, state_dim, hidden_dim=128):
        super().__init__()
        self.n_agents = n_agents

        # Attention mechanism
        self.query_net = nn.Linear(state_dim, hidden_dim)
        self.key_net = nn.Linear(1, hidden_dim)  # Per-agent advantage difference
        self.value_net = nn.Linear(1, hidden_dim)

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # State value network
        self.state_value = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, advantage_diffs, state):
        """
        advantage_diffs: [batch, n_agents] - A_i(a_i) - A_i(a_i*)
        state: [batch, state_dim]
        """
        batch_size = state.size(0)

        # V_tot(s)
        v_tot = self.state_value(state)  # [batch, 1]

        # Attention over advantage differences
        query = self.query_net(state)  # [batch, hidden]
        keys = self.key_net(advantage_diffs.unsqueeze(-1))  # [batch, n_agents, hidden]
        values = self.value_net(advantage_diffs.unsqueeze(-1))  # [batch, n_agents, hidden]

        # Compute attention weights
        attention = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2))  # [batch, 1, n_agents]
        attention = F.softmax(attention / (hidden_dim ** 0.5), dim=-1)

        # Weighted sum
        attended = torch.bmm(attention, values).squeeze(1)  # [batch, hidden]
        a_tot = self.output_head(attended)  # [batch, 1]

        # Q_tot = V_tot + A_tot
        q_tot = v_tot + a_tot

        return q_tot, v_tot, a_tot
```

### Training Loop (Proposed)

```python
def update(batch):
    # Get agent Q-values and advantages
    agent_qs, agent_vs, agent_advs = [], [], []
    for i, agent_net in enumerate(agent_networks):
        q, v, adv = agent_net(batch['obs'][i])
        agent_qs.append(q)
        agent_vs.append(v)
        agent_advs.append(adv)

    # Get chosen advantages and greedy advantages
    chosen_advs = [advs.gather(-1, batch['actions'][i])
                   for i, advs in enumerate(agent_advs)]
    greedy_advs = [advs.max(dim=-1, keepdim=True)[0]
                   for advs in agent_advs]

    # Advantage differences
    adv_diffs = [chosen - greedy
                 for chosen, greedy in zip(chosen_advs, greedy_advs)]
    adv_diffs = torch.cat(adv_diffs, dim=-1)  # [batch, n_agents]

    # Transform to joint Q
    q_tot, v_tot, a_tot = transformation_net(adv_diffs, batch['state'])

    # Target
    with torch.no_grad():
        # Target agents pick greedy actions
        target_actions = [target_net(batch['next_obs'][i])[0].argmax(dim=-1)
                         for i, target_net in enumerate(target_agent_networks)]

        # Target advantages (should be 0 for greedy)
        target_adv_diffs = torch.zeros_like(adv_diffs)
        target_q_tot, target_v_tot, _ = target_transformation_net(
            target_adv_diffs, batch['next_state']
        )

        targets = batch['reward'] + gamma * (1 - batch['done']) * target_q_tot

    # TD loss
    td_loss = F.mse_loss(q_tot, targets)

    # Consistency loss (greedy actions should give V_tot)
    consistency_loss = F.mse_loss(v_tot, q_tot.detach())

    # Total loss
    loss = td_loss + lambda_consistency * consistency_loss

    # Optimize
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(parameters, max_grad_norm=10.0)
    optimizer.step()

    return {
        'loss': loss.item(),
        'td_loss': td_loss.item(),
        'consistency_loss': consistency_loss.item(),
        'q_tot': q_tot.mean().item()
    }
```

## 5. Expected Performance

Based on the paper's results:

### SMAC Benchmark (Expected)

| Map | QMIX | WQMIX | QPLEX |
|-----|------|-------|-------|
| 2s3z | 95% | 97% | 98% |
| 3s5z | 90% | 95% | 97% |
| MMM | 80% | 92% | 95% |
| 1c3s5z | 85% | 93% | 96% |
| corridor | 98% | 99% | 99% |

**Expected Benefits**:
- Best-in-class performance on complex coordination tasks
- Handles non-monotonic payoff structures
- More stable than QTRAN
- Complete representational capacity

## 6. Implementation Roadmap

### Phase 1: Core Components
- [ ] Agent dueling networks (V + A streams)
- [ ] Transformation network with attention
- [ ] State value network
- [ ] Target networks and soft updates

### Phase 2: Training Infrastructure
- [ ] Episode replay buffer
- [ ] Consistency regularization
- [ ] Gradient clipping and normalization
- [ ] Epsilon-greedy exploration

### Phase 3: Optimizations
- [ ] Prioritized experience replay
- [ ] Multi-step returns
- [ ] Distributional RL extension
- [ ] N-agent scaling tests

### Phase 4: Evaluation
- [ ] SMAC benchmark integration
- [ ] Comparison with QMIX/WQMIX
- [ ] Ablation studies
- [ ] Scalability analysis

## 7. Anticipated Challenges

1. **Training Stability**: Duplex architecture may be sensitive to hyperparameters
2. **Attention Mechanism**: Requires careful initialization
3. **Consistency Regularization**: Balancing λ_consistency is critical
4. **Computational Cost**: More complex than QMIX (attention overhead)
5. **Sample Efficiency**: May need more samples to converge

## 8. Comparison with Alternatives

| Feature | QMIX | WQMIX | QTRAN | QPLEX |
|---------|------|-------|-------|-------|
| Expressiveness | Limited | Moderate | Complete | Complete |
| Training Stability | High | High | Low | Moderate |
| Computational Cost | Low | Low | Moderate | Moderate |
| Sample Efficiency | Good | Good | Poor | Good |
| Implementation Complexity | Low | Low | High | Moderate |

## 9. Related Work

1. **Dueling DQN**: Wang et al., ICML 2016 - Foundation for value/advantage decomposition
2. **QMIX**: Rashid et al., ICML 2018 - Monotonic value factorization
3. **QTRAN**: Son et al., ICML 2019 - Previous complete factorization approach
4. **WQMIX**: Rashid et al., NeurIPS 2020 - Weighted relaxation of monotonicity

## 10. References

1. **QPLEX**: Wang et al., "QPLEX: Duplex Dueling Multi-Agent Q-Learning", ICLR 2021 [arXiv:2008.01062](https://arxiv.org/abs/2008.01062)

2. **Dueling Networks**: Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning", ICML 2016

3. **SMAC**: Samvelyan et al., "The StarCraft Multi-Agent Challenge", AAMAS 2019

4. **Value Factorization Survey**: Rashid et al., "Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning", 2020

**Implementation Status**: This algorithm is documented but not yet implemented. Contributions welcome! See implementation roadmap above for guidance.
