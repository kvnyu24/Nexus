# Multi-Agent Reinforcement Learning

Multi-Agent Reinforcement Learning (MARL) extends single-agent RL to settings where multiple agents interact within a shared environment. Agents may cooperate, compete, or operate in mixed scenarios, requiring coordination mechanisms and specialized training techniques.

## Key Concepts

### Centralized Training with Decentralized Execution (CTDE)

The dominant paradigm in cooperative MARL, where:
- **Training**: Agents have access to global information (all observations, actions, or the global state)
- **Execution**: Each agent acts independently using only its local observations
- **Benefit**: Enables better credit assignment while maintaining scalability at execution time

### Value Function Factorization

For cooperative tasks, the key challenge is decomposing a joint value function into individual agent contributions while maintaining representational capacity:
- **Individual Condition of Maximum (ICM)**: If each agent independently takes the action that maximizes its local Q-value, this should correspond to the team-optimal joint action
- **Monotonicity Constraint**: Q_total should be monotonically increasing in each agent's Q-value (used by QMIX)

### Challenges in MARL

1. **Non-stationarity**: From each agent's perspective, the environment is non-stationary as other agents' policies evolve
2. **Credit Assignment**: Determining each agent's contribution to team success
3. **Scalability**: Complexity grows exponentially with the number of agents
4. **Partial Observability**: Agents often have limited local observations
5. **Communication**: Designing efficient communication protocols between agents

## Algorithms Covered

### Value-Based Methods
- **[QMIX](./qmix.md)**: Monotonic value function factorization using a mixing network
- **[WQMIX](./wqmix.md)**: Weighted QMIX with relaxed monotonicity constraints
- **[QPLEX](./qplex.md)**: Duplex dueling architecture for complete IGM factorization

### Policy-Based Methods
- **[MAPPO](./mappo.md)**: Multi-Agent PPO with shared centralized critic
- **[MADDPG](./maddpg.md)**: Multi-Agent DDPG for continuous control

## Comparison of Approaches

| Algorithm | Type | Action Space | Best For | Key Innovation |
|-----------|------|--------------|----------|----------------|
| MAPPO | Policy-gradient | Continuous/Discrete | Cooperative tasks with continuous actions | Shared centralized critic with GAE |
| QMIX | Value-based | Discrete | Cooperative tasks with discrete actions | Monotonic value mixing |
| WQMIX | Value-based | Discrete | Non-monotonic cooperative tasks | Importance-weighted mixing |
| QPLEX | Value-based | Discrete | Complex factorization needs | Duplex dueling structure |
| MADDPG | Policy-gradient | Continuous | Mixed cooperative-competitive | Per-agent critics with global info |

## When to Use Which Algorithm

**MAPPO**:
- Continuous action spaces (robot control, autonomous vehicles)
- Environments with high-dimensional observations
- When stability and ease of tuning are priorities

**QMIX/WQMIX/QPLEX**:
- Discrete action spaces (StarCraft, traffic control)
- Strict cooperation requirements
- When sample efficiency is critical
- WQMIX when optimal policy violates monotonicity
- QPLEX when you need the strongest representational capacity

**MADDPG**:
- Mixed cooperative-competitive scenarios
- Continuous control tasks
- When you can afford the computational cost of per-agent critics

## Common Implementation Patterns

### Experience Replay
Multi-agent replay buffers typically store:
- Per-agent observations and actions
- Global state (if available)
- Shared team reward (cooperative) or individual rewards (competitive)
- Episode information for proper trajectory handling

### Network Architectures
- **Observation encoding**: CNN for visual inputs, MLP for vectors
- **Agent networks**: Often parameter-shared across agents to improve generalization
- **Centralized components**: Mixing networks or critics process concatenated information
- **RNNs**: Often used to handle partial observability (QMIX with GRU, etc.)

### Training Tricks
1. **Parameter sharing**: Share weights across agents to reduce parameters and improve generalization
2. **Gradient clipping**: Essential for stability in multi-agent settings
3. **Target networks**: Use slower-updating targets to stabilize learning
4. **Episode-based training**: Train on complete episodes for proper credit assignment
5. **Exploration**: Epsilon-greedy, action noise, or entropy bonuses scaled per agent

## References

- **QMIX**: [Rashid et al., 2018](https://arxiv.org/abs/1803.11485)
- **WQMIX**: [Rashid et al., 2020](https://arxiv.org/abs/2006.10800)
- **QPLEX**: [Wang et al., 2020](https://arxiv.org/abs/2008.01062)
- **MAPPO**: [Yu et al., 2022](https://arxiv.org/abs/2103.01955)
- **MADDPG**: [Lowe et al., 2017](https://arxiv.org/abs/1706.02275)

## Related Topics

- [Offline RL](../offline_rl/README.md): Multi-agent offline RL is an emerging research area
- [Reward Modeling](../reward_modeling/README.md): Designing reward functions for multi-agent cooperation
- [Exploration](../exploration/README.md): Multi-agent exploration strategies
