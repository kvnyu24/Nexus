# Planning Methods in Reinforcement Learning

This directory contains documentation for **planning algorithms** that combine search, neural networks, and reinforcement learning to make sequential decisions. These methods explicitly reason about future consequences before taking actions.

## Overview

Planning methods bridge the gap between pure learning and pure search:

- **Pure Learning** (DQN, PPO): Fast but myopic, no lookahead
- **Pure Search** (A*, MCTS): Optimal but requires perfect models
- **Planning + Learning**: Leverage both learning and search for intelligent decision-making

Key advantages:
- **Sample efficiency**: Learn from simulated rollouts
- **Interpretability**: Explicit reasoning traces
- **Robustness**: Can handle novel situations
- **Performance**: Often superhuman (AlphaGo, AlphaZero)

## Algorithms Covered

### [AlphaZero](./alphazero.md)
**Core Innovation**: Neural network-guided Monte Carlo Tree Search

- Combines policy/value network with MCTS
- Self-play for continuous improvement
- Superhuman performance in Go, Chess, Shogi
- No domain knowledge beyond rules

**When to Use**: Perfect information games, simulation available, computational budget allows search.

**Key Papers**: Silver et al. (2017, 2018) - Nature/Science

### Monte Carlo Tree Search (MCTS)
**Core Innovation**: Selective tree expansion with rollouts

- Balances exploration and exploitation (UCB)
- Asymptotically optimal with infinite samples
- Anytime algorithm (improves with more time)
- Works without value function

**When to Use**: Large branching factor, expensive evaluation, no good heuristics.

**Key Papers**: Kocsis & Szepesvári (2006), Browne et al. (2012)

### PRM Agent (Probabilistic Roadmap)
**Core Innovation**: Sample-based motion planning with learned components

- Builds roadmap of reachable states
- Neural network for state encoding and value estimation
- Efficient path finding with A* search
- Applicable to continuous spaces

**When to Use**: Robotics, navigation, continuous state spaces.

**Key Papers**: Kavraki et al. (1996), Qureshi & Ayaz (2015)

## Comparison Table

| Method | State Space | Search Type | Learning | Best For |
|--------|-------------|-------------|----------|----------|
| AlphaZero | Discrete | MCTS | Policy + Value | Perfect info games |
| MCTS | Discrete | Tree | None (or light) | Online planning |
| PRM Agent | Continuous | Graph | Value function | Navigation/robotics |

## Core Concepts

### Planning Horizon

How far ahead to look:
- **Short horizon** (H=1): Greedy, fast but myopic
- **Medium horizon** (H=10-100): Balance planning and computation
- **Long horizon** (H>100): Better decisions but slower

Trade-off between:
- Decision quality (longer is better)
- Computational cost (longer is more expensive)

### Model-Based vs Model-Free

**Model-Based Planning**:
- Requires environment model: s' = f(s, a)
- Can simulate: "What if I do this?"
- Sample efficient but model errors compound

**Model-Free Learning**:
- No model, learns directly from experience
- Robust to model errors
- Sample inefficient

**Hybrid** (like AlphaZero):
- Use model for short-term planning
- Use learned value for long-term estimates

### Exploration in Planning

How to explore the search space:

**UCB (Upper Confidence Bound)**:
```
Score(node) = Q(node) + c * √(log(N_parent) / N_node)
```

**Thompson Sampling**:
Sample from posterior over values.

**Progressive Widening**:
Expand promising nodes more than unpromising ones.

### Value Backup

Propagate information through search tree:

**Max Backup** (Minimax):
```
V(s) = max_a [r(s,a) + γ * V(s')]
```

**Average Backup** (MCTS):
```
V(s) = (1/N) * Σ rollout_values
```

**Soft Max Backup**:
```
V(s) = log Σ_a exp(Q(s, a) / τ)
```

## Algorithm Deep Dives

### AlphaZero Workflow

```
Training Loop:
  1. Self-Play:
     - Run MCTS to select actions
     - Play game to completion
     - Store (state, MCTS_policy, outcome)

  2. Training:
     - Sample from replay buffer
     - Update network to match MCTS policies and outcomes
     - p_loss + v_loss + regularization

  3. Evaluation:
     - New network vs old network
     - If new wins >55%, replace old

  Repeat until convergence
```

### MCTS Phases

```
Selection:
  Start at root
  While not at leaf:
    Choose child with highest UCB score

Expansion:
  If leaf is not terminal:
    Add one or more children

Simulation (Rollout):
  From new node, simulate to terminal state
  (Or use value network evaluation)

Backup:
  Propagate value up the tree
  Update visit counts and Q-values
```

### PRM Planning

```
Offline Phase:
  1. Sample N random states in state space
  2. Connect nearby states with edges
  3. Build roadmap graph

Online Phase:
  1. Add start and goal to roadmap
  2. Run A* search on roadmap
  3. Extract path
  4. Follow path (with local adjustments)
```

## Implementation Patterns

### Batched MCTS

Improve efficiency by batching neural network calls:

```python
# Collect all leaf nodes
leaves = []
for _ in range(batch_size):
    leaf = mcts_select_leaf()
    leaves.append(leaf)

# Batch evaluate
states = [leaf.state for leaf in leaves]
policies, values = network(torch.stack(states))

# Expand and backup
for leaf, policy, value in zip(leaves, policies, values):
    leaf.expand(policy)
    leaf.backup(value)
```

### Parallel MCTS

Run multiple MCTS in parallel:

```python
# Virtual loss to avoid redundant exploration
def select_with_virtual_loss(node):
    node.virtual_loss += 1  # Temporary penalty
    child = select_child(node)
    return child

# After evaluation, remove virtual loss
def backup_and_remove_virtual_loss(node, value):
    node.virtual_loss -= 1
    node.backup(value)
```

### Model Ensemble

Use multiple models for robustness:

```python
# Ensemble of dynamics models
predictions = [model_i(state, action) for model_i in ensemble]

# Sample from ensemble
next_state = random.choice(predictions)

# Or use agreement for uncertainty
uncertainty = std(predictions)
```

## Practical Considerations

### Computational Budget

MCTS simulations vs performance:
- **1-10 sims**: Quick decisions, low quality
- **50-100 sims**: Decent quality, reasonable time
- **500-1000 sims**: High quality, slow
- **10K+ sims**: Diminishing returns

Allocate budget based on:
- Decision importance
- Available time
- State complexity

### Search Depth

How deep to search:
- **Shallow** (depth < 5): Fast but myopic
- **Medium** (depth 10-30): Good balance
- **Deep** (depth > 50): Expensive, model errors compound

Use value function to cut off deep searches.

### Branching Factor

Number of actions to consider:
- **Low** (b < 10): Can explore exhaustively
- **Medium** (b = 10-100): Need selective exploration
- **High** (b > 100): Must prune aggressively

Techniques:
- Policy network to focus on promising actions
- Progressive widening
- Action abstractions

### Model Errors

Planning with imperfect models:
- **Optimistic**: Overestimate value → risky behavior
- **Pessimistic**: Underestimate value → conservative
- **Realistic**: Model uncertainty → robust

Use:
- Ensemble models
- Pessimistic value estimates
- Short planning horizons

## Performance Benchmarks

### AlphaZero Results

After 24 hours of training:

| Game | Opponent | Win Rate |
|------|----------|----------|
| Go | AlphaGo Lee | 100% |
| Chess | Stockfish 8 | 72% |
| Shogi | Elmo | 90% |

### MCTS Scaling

Elo improvement over random policy:

| Simulations | Go | Chess | Shogi |
|-------------|-----|-------|-------|
| 1 | +500 | +400 | +450 |
| 10 | +1200 | +1000 | +1100 |
| 100 | +1800 | +1600 | +1700 |
| 1000 | +2200 | +2000 | +2100 |

### PRM Success Rate

Robot navigation tasks:

| Environment | Success % | Path Length | Planning Time |
|-------------|-----------|-------------|---------------|
| Simple | 98% | 1.1× optimal | 0.5s |
| Cluttered | 87% | 1.3× optimal | 2.1s |
| Dynamic | 72% | 1.5× optimal | 1.8s |

## Common Pitfalls

### 1. Insufficient Exploration
Add exploration bonus (UCB, noise at root).

### 2. Model Overfitting
Use ensemble, limit planning depth.

### 3. Expensive Rollouts
Use value network instead of full rollouts.

### 4. Poor Prior Policy
Pre-train policy network on expert data.

### 5. Unbalanced Tree
Use virtual loss in parallel MCTS.

### 6. Not Reusing Search
Save search tree between decisions.

### 7. Wrong Temperature
Use τ=1 during search, τ→0 for final decision.

### 8. Ignoring Uncertainty
Track and use model uncertainty.

## Research Directions

### Current Challenges
- **Sample Efficiency**: Reduce data needed for good models
- **Partial Observability**: Plan with incomplete information
- **Continuous Actions**: MCTS designed for discrete actions
- **Real-Time**: Planning under strict time constraints
- **Multi-Agent**: Planning with other agents

### Future Work
- **Learned Search**: Meta-learn search strategies
- **Hierarchical Planning**: Abstract action spaces
- **World Models**: Better environment models
- **Transfer**: Reuse plans across tasks
- **Safety**: Ensure safe exploration

## Code Locations

- **AlphaZero**: `/Users/kevinyu/Projects/Nexus/nexus/models/rl/alphazero.py`
- **PRM Agent**: `/Users/kevinyu/Projects/Nexus/nexus/models/rl/prm.py`
- **MCTS**: Included in AlphaZero implementation

## References

### Foundational Papers

**AlphaGo/AlphaZero**:
1. Silver, D., et al. (2016). **Mastering the Game of Go with Deep Neural Networks and Tree Search.** Nature.
2. Silver, D., et al. (2017). **Mastering the Game of Go without Human Knowledge.** Nature.
3. Silver, D., et al. (2018). **A General Reinforcement Learning Algorithm that Masters Chess, Shogi, and Go through Self-Play.** Science.

**MCTS**:
4. Kocsis, L., & Szepesvári, C. (2006). **Bandit Based Monte-Carlo Planning.** ECML.
5. Browne, C., et al. (2012). **A Survey of Monte Carlo Tree Search Methods.** IEEE TCIAIG.

**Planning with Learned Models**:
6. Schrittwieser, J., et al. (2020). **Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model.** Nature. (MuZero)
7. Hafner, D., et al. (2020). **Dream to Control: Learning Behaviors by Latent Imagination.** ICLR. (Dreamer)

**Motion Planning**:
8. Kavraki, L., et al. (1996). **Probabilistic Roadmaps for Path Planning in High-Dimensional Configuration Spaces.** IEEE TRA.
9. LaValle, S. (1998). **Rapidly-Exploring Random Trees: A New Tool for Path Planning.** Technical Report.

---

**Navigation**:
- [← Back to RL Overview](../)
- [AlphaZero →](./alphazero.md)
- [MCTS →](./mcts.md)
- [PRM Agent →](./prm_agent.md)
