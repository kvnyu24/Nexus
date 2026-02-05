# Go-Explore: Archive-Based Exploration

## 1. Overview

Go-Explore is a family of algorithms that solve hard-exploration problems by maintaining an archive of interesting states, deterministically returning to them, and exploring from there. It achieved first superhuman performance on Montezuma's Revenge and Pitfall, two notoriously difficult Atari games.

**Paper**: "First Return, Then Explore" (Ecoffet et al., Nature 2021)

**Status**: ⚠️ **NOT YET IMPLEMENTED** - Documentation prepared for future implementation

**Key Innovation**: Separates exploration into two phases:
1. **Exploration phase**: Remember and return to promising states
2. **Robustification phase**: Learn robust policy from found solutions

**Use Cases**:
- Extremely sparse rewards (Montezuma's Revenge, Pitfall)
- Deterministic or resettable environments
- Puzzles and games requiring long action sequences
- Environments with deceptive rewards

## 2. Theory and Background

### 2.1 The Two Phases

**Phase 1: Archive-Based Exploration**
```
1. Maintain archive of cells (state abstractions)
2. Select promising cell from archive
3. Return to that cell (deterministically)
4. Explore from there
5. Add newly discovered cells to archive
6. Repeat
```

**Phase 2: Robustification**
```
1. Use trajectory from Phase 1 as demonstration
2. Train robust policy via imitation or RL
3. Add noise/stochasticity for robustness
4. Validate in real (stochastic) environment
```

### 2.2 Cell Representation

States are discretized into "cells" using domain knowledge:
```
cell = downscale(state)  # E.g., downscale image to 8x8
```

Cells serve as:
- Keys in archive (avoid revisiting exact states)
- Goal representations (return to this cell)
- Exploration landmarks

### 2.3 Selection Strategy

Choose which cell to explore from:
- **Random**: Uniform selection
- **Novelty-based**: Rarely visited cells
- **Frontier-based**: Cells near unexplored regions
- **Max-returns**: Cells with high discovered rewards

### 2.4 Return Mechanism

**Deterministic Reset**: If environment supports it
```
env.reset_to_state(cell.state)
```

**Goal-Conditioned Policy**: Otherwise
```
policy(s | goal=cell.state)
```

## 3. Mathematical Formulation

### Archive Structure

```
Archive = {cell_1, cell_2, ..., cell_N}

Each cell contains:
- representation: Downscaled state
- actual_state: Full state for restoration
- trajectory: Sequence to reach this cell
- return: Max return found from this cell
- visit_count: Number of times explored from here
- discovered_timestamp: When first found
```

### Selection Function

Priority-based selection:
```
score(cell) = α · novelty(cell) + β · potential(cell) + γ · return(cell)

novelty(cell) = 1 / (visit_count + 1)
potential(cell) = |neighbors(cell) ∩ unvisited|
return(cell) = max return found from cell
```

### Exploration from Cell

```
1. Reset to cell.actual_state
2. Take K random actions
3. For each new state:
   - Create cell representation
   - If cell not in archive:
       Add to archive
       Store full state and trajectory
4. Update cell statistics
```

### Robustification

Train policy π via:
```
Option 1 (Imitation):
L = E[||π(s) - a_demo||^2]

Option 2 (Backward Algorithm):
Start from goal, work backwards with stochastic policy
π(a|s,t) = P(reach goal from s,a)

Option 3 (RL):
Train policy with dense waypoint rewards
r(s) = -distance(s, next_waypoint)
```

## 4. Implementation Sketch

### Cell Class

```python
class Cell:
    def __init__(self, representation, state, trajectory, return_value):
        self.representation = representation  # Downscaled state
        self.state = state  # Full state for reset
        self.trajectory = trajectory  # Actions to reach here
        self.return_value = return_value  # Max return found
        self.visit_count = 0
        self.timestamp = time.time()

    def score(self, alpha=1.0, beta=1.0, gamma=1.0):
        novelty = 1.0 / (self.visit_count + 1)
        return alpha * novelty + gamma * self.return_value
```

### Archive

```python
class Archive:
    def __init__(self):
        self.cells = {}  # representation -> Cell

    def add(self, cell):
        if cell.representation not in self.cells:
            self.cells[cell.representation] = cell
        else:
            # Update if better return found
            existing = self.cells[cell.representation]
            if cell.return_value > existing.return_value:
                existing.return_value = cell.return_value
                existing.trajectory = cell.trajectory

    def select(self, strategy='weighted'):
        if strategy == 'uniform':
            return random.choice(list(self.cells.values()))
        elif strategy == 'weighted':
            cells = list(self.cells.values())
            scores = [cell.score() for cell in cells]
            probs = softmax(scores)
            return np.random.choice(cells, p=probs)
```

### Exploration Phase

```python
def explore_phase(env, archive, num_iterations):
    for _ in range(num_iterations):
        # Select cell to explore from
        cell = archive.select()

        # Return to cell
        if env.can_reset_to_state():
            env.reset_to_state(cell.state)
        else:
            # Replay trajectory to reach cell
            env.reset()
            for action in cell.trajectory:
                env.step(action)

        # Explore from cell
        state = cell.state
        trajectory = cell.trajectory.copy()
        total_return = 0

        for _ in range(exploration_steps):
            # Random action (or epsilon-greedy)
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)

            trajectory.append(action)
            total_return += reward

            # Create cell for new state
            cell_repr = downscale(next_state)
            new_cell = Cell(
                representation=cell_repr,
                state=next_state,
                trajectory=trajectory.copy(),
                return_value=total_return
            )

            # Add to archive
            archive.add(new_cell)

            state = next_state
            if done:
                break

        # Update visit count
        cell.visit_count += 1
```

### Robustification Phase

```python
def robustification_phase(env, successful_trajectory, method='imitation'):
    if method == 'imitation':
        # Behavioral cloning on trajectory
        policy = train_imitation_policy(successful_trajectory)

    elif method == 'backward':
        # Backward algorithm (work from goal backwards)
        policy = train_backward_algorithm(env, successful_trajectory)

    elif method == 'rl':
        # RL with waypoint rewards
        waypoints = sample_waypoints(successful_trajectory)
        policy = train_rl_with_waypoints(env, waypoints)

    return policy

def train_imitation_policy(trajectory):
    """Simple behavioral cloning"""
    policy = Policy()
    optimizer = torch.optim.Adam(policy.parameters())

    for epoch in range(num_epochs):
        for state, action in trajectory:
            pred_action = policy(state)
            loss = F.mse_loss(pred_action, action)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return policy
```

## 5. Implementation Roadmap

### Phase 1: Core Archive
- [ ] Cell representation (downscaling)
- [ ] Archive data structure
- [ ] Cell selection strategies
- [ ] Visit counting and scoring

### Phase 2: Exploration
- [ ] State restoration (for simulators)
- [ ] Trajectory replay (for environments without reset)
- [ ] Random exploration from cells
- [ ] Archive growth monitoring

### Phase 3: Robustification
- [ ] Imitation learning baseline
- [ ] Backward algorithm
- [ ] RL with waypoint rewards
- [ ] Trajectory demonstration storage

### Phase 4: Integration
- [ ] Atari domain adapter
- [ ] Image downscaling for cells
- [ ] Visualization tools
- [ ] Benchmark suite (Montezuma, Pitfall)

## 6. Expected Performance

Based on paper results:

### Montezuma's Revenge

| Algorithm | Mean Score | Max Score | % of Game |
|-----------|------------|-----------|-----------|
| Go-Explore | 43,791 | 2,500,000+ | 100% |
| RND | 8,152 | 10,070 | ~5% |
| ICM | 3,340 | 4,800 | ~3% |
| Human | 4,753 | - | Variable |

Go-Explore completes **all 24 rooms** of the game.

### Pitfall

| Algorithm | Mean Score | Rooms Solved |
|-----------|------------|--------------|
| Go-Explore | 26,860 | 26 / 255 |
| RND | <100 | 0-2 / 255 |
| All others | 0 | 0 / 255 |

First algorithm to make progress on Pitfall.

## 7. Challenges and Limitations

### 7.1 Determinism Requirement

**Challenge**: Needs deterministic environment or ability to restore states

**Solutions**:
- Simulator-based environments (Atari via ALE)
- Trajectory replay for partially stochastic
- Goal-conditioned policies for fully stochastic

### 7.2 Cell Representation

**Challenge**: Domain-specific downscaling needed

**Solutions**:
- Learned cell representations (autoencoders)
- Adaptive discretization
- Graph-based representations

### 7.3 Scalability

**Challenge**: Archive grows without bound

**Solutions**:
- Prune rarely visited cells
- Merge similar cells
- Hierarchical archives

### 7.4 Robustification Gap

**Challenge**: Phase 1 trajectory doesn't transfer to robust policy

**Solutions**:
- Backward algorithm (more robust)
- Data augmentation during robustification
- Adversarial training

## 8. Comparison with Other Methods

| Feature | Go-Explore | RND | ICM | MCTS |
|---------|------------|-----|-----|------|
| Requires Determinism | Yes* | No | No | Partial |
| Exploration Strategy | Archive | Prediction error | Prediction error | Tree search |
| Memory Overhead | High | Low | Moderate | Very High |
| Robustification Needed | Yes | No | No | No |
| Hard Exploration | Excellent | Good | Moderate | Good |
| Sample Efficiency | Moderate | Moderate | Moderate | High |

*Relaxed in later variants

## 9. Extensions and Variants

### 9.1 Policy-Based Go-Explore

Replace random actions with learned policy:
```python
action = policy(state, temperature=high)  # High temperature for exploration
```

### 9.2 Continuous Go-Explore

For continuous state spaces:
```python
cell = kmeans_cluster(state)  # Use clustering instead of downscaling
```

### 9.3 Multi-Agent Go-Explore

Archive for multi-agent states:
```python
cell = (downscale(state), tuple(agent_positions))
```

### 9.4 Transfer Go-Explore

Reuse archive across related tasks:
```python
task2_archive = filter(task1_archive, relevance_threshold)
```

## 10. References

### Original Papers

1. **Go-Explore (Nature)**: Ecoffet et al., "First Return, Then Explore", Nature 2021 [arXiv:2004.12919](https://arxiv.org/abs/2004.12919)

2. **Go-Explore (arXiv)**: Ecoffet et al., "Go-Explore: a New Approach for Hard-Exploration Problems", 2019 [arXiv:1901.10995](https://arxiv.org/abs/1901.10995)

3. **Backward Algorithm**: Ecoffet et al., "Montezuma's Revenge Solved by Go-Explore", 2019

### Related Work

4. **RND**: Burda et al., "Exploration by Random Network Distillation", ICLR 2019

5. **ICM**: Pathak et al., "Curiosity-driven Exploration by Self-Supervised Prediction", ICML 2017

6. **Archive-Based Methods**: Cully & Demiris, "Quality and Diversity Optimization: A Unifying Modular Framework", IEEE Trans, 2018

### Analysis

7. **Hard Exploration**: Osband et al., "Deep Exploration via Bootstrapped DQN", NeurIPS 2016

8. **Montezuma Analysis**: Machado et al., "Revisiting the Arcade Learning Environment: Evaluation Protocols", JAIR 2018

### Implementation References

9. [Uber Go-Explore](https://github.com/uber-research/go-explore): Original implementation

10. [Adrien Ecoffet's Blog](https://eng.uber.com/go-explore/): Detailed explanation and insights

**Implementation Status**: This algorithm is documented but not yet implemented. Due to the need for deterministic environments and state restoration, implementation requires careful environment interface design. Contributions welcome!
