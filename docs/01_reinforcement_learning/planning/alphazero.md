# AlphaZero

## 1. Overview & Motivation

AlphaZero combines Monte Carlo Tree Search (MCTS) with deep neural networks to master games through pure self-play, without any human knowledge beyond the rules. It unified the approaches of AlphaGo, AlphaGo Zero, and extended to chess and shogi.

### Revolutionary Achievements
- **Superhuman play** in Go, Chess, and Shogi from scratch
- **Tabula rasa learning**: No human games, no domain heuristics
- **General algorithm**: Same approach for different games
- **Elegant simplicity**: Policy + value network + MCTS

### Key Innovations
- Combined policy and value network (shared trunk)
- MCTS guided by neural network predictions
- Self-play for continuous improvement
- Minimal domain knowledge required

## 2. Theoretical Background

### Components

**1. Neural Network**
```
f_θ(s) → (p, v)
where:
- p: Policy vector (action probabilities)
- v: Value estimate (expected outcome)
```

**2. MCTS**
Uses neural network to guide tree search:
- **Selection**: Pick actions using UCB with neural network prior
- **Expansion**: Add new nodes when leaf reached
- **Evaluation**: Use neural network value
- **Backup**: Propagate value up the tree

**3. Self-Play**
Generate training data by playing against itself:
```
(s_t, π_t, z_t)
where:
- s_t: Board state
- π_t: MCTS visit distribution (improved policy)
- z_t: Game outcome (+1/0/-1)
```

### UCB Formula

AlphaZero selection uses PUCT:
```
UCB(s, a) = Q(s, a) + c_puct · P(s, a) · √(Σ_b N(s, b)) / (1 + N(s, a))

where:
- Q(s, a): Mean action value
- P(s, a): Prior probability from neural network
- N(s, a): Visit count
- c_puct: Exploration constant
```

### Training Objective

```
L = (z - v)² - π^T log p + c||θ||²

Components:
- Value loss: MSE between predicted value and game outcome
- Policy loss: Cross-entropy between MCTS policy and network policy
- L2 regularization: Prevent overfitting
```

## 3. Mathematical Formulation

### MCTS Search

For each simulation:
```
1. Selection: a_t = argmax_a UCB(s_t, a)
2. Expansion: If leaf, add children
3. Evaluation: v = f_θ(s_leaf)
4. Backup: Q(s, a) ← (N·Q + v) / (N+1), N ← N+1
```

### Visit Count to Policy

Temperature-based policy:
```
π(a|s) = N(s, a)^(1/τ) / Σ_b N(s, b)^(1/τ)

where τ controls exploration:
- τ = 1: Proportional to visits
- τ → 0: Argmax (deterministic)
- τ → ∞: Uniform (maximum exploration)
```

### Self-Play Data Collection

```
For each game:
  Initialize s_0
  For t = 0, 1, ..., T-1:
    Run MCTS(s_t, num_simulations)
    Sample a_t ~ π_t
    Execute: s_{t+1} = transition(s_t, a_t)
    Store: (s_t, π_t, None)  # Outcome unknown yet
  After game ends with outcome z:
    Update all stored tuples: (s_t, π_t, z)
```

## 4. High-Level Intuition

### The AlphaZero Recipe

1. **Neural Network = Intuition**: Fast pattern recognition
2. **MCTS = Deliberation**: Slow but accurate search
3. **Self-Play = Practice**: Continuously improve against itself

Like a chess player who:
- Has intuition (neural network)
- Thinks ahead (MCTS)
- Gets better by analyzing their games (self-play)

### Why It Works

**Virtuous cycle:**
```
Better network → Better MCTS → Better training data → Better network → ...
```

Each component improves the others.

### MCTS vs Pure Neural Network

- **Pure NN**: Fast (1ms) but less accurate
- **Pure MCTS**: Accurate but slow without guidance
- **AlphaZero**: Best of both worlds

MCTS typically improves play by 200-400 Elo over raw neural network.

## 5. Implementation Details

From `/Users/kevinyu/Projects/Nexus/nexus/models/rl/alphazero.py`:

```python
config = {
    "state_dim": 19*19*17,   # Go: 19x19 board, 17 feature planes
    "action_dim": 19*19+1,   # All positions + pass
    "hidden_dim": 256,       # Network size
    "num_simulations": 800,  # MCTS simulations per move
    "c_puct": 1.0,          # Exploration constant
    "temperature": 1.0,      # Sampling temperature
    "learning_rate": 0.001,
    "value_loss_weight": 1.0,
    "l2_weight": 1e-4,
}
```

### Network Architecture

```python
class AlphaZeroNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        # Shared trunk (ResNet-style for games)
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Policy head
        self.policy_head = nn.Linear(hidden_dim, action_dim)

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),  # Value in [-1, 1]
        )

    def forward(self, state):
        features = self.trunk(state)
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        return policy_logits, value
```

## 6. Code Walkthrough

### MCTS Node

```python
class MCTSNode:
    def __init__(self, prior):
        self.prior = prior          # P(a|s) from neural network
        self.visit_count = 0        # N(s, a)
        self.value_sum = 0.0        # W(s, a)
        self.children = {}          # Child nodes

    def q_value(self):
        """Mean action value Q(s, a) = W(s, a) / N(s, a)"""
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

    def ucb_score(self, parent_visit_count, c_puct=1.0):
        """UCB score for action selection"""
        u = c_puct * self.prior * math.sqrt(parent_visit_count) / (1 + self.visit_count)
        return self.q_value() + u
```

### MCTS Search

```python
def mcts_search(self, root_state, legal_actions):
    """Run MCTS from root state."""
    # Get neural network predictions
    policy_probs, _ = self.predict(root_state)

    # Initialize root
    root = MCTSNode(prior=0.0)
    for action in legal_actions:
        root.children[action] = MCTSNode(prior=policy_probs[action])

    # Run simulations
    for _ in range(self.num_simulations):
        # Selection
        node = root
        search_path = [node]
        current_state = root_state

        while len(node.children) > 0:
            action = self._select_action(node, legal_actions)
            node = node.children[action]
            search_path.append(node)
            current_state = self.env.step(current_state, action)

        # Evaluation
        _, value = self.predict(current_state)

        # Backup
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Negate for opponent

    # Return visit count distribution
    visit_counts = np.zeros(self.action_dim)
    for action in legal_actions:
        visit_counts[action] = root.children[action].visit_count

    return visit_counts

def _select_action(self, node, legal_actions):
    """Select action with highest UCB."""
    best_score = -float('inf')
    best_action = legal_actions[0]

    for action in legal_actions:
        if action in node.children:
            score = node.children[action].ucb_score(
                node.visit_count, self.c_puct
            )
            if score > best_score:
                best_score = score
                best_action = action

    return best_action
```

### Training

```python
def update(self, batch):
    """Update network from self-play data."""
    states = batch["states"]
    policy_targets = batch["policy_targets"]  # MCTS visit distributions
    value_targets = batch["value_targets"]    # Game outcomes

    # Forward pass
    policy_logits, value_preds = self.network(states)

    # Policy loss: Cross-entropy with MCTS policy
    policy_loss = -torch.sum(
        policy_targets * F.log_softmax(policy_logits, dim=-1), dim=-1
    ).mean()

    # Value loss: MSE with game outcome
    value_loss = F.mse_loss(value_preds.squeeze(-1), value_targets)

    # Total loss with regularization
    loss = policy_loss + self.value_loss_weight * value_loss

    # Optimize
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    return {
        "loss": loss.item(),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
    }
```

## 7. Optimization Tricks

### 1. Dirichlet Noise for Exploration

Add noise to root prior during self-play:
```python
# Encourage exploration at root
noise = np.random.dirichlet([alpha] * num_actions)
prior = (1 - epsilon) * prior + epsilon * noise
```

### 2. Temperature Schedule

Decrease temperature during game:
```python
if move_number < 30:
    temperature = 1.0  # Explore
else:
    temperature = 0.0  # Exploit
```

### 3. Resign Threshold

Resign early if value is very negative:
```python
if value < resign_threshold:
    return RESIGN
```

Saves computation and focuses training on competitive games.

### 4. Data Augmentation

Use symmetries (rotation, reflection):
```python
# For Go/Chess, augment with board symmetries
augmented_data = apply_symmetries(game_data)
```

### 5. Prioritized Replay

Sample recent games more often:
```python
sample_prob = games[-recent_window:]
```

### 6. Batch MCTS

Run multiple MCTS in parallel:
```python
# Batch neural network evaluations
states_batch = [s for s in leaves]
policies, values = network(states_batch)
```

### 7. Value Target Bootstrapping

Use n-step returns:
```python
value_target = z  # Final outcome
# Or use bootstrapping for unfinished games
value_target = reward + γ * v_next
```

### 8. Learning Rate Schedule

Decay learning rate over training:
```python
lr = initial_lr * 0.1 ** (step / decay_steps)
```

## 8. Experiments & Results

### Performance

AlphaZero after 24 hours of training:
- **Chess**: Defeated Stockfish 28-72-0 (wins-losses-draws)
- **Shogi**: Defeated Elmo 90-8-2
- **Go**: Surpassed AlphaGo Lee (4-0)

### MCTS Simulations vs Strength

```
Simulations    Elo (relative to network only)
1             +0 (baseline)
10            +100
100           +250
800           +400
```

Diminishing returns beyond 800-1000 simulations.

### Training Progression

```
Hours  Chess Elo
1      2000
3      2500
9      3000
24     3500+
```

Rapid improvement in early hours, then slower gains.

## 9. Common Pitfalls

### 1. Insufficient Exploration
Add Dirichlet noise at root.

### 2. Overfitting
Use L2 regularization and data augmentation.

### 3. Wrong Temperature
Use τ=1 during search, τ=0 during evaluation.

### 4. Imbalanced Training
Balance wins/losses in training data.

### 5. Too Few Simulations
At least 100-800 simulations per move.

### 6. Not Using Symmetries
Augment with board transformations.

### 7. Poor Initialization
Use proper weight initialization (orthogonal/xavier).

### 8. Learning Rate Too High
Start with small LR (1e-3 to 1e-4).

### 9. Not Batching
Batch neural network calls for efficiency.

### 10. Ignoring Game Phase
Different strategies for opening/midgame/endgame.

## 10. References

### Primary Papers
- Silver, D., et al. (2017). **Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm.** ArXiv.
- Silver, D., et al. (2017). **Mastering the Game of Go without Human Knowledge.** Nature.
- Silver, D., et al. (2018). **A General Reinforcement Learning Algorithm that Masters Chess, Shogi, and Go through Self-Play.** Science.

### Related Work
- Silver, D., et al. (2016). **Mastering the Game of Go with Deep Neural Networks and Tree Search.** Nature. (AlphaGo)
- Schrittwieser, J., et al. (2020). **Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model.** Nature. (MuZero)

### Implementation
- Nexus: `/Users/kevinyu/Projects/Nexus/nexus/models/rl/alphazero.py`

---

**Key Takeaways:**
- Combines neural networks with MCTS
- Self-play generates training data
- Superhuman performance from scratch
- Generalizes across different games
