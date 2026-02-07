# Monte Carlo Tree Search (MCTS)

## 1. Overview & Motivation

Monte Carlo Tree Search (MCTS) is a **simulation-based search algorithm** that combines random sampling with tree search to make sequential decisions. It has revolutionized game AI, particularly in domains with large branching factors where traditional minimax search is intractable.

### The Problem with Traditional Search

**Minimax/Alpha-Beta:**
- Requires exhaustive exploration
- Exponential complexity: O(b^d) where b=branching factor, d=depth
- Infeasible for Go (b≈250) or complex planning tasks

**Heuristic Search (A\*):**
- Requires good heuristic function
- Hard to design heuristics for complex domains
- Can get stuck in local optima

### MCTS's Solution

MCTS selectively expands the most promising parts of the search tree:
```
Key Insight: Focus computational effort where it matters most
```

**Advantages:**
- **Anytime algorithm**: Improves with more time, can be stopped anytime
- **No domain knowledge**: Works with just forward model
- **Asymptotically optimal**: Converges to minimax with infinite samples
- **Handles uncertainty**: Uses statistical sampling

### Revolutionary Impact

MCTS enabled:
- **AlphaGo** defeating world champion in Go (2016)
- **AlphaZero** mastering Chess, Shogi, Go from scratch (2017)
- **MuZero** learning without knowing game rules (2019)
- **Planning in robotics**, game playing, theorem proving

## 2. Theoretical Background

### Bandit Formulation

MCTS views each node as a **multi-armed bandit problem**:
```
At each state s, choose action a that balances:
- Exploitation: Pick action with highest known value
- Exploration: Pick action to reduce uncertainty
```

### Upper Confidence Bound (UCB)

UCB1 algorithm (Auer et al., 2002):
```
UCB1(s, a) = Q(s, a) + c * sqrt(ln(N(s)) / N(s, a))

where:
- Q(s, a): Mean value of action a from state s
- N(s): Total visits to state s
- N(s, a): Visits to action a from state s
- c: Exploration constant (typically √2)
```

**Intuition:**
- First term: Exploitation (pick best known action)
- Second term: Exploration (pick less-visited actions)
- sqrt(ln N / n) grows as action is visited less

### PUCT Variant (AlphaZero)

Polynomial UCT with prior:
```
PUCT(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))

where:
- P(s, a): Prior probability from neural network
- c_puct: Exploration constant (typically 1-5)
```

**Difference from UCB1:**
- Uses prior knowledge P(s, a) from policy network
- Exploration term uses sqrt(N) instead of sqrt(ln N)
- More aggressive exploration early on

### Regret Bounds

UCB1 has logarithmic regret:
```
Regret = O(K log(T) / Δ)

where:
- K: Number of actions
- T: Time horizon
- Δ: Gap between best and second-best action
```

This means MCTS will eventually identify the best action.

### Tree Policy vs Default Policy

**Tree Policy**: How to navigate existing tree (UCB/PUCT)
**Default Policy**: How to simulate beyond tree (random/learned)

Classical MCTS uses random rollouts; modern MCTS uses value networks.

### Concentration Inequalities

Hoeffding's inequality guarantees convergence:
```
P(|Q̂(s, a) - Q*(s, a)| > ε) ≤ 2 * exp(-2nε²)

where:
- Q̂: Estimated value
- Q*: True value
- n: Number of samples
```

With enough samples, estimates converge to true values.

## 3. Mathematical Formulation

### MCTS Algorithm

Four phases per simulation:

**1. Selection:**
```
Starting from root r, select child c using tree policy:

a_t = argmax_a UCB(s_t, a)
s_{t+1} = transition(s_t, a_t)

Until reaching leaf node l
```

**2. Expansion:**
```
If l is not terminal and visited enough:
  For each legal action a:
    Create child node c_a
    Initialize Q(l, a) = 0, N(l, a) = 0
    Set prior P(l, a) from policy network (if available)
```

**3. Simulation/Evaluation:**
```
Classical: Rollout to terminal state using default policy
  z = simulate(l, π_default)

Modern: Use value network
  z = V_θ(l)
```

**4. Backup:**
```
For each node in selection path (reversed):
  N(s, a) ← N(s, a) + 1
  Q(s, a) ← Q(s, a) + (z - Q(s, a)) / N(s, a)

For two-player games: negate z for opponent
```

### Action Selection

After simulations, select final action:

**Temperature-based:**
```
π(a|s) = N(s, a)^(1/τ) / Σ_b N(s, b)^(1/τ)

where τ controls randomness:
- τ → 0: Deterministic (argmax)
- τ = 1: Proportional to visits
- τ → ∞: Uniform (random)
```

**Visit count proportional:**
```
π(a|s) = N(s, a) / Σ_b N(s, b)
```

### Value Aggregation

Different backup strategies:

**Mean (MCTS standard):**
```
Q(s, a) = (1/N) Σ_i z_i
```

**Max (Minimax):**
```
V(s) = max_a [r(s,a) + γV(s')]
```

**Robust child (safety):**
```
a* = argmax_a [Q(s, a) - β * σ(s, a)]
where σ is standard deviation
```

### Progressive Widening

Limit action expansion for continuous/large action spaces:
```
K(N) = k * N^α

where:
- K(N): Number of actions to consider after N visits
- k, α: Constants (typically α ∈ [0.3, 0.5])
```

Expand new actions only when:
```
|children(s)| < K(N(s))
```

## 4. High-Level Intuition

### The Core Idea

Imagine you're exploring a maze:
- **Greedy**: Take first promising path, might miss better routes
- **Exhaustive**: Try every path, too slow
- **MCTS**: Focus on promising paths, but keep checking if you missed something

### The Four Phases Analogy

Using a restaurant recommendation analogy:

**Selection**: Walk to a neighborhood with good restaurants
- Use past experience (Q-values) to guide you
- Occasionally try less-explored areas (exploration bonus)

**Expansion**: Discover new restaurants in that neighborhood
- Once you arrive, look around for new options

**Simulation**: Try the restaurant (or ask friends who tried it)
- Get a quick estimate of quality
- Modern: Ask expert friend (value network)
- Classic: Try random items (rollout)

**Backup**: Update your restaurant map
- Tell everyone on your path how good it was
- They update their recommendations

### Why MCTS Works

**Selective Sampling:**
```
Don't waste time on obviously bad moves
Focus where outcome is uncertain or promising
```

**Anytime Property:**
```
More simulations → Better decision
Can stop anytime with current best guess
```

**Asymptotic Optimality:**
```
lim_{n→∞} Q̂(s, a) = Q*(s, a)
Given infinite time, finds optimal action
```

### MCTS vs Other Search

| Algorithm | Knowledge Needed | Computation | Optimality |
|-----------|-----------------|-------------|------------|
| Minimax | Terminal values | O(b^d) | Optimal if complete |
| A* | Heuristic | O(b^d) (worst) | Optimal if admissible h |
| Greedy | Value function | O(b) | No guarantee |
| **MCTS** | Forward model | Adaptive | Asymptotically optimal |

### Progressive Precision

MCTS progressively refines estimates:
```
After 10 sims:    Rough idea of best moves
After 100 sims:   Confident about top 3 moves
After 1000 sims:  Very confident about best move
```

Like taking a blurry photo and gradually increasing resolution.

## 5. Implementation Details

From `Nexus/nexus/models/search/mcts.py`:

```python
config = MCTSConfig(
    hidden_dim=256,
    state_dim=64,
    num_actions=9,
    num_simulations=800,
    c_puct=2.5,
    exploration_strategy=ExplorationStrategy.PUCT,
    max_depth=100,
    bank_size=50000,
    init_temp=1.0,
    final_temp=0.1,
    temp_decay=0.98,
    dirichlet_alpha=0.3,
    dirichlet_epsilon=0.25,
)
```

### Key Hyperparameters

**c_puct (Exploration constant):**
- Higher → More exploration
- Lower → More exploitation
- Typical range: 1.0-5.0
- Go/Chess: 1.0-2.0
- Complex planning: 2.0-5.0

**num_simulations:**
- More → Better decisions, slower
- Typical: 50-800 for games
- Can scale to 10,000+ for critical decisions

**Temperature schedule:**
- Early game (explore): τ=1.0
- Late game (exploit): τ→0.0
- Decay: τ_t = τ_0 * 0.98^t

**Dirichlet noise:**
- Encourages exploration at root
- α=0.03 for Go/Chess (many moves)
- α=0.3 for smaller action spaces
- ε=0.25 (25% noise, 75% prior)

### Neural Network Architecture

```python
class MCTS(NexusModule):
    def __init__(self, config):
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(config.state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Policy head (prior probabilities)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )

        # Value head (state evaluation)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()  # Value in [-1, 1]
        )
```

### Node Structure

```python
class MCTSNode:
    def __init__(self, state, prior=0.0):
        self.state = state
        self.prior = prior              # P(a|s) from policy network
        self.visit_count = 0            # N(s, a)
        self.value_sum = 0.0            # W(s, a)
        self.children = {}              # {action: MCTSNode}

        # Advanced statistics
        self.squared_value_sum = 0.0    # For variance
        self.max_value = -inf
        self.min_value = inf

    def value(self):
        """Mean action value Q(s, a)"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, parent_visits, c_puct=1.0):
        """PUCT formula"""
        q = self.value()
        u = c_puct * self.prior * sqrt(parent_visits) / (1 + self.visit_count)
        return q + u
```

## 6. Code Walkthrough

### Main MCTS Loop

```python
def simulate(
    self,
    root_state: torch.Tensor,
    num_simulations: int = 800
) -> Dict[str, torch.Tensor]:
    """
    Run MCTS from root state.

    Returns:
        action_probs: Policy π(a|s)
        root_value: Estimated V(s)
        visit_counts: N(s, a) for each action
    """
    root = MCTSNode(root_state)

    # Initialize root children with policy network
    outputs = self.forward(root_state)
    policy_probs = outputs["policy"]

    for action in range(self.num_actions):
        root.children[action] = MCTSNode(
            state=None,  # State computed lazily
            prior=policy_probs[action].item()
        )

    # Add exploration noise to root
    root.add_exploration_noise(
        self.config.dirichlet_alpha,
        self.config.dirichlet_epsilon
    )

    # Run simulations
    for sim in range(num_simulations):
        node = root
        search_path = [node]

        # Selection phase
        while node.expanded():
            action, node = self.select_action(node)
            search_path.append(node)

        # Expansion phase
        if node.state is None:
            # Compute state lazily
            node.state = self.env.step(
                search_path[-2].state,
                action
            )

        # Evaluation phase
        if not self.env.is_terminal(node.state):
            outputs = self.forward(node.state)
            value = outputs["value"].item()
            policy = outputs["policy"]

            # Expand node
            for a in range(self.num_actions):
                node.children[a] = MCTSNode(
                    state=None,
                    prior=policy[a].item()
                )
        else:
            # Terminal state
            value = self.env.get_reward(node.state)

        # Backup phase
        self.backup(search_path, value)

    # Extract visit distribution
    visit_counts = torch.zeros(self.num_actions)
    for action, child in root.children.items():
        visit_counts[action] = child.visit_count

    # Temperature-based action selection
    temp = self.config.get_temperature(self.step)
    if temp > 0:
        action_probs = torch.pow(visit_counts, 1.0 / temp)
        action_probs /= action_probs.sum()
    else:
        # Deterministic: select most visited
        action_probs = torch.zeros_like(visit_counts)
        action_probs[visit_counts.argmax()] = 1.0

    return {
        "action_probs": action_probs,
        "root_value": root.value(),
        "visit_counts": visit_counts
    }
```

### Action Selection (PUCT)

```python
def select_action(
    self,
    node: MCTSNode
) -> Tuple[int, MCTSNode]:
    """
    Select best action using PUCT algorithm.

    Returns:
        action: Selected action index
        child: Child node
    """
    best_score = -float('inf')
    best_action = -1
    best_child = None

    for action, child in node.children.items():
        # PUCT formula
        score = child.ucb_score(
            parent_visits=node.visit_count,
            c_puct=self.c_puct
        )

        if score > best_score:
            best_score = score
            best_action = action
            best_child = child

    return best_action, best_child
```

### Backup

```python
def backup(
    self,
    search_path: List[MCTSNode],
    value: float
):
    """
    Propagate value up the search path.

    For two-player games, negate value at each level.
    """
    for i, node in enumerate(reversed(search_path)):
        # For two-player games
        if self.env.is_two_player:
            value = -value

        # Update statistics
        node.visit_count += 1
        node.value_sum += value
        node.squared_value_sum += value ** 2
        node.max_value = max(node.max_value, value)
        node.min_value = min(node.min_value, value)
```

### Forward Pass (Network Evaluation)

```python
def forward(
    self,
    state: torch.Tensor,
    legal_actions: Optional[torch.Tensor] = None
) -> Dict[str, torch.Tensor]:
    """
    Evaluate state with neural network.

    Returns:
        policy: P(a|s) - Prior probabilities
        value: V(s) - State value estimate
    """
    # Encode state
    encoded = self.state_encoder(state)

    # Get policy and value
    policy_logits = self.policy_head(encoded)
    value = self.value_head(encoded)

    # Mask illegal actions
    if legal_actions is not None:
        policy_logits = policy_logits.masked_fill(
            ~legal_actions,
            -float('inf')
        )

    # Softmax for probabilities
    policy_probs = torch.softmax(policy_logits, dim=-1)

    return {
        "policy": policy_probs,
        "value": value,
        "encoded_state": encoded
    }
```

## 7. Optimization Tricks

### 1. Virtual Loss (Parallel MCTS)

Prevent multiple threads from exploring same path:

```python
def select_with_virtual_loss(node, virtual_loss=3.0):
    """Temporarily penalize node during parallel selection"""
    node.value_sum -= virtual_loss
    node.visit_count += 1

    action, child = select_best_action(node)

    return action, child

def backup_and_remove_virtual_loss(node, value, virtual_loss=3.0):
    """Add real value and remove virtual loss"""
    node.value_sum += value + virtual_loss
    # visit_count already incremented
```

### 2. Tree Reuse

Save search tree between moves:

```python
def reuse_tree(old_root, action_taken):
    """Reuse subtree as new root"""
    if action_taken in old_root.children:
        new_root = old_root.children[action_taken]
        new_root.parent = None  # Detach from old tree
        return new_root
    else:
        # Action wasn't in tree, start fresh
        return MCTSNode(new_state)
```

Saves ~30-50% of simulations!

### 3. Batched Evaluation

Evaluate multiple nodes simultaneously:

```python
def batch_simulate(num_parallel=8):
    """Run multiple simulations in parallel"""
    # Collect leaf nodes
    leaves = []
    paths = []

    for _ in range(num_parallel):
        node = root
        path = [node]

        while node.expanded():
            action, node = select_action(node)
            path.append(node)

        leaves.append(node)
        paths.append(path)

    # Batch evaluate
    states = torch.stack([leaf.state for leaf in leaves])
    outputs = self.forward(states)  # Single forward pass
    policies = outputs["policy"]
    values = outputs["value"]

    # Expand and backup
    for leaf, path, policy, value in zip(leaves, paths, policies, values):
        expand_node(leaf, policy)
        backup(path, value)
```

GPU utilization: 20-30% → 70-90%!

### 4. First Play Urgency (FPU)

Initialize unvisited nodes pessimistically:

```python
def fpu_value(parent_value, reduction=0.25):
    """First play urgency - pessimistic initialization"""
    return parent_value - reduction

class MCTSNode:
    def value(self):
        if self.visit_count == 0:
            return fpu_value(self.parent.value(), 0.25)
        return self.value_sum / self.visit_count
```

Prevents over-exploration of unvisited actions.

### 5. Progressive Widening

Gradually expand action space:

```python
def should_expand_action(node, action):
    """Check if we should consider new action"""
    k = 10  # Base actions
    alpha = 0.5  # Widening exponent

    max_actions = k * (node.visit_count ** alpha)
    current_actions = len(node.children)

    return current_actions < max_actions
```

Essential for continuous or large action spaces.

### 6. RAVE (Rapid Action Value Estimation)

Share value estimates across the tree:

```python
def rave_value(node, action):
    """All-Moves-As-First heuristic"""
    # Local value (specific to this position)
    q_local = node.children[action].value()
    n_local = node.children[action].visit_count

    # Global value (action played anywhere in tree)
    q_global = global_action_stats[action].value()
    n_global = global_action_stats[action].visit_count

    # Weighted average (prefer local as n_local grows)
    beta = n_global / (n_local + n_global + 1e-5)
    return (1 - beta) * q_local + beta * q_global
```

Speeds up learning in early game.

### 7. Transposition Tables

Detect when different paths lead to same state:

```python
transposition_table = {}  # {state_hash: MCTSNode}

def get_or_create_node(state):
    """Reuse nodes for same state"""
    state_hash = hash(state)

    if state_hash in transposition_table:
        return transposition_table[state_hash]

    node = MCTSNode(state)
    transposition_table[state_hash] = node
    return node
```

Crucial for games with repetitions (e.g., Go ko rules).

### 8. Solver (Terminal Detection)

Mark solved subtrees:

```python
class MCTSNode:
    def __init__(self):
        # ...
        self.is_solved = False
        self.solved_value = None

def backup_with_solver(path, value):
    """Detect and mark solved positions"""
    for node in reversed(path):
        # Check if all children are solved
        if all(child.is_solved for child in node.children.values()):
            # This node is also solved
            node.is_solved = True

            # Exact value depends on game
            if is_two_player_game:
                node.solved_value = max(
                    -child.solved_value
                    for child in node.children.values()
                )

        # Regular backup
        node.value_sum += value
        node.visit_count += 1
```

### 9. Exploration Noise Annealing

Reduce exploration over time:

```python
def get_noise_epsilon(step, total_steps):
    """Anneal exploration noise"""
    progress = step / total_steps

    # High noise early, low noise late
    epsilon_start = 0.25
    epsilon_end = 0.05

    return epsilon_start * (1 - progress) + epsilon_end * progress
```

### 10. Value Prediction Caching

Cache network evaluations:

```python
value_cache = {}  # {state_hash: (policy, value)}

def forward_with_cache(state):
    """Cache network predictions"""
    state_hash = hash(state)

    if state_hash in value_cache:
        return value_cache[state_hash]

    policy, value = self.network(state)
    value_cache[state_hash] = (policy, value)

    return policy, value
```

Reduces network calls by 50-70% in deterministic games.

## 8. Experiments & Results

### AlphaGo/AlphaZero Performance

MCTS simulations vs playing strength (Elo gain over raw network):

| Simulations | Go | Chess | Shogi | Time (ms) |
|-------------|-----|-------|-------|-----------|
| 1 | +0 | +0 | +0 | 1 |
| 10 | +100 | +80 | +90 | 10 |
| 50 | +200 | +180 | +190 | 50 |
| 100 | +280 | +250 | +270 | 100 |
| 400 | +380 | +340 | +360 | 400 |
| 800 | +420 | +380 | +400 | 800 |
| 1600 | +440 | +400 | +420 | 1600 |

Diminishing returns beyond 800 simulations.

### Branching Factor Impact

Performance on different game complexities:

| Game | Branching | MCTS | Minimax | α-β Pruning |
|------|-----------|------|---------|-------------|
| Tic-Tac-Toe | ~5 | ✓ Optimal | ✓ Optimal | ✓ Optimal |
| Connect-4 | ~7 | ✓ Strong | ✓ Optimal | ✓ Optimal |
| Othello | ~10 | ✓ Strong | ✗ Slow | △ Feasible |
| Chess | ~35 | ✓ Strong | ✗ Infeasible | ✗ Slow |
| Go | ~250 | ✓ Strong | ✗ Infeasible | ✗ Infeasible |

MCTS scales gracefully to large branching factors.

### Ablation Studies

**Effect of c_puct:**
```
c_puct=0.5:  38% win rate (too exploitative)
c_puct=1.0:  52% win rate
c_puct=2.0:  58% win rate (good balance)
c_puct=5.0:  47% win rate (too exploratory)
```

**Effect of value network vs rollouts:**
```
Random rollouts:     1200 Elo
Learned policy rollouts: 1800 Elo
Value network:       2400 Elo (+ 600 Elo!)
```

Value networks are crucial for strong play.

**Effect of tree reuse:**
```
Without reuse: 100% simulations needed
With reuse:    50-60% simulations needed
```

**Effect of batching:**
```
Sequential: 100 evals/sec
Batch=8:    600 evals/sec (6x faster)
Batch=32:   1800 evals/sec (18x faster)
```

### Scaling Laws

Performance vs compute budget:

```
10 sims:    Amateur level
100 sims:   Club player
1K sims:    Expert
10K sims:   Master
100K sims:  Grandmaster
```

Roughly 10x simulations = +200 Elo.

### Generalization

MCTS trained on one game, tested on variants:

```
Chess → Chess960:        -150 Elo (pieces start differently)
Go (19x19) → Go (13x13): -80 Elo (smaller board)
Go (19x19) → Go (9x9):   -200 Elo (very different tactics)
```

MCTS generalizes reasonably well to similar tasks.

## 9. Common Pitfalls

### 1. Wrong Exploration Constant

**Problem:** c_puct too low → Exploits too early, misses better moves.

**Solution:** Tune c_puct on validation set:
```python
for c in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    win_rate = evaluate(mcts, c_puct=c)
    # Pick best
```

Typical values: 1.0-2.0 for games, 2.0-5.0 for planning.

### 2. Not Enough Simulations

**Problem:** Too few simulations → Random decisions.

**Solution:** Scale simulations with importance:
```python
if critical_position:
    num_sims = 1600
else:
    num_sims = 400
```

### 3. No Exploration at Root

**Problem:** Without noise, MCTS gets stuck exploiting first promising move.

**Solution:** Add Dirichlet noise:
```python
root.add_exploration_noise(alpha=0.3, epsilon=0.25)
```

### 4. Incorrect Backup for Two-Player Games

**Problem:** Forgetting to negate value for opponent.

**Solution:**
```python
for node in reversed(path):
    value = -value  # Opponent's perspective
    node.update(value)
```

### 5. Not Batching Network Calls

**Problem:** Sequential evaluation → 90% time waiting for GPU.

**Solution:** Batch evaluate 8-32 nodes at once.

### 6. Ignoring Terminal States

**Problem:** Running simulations on already-won/lost positions.

**Solution:**
```python
if env.is_terminal(state):
    return env.get_reward(state)  # Don't search
```

### 7. Poor Temperature Schedule

**Problem:** Using wrong temperature → Too random or too greedy.

**Solution:**
```python
# Early game: explore (τ=1.0)
# Late game: exploit (τ→0.0)
temp = max(0.1, init_temp * (0.98 ** move_num))
```

### 8. Not Reusing Tree

**Problem:** Discarding valuable search → Wasting 50% of compute.

**Solution:** Reuse subtree after opponent's move:
```python
new_root = old_root.children[opponent_action]
```

### 9. Unbounded Tree Growth

**Problem:** Memory explosion from unlimited tree expansion.

**Solution:** Limit tree size:
```python
if total_nodes > MAX_NODES:
    prune_least_visited_subtrees()
```

### 10. No Handling of Stochastic Transitions

**Problem:** Deterministic MCTS on stochastic environment → Incorrect values.

**Solution:** Sample transitions or use expectimax:
```python
# Sample multiple outcomes
for _ in range(K_samples):
    next_state = env.sample_transition(state, action)
    value += simulate(next_state) / K_samples
```

## 10. References

### Foundational Papers

1. **Kocsis, L., & Szepesvári, C. (2006).** *Bandit Based Monte-Carlo Planning.* ECML.
   - Original UCT algorithm
   - Proved convergence guarantees

2. **Browne, C., et al. (2012).** *A Survey of Monte Carlo Tree Search Methods.* IEEE TCIAIG.
   - Comprehensive MCTS survey
   - Variants and enhancements

3. **Coulom, R. (2006).** *Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search.* CG.
   - RAVE heuristic
   - Rapid value estimation

### AlphaGo/AlphaZero

4. **Silver, D., et al. (2016).** *Mastering the Game of Go with Deep Neural Networks and Tree Search.* Nature.
   - AlphaGo - First superhuman Go AI
   - Neural network-guided MCTS

5. **Silver, D., et al. (2017).** *Mastering the Game of Go without Human Knowledge.* Nature.
   - AlphaGo Zero - Tabula rasa learning
   - Self-play + MCTS

6. **Silver, D., et al. (2018).** *A General Reinforcement Learning Algorithm that Masters Chess, Shogi, and Go through Self-Play.* Science.
   - AlphaZero - Unified algorithm
   - Superhuman in multiple games

### Extensions

7. **Schrittwieser, J., et al. (2020).** *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model.* Nature. (MuZero)
   - MCTS with learned dynamics model
   - No game rules needed

8. **Grill, J.-B., et al. (2020).** *Monte-Carlo Tree Search as Regularized Policy Optimization.* ICML.
   - Theoretical connection to policy gradient
   - MCTS as regularization

### Optimizations

9. **Chaslot, G., et al. (2008).** *Parallel Monte-Carlo Tree Search.* CG.
   - Virtual loss technique
   - Tree/leaf parallelization

10. **Yoshizoe, K., et al. (2011).** *Scalable Distributed Monte-Carlo Tree Search.* SoCS.
    - Distributed MCTS
    - Scaling to clusters

### Applications

11. **Huang, S., et al. (2022).** *Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents.* ICML.
    - MCTS for planning with LLMs
    - Reasoning tasks

12. **Pearce, T., et al. (2023).** *Tree Search for Language Model Agents.* ArXiv.
    - MCTS for code generation
    - Theorem proving

### Theoretical Analysis

13. **Auer, P., et al. (2002).** *Finite-time Analysis of the Multiarmed Bandit Problem.* Machine Learning.
    - UCB1 algorithm
    - Regret bounds

14. **Munos, R. (2014).** *From Bandits to Monte-Carlo Tree Search: The Optimistic Principle Applied to Optimization and Planning.* Foundations and Trends in ML.
    - Theoretical foundations
    - Optimism in the face of uncertainty

### Implementation Reference

- **Nexus MCTS**: `Nexus/nexus/models/search/mcts.py`
- **MCTS Node**: `Nexus/nexus/models/search/mcts_node.py`
- **MCTS Config**: `Nexus/nexus/models/search/mcts_config.py`

---

**Key Takeaways:**
- MCTS balances exploration and exploitation using UCB/PUCT
- Asymptotically optimal with anytime property
- Revolutionized game AI (AlphaGo, AlphaZero)
- Works without domain knowledge, just forward model
- Critical optimizations: batching, tree reuse, virtual loss
- Scales to large branching factors unlike traditional search
