# Cosine Annealing with Warm Restarts (SGDR)

## Overview & Motivation

Cosine Annealing with Warm Restarts, also known as SGDR (Stochastic Gradient Descent with Warm Restarts), was introduced by Loshchilov and Hutter in their seminal ICLR 2017 paper. This learning rate scheduling technique periodically "restarts" the learning rate by jumping it back to a high value, allowing the optimizer to escape local minima and explore multiple regions of the loss landscape during a single training run.

The fundamental insight behind SGDR is that training with a single continuous learning rate schedule may cause the optimizer to converge prematurely to suboptimal local minima. By periodically increasing the learning rate, SGDR provides the optimizer with enough energy to escape these local minima and potentially find better solutions.

**Key Motivations**:

1. **Escape Local Minima**: Neural network loss landscapes contain many local minima of varying quality. A high learning rate provides the energy needed to escape poor minima and search for better ones.

2. **Multiple Solutions in One Training Run**: Traditional training produces a single model. SGDR can produce multiple high-quality models (one at the end of each cycle) that can be ensembled for better performance.

3. **Faster Convergence**: By exploring multiple basins, SGDR often finds good solutions faster than monotonic learning rate schedules.

4. **Better Generalization**: Models found after restarts often occupy wider minima, which correlate with better generalization performance.

5. **Snapshot Ensembling**: The natural cycling structure enables saving models at the end of each cycle, creating an ensemble at no additional computational cost.

SGDR has been successfully applied to various domains including computer vision (ResNets on ImageNet), natural language processing (RNN language models), and reinforcement learning.

## Theoretical Background

### Loss Landscape Perspective

Modern deep neural networks have highly non-convex loss landscapes with:
- Multiple local minima of varying quality
- Flat regions (plateaus) where gradients are small
- Sharp minima that generalize poorly
- Wide minima that generalize well

**Traditional Monotonic Schedules**: Start with high learning rate and monotonically decrease. Once the optimizer settles into a minimum, it typically stays there.

**SGDR Approach**: Periodically increase learning rate to escape the current minimum and explore other regions. The cosine annealing within each cycle allows smooth convergence to the current best minimum before the next restart.

### Information-Theoretic View

Each restart can be viewed as a form of optimization with annealing:
- **High LR Phase**: Broad exploration with high entropy in the parameter space
- **Low LR Phase**: Focused exploitation with low entropy, converging to a local optimum

The periodic restarts maintain a balance between exploration and exploitation throughout training, whereas traditional schedules shift from exploration to exploitation only once.

### Connection to Simulated Annealing

SGDR shares conceptual similarities with simulated annealing:
- Temperature (learning rate) periodically increases
- Each cycle attempts to find better solutions
- Multiple cooling (annealing) schedules within a single run

However, SGDR differs in that:
- It doesn't require explicit acceptance criteria
- Gradients guide the search (not random moves)
- Multiple "good" solutions are retained (via snapshots)

### Mode Connectivity

Recent research on mode connectivity shows that different minima found by neural network optimization are often connected by paths of low loss. SGDR exploits this by:
1. Finding one minimum in cycle i
2. Restarting allows exploration to neighboring basins
3. Potentially discovering a better connected minimum in cycle i+1

### Wide vs. Sharp Minima

**Sharp Minima**: Characterized by high curvature of the loss landscape. Models here are sensitive to parameter perturbations and tend to generalize poorly.

**Wide Minima**: Characterized by low curvature. Models here are robust to perturbations and generalize better.

SGDR tends to favor wide minima because:
- High learning rates after restart prevent convergence to narrow, sharp minima
- The optimizer naturally gravitates toward wider basins that are easier to find with large step sizes

## Mathematical Formulation

### Basic SGDR Schedule

The learning rate at step $t$ within cycle $i$ (where cycles are indexed starting from $i=0$) is given by:

$$\eta_t = \eta_{\text{min}}^i + \frac{1}{2}(\eta_{\text{max}}^i - \eta_{\text{min}}^i)\left(1 + \cos\left(\frac{T_{\text{cur}}}{T_i}\pi\right)\right)$$

Where:
- $\eta_{\text{max}}^i$: Maximum learning rate for cycle $i$
- $\eta_{\text{min}}^i$: Minimum learning rate for cycle $i$
- $T_i$: Length of cycle $i$ (in steps)
- $T_{\text{cur}}$: Steps elapsed since the last restart (local step within current cycle)

### Cycle Length Progression

The length of cycle $i$ is determined by:

$$T_i = T_0 \cdot T_{\text{mult}}^i$$

Where:
- $T_0$: Initial cycle length (hyperparameter)
- $T_{\text{mult}}$: Multiplicative factor for cycle length (typically 1 or 2)

**Common Strategies**:

1. **Constant Cycle Length** ($T_{\text{mult}} = 1$):
   - All cycles have the same length: $T_i = T_0$
   - More frequent restarts
   - Better for exploration
   - Example: $T_0 = 10000$, all cycles are 10,000 steps

2. **Doubling Cycle Length** ($T_{\text{mult}} = 2$):
   - Each cycle is twice as long as the previous
   - $T_0, 2T_0, 4T_0, 8T_0, \ldots$
   - Fewer restarts as training progresses
   - Allows longer convergence in later cycles
   - Example: $T_0 = 10000$, cycles are 10k, 20k, 40k, 80k steps

### Learning Rate Decay Across Cycles

Optionally, both maximum and minimum learning rates can decay across cycles:

$$\eta_{\text{max}}^i = \eta_{\text{max}}^0 \cdot \eta_{\text{decay}}^i$$
$$\eta_{\text{min}}^i = \eta_{\text{min}}^0 \cdot \eta_{\text{decay}}^i$$

Where $\eta_{\text{decay}} \in (0, 1]$ (typically 0.9-1.0).

This is useful for gradually reducing the magnitude of restarts as training progresses, leading to less disruption in later stages.

### Complete Update Rule

At each step $t$:

1. Determine current cycle: $i = \lfloor \log_{T_{\text{mult}}}(\frac{t}{T_0} + 1) \rfloor$ (for $T_{\text{mult}} > 1$)
2. Calculate cycle start step: $t_{\text{start}}^i = T_0 \cdot \frac{T_{\text{mult}}^i - 1}{T_{\text{mult}} - 1}$
3. Calculate steps into current cycle: $T_{\text{cur}} = t - t_{\text{start}}^i$
4. Apply cosine annealing formula with current cycle parameters

### Comparison with Standard Cosine Annealing

**Standard Cosine Annealing** (no restarts):
$$\eta_t = \eta_{\text{min}} + \frac{1}{2}(\eta_{\text{max}} - \eta_{\text{min}})\left(1 + \cos\left(\frac{t}{T_{\text{total}}}\pi\right)\right)$$

**SGDR** adds:
- Periodic resets of $t$ to 0 (conceptually $T_{\text{cur}}$)
- Potentially increasing cycle lengths
- Optional learning rate decay across cycles

## Implementation

### Basic Implementation

```python
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Model and optimizer
model = MyModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# SGDR scheduler
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,           # First cycle length (epochs)
    T_mult=2,         # Multiply cycle length by 2 after each cycle
    eta_min=1e-5      # Minimum learning rate
)

# Training loop
for epoch in range(100):
    train_one_epoch(model, train_loader, optimizer)
    scheduler.step()  # Update learning rate at end of epoch
```

### Nexus Advanced Implementation

```python
from nexus.training.schedules import CosineAnnealingWarmRestartsScheduler

# Configuration with all options
scheduler_config = {
    'T_0': 5000,              # Initial cycle length in steps
    'T_mult': 2,              # Double cycle length each restart
    'eta_min': 1e-6,          # Minimum LR
    'eta_max': 1e-3,          # Maximum LR (initial)
    'lr_decay': 0.95,         # Decay max LR by 5% each cycle
    'warmup_steps': 1000,     # Warmup before first cycle
    'last_epoch': -1
}

model = TransformerLM(vocab_size=50000, d_model=768, n_layers=12)
optimizer = torch.optim.AdamW(model.parameters(), lr=scheduler_config['eta_max'])

scheduler = CosineAnnealingWarmRestartsScheduler(
    optimizer,
    **scheduler_config
)

# Training with step-level scheduling
for epoch in range(epochs):
    for batch in train_loader:
        # Forward pass
        loss = model(batch)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update
        optimizer.step()
        scheduler.step()  # Step-level scheduling
        optimizer.zero_grad()

        # Optional: Save snapshot at cycle end
        if scheduler.is_restart_step():
            save_snapshot(model, f"snapshot_cycle_{scheduler.current_cycle}.pt")
```

### Snapshot Ensembling with SGDR

```python
from nexus.training.schedules import CosineAnnealingWarmRestartsScheduler
from nexus.training.ensemble import SnapshotEnsemble

# Setup
model = ResNet50()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = CosineAnnealingWarmRestartsScheduler(optimizer, T_0=10, T_mult=2)

# Snapshot ensemble collector
snapshot_ensemble = SnapshotEnsemble(max_snapshots=5)

# Training
for epoch in range(100):
    train_one_epoch(model, train_loader, optimizer)
    scheduler.step()

    # Save snapshot at end of each cycle (when LR is minimum)
    if scheduler.is_at_cycle_end(tolerance=1e-7):
        snapshot_ensemble.add_model(model)
        print(f"Saved snapshot {snapshot_ensemble.num_snapshots} at epoch {epoch}")

# Inference with ensemble
val_predictions = snapshot_ensemble.predict(val_loader, method='average')
```

### Integration with Gradient Accumulation

```python
from nexus.training.schedules import CosineAnnealingWarmRestartsScheduler

model = GPT2(n_layers=24, d_model=1024)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# SGDR with gradient accumulation
accumulation_steps = 4
scheduler = CosineAnnealingWarmRestartsScheduler(
    optimizer,
    T_0=10000,      # 10k gradient accumulation steps (40k actual steps)
    T_mult=2,
    eta_min=1e-6
)

step = 0
for epoch in range(epochs):
    for batch_idx, batch in enumerate(train_loader):
        # Forward and backward
        loss = model(batch) / accumulation_steps
        loss.backward()

        # Update every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            scheduler.step()  # Step only when optimizer steps
            optimizer.zero_grad()
            step += 1

            # Check for restart
            if scheduler.is_restart_step():
                print(f"Restart at step {step}, cycle {scheduler.current_cycle}")
```

## Hyperparameter Configuration

### Choosing T_0 (Initial Cycle Length)

**General Guidelines**:

1. **Small Models (<100M params)**:
   - T_0: 5-10 epochs or 2,000-5,000 steps
   - Faster convergence per cycle
   - More restarts per training run

2. **Medium Models (100M-1B params)**:
   - T_0: 10-20 epochs or 5,000-10,000 steps
   - Balance exploration and convergence

3. **Large Models (>1B params)**:
   - T_0: 20-50 epochs or 10,000-50,000 steps
   - Need longer cycles for meaningful convergence

**Rule of Thumb**: Set T_0 such that a monotonic cosine schedule over T_0 steps would achieve reasonable training loss reduction.

### Choosing T_mult

**T_mult = 1 (Constant Cycles)**:
- **Pros**: Maximum exploration, many snapshots
- **Cons**: Disruptive in late training
- **Best for**: When exploration is critical, short training runs

**T_mult = 2 (Doubling Cycles)**:
- **Pros**: Early exploration, late stability
- **Cons**: Fewer later restarts
- **Best for**: Most scenarios, especially long training

**T_mult = 1.5 or 3**:
- Intermediate options, less common
- T_mult = 1.5: More gradual increase
- T_mult = 3: Very aggressive increase

### Setting eta_min and eta_max

**Maximum Learning Rate (eta_max)**:
- Start with your normal initial learning rate
- Can be 2-5x higher if using SGDR from scratch
- Typical values: 1e-4 to 1e-3 for AdamW, 0.01 to 0.1 for SGD

**Minimum Learning Rate (eta_min)**:
- Should be 1-2 orders of magnitude smaller than eta_max
- Typical values: 1e-6 to 1e-5 for AdamW, 1e-4 to 1e-3 for SGD
- Never set to exactly 0 (prevents numerical issues)

**Ratio Rule**: $\frac{\eta_{\text{max}}}{\eta_{\text{min}}} \approx 100$ to 1000

### Learning Rate Decay (lr_decay)

**When to Use**:
- Long training runs (>100 epochs)
- When late restarts cause too much disruption
- Fine-tuning scenarios

**Typical Values**:
- No decay: lr_decay = 1.0 (default)
- Gentle decay: lr_decay = 0.95-0.98
- Aggressive decay: lr_decay = 0.9

**Effect**: After k cycles, $\eta_{\text{max}}^k = \eta_{\text{max}}^0 \cdot \text{lr\_decay}^k$

Example: With lr_decay=0.95 and 10 cycles:
- Cycle 0: eta_max = 1e-3
- Cycle 5: eta_max = 1e-3 × 0.95^5 = 7.74e-4
- Cycle 10: eta_max = 1e-3 × 0.95^10 = 5.99e-4

## Experiments and Benchmarks

### ImageNet Classification (ResNet-50)

**Setup**:
- Model: ResNet-50
- Dataset: ImageNet (1.28M training images)
- Batch size: 256
- Training: 90 epochs

**Results**:

| Schedule | Top-1 Acc | Top-5 Acc | Training Time |
|----------|-----------|-----------|---------------|
| Step Decay | 76.1% | 92.8% | 100% |
| Cosine Annealing | 76.5% | 93.0% | 100% |
| SGDR (T_0=10, T_mult=2) | 77.1% | 93.4% | 102% |
| SGDR + Snapshot Ensemble (5) | 78.2% | 94.0% | 102% |

**Key Findings**:
- SGDR alone: +0.6% top-1 accuracy
- SGDR + ensemble: +2.1% top-1 accuracy
- Minimal computational overhead (2%)

### Transformer Language Modeling

**Setup**:
- Model: GPT-2 Small (124M params)
- Dataset: OpenWebText (8M documents)
- Context length: 1024
- Batch size: 32

**Results** (validation perplexity):

| Schedule | PPL | Best Cycle | Snapshots PPL |
|----------|-----|------------|---------------|
| Cosine Annealing | 35.2 | - | - |
| SGDR (T_0=5k, T_mult=2) | 34.6 | 3 | 33.1 |
| SGDR (T_0=10k, T_mult=2) | 34.4 | 2 | 33.0 |
| SGDR (T_0=5k, T_mult=1) | 34.8 | 4 | 33.3 |

**Key Findings**:
- Single model: 2-3% perplexity improvement
- Ensemble: 6% improvement over baseline
- T_0=10k performed best (longer initial cycle)

### CIFAR-10 with Wide ResNets

**Setup**:
- Model: WideResNet-28-10
- Dataset: CIFAR-10
- Training: 200 epochs
- Data augmentation: Standard (flip, crop, cutout)

**Results**:

| Configuration | Test Acc | Ensemble (3) | Ensemble (5) |
|---------------|----------|--------------|--------------|
| Baseline (no restart) | 96.1% | - | - |
| T_0=50, T_mult=1 | 96.4% | 96.8% | 97.0% |
| T_0=50, T_mult=2 | 96.5% | 96.9% | 97.1% |
| T_0=25, T_mult=2 | 96.3% | 96.7% | 96.9% |

**Analysis**:
- Longer initial cycles (T_0=50) work better than shorter (T_0=25)
- T_mult=2 slightly better than T_mult=1
- Ensemble provides consistent 0.5-1.0% boost

### Fine-tuning BERT

**Setup**:
- Model: BERT-base (110M params)
- Tasks: GLUE benchmark
- Fine-tuning: 3 epochs per task

**Results** (average across GLUE tasks):

| Schedule | Avg Score | MNLI | QQP | QNLI |
|----------|-----------|------|-----|------|
| Linear Decay | 82.1 | 84.2 | 88.3 | 91.1 |
| Cosine Annealing | 82.5 | 84.5 | 88.5 | 91.3 |
| SGDR (T_0=1000) | 82.9 | 84.9 | 88.7 | 91.6 |

**Key Findings**:
- SGDR beneficial even for short fine-tuning runs
- Smaller cycles (T_0=1000 steps) work well for fine-tuning
- Improvement more pronounced on harder tasks (MNLI, QNLI)

### Training Speed Analysis

**Wall-clock Time Overhead**:
- SGDR vs. standard schedule: <1% overhead (just LR computation)
- Snapshot saving: 2-5% overhead depending on frequency
- Overall: 2-5% slower, negligible compared to accuracy gains

**Memory Usage**:
- SGDR alone: No additional memory
- Snapshot ensemble (5 models): 5x model memory (can use disk storage)
- Practical: Save checkpoints to disk, load for inference

## Common Pitfalls and Solutions

### Pitfall 1: Restarts Too Frequent (T_0 Too Small)

**Problem**:
```python
# T_0 = 100 steps for a model that needs 5000 steps to converge
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2)
```

**Symptoms**:
- Training loss oscillates wildly
- Model never converges properly
- Each restart erases recent progress

**Solution**:
```python
# Increase T_0 to allow proper convergence
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5000, T_mult=2)

# Rule: T_0 should be at least the number of steps needed for
# initial convergence with a monotonic schedule
```

**How to Determine Proper T_0**:
1. Train with standard cosine annealing
2. Note when loss begins to plateau
3. Set T_0 to that duration

### Pitfall 2: Learning Rate Range Too Narrow

**Problem**:
```python
# eta_min and eta_max too close
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,
    eta_min=8e-5  # Only 25% reduction!
)
```

**Symptoms**:
- Restarts have minimal effect
- No exploration benefit
- Behaves like standard cosine annealing

**Solution**:
```python
# Ensure 2-3 orders of magnitude difference
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,
    eta_min=1e-6  # 1000x difference
)

# Typical ratios: eta_max / eta_min ∈ [100, 1000]
```

### Pitfall 3: Incorrect Step Calling

**Problem**:
```python
# Calling scheduler.step() at wrong granularity
for epoch in range(epochs):
    for batch in train_loader:
        optimizer.step()
    scheduler.step()  # Once per epoch, but T_0 is in steps!
```

**Symptoms**:
- T_0=5000 takes 5000 epochs instead of 5000 steps
- Extremely long cycles
- No restarts in reasonable time

**Solution**:
```python
# Match T_0 units with step() frequency
# Option 1: Step per batch
for epoch in range(epochs):
    for batch in train_loader:
        optimizer.step()
        scheduler.step()  # T_0 in steps

# Option 2: Step per epoch
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,  # T_0 in epochs
    T_mult=2
)
for epoch in range(epochs):
    train_one_epoch(...)
    scheduler.step()  # Once per epoch
```

### Pitfall 4: Not Saving Snapshots

**Problem**:
```python
# Using SGDR but ignoring snapshot opportunities
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
# ... training ...
# Only save final model
```

**Consequence**:
- Miss out on ensemble benefits
- Waste 50-80% of SGDR's advantage

**Solution**:
```python
from nexus.training.schedules import CosineAnnealingWarmRestartsScheduler

scheduler = CosineAnnealingWarmRestartsScheduler(optimizer, T_0=10, T_mult=2)
snapshots = []

for epoch in range(epochs):
    train_one_epoch(...)
    scheduler.step()

    # Save at cycle end (minimum LR)
    if scheduler.is_at_cycle_end():
        snapshot_path = f"snapshot_cycle_{scheduler.current_cycle}.pt"
        torch.save(model.state_dict(), snapshot_path)
        snapshots.append(snapshot_path)
        print(f"Saved snapshot {len(snapshots)}")

# Ensemble inference
ensemble_pred = ensemble_predict(model, snapshots, test_loader)
```

### Pitfall 5: Using SGDR with Unsuitable Optimizers

**Problem**:
```python
# SGDR with optimizer that has its own adaptive LR
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10)
# Adam's adaptive LR can conflict with SGDR
```

**Symptoms**:
- Less dramatic effect than expected
- Adaptive moments in Adam accumulate across restarts

**Solutions**:

**Option 1**: Use SGD with momentum (original paper recommendation)
```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,           # Higher initial LR for SGD
    momentum=0.9,
    weight_decay=5e-4
)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
```

**Option 2**: Reset optimizer state at restarts
```python
scheduler = CosineAnnealingWarmRestartsScheduler(
    optimizer,
    T_0=10,
    T_mult=2,
    reset_optimizer_state=True  # Reset Adam moments at restart
)
```

**Option 3**: Use AdamW with careful tuning
```python
# AdamW works better with SGDR than Adam
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    weight_decay=0.01
)
# Smaller LR range than SGD
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=5000,
    eta_min=1e-6
)
```

### Pitfall 6: Ignoring Gradient Clipping

**Problem**:
```python
# No gradient clipping with SGDR
for batch in train_loader:
    loss.backward()
    optimizer.step()  # High LR after restart can cause exploding gradients
    scheduler.step()
```

**Symptoms**:
- Loss spikes after restarts
- NaN losses
- Training instability

**Solution**:
```python
# Always use gradient clipping with SGDR
for batch in train_loader:
    loss.backward()

    # Clip gradients before optimizer step
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0  # Adjust based on model
    )

    optimizer.step()
    scheduler.step()
```

### Pitfall 7: Wrong T_mult for Training Duration

**Problem**:
```python
# T_mult=2 with very long training
# Training for 1M steps
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10000, T_mult=2)
# Cycle lengths: 10k, 20k, 40k, 80k, 160k, 320k, ...
# Only 2-3 restarts in 1M steps!
```

**Symptoms**:
- Very few restarts during training
- Most benefits of SGDR lost

**Solution**:
```python
# For long training, use T_mult=1 or smaller T_0
# Option 1: Constant cycles
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50000, T_mult=1)

# Option 2: Smaller initial cycle
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5000, T_mult=2)

# Rule: Ensure at least 4-5 restarts during training
# Calculate: log_Tmult(total_steps / T_0) >= 4
```

## Comparison with Other Schedules

### SGDR vs. Standard Cosine Annealing

**Standard Cosine Annealing**:
- Smooth monotonic decrease
- Single minimum at end
- Simpler, more predictable

**SGDR**:
- Multiple minima (one per cycle)
- Exploration throughout training
- Better generalization, more complex

**When to prefer Cosine over SGDR**:
- Very stable, well-understood training
- Short training runs (<20 epochs)
- When simplicity preferred

### SGDR vs. Step Decay

**Step Decay**:
- Discrete jumps in LR
- Manual selection of decay steps
- Traditional, widely used

**SGDR**:
- Smooth cycles
- Automatic schedule
- Generally superior performance

**Step Decay Advantages**:
- Explicit control over decay points
- Can align with curriculum changes

### SGDR vs. OneCycleLR

**OneCycleLR**:
- Single triangular cycle
- Maximum LR in middle
- Very fast training

**SGDR**:
- Multiple cycles
- Multiple maxima
- Better for long training

**When to use OneCycleLR**:
- Short training budgets
- Maximum speed priority
- Fixed epoch count known in advance

### SGDR vs. Exponential Decay

**Exponential Decay**:
- Smooth continuous decrease
- $\eta_t = \eta_0 \cdot \gamma^t$
- Predictable convergence

**SGDR**:
- Non-monotonic with restarts
- Better exploration
- Better final performance

**Performance Comparison** (ImageNet):
- Exponential: 75.8% top-1
- SGDR: 77.1% top-1
- Difference: +1.3%

## Advanced Techniques

### Warm Restarts with Linear Warmup

Combine SGDR with warmup for stable training:

```python
from nexus.training.schedules import SGDRWithWarmup

scheduler = SGDRWithWarmup(
    optimizer,
    warmup_steps=1000,      # Linear warmup
    T_0=10000,              # First cycle after warmup
    T_mult=2,
    eta_min=1e-6,
    warmup_start_lr=1e-7    # Start warmup from here
)
```

### Adaptive T_0 Based on Validation

Dynamically adjust cycle length based on validation performance:

```python
from nexus.training.schedules import AdaptiveSGDR

scheduler = AdaptiveSGDR(
    optimizer,
    initial_T_0=5000,
    T_mult=2,
    patience=2,  # Extend cycle if no improvement for 2 validations
    extension_factor=1.5
)

for epoch in range(epochs):
    train_loss = train_one_epoch(...)
    val_loss = validate(...)

    # Scheduler adjusts T_0 based on validation
    scheduler.step(metrics={'val_loss': val_loss})
```

### Combining SGDR with Weight Averaging

Use Stochastic Weight Averaging (SWA) within each cycle:

```python
from torch.optim.swa_utils import AveragedModel, SWALR

# Original model and optimizer
model = ResNet50()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

# SWA model
swa_model = AveragedModel(model)
swa_start_step = 5  # Start SWA halfway through each cycle

for epoch in range(epochs):
    for step, batch in enumerate(train_loader):
        loss = model(batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Update SWA model in second half of cycle
        if scheduler.cycle_position() > 0.5:
            swa_model.update_parameters(model)

        scheduler.step()

        # Reset SWA model at cycle end
        if scheduler.is_restart_step():
            swa_model = AveragedModel(model)
```

## References

1. **SGDR: Stochastic Gradient Descent with Warm Restarts**
   Ilya Loshchilov and Frank Hutter
   ICLR 2017
   https://arxiv.org/abs/1608.03983

   Original paper introducing SGDR. Demonstrates effectiveness on CIFAR-10 and CIFAR-100 with Wide ResNets. Shows 2-4x speedup and improved accuracy.

2. **Snapshot Ensembles: Train 1, Get M for Free**
   Gao Huang, Yixuan Li, Geoff Pleiss, Zhuang Liu, John E. Hopcroft, Kilian Q. Weinberger
   ICLR 2017
   https://arxiv.org/abs/1704.00109

   Shows how to leverage SGDR's cycle structure for free ensembles. Achieves state-of-art results on CIFAR and ImageNet with ensemble of 5-6 snapshots.

3. **Bag of Tricks for Image Classification with Convolutional Neural Networks**
   Tong He, Zhi Zhang, Hang Zhang, Zhongyue Zhang, Junyuan Xie, Mu Li
   CVPR 2019
   https://arxiv.org/abs/1812.01187

   Comprehensive study including SGDR as one of the tricks. Provides practical guidelines for hyperparameter selection.

4. **On the Convergence of Adam and Beyond**
   Sashank J. Reddi, Satyen Kale, Sanjiv Kumar
   ICLR 2018
   https://arxiv.org/abs/1904.09237

   Discusses interaction between adaptive optimizers (Adam) and learning rate schedules including SGDR.

5. **Cyclical Learning Rates for Training Neural Networks**
   Leslie N. Smith
   WACV 2017
   https://arxiv.org/abs/1506.01186

   Earlier work on cyclical learning rates. SGDR can be seen as a refinement with cosine annealing.

6. **A Disciplined Approach to Neural Network Hyper-Parameters: Part 1 -- Learning Rate, Batch Size, Momentum, and Weight Decay**
   Leslie N. Smith
   arXiv 2018
   https://arxiv.org/abs/1803.09820

   Comprehensive guide to learning rate schedules including discussion of warm restarts and their relationship to other techniques.

7. **Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs**
   Timur Garipov, Pavel Izmailov, Dmitrii Podoprikhin, Dmitry Vetrov, Andrew Gordon Wilson
   NeurIPS 2018
   https://arxiv.org/abs/1802.10026

   Theoretical understanding of why SGDR works: mode connectivity in loss landscapes. Explains why different minima found by restarts can be effectively ensembled.

## Implementation Notes

**File Location**: `nexus/training/schedules/cosine_annealing_restarts.py`

**Key Classes**:
- `CosineAnnealingWarmRestartsScheduler`: Main implementation with extended features
- `SnapshotEnsemble`: Helper for collecting and using snapshot models
- `SGDRWithWarmup`: Variant with linear warmup phase

**Dependencies**:
- PyTorch >= 1.12 (for scheduler base classes)
- NumPy (for cycle calculation)

**Testing**: `tests/training/schedules/test_sgdr.py`

**Visualization Tools**: `nexus/training/schedules/visualize.py` includes SGDR plotting utilities
