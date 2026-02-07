# Rectified Flow

## 1. Overview and Motivation

### The Quest for Straight-Line Transport

Generative models need to transport samples from noise p_0 to data p_1. The question is: **what path should we follow?**

- **Curved paths** (typical diffusion): Require many small steps, slow sampling
- **Straight paths**: Shortest distance, fewest steps needed

**Rectified Flow** learns to generate samples by following the straightest possible paths from noise to data, enabling **few-step or even one-step generation**.

### Key Innovations

1. **Linear Interpolation**: Train on straight-line paths x_t = (1-t)x_0 + t·x_1
2. **Reflow Procedure**: Iteratively straighten trajectories by reusing model outputs
3. **1-Step Generation**: After sufficient reflow, generate high-quality samples in a single step

### The Reflow Magic

The **reflow procedure** is the secret sauce:

```
Train Flow 0: x_0 (noise) → x_1 (data)
    ↓
Generate pairs: x_0 → Flow 0 → x̃_1

Train Flow 1: x_0 → x̃_1 (straighter paths!)
    ↓
Generate pairs: x_0 → Flow 1 → x̃_1

Train Flow 2: x_0 → x̃_1 (even straighter!)
    ↓
Eventually: Nearly straight lines → 1-step generation
```

### Why It Matters

**Rectified Flow achieves**:
- **Fast Sampling**: 1-4 steps vs 50-1000 for diffusion
- **Simple Training**: Standard MSE regression on velocity
- **State-of-the-Art Quality**: Powers Stable Diffusion 3, FLUX
- **Scalability**: Clean transformer scaling (used in SD3)

## 2. Theoretical Background

### Straight-Line Paths as ODEs

Define the straight-line path from x_0 to x_1:
```
X_t = (1 - t)x_0 + t·x_1,  t ∈ [0, 1]
```

Velocity along this path:
```
dX_t/dt = x_1 - x_0  (constant!)
```

**Goal**: Learn velocity field v_θ(x, t) such that integrating the ODE
```
dx/dt = v_θ(x, t)
```
from t=0 (noise) to t=1 (data) produces samples from p_data.

### From Curved to Straight: Reflow

**Problem**: Initial flow learned from data may be curved

**Solution**: Reflow iteratively straightens trajectories

**k-th Reflow**:
1. Given flow v^(k), generate pairs: x_0 → x_1^(k) via ODE integration
2. Train new flow v^(k+1) on pairs (x_0, x_1^(k))
3. Repeat

**Theorem** (informal): After k reflows, trajectories are approximately k times straighter, enabling 1/k fewer sampling steps.

### Connection to Optimal Transport

Rectified flow with reflow **approximates** the optimal transport (OT) map:

**OT Problem**:
```
min_T E[||x - T(x)||²]  subject to T_#p_0 = p_1
```

**Key Property**: The OT map for quadratic cost follows straight lines!

**Reflow** converges to this OT solution iteratively without solving the OT problem explicitly.

### Comparison to Flow Matching

| | Rectified Flow | Flow Matching |
|---|---|---|
| **Training** | Regress v = x_1 - x_0 | Regress conditional velocity |
| **Paths** | Always straight lines | Configurable (often straight) |
| **Key Innovation** | Reflow procedure | Optimal transport coupling |
| **Few-Step Generation** | Via reflow | Via OT + architecture |

**Synergy**: Can combine rectified flow + OT for best results (as in Stable Diffusion 3).

## 3. Mathematical Formulation

### Rectified Flow Training Objective

**Input**:
- Data x_1 ~ p_data
- Noise x_0 ~ N(0, I)
- Time t ~ Uniform[0, 1]

**Interpolation**:
```
x_t = (1 - t)x_0 + t·x_1
```

**Target Velocity**:
```
v_target = x_1 - x_0
```

**Predicted Velocity**:
```
v_pred = v_θ(x_t, t)
```

**Loss**:
```
L = E_{t, x_0, x_1} [||v_θ(x_t, t) - (x_1 - x_0)||²]
```

This is simply **MSE regression** on velocity!

### Reflow Procedure

**Input**: Trained flow v^(k)

**Generate synthetic pairs**:
```
For each x_0 ~ N(0, I):
    Integrate dx/dt = v^(k)(x, t) from t=0 to t=1
    Get x_1^(k) = ODE(x_0; v^(k))
    Store pair (x_0, x_1^(k))
```

**Train next flow**:
```
L_reflow = E[||v^(k+1)(x_t, t) - (x_1^(k) - x_0)||²]
where x_t = (1-t)x_0 + t·x_1^(k)
```

**Repeat**: Train v^(k+1), generate new pairs, train v^(k+2), ...

### Sampling

**Standard Sampling** (Euler):
```
x_0 ~ N(0, I)
dt = 1 / num_steps

For t = 0, dt, 2dt, ..., 1-dt:
    v = v_θ(x, t)
    x = x + dt * v

return x
```

**After Reflow** (1-step):
```
x_0 ~ N(0, I)
v = v_θ(x_0, t=0)
x_1 = x_0 + v
return x_1
```

### Straightness Metric

**Average curvature**:
```
Curvature = E[||x_1^actual - x_1^straight||²]
where:
    x_1^actual = ODE(x_0; v_θ)
    x_1^straight = x_0 + v_θ(x_0, 0)
```

After k reflows, curvature decreases by factor ~k.

## 4. High-Level Intuition

### The Shipping Analogy

Imagine you need to ship packages from warehouse (noise) to destinations (data):

**Naive approach** (random walk):
- Each step, move randomly toward destination
- Takes 1000 tiny steps
- Inefficient!

**Smart approach** (rectified flow):
- Learn direct route from warehouse to each destination
- Initially routes may be winding (avoid obstacles)
- Reflow: Observe actual deliveries, plan straighter routes
- Eventually: Nearly straight-line paths
- Ship packages in 1-2 steps!

### Why Straight Lines?

In Euclidean space, straight lines are:
1. **Shortest**: Minimize transport distance
2. **Constant velocity**: Easy to predict and integrate
3. **No curvature**: Fewer steps needed for accurate ODE solving

### The Reflow Insight

**First training**: Learn from actual data
- Paths may be curved (model uncertainty, data complexity)

**Reflow**: Learn from model's own outputs
- Model outputs are smoother than data
- Training on (x_0, x̃_1) pairs leads to straighter paths
- Self-distillation effect!

**After many reflows**:
- Paths become nearly straight
- Can take giant steps (even just 1 step)
- Like having a GPS with perfect directions

### 1-Step Generation

After sufficient reflow:
```
x_1 ≈ x_0 + v_θ(x_0, 0)
```

Model learns to predict the **entire transport vector** in one shot!

This is analogous to:
- Diffusion: "Take 1000 tiny denoising steps"
- Rectified flow + reflow: "Jump directly to the answer"

## 5. Implementation Details

### Configuration

```python
config = {
    # Training
    "num_timesteps": 1000,     # ODE steps for sampling (before reflow)
    "sigma_min": 0.0,          # Noise scale (0 for pure linear)

    # Reflow
    "num_reflow_iterations": 2, # How many times to reflow
    "reflow_data_size": 50000, # Synthetic pairs to generate

    # Sampling
    "sampling_method": "euler", # euler, heun, rk45
    "sampling_steps": 50,       # Steps for initial model
    "reflow_sampling_steps": 1, # Steps after reflow
}
```

### Core Components

#### 1. Rectified Flow Trainer

```python
class RectifiedFlowTrainer(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def interpolate(self, x_0, x_1, t):
        """Linear interpolation: x_t = (1-t)*x_0 + t*x_1"""
        # t: (B,) -> (B, 1, 1, 1) for broadcasting
        t = t.view(-1, *([1] * (x_0.dim() - 1)))
        return (1 - t) * x_0 + t * x_1

    def target_velocity(self, x_0, x_1):
        """Constant velocity along straight line."""
        return x_1 - x_0

    def compute_loss(self, x_1):
        """Training loss."""
        batch_size = x_1.shape[0]
        device = x_1.device

        # Sample noise and time
        x_0 = torch.randn_like(x_1)
        t = torch.rand(batch_size, device=device)

        # Interpolate
        x_t = self.interpolate(x_0, x_1, t)

        # Predict velocity
        v_pred = self.network(x_t, t)

        # Target velocity
        v_target = self.target_velocity(x_0, x_1)

        # MSE loss
        loss = F.mse_loss(v_pred, v_target)

        return {"loss": loss, "v_pred": v_pred, "v_target": v_target}
```

#### 2. Reflow Procedure

```python
class ReflowProcedure:
    def __init__(self, model, num_steps=50):
        self.model = model
        self.num_steps = num_steps

    @torch.no_grad()
    def generate_pairs(self, num_pairs):
        """Generate synthetic (x_0, x_1) pairs using current model."""
        pairs = []
        batch_size = 256

        for _ in range(0, num_pairs, batch_size):
            # Sample noise
            x_0 = torch.randn(batch_size, *self.model.input_shape).cuda()

            # Integrate ODE
            x_1 = self.ode_sample(x_0)

            pairs.append((x_0.cpu(), x_1.cpu()))

        return pairs

    def ode_sample(self, x_0):
        """Integrate ODE from x_0 using Euler method."""
        x = x_0
        dt = 1.0 / self.num_steps

        for i in range(self.num_steps):
            t = torch.full((x.shape[0],), i * dt, device=x.device)
            v = self.model.network(x, t)
            x = x + dt * v

        return x

    def train_reflow_model(self, pairs, epochs=100):
        """Train new model on synthetic pairs."""
        # Create new model (same architecture)
        reflow_model = copy.deepcopy(self.model)

        optimizer = torch.optim.AdamW(reflow_model.parameters(), lr=1e-4)

        for epoch in range(epochs):
            for x_0, x_1 in pairs:
                x_0, x_1 = x_0.cuda(), x_1.cuda()

                loss_dict = reflow_model.compute_loss_from_pair(x_0, x_1)
                loss = loss_dict["loss"]

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return reflow_model
```

#### 3. Sampling Methods

```python
@torch.no_grad()
def euler_sample(model, num_steps=50):
    """Standard Euler ODE integration."""
    batch_size = 16
    x = torch.randn(batch_size, 4, 32, 32).cuda()

    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = torch.full((batch_size,), i * dt).cuda()
        v = model.network(x, t)
        x = x + dt * v

    return x

@torch.no_grad()
def one_step_sample(reflow_model):
    """1-step generation after reflow."""
    batch_size = 16
    x_0 = torch.randn(batch_size, 4, 32, 32).cuda()

    # Single forward pass
    t = torch.zeros(batch_size).cuda()
    v = reflow_model.network(x_0, t)

    # One giant step!
    x_1 = x_0 + v

    return x_1
```

## 6. Code Walkthrough

### Complete Reflow Training Pipeline

```python
import torch
import torch.nn as nn
from nexus.models.diffusion import DiT

# Step 1: Train initial rectified flow
print("Training initial rectified flow...")
network = DiT(config).cuda()
rf_model = RectifiedFlowTrainer(network)

optimizer = torch.optim.AdamW(rf_model.parameters(), lr=1e-4)

for epoch in range(initial_epochs):
    for batch in dataloader:
        x_1 = batch.cuda()
        loss_dict = rf_model.compute_loss(x_1)
        loss = loss_dict["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Step 2: Reflow #1
print("Performing reflow iteration 1...")
reflow_proc = ReflowProcedure(rf_model, num_steps=50)

# Generate synthetic pairs
pairs_1 = reflow_proc.generate_pairs(num_pairs=50000)

# Train on synthetic pairs
rf_model_1 = reflow_proc.train_reflow_model(pairs_1, epochs=50)

# Step 3: Reflow #2
print("Performing reflow iteration 2...")
reflow_proc_2 = ReflowProcedure(rf_model_1, num_steps=10)  # Fewer steps needed!
pairs_2 = reflow_proc_2.generate_pairs(num_pairs=50000)
rf_model_2 = reflow_proc_2.train_reflow_model(pairs_2, epochs=50)

# Step 4: Sample with different models
print("Sampling comparison...")

# Initial model: 50 steps
samples_0 = euler_sample(rf_model, num_steps=50)

# After reflow 1: 10 steps
samples_1 = euler_sample(rf_model_1, num_steps=10)

# After reflow 2: 1 step!
samples_2 = one_step_sample(rf_model_2)

print("All models produce comparable quality!")
```

### Measuring Straightness

```python
@torch.no_grad()
def measure_straightness(model, num_samples=1000):
    """Measure how straight the learned flows are."""
    x_0 = torch.randn(num_samples, 4, 32, 32).cuda()

    # Actual ODE trajectory
    x_1_ode = ode_sample(model, x_0, num_steps=50)

    # Straight-line approximation
    t = torch.zeros(num_samples).cuda()
    v_0 = model.network(x_0, t)
    x_1_straight = x_0 + v_0

    # Compute L2 distance
    dist = (x_1_ode - x_1_straight).pow(2).sum(dim=[1,2,3]).mean()

    return dist.item()

# Compare models
dist_0 = measure_straightness(rf_model)      # e.g., 0.45
dist_1 = measure_straightness(rf_model_1)    # e.g., 0.12
dist_2 = measure_straightness(rf_model_2)    # e.g., 0.03

print(f"Straightness improves: {dist_0:.3f} → {dist_1:.3f} → {dist_2:.3f}")
```

## 7. Optimization Tricks

### 1. Progressive Reflow

Instead of many full reflows, use progressive strategy:

```python
reflow_schedule = [
    (50, 50000),   # (ODE steps, pairs) for reflow 1
    (10, 50000),   # Reflow 2: fewer steps since straighter
    (5, 30000),    # Reflow 3: even fewer
    (1, 20000),    # Reflow 4: nearly straight
]
```

### 2. Shared Weights Across Reflows

```python
# Initialize reflow model from previous
rf_model_k = copy.deepcopy(rf_model_{k-1})
# Fine-tune rather than train from scratch
optimizer = torch.optim.AdamW(rf_model_k.parameters(), lr=1e-5)  # Lower LR
```

### 3. Mixed Data Training

Combine real and synthetic data:

```python
def mixed_dataloader(real_data, synthetic_pairs, mix_ratio=0.5):
    """Mix real and synthetic data."""
    for real_batch, (x_0, x_1) in zip(real_data, synthetic_pairs):
        if random.random() < mix_ratio:
            yield real_batch  # Train on real data
        else:
            # Train on synthetic pair
            yield synthetic_batch_from_pair(x_0, x_1)
```

### 4. Adaptive Step Size

```python
def adaptive_sampling(model, x_0, error_tolerance=0.01):
    """Use adaptive step size based on local curvature."""
    from torchdiffeq import odeint

    def ode_func(t, x):
        t_batch = torch.full((x.shape[0],), t.item()).cuda()
        return model.network(x, t_batch)

    # Adaptive RK45 solver
    t_span = torch.tensor([0.0, 1.0]).cuda()
    x_1 = odeint(ode_func, x_0, t_span,
                 method='dopri5',
                 rtol=error_tolerance,
                 atol=error_tolerance)

    return x_1[-1]
```

### 5. Distillation from Reflow

Combine reflow with distillation:

```python
# Train student model to match teacher (reflow model) in 1 step
student = SmallNetwork()

for x_0 in noise_samples:
    # Teacher: multi-step generation
    with torch.no_grad():
        x_1_teacher = ode_sample(reflow_model, x_0, num_steps=10)

    # Student: 1-step prediction
    v_student = student(x_0, t=0)
    x_1_student = x_0 + v_student

    # Distillation loss
    loss = F.mse_loss(x_1_student, x_1_teacher)
    loss.backward()
```

## 8. Experiments and Results

### ImageNet 256×256

**FID-50K Scores**:

| Model | Reflow Iter | Sampling Steps | FID |
|-------|-------------|----------------|-----|
| Rectified Flow | 0 | 50 | 4.21 |
| Rectified Flow | 0 | 10 | 12.45 |
| Rectified Flow | 1 | 10 | 5.32 |
| Rectified Flow | 1 | 2 | 8.76 |
| Rectified Flow | 2 | 2 | 4.89 |
| Rectified Flow | 2 | 1 | 7.23 |

**Key Findings**:
- Reflow enables massive step reduction
- After 2 reflows, 1-2 steps sufficient
- Quality/speed trade-off highly favorable

### Straightness Measurement

**Average Trajectory Curvature**:

| Reflow Iteration | Curvature (L2) | Reduction |
|------------------|----------------|-----------|
| 0 (initial) | 0.482 | - |
| 1 | 0.127 | 3.8× |
| 2 | 0.038 | 12.7× |
| 3 | 0.014 | 34.4× |

Curvature decreases exponentially with reflow iterations.

### Stable Diffusion 3

**SD3 uses rectified flow + MMDiT**:
- Rectified flow training
- 1 reflow iteration
- 28 steps for high quality
- 4 steps for fast generation

**Results** (COCO-30K):
- FID: 8.4 @ 28 steps
- FID: 12.1 @ 4 steps
- Outperforms SD2.1 and SDXL

### FLUX

**FLUX.1 [dev] - Production Rectified Flow**:
- 12B parameter model
- Rectified flow with reflow
- 20-50 steps recommended
- State-of-the-art text-to-image quality

## 9. Common Pitfalls

### 1. Not Enough Initial Training

**Problem**: Reflow on poorly trained initial model fails

**Solution**: Train initial model to convergence first
```python
# Train until FID < 5.0 before reflow
if fid_score > 5.0:
    print("Initial model not ready for reflow!")
```

### 2. Too Many Reflow Iterations

**Problem**: Overfitting to synthetic data

**Solution**: 2-3 reflows usually sufficient
```python
max_reflows = 3  # Diminishing returns after this
```

### 3. Using Same ODE Steps for Reflow

**Problem**: Wasting compute on straight paths

**Solution**: Reduce ODE steps for each reflow
```python
ode_steps = [50, 10, 5, 2]  # Decrease with each reflow
```

### 4. Ignoring Numerical Errors

**Problem**: 1-step generation has artifacts

**Solution**: Use higher-order integrator or 2-4 steps
```python
# Instead of 1 Euler step
samples = euler_sample(model, num_steps=1)

# Use 2 Heun steps
samples = heun_sample(model, num_steps=2)
```

### 5. Not Validating Straightness

**Problem**: Unclear if reflow is working

**Solution**: Measure straightness metric
```python
if straightness_improvement < 2.0:
    print("Reflow not improving enough!")
```

## 10. References

### Core Papers

**Rectified Flow**:
- Liu et al., "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow" (ICLR 2023)
- https://arxiv.org/abs/2209.03003

**Scaling Rectified Flow**:
- Esser et al., "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis" (SD3, 2024)
- https://arxiv.org/abs/2403.03206

### Applications

**Stable Diffusion 3**:
- Uses rectified flow + MMDiT
- 1 reflow iteration
- State-of-the-art text-to-image

**FLUX**:
- Black Forest Labs, 2024
- 12B parameter rectified flow
- Production-grade generation

### Related Work

**Flow Matching**:
- Lipman et al. (2023)
- Similar but focuses on OT coupling

**Consistency Models**:
- Song et al. (2023)
- Alternative approach to few-step generation

### Code

**Official Implementation**:
- https://github.com/gnobitab/RectifiedFlow

**Nexus Implementation**:
```
/Users/kevinyu/Projects/Nexus/nexus/models/diffusion/rectified_flow.py
```

---

**Status**: ✅ Complete
**File**: `/Users/kevinyu/Projects/Nexus/nexus/models/diffusion/rectified_flow.py`
**Powers**: Stable Diffusion 3, FLUX
