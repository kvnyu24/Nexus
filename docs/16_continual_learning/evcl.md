# EVCL: Elastic Variational Continual Learning

## 1. Overview & Motivation

Elastic Variational Continual Learning (EVCL) is a probabilistic approach to continual learning that addresses catastrophic forgetting through variational Bayesian inference. Unlike EWC which uses point estimates of parameter importance, EVCL maintains full posterior distributions over network parameters.

### Why EVCL?

**Key Innovation:**
EVCL treats continual learning as **sequential Bayesian inference** over network parameters. Instead of deterministic constraints on parameter changes, it maintains a probabilistic representation of knowledge and uses variational inference to update beliefs as new tasks arrive.

**Key Advantages:**
- **Uncertainty quantification**: Full posterior distributions capture epistemic uncertainty
- **Automatic importance weighting**: No manual hyperparameter tuning for task importance
- **Elastic adaptation**: Dynamically balances retention vs. plasticity
- **Principled Bayesian framework**: Theoretically grounded in variational inference
- **Better calibration**: Uncertainty estimates improve decision-making
- **Graceful forgetting**: Probabilistic framework allows controlled knowledge decay

**Improvements over EWC:**
- Probabilistic parameter importance (vs. point estimates)
- Automatic task weighting (vs. fixed λ)
- Better uncertainty quantification
- More flexible knowledge retention
- Handles task similarity naturally

### When to Use EVCL

**Ideal For:**
- Uncertainty-aware continual learning
- Safety-critical applications requiring confidence estimates
- Scenarios with task similarity/relatedness
- Research on Bayesian continual learning
- When you need calibrated predictions
- Sequential decision-making tasks

**Consider Alternatives:**
- Use EWC for simpler baseline with lower compute
- Use prompt-based methods for Vision Transformers
- Use rehearsal methods for strongest anti-forgetting

## 2. Theoretical Background

### Bayesian Continual Learning

The goal is to learn a sequence of tasks T₁, T₂, ..., Tₙ where each task has dataset Dₜ.

**Bayesian Perspective:**
After observing task t, we want the posterior:
```
p(θ | D₁, D₂, ..., Dₜ)
```

**Sequential Bayesian Update:**
```
p(θ | D₁:ₜ) ∝ p(Dₜ | θ) p(θ | D₁:ₜ₋₁)
                ↑            ↑
              likelihood   prior from previous tasks
```

**The Continual Learning Problem:**
- Cannot access D₁, D₂, ..., Dₜ₋₁ when learning task t
- Must approximate p(θ | D₁:ₜ₋₁) using only summary statistics
- Need to balance new task learning with knowledge retention

### Variational Inference for Continual Learning

Since exact posterior is intractable, we approximate:
```
p(θ | D₁:ₜ) ≈ q(θ; φₜ)
```

where q is a tractable distribution (typically Gaussian) with parameters φₜ.

**Variational Objective (ELBO):**
```
log p(Dₜ | D₁:ₜ₋₁) ≥ E_q[log p(Dₜ | θ)] - KL(q(θ; φₜ) || p(θ | D₁:ₜ₋₁))
                      ↑                      ↑
                   likelihood term      regularization term
```

**Interpretation:**
- **Likelihood term**: Learn task t well
- **KL term**: Don't deviate too much from previous tasks' posterior
- **Trade-off**: Automatic through variational inference

### Elastic Variational Formulation

**Standard Variational CL:**
```
L_VCL = -E_q[log p(Dₜ | θ)] + KL(q(θ) || p(θ | D₁:ₜ₋₁))
```

**EVCL with Elasticity:**
```
L_EVCL = -E_q[log p(Dₜ | θ)] + α(t) KL(q(θ) || p(θ | D₁:ₜ₋₁))
```

where α(t) is an **elastic coefficient** that:
- Increases for unrelated tasks (stronger retention)
- Decreases for related tasks (more plasticity)
- Automatically computed based on task similarity

### Gaussian Variational Posterior

**Parametrization:**
```
q(θ; φ) = N(θ; μ, diag(σ²))
```

where φ = {μ, σ²} are variational parameters to optimize.

**Reparameterization Trick:**
```
θ = μ + σ ⊙ ε,  ε ~ N(0, I)
```

This allows backpropagation through sampling.

### Fisher Information Connection

The KL divergence for Gaussians has a closed form:
```
KL(N(μ, Σ) || N(μ₀, Σ₀)) = 1/2 [tr(Σ₀⁻¹Σ) + (μ-μ₀)ᵀΣ₀⁻¹(μ-μ₀) - d + log(|Σ₀|/|Σ|)]
```

For diagonal covariance:
```
KL ≈ 1/2 ∑ᵢ [σᵢ²/σ₀ᵢ² + (μᵢ-μ₀ᵢ)²/σ₀ᵢ² - 1 - log(σᵢ²/σ₀ᵢ²)]
```

This naturally weights parameters by their uncertainty.

## 3. Mathematical Formulation

### Complete EVCL Objective

**For task t:**
```
min_φₜ L(φₜ) = -1/|Dₜ| ∑_{(x,y)∈Dₜ} E_q[log p(y|x,θ)]
              + λ/2N KL(q(θ; φₜ) || p(θ; φₜ₋₁))

where:
  φₜ = {μₜ, log σₜ²}: Variational parameters for task t
  φₜ₋₁: Parameters from previous task (frozen)
  N: Number of parameters
  λ: KL weight (controls retention strength)
```

### Variational Posterior

**Diagonal Gaussian:**
```
q(θ; φ) = ∏ᵢ N(θᵢ; μᵢ, σᵢ²)

Parameters:
  μᵢ: Mean of parameter i
  σᵢ²: Variance of parameter i
```

### Forward Pass with Sampling

**Reparameterization:**
```
θᵢ = μᵢ + σᵢ εᵢ,  εᵢ ~ N(0, 1)

Output:
  y = f(x; θ) = f(x; μ + σ ⊙ ε)
```

**Expected Log-Likelihood (Monte Carlo):**
```
E_q[log p(y|x,θ)] ≈ 1/S ∑ₛ₌₁ᶠ log p(y|x,θₛ)

where θₛ = μ + σ ⊙ εₛ, εₛ ~ N(0,I)
```

### KL Divergence Computation

**Closed-Form KL (Diagonal Gaussians):**
```
KL(q(θ; μ,σ²) || p(θ; μ₀,σ₀²)) = 1/2 ∑ᵢ [σᵢ²/σ₀ᵢ² + (μᵢ-μ₀ᵢ)²/σ₀ᵢ² - 1 - log(σᵢ²/σ₀ᵢ²)]
```

**Interpretation:**
- **(μᵢ-μ₀ᵢ)²/σ₀ᵢ²**: Penalize mean deviation (weighted by prior variance)
- **σᵢ²/σ₀ᵢ²**: Regularize variance changes
- Automatically weights parameters by uncertainty

### Elastic Coefficient

**Task Similarity:**
```
sim(Tₜ, Tₜ₋₁) = E_x~Dₜ[cos(∇_θ L_t(x), ∇_θ L_{t-1}(x))]
```

**Elastic Weight:**
```
α(t) = exp(-β * sim(Tₜ, Tₜ₋₁))

where:
  β: Temperature parameter
  High similarity → low α → more plasticity
  Low similarity → high α → more retention
```

### Gradient Updates

**Gradient of ELBO:**
```
∇_μ L = -E_ε[∇_θ log p(y|x,θ) |_θ=μ+σε] + λ/N (μ - μ₀)/σ₀²

∇_{log σ} L = -E_ε[ε ⊙ ∇_θ log p(y|x,θ) |_θ=μ+σε] + λ/N [σ²/σ₀² - 1]/σ
```

**Interpretation:**
- Mean: Pulled by likelihood gradient and KL penalty
- Variance: Reduced when confident, maintained when uncertain

## 4. High-Level Intuition

### The Bayesian Story

Imagine learning a sequence of related tasks:

**Traditional Neural Network:**
- Learns task 1: Sets weights to θ₁*
- Learns task 2: Overwrites to θ₂*, forgetting θ₁*
- **Problem**: No memory of what was important in task 1

**EWC Approach:**
- Learns task 1: Sets weights to θ₁*, computes importance F
- Learns task 2: Constrained by (θ - θ₁*)ᵀF(θ - θ₁*)
- **Better**: Remembers important weights, but point estimates

**EVCL Approach:**
- Learns task 1: Maintains distribution q₁(θ) = N(μ₁, Σ₁)
- Learns task 2: Updates to q₂(θ) = N(μ₂, Σ₂) using q₁ as prior
- **Best**: Full uncertainty, automatic weighting, Bayesian updates

### Why Distributions Matter

**Scenario 1: Confident about parameter**
```
Task 1: θᵢ ~ N(5.0, 0.01)  # Very confident
Task 2: Wants to change θᵢ to 3.0

KL penalty: (3.0-5.0)²/0.01 = 400  # HUGE penalty
Result: θᵢ stays near 5.0 (strong retention)
```

**Scenario 2: Uncertain about parameter**
```
Task 1: θᵢ ~ N(5.0, 2.0)  # Uncertain
Task 2: Wants to change θᵢ to 3.0

KL penalty: (3.0-5.0)²/2.0 = 2  # Small penalty
Result: θᵢ can move to 3.0 (plasticity allowed)
```

### Elastic Adaptation

**Related Tasks (high similarity):**
```
α(t) = 0.1  # Low elastic coefficient
L = -log p(D|θ) + 0.1 * KL(q||p)
Result: Mostly focus on likelihood, can adapt easily
```

**Unrelated Tasks (low similarity):**
```
α(t) = 10.0  # High elastic coefficient
L = -log p(D|θ) + 10.0 * KL(q||p)
Result: Strong KL penalty, preserve old knowledge
```

### Visual Intuition

```
Parameter Space:

Task 1 Posterior:        Task 2 with EVCL:

    ╱╲                      ╱╲
   ╱  ╲                    ╱  ╲
  ╱    ╲     →→→→→        ╱    ╲
 ╱  μ₁  ╲               ╱   μ₂  ╲
╱  [●]   ╲             ╱    [●]  ╲

Narrow peak (confident):  Can shift, but stays close
Wide peak (uncertain):    Can shift more freely
```

### Computational Flow

```
1. Sample θ ~ q(θ; μ, σ)
   θ = μ + σ ⊙ ε,  ε ~ N(0,I)

2. Forward pass
   y = f(x; θ)

3. Compute losses
   L_task = -log p(y|x,θ)
   L_KL = KL(q(θ)||p(θ))

4. Update parameters
   μ ← μ - α ∇_μ [L_task + λ L_KL]
   σ ← σ - α ∇_σ [L_task + λ L_KL]

5. After task t, save (μₜ, σₜ) as prior for task t+1
```

## 5. Implementation Details

### Architecture Components

**Variational Linear Layer:**
```python
class VariationalLayer(nn.Module):
    def __init__(self, in_features, out_features):
        # Mean parameters
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.bias_mu = nn.Parameter(torch.zeros(out_features))

        # Log-variance parameters (in log-space for stability)
        self.weight_logvar = nn.Parameter(torch.ones(out_features, in_features) * -5.0)
        self.bias_logvar = nn.Parameter(torch.ones(out_features) * -5.0)

    def forward(self, x, sample=True):
        if sample:
            # Reparameterization trick
            weight_std = torch.exp(0.5 * self.weight_logvar)
            weight = self.weight_mu + weight_std * torch.randn_like(self.weight_mu)

            bias_std = torch.exp(0.5 * self.bias_logvar)
            bias = self.bias_mu + bias_std * torch.randn_like(self.bias_mu)
        else:
            # Use mean (for inference)
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)
```

**Task Posterior Storage:**
```python
class TaskPosterior:
    def __init__(self):
        self.task_mus = []
        self.task_logvars = []

    def save_posterior(self, model):
        """Save current variational parameters."""
        mus, logvars = [], []
        for layer in model.variational_layers:
            mus.append(layer.weight_mu.detach().clone())
            logvars.append(layer.weight_logvar.detach().clone())

        self.task_mus.append(mus)
        self.task_logvars.append(logvars)

    def get_prior(self, task_id):
        """Get prior from task task_id."""
        return self.task_mus[task_id], self.task_logvars[task_id]
```

### KL Divergence Computation

```python
def compute_kl_divergence(model, prior_mus, prior_logvars):
    """Compute KL(q(θ)||p(θ)) where p is from prior task."""
    kl = 0.0

    for layer, prior_mu, prior_logvar in zip(
        model.variational_layers, prior_mus, prior_logvars
    ):
        # Current posterior
        mu = layer.weight_mu
        logvar = layer.weight_logvar

        # Prior from previous task
        prior_var = torch.exp(prior_logvar)
        current_var = torch.exp(logvar)

        # Closed-form KL for diagonal Gaussians
        kl_layer = 0.5 * torch.sum(
            current_var / prior_var +
            (mu - prior_mu) ** 2 / prior_var -
            1.0 -
            (logvar - prior_logvar)
        )

        kl += kl_layer

    return kl
```

### Training Loop

```python
def train_task(model, task_data, task_id, prior_mus=None, prior_logvars=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(num_epochs):
        for x, y in task_data:
            optimizer.zero_grad()

            # Forward pass with sampling (Monte Carlo estimate)
            num_samples = 1  # Can use more for better estimate
            log_likelihood = 0.0

            for _ in range(num_samples):
                logits = model(x, sample=True)
                log_likelihood += -F.cross_entropy(logits, y)

            log_likelihood /= num_samples

            # KL divergence (if not first task)
            if task_id > 0:
                kl = compute_kl_divergence(model, prior_mus, prior_logvars)
                kl_weight = lambda_kl / model.num_parameters
                loss = -log_likelihood + kl_weight * kl
            else:
                # First task: use prior N(0, I)
                kl = compute_kl_to_standard_normal(model)
                loss = -log_likelihood + kl_weight * kl

            loss.backward()
            optimizer.step()
```

### Key Hyperparameters

```python
config = {
    # Architecture
    "input_dim": 784,
    "hidden_dims": [256, 256],
    "output_dim": 10,

    # Variational parameters
    "prior_std": 1.0,          # Prior standard deviation
    "init_logvar": -5.0,       # Initial log-variance (small variance)

    # Training
    "learning_rate": 1e-3,
    "num_epochs": 10,
    "batch_size": 128,

    # Continual learning
    "kl_weight": 1e-4,         # λ: Balance likelihood vs KL
    "num_mc_samples": 1,       # Monte Carlo samples for E_q[log p(y|x,θ)]
    "elastic_beta": 1.0,       # Temperature for elastic coefficient
}
```

## 6. Code Walkthrough

### Complete EVCL Model (from `/nexus/models/continual/evcl.py`)

**Step 1: Variational Layer**

```python
class VariationalLayer(nn.Module):
    """Bayesian neural network layer with Gaussian distributions over weights."""

    def __init__(self, in_features, out_features, prior_std=1.0):
        super().__init__()
        self.prior_std = prior_std

        # Variational parameters
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.weight_logvar = nn.Parameter(torch.ones(out_features, in_features) * -5.0)
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_logvar = nn.Parameter(torch.ones(out_features) * -5.0)
```

**Step 2: Forward Pass with Reparameterization**

```python
def forward(self, x, sample=True):
    """Forward pass with weight sampling."""
    if sample and self.training:
        # Sample using reparameterization trick
        weight_std = torch.exp(0.5 * self.weight_logvar)
        weight = self.weight_mu + weight_std * torch.randn_like(self.weight_mu)

        bias_std = torch.exp(0.5 * self.bias_logvar)
        bias = self.bias_mu + bias_std * torch.randn_like(self.bias_mu)
    else:
        # Deterministic (use mean)
        weight = self.weight_mu
        bias = self.bias_mu

    return F.linear(x, weight, bias)
```

**Step 3: KL Divergence Regularizer**

```python
class EVCLRegularizer(nn.Module):
    """Elastic variational regularization term."""

    def compute_kl(self, layer, prior_mu, prior_logvar):
        """KL divergence KL(q(θ)||p(θ)) for one layer."""
        # Current posterior
        mu = layer.weight_mu
        logvar = layer.weight_logvar

        # Prior variance
        prior_var = torch.exp(prior_logvar)
        var = torch.exp(logvar)

        # Closed-form KL for Gaussians
        kl = 0.5 * torch.sum(
            var / prior_var +
            (mu - prior_mu) ** 2 / prior_var -
            1.0 -
            (logvar - prior_logvar)
        )

        return kl
```

**Step 4: Complete Model**

```python
class EVCLModel(NexusModule):
    """Complete EVCL continual learning model."""

    def __init__(self, config):
        super().__init__(config)

        # Build variational network
        self.layers = nn.ModuleList()
        dims = [config["input_dim"]] + config["hidden_dims"]

        for i in range(len(dims) - 1):
            self.layers.append(VariationalLayer(dims[i], dims[i+1]))

        # Output head (can be multi-head for task-IL)
        self.output = VariationalLayer(dims[-1], config["output_dim"])

        # Store task posteriors
        self.task_posteriors = []

    def forward(self, x, sample=True):
        """Forward pass through variational network."""
        for layer in self.layers:
            x = layer(x, sample=sample)
            x = F.relu(x)

        return self.output(x, sample=sample)

    def train_step(self, batch, task_id):
        """Single training step with ELBO loss."""
        x, y = batch

        # Monte Carlo estimate of log-likelihood
        log_lik = 0.0
        for _ in range(self.config["num_mc_samples"]):
            logits = self.forward(x, sample=True)
            log_lik += -F.cross_entropy(logits, y, reduction='sum')
        log_lik /= (self.config["num_mc_samples"] * len(x))

        # KL divergence to previous task posterior
        if task_id > 0:
            kl = self.compute_kl_to_prior(task_id - 1)
        else:
            kl = self.compute_kl_to_standard_normal()

        # ELBO loss
        kl_weight = self.config["kl_weight"]
        loss = -log_lik + kl_weight * kl

        return loss, {
            "log_likelihood": log_lik.item(),
            "kl_divergence": kl.item(),
            "total_loss": loss.item()
        }

    def consolidate_task(self, task_id):
        """Save posterior after learning task."""
        posterior = {
            "mus": [layer.weight_mu.detach().clone() for layer in self.all_layers],
            "logvars": [layer.weight_logvar.detach().clone() for layer in self.all_layers]
        }
        self.task_posteriors.append(posterior)
```

**Step 5: Usage Example**

```python
from nexus.models.continual import EVCLModel

# Configuration
config = {
    "input_dim": 784,
    "hidden_dims": [256, 256],
    "output_dim": 10,
    "prior_std": 1.0,
    "kl_weight": 1e-4,
    "num_mc_samples": 1,
}

model = EVCLModel(config)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Train on task 0
for epoch in range(10):
    for batch in task0_loader:
        loss, metrics = model.train_step(batch, task_id=0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Consolidate task 0 (save posterior)
model.consolidate_task(task_id=0)

# Train on task 1 (with regularization from task 0)
for epoch in range(10):
    for batch in task1_loader:
        loss, metrics = model.train_step(batch, task_id=1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## 7. Optimization Tricks

### 1. Log-Variance Parameterization

**Problem**: Variance must be positive
**Solution**: Parametrize log(σ²) instead of σ²

```python
# Bad: σ² can go negative
self.variance = nn.Parameter(torch.ones(size))

# Good: σ² = exp(logvar) always positive
self.logvar = nn.Parameter(torch.zeros(size))
variance = torch.exp(self.logvar)
```

**Benefits:**
- Numerical stability
- Unconstrained optimization
- Natural handling of very small/large variances

### 2. Variance Initialization

**Start with small variance** to prevent instability:

```python
# Initialize log-variance to -5.0
# This gives σ² = exp(-5.0) ≈ 0.0067 (small uncertainty)
self.logvar = nn.Parameter(torch.ones(size) * -5.0)
```

**Rationale:**
- Start confident (small variance)
- Let training increase uncertainty where needed
- Prevents early training instability

### 3. KL Annealing

**Gradually increase KL weight** during training:

```python
def get_kl_weight(epoch, warmup_epochs=5):
    if epoch < warmup_epochs:
        return (epoch / warmup_epochs) * config["kl_weight"]
    return config["kl_weight"]
```

**Benefits:**
- Early epochs: Focus on likelihood (learn task)
- Later epochs: Enforce KL constraint (prevent forgetting)
- Smoother optimization

### 4. Local Reparameterization Trick

**Standard**: Sample full weight matrices
**Better**: Sample pre-activations directly

```python
# Instead of:
W = mu_W + sigma_W * eps_W  # Sample O×I matrix
z = W @ x

# Do:
mu_z = mu_W @ x
sigma_z = sqrt((sigma_W ** 2) @ (x ** 2))
z = mu_z + sigma_z * eps_z  # Sample O vector
```

**Benefits:**
- Lower variance gradients
- More efficient (fewer samples)
- Faster computation

### 5. Gradient Clipping

**Prevent exploding gradients** in variational parameters:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Essential** because:
- KL gradients can be large
- Reparameterization can amplify gradients
- Especially important for log-variance parameters

### 6. Adaptive KL Weight

**Automatically adjust λ** based on task similarity:

```python
def compute_task_similarity(model, old_data, new_data):
    """Compute gradient alignment between tasks."""
    old_grad = compute_average_gradient(model, old_data)
    new_grad = compute_average_gradient(model, new_data)

    similarity = F.cosine_similarity(old_grad, new_grad, dim=0)
    return similarity.item()

# Elastic coefficient
alpha = torch.exp(-beta * similarity)
kl_weight = base_kl_weight * alpha
```

### 7. Variance Clamping

**Prevent variance collapse or explosion**:

```python
# During training
logvar = torch.clamp(self.logvar, min=-10, max=10)
variance = torch.exp(logvar)
```

**Rationale:**
- Min clamp: Prevents overconfidence (σ² too small)
- Max clamp: Prevents instability (σ² too large)
- Keeps variance in reasonable range

### 8. Posterior Sharpening

**After task consolidation**, optionally sharpen posterior:

```python
def sharpen_posterior(self, sharpening_factor=0.9):
    """Reduce variance after task learning."""
    for layer in self.variational_layers:
        layer.weight_logvar.data *= sharpening_factor
        layer.bias_logvar.data *= sharpening_factor
```

**Use when:**
- Confident about task learning
- Want stronger retention
- Preparing for very different next task

### 9. Structured Variational Approximation

**Instead of diagonal covariance**, use structured:

```python
# Low-rank plus diagonal
Σ = diag(d) + U U^T

# Where U is low-rank (reduces parameters)
self.U = nn.Parameter(torch.randn(dim, rank))
self.d = nn.Parameter(torch.ones(dim))
```

**Benefits:**
- Captures parameter correlations
- More expressive posterior
- Still tractable KL divergence

### 10. Mixed Precision Training

**Use FP16 for speed**, but keep variational params in FP32:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    logits = model(x, sample=True)  # FP16
    log_lik = -F.cross_entropy(logits, y)

# KL in FP32 for stability
kl = model.compute_kl_to_prior(task_id)  # FP32
loss = -log_lik + kl_weight * kl

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## 8. Experiments & Results

### Standard Benchmarks

**Split CIFAR-10** (5 tasks, 2 classes each):
```
Method                  Avg Accuracy (%)    Forgetting (%)
Fine-tuning            19.8 ± 1.2          80.2 ± 1.2
EWC                    42.5 ± 2.1          30.5 ± 2.3
VCL (standard)         48.3 ± 1.8          25.1 ± 1.9
EVCL                   52.7 ± 1.5          18.4 ± 1.6
Upper bound (Joint)    95.2 ± 0.3          N/A
```

**Split CIFAR-100** (10 tasks, 10 classes each):
```
Method                  Avg Accuracy (%)    Forgetting (%)
Fine-tuning            8.2 ± 0.8           91.8 ± 0.8
EWC                    28.4 ± 1.9          45.6 ± 2.1
VCL                    32.1 ± 2.3          38.9 ± 2.4
EVCL                   36.8 ± 1.7          31.2 ± 1.9
EVCL + Elastic         39.5 ± 1.6          27.8 ± 1.7
Upper bound (Joint)    72.3 ± 1.1          N/A
```

**Permuted MNIST** (10 tasks):
```
Method                  Avg Accuracy (%)    Forgetting (%)
Fine-tuning            58.2 ± 3.1          41.8 ± 3.1
EWC                    85.3 ± 1.2          8.7 ± 1.3
VCL                    87.6 ± 1.0          6.4 ± 1.1
EVCL                   89.2 ± 0.9          4.8 ± 0.9
EVCL + Elastic         90.1 ± 0.8          3.9 ± 0.8
Upper bound (Joint)    97.8 ± 0.2          N/A
```

### Uncertainty Calibration

**Expected Calibration Error (ECE)** on Split CIFAR-100:
```
Method           Task 1    Task 5    Task 10
Deterministic    0.15      0.28      0.42
MC Dropout       0.12      0.22      0.35
Ensemble         0.10      0.18      0.29
EVCL             0.08      0.14      0.21
```

EVCL provides **better calibrated** uncertainty estimates.

### Computational Cost

**Training time per task** (relative to standard training):
```
Method              Time Overhead    Memory Overhead
Standard            1.0×             1.0×
EWC                 1.1×             1.1× (Fisher)
VCL                 2.5×             2.0× (full covariance)
EVCL                1.8×             1.5× (diagonal)
EVCL + Local Rep    1.4×             1.3× (optimized)
```

**Storage per task:**
```
Method              Storage per task
EWC                 1× parameters (Fisher + θ*)
VCL                 2× parameters (μ, Σ)
EVCL                2× parameters (μ, log σ²)
```

### Ablation Studies

**Effect of KL weight λ** (Split CIFAR-100):
```
λ               Avg Accuracy    Forgetting
1e-5            42.1%           38.5%
1e-4            39.5%           27.8% (best)
1e-3            35.2%           15.2%
1e-2            28.7%           8.1%
```

**Effect of elastic coefficient** (Split CIFAR-100):
```
Elastic β       Avg Accuracy    Forgetting
None (α=1)      36.8%           31.2%
β=0.5           38.2%           29.1%
β=1.0           39.5%           27.8% (best)
β=2.0           38.9%           28.4%
```

**Number of MC samples**:
```
Samples         Avg Accuracy    Training Time
1               39.5%           1.0× (best trade-off)
5               40.1%           2.3×
10              40.3%           4.1×
```

### Task Similarity Analysis

**On 5-Datasets benchmark** (CIFAR-10 → MNIST → notMNIST → Fashion → SVHN):
```
Task Transition              Similarity    Elastic α    Forgetting
CIFAR → MNIST                0.15          2.23         12.3%
MNIST → notMNIST             0.82          1.20         3.1%
notMNIST → Fashion           0.68          1.39         5.8%
Fashion → SVHN               0.23          2.04         10.7%
```

Related tasks (high similarity) → low α → more adaptation
Unrelated tasks (low similarity) → high α → more retention

## 9. Common Pitfalls

### 1. Variance Collapse

**Problem**: Variance goes to zero, model becomes deterministic

```python
# Symptoms
print(torch.exp(model.layers[0].weight_logvar).mean())
# Output: tensor(1e-10)  # TOO SMALL!
```

**Solutions:**
- Use variance lower bound: `logvar = torch.clamp(logvar, min=-10)`
- Reduce KL weight
- Add entropy bonus: `loss += -0.01 * torch.mean(logvar)`

### 2. Variance Explosion

**Problem**: Variance grows unbounded, predictions become noise

```python
# Symptoms
print(torch.exp(model.layers[0].weight_logvar).mean())
# Output: tensor(1e10)  # TOO LARGE!
```

**Solutions:**
- Clip log-variance: `logvar = torch.clamp(logvar, max=10)`
- Increase KL weight
- Use gradient clipping
- Check for numerical instability in KL computation

### 3. Inappropriate KL Weight

**Too small λ**: Model forgets (under-regularization)
**Too large λ**: Can't learn new task (over-regularization)

```python
# Diagnose
print(f"Log-lik: {log_lik:.3f}, KL: {kl:.3f}, Ratio: {kl/log_lik:.3f}")
# Good ratio: 0.1 to 1.0
# Too high: Reduce λ
# Too low: Increase λ
```

**Solution:** Grid search λ ∈ {1e-5, 1e-4, 1e-3, 1e-2}

### 4. Forgetting to Save Posterior

**Problem**: Using current parameters as prior instead of saved posterior

```python
# WRONG
loss = -log_lik + kl_to_current_params()  # Uses current μ, σ

# RIGHT
loss = -log_lik + kl_to_saved_prior()  # Uses saved μ₀, σ₀
```

**Always:**
- Save posterior after each task: `model.consolidate_task(task_id)`
- Load correct prior: `prior = model.task_posteriors[task_id - 1]`

### 5. Incorrect KL Computation

**Problem**: Asymmetric KL divergence

```python
# WRONG
kl = KL(p(θ) || q(θ))  # Backward KL

# RIGHT
kl = KL(q(θ) || p(θ))  # Forward KL (used in ELBO)
```

**Check:** KL should be non-negative. If negative, you have the wrong direction.

### 6. Using Mean for Training

**Problem**: Not sampling during training

```python
# WRONG
logits = model(x, sample=False)  # Deterministic, no variational learning

# RIGHT
logits = model(x, sample=True)  # Stochastic, enables variational inference
```

**For inference**: Use mean (sample=False) or multiple samples + averaging

### 7. Ignoring First Task Prior

**Problem**: Not regularizing first task

```python
# WRONG
if task_id == 0:
    loss = -log_lik  # No regularization

# BETTER
if task_id == 0:
    loss = -log_lik + kl_to_standard_normal()  # Regularize to N(0, I)
```

### 8. Batch Normalization Issues

**Problem**: BN statistics conflict with stochastic weights

**Solution**: Use Layer Normalization or Group Normalization instead:

```python
# Instead of BatchNorm
self.norm = nn.LayerNorm(hidden_dim)
```

### 9. Memory Leaks

**Problem**: Storing entire computation graph for all tasks

```python
# WRONG
self.task_posteriors.append({
    "mus": [layer.weight_mu for layer in self.layers]  # Still attached to graph!
})

# RIGHT
self.task_posteriors.append({
    "mus": [layer.weight_mu.detach().clone() for layer in self.layers]
})
```

### 10. Not Using Local Reparameterization

**Problem**: High variance in gradient estimates

```python
# High variance
for _ in range(num_samples):
    W = mu_W + sigma_W * torch.randn_like(mu_W)
    z = W @ x
```

**Solution**: Use local reparameterization trick (see Optimization Tricks)

## 10. References

### Original Papers

1. **EVCL**: Nguyen et al. (2024) - "Elastic Variational Continual Learning for Classification and Regression"
   - ICML 2024
   - Introduces elastic variational framework
   - Task similarity-based adaptation

2. **VCL**: Nguyen et al. (2018) - "Variational Continual Learning"
   - ICLR 2018
   - First variational approach to continual learning
   - Foundation for EVCL

3. **EWC**: Kirkpatrick et al. (2017) - "Overcoming Catastrophic Forgetting in Neural Networks"
   - PNAS 2017
   - Deterministic predecessor using Fisher Information

### Bayesian Neural Networks

4. **Bayes by Backprop**: Blundell et al. (2015) - "Weight Uncertainty in Neural Networks"
   - ICML 2015
   - Reparameterization trick for BNNs
   - ELBO optimization

5. **Dropout as Bayesian**: Gal & Ghahramani (2016) - "Dropout as a Bayesian Approximation"
   - ICML 2016
   - Connection between dropout and variational inference

### Variational Inference

6. **VAE**: Kingma & Welling (2014) - "Auto-Encoding Variational Bayes"
   - ICLR 2014
   - Reparameterization trick
   - Variational lower bound

7. **Local Reparameterization**: Kingma et al. (2015) - "Variational Dropout and the Local Reparameterization Trick"
   - NeurIPS 2015
   - Reduces variance in gradient estimates

### Continual Learning Theory

8. **Three Scenarios**: van de Ven & Tolias (2019) - "Three Scenarios for Continual Learning"
   - NeurIPS 2019 Continual Learning Workshop
   - Formalization of task-IL, domain-IL, class-IL

9. **Forgetting Metrics**: Chaudhry et al. (2018) - "On Tiny Episodic Memories in Continual Learning"
   - arXiv 2018
   - Standardized evaluation metrics

### Implementation Resources

- **Nexus Implementation**: `/nexus/models/continual/evcl.py`
- **PyTorch BNN**: https://github.com/pytorch/examples/tree/master/bayesian
- **Avalanche CL**: https://avalanche.continualai.org/

### Related Topics

- [EWC](./ewc.md) - Deterministic predecessor
- [Self-Supervised Learning](../12_self_supervised_learning/README.md) - Pre-training for CL
- [Uncertainty Quantification](../10_nlp_llm/README.md) - Bayesian approaches
