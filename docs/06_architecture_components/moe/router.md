# Expert Router

## Overview & Motivation

The Expert Router is the decision-making component in MoE architectures. It determines which tokens go to which experts, implementing the "mixture" in "Mixture of Experts." A well-designed router is critical for both model quality and training stability.

**Key Responsibilities**:
1. Compute routing scores for each token-expert pair
2. Select top-k experts per token
3. Normalize routing weights
4. Implement load balancing strategies

## Routing Strategies

### 1. Top-K Token Choice

Each token selects its top-k experts (most common):

```python
from nexus.components.moe.router import ExpertRouter

router = ExpertRouter(
    dim=2048,
    num_experts=64,
    top_k=2,
    gating_type='softmax'
)

weights, indices, aux_loss = router(hidden_states)
```

**Code**: `/Users/kevinyu/Projects/Nexus/nexus/components/moe/router.py` - `ExpertRouter`

### 2. Expert Choice

Experts select their top-k tokens (inverted paradigm):

```python
from nexus.components.moe.router import ExpertChoiceRouter

router = ExpertChoiceRouter(
    dim=2048,
    num_experts=32,
    capacity_factor=1.0
)

expert_weights, token_indices, scores = router(hidden_states)
```

**Benefit**: Natural load balancing, no auxiliary loss needed

**Code**: `/Users/kevinyu/Projects/Nexus/nexus/components/moe/router.py` - `ExpertChoiceRouter`

### 3. Noisy Top-K

Adds learned noise for load balancing exploration:

```python
router = ExpertRouter(
    dim=2048,
    num_experts=64,
    top_k=2,
    gating_type='noisy_top_k',  # Adds learned noise
    jitter_noise=0.01
)
```

**Use Case**: Encourage expert diversity during early training

## Load Balancing

### Standard Auxiliary Loss

Encourages uniform token distribution:

```python
from nexus.components.moe.router import LoadBalancingLoss

load_balance = LoadBalancingLoss(
    num_experts=64,
    loss_type='standard',
    loss_weight=0.01
)

aux_loss = load_balance(router_logits, expert_indices)
total_loss = task_loss + aux_loss
```

**Code**: `/Users/kevinyu/Projects/Nexus/nexus/components/moe/router.py` - `LoadBalancingLoss`

### Z-Loss

Prevents router logit divergence:

```python
load_balance = LoadBalancingLoss(
    num_experts=64,
    loss_type='z_loss',
    loss_weight=0.001
)

aux_loss = load_balance(router_logits)
```

**Use Case**: Training stability, especially with many experts

### Loss-Free Balancing

DeepSeek-V3 approach using learnable bias:

```python
from nexus.components.moe.router import LossFreeBalancing

balancer = LossFreeBalancing(
    num_experts=64,
    update_rate=0.01,
    balance_factor=0.1
)

adjusted_logits = balancer(router_logits, expert_indices)
```

**Benefit**: No hyperparameter tuning for loss coefficient

**Code**: `/Users/kevinyu/Projects/Nexus/nexus/components/moe/router.py` - `LossFreeBalancing`

## Mathematical Formulation

### Top-K Routing

**Step 1**: Compute routing scores
```
logits = xW_gate ∈ ℝ^(batch×seq×num_experts)
```

**Step 2**: Add optional noise
```
logits = logits + ε, where ε ~ N(0, σ²)
```

**Step 3**: Select top-k
```
weights, indices = TopK(logits, k)
```

**Step 4**: Normalize
```
weights = Softmax(weights) ∈ ℝ^(batch×seq×k)
```

### Load Balancing Loss

Standard formulation:
```
L_aux = α · num_experts · Σᵢ (fᵢ · Pᵢ)

where:
fᵢ = (tokens routed to expert i) / (total tokens)
Pᵢ = mean(routing_probabilities[:, :, i])
α = loss coefficient (typically 0.01)
```

Minimized when all experts have equal usage.

## Implementation Example

```python
import torch
import torch.nn.functional as F
from nexus.components.moe.router import ExpertRouter

# Create router
router = ExpertRouter(
    dim=2048,
    num_experts=8,
    top_k=2,
    jitter_noise=0.01
)

# Input
x = torch.randn(2, 100, 2048)  # (batch, seq, dim)

# Route
weights, indices, aux_loss = router(x, return_aux_loss=True)

print(f"Weights shape: {weights.shape}")  # (2, 100, 2)
print(f"Indices shape: {indices.shape}")  # (2, 100, 2)
print(f"Aux loss: {aux_loss.item():.4f}")

# Use routing decisions
for expert_idx in range(8):
    mask = (indices == expert_idx)
    tokens_for_expert = x[mask]
    print(f"Expert {expert_idx}: {mask.sum().item()} tokens")
```

## Optimization Tricks

### 1. Capacity Constraints

Limit tokens per expert to prevent overload:

```python
capacity = (total_tokens // num_experts) * capacity_factor

# Drop tokens exceeding capacity
if tokens_to_expert > capacity:
    keep_first_k_tokens(capacity)
```

### 2. Expert Dropout

Randomly drop expert assignments during training:

```python
if training and expert_dropout > 0:
    drop_mask = torch.rand_like(weights) > expert_dropout
    weights = weights * drop_mask
```

### 3. Temperature Scaling

Control routing sharpness:

```python
# Lower temperature = sharper routing
logits = logits / temperature
weights, indices = topk(logits, k)
```

## Common Pitfalls

### 1. Forgetting to Normalize Weights

```python
# WRONG
weights, indices = torch.topk(logits, k)
# weights don't sum to 1!

# CORRECT
weights, indices = torch.topk(logits, k)
weights = F.softmax(weights, dim=-1)
```

### 2. Not Monitoring Expert Utilization

```python
# Track expert usage
expert_counts = torch.bincount(indices.flatten(), minlength=num_experts)
print(f"Expert usage std: {expert_counts.std().item():.2f}")

# High std = imbalanced, tune load balancing
```

### 3. Auxiliary Loss Not Applied

```python
# WRONG: Compute but don't use
weights, indices, aux_loss = router(x)
loss.backward()  # aux_loss ignored!

# CORRECT
weights, indices, aux_loss = router(x)
total_loss = task_loss + 0.01 * aux_loss
total_loss.backward()
```

## References

1. **Shazeer et al. (2017)** - "Outrageously Large Neural Networks"
   - Original top-k routing and load balancing
   - https://arxiv.org/abs/1701.06538

2. **Zhou et al. (2022)** - "Mixture-of-Experts with Expert Choice Routing"
   - Expert-choice routing paradigm
   - https://arxiv.org/abs/2202.09368

3. **DeepSeek-V3 (2024)** - Loss-free load balancing
