# Muon: Momentum + Orthogonalization Optimizer

## Overview

Muon (Jordan et al., 2024) applies Nesterov momentum SGD followed by Newton-Schulz orthogonalization to 2D parameters (weights), while using AdamW for non-2D parameters (embeddings, biases). Particularly effective for transformer training.

## Mathematical Foundation

### Newton-Schulz Orthogonalization

For matrix $A$, find nearest orthogonal matrix via iteration:
$$X_{k+1} = \\frac{1}{2} X_k (3I - X_k^T X_k)$$

Converges quadratically to polar factor of $A = US$ decomposition.

### Muon Algorithm

**For 2D parameters** (e.g., linear layers):
```
1. Momentum update:
   buf = momentum * buf + grad
   nesterov_grad = grad + momentum * buf

2. Orthogonalize:
   update = NewtonSchulz(nesterov_grad, steps=5)

3. Scale to match original norm:
   update = update * (||grad|| / ||update||)

4. Apply: param -= lr * update
```

**For non-2D parameters** (embeddings, biases):
- Use standard AdamW

## Implementation

```python
from nexus.training.optimizers import Muon

optimizer = Muon(
    model.parameters(),
    lr=0.02,  # For 2D parameters
    momentum=0.95,
    ns_steps=5,  # Newton-Schulz iterations
    weight_decay=0.0,
    adamw_lr=3e-4,  # For non-2D parameters
    adamw_betas=(0.95, 0.95),
    adamw_wd=0.0
)

for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### Hyperparameters

**Learning Rate (2D)**: 0.02 (for transformers)  
**Momentum**: 0.95 (high momentum recommended)  
**NS Steps**: 5 (sufficient for convergence)  
**AdamW LR (non-2D)**: 3e-4 (standard)

## Performance

**Training Speed**: Comparable to Adam  
**Convergence**: Often better sample efficiency on transformers  
**Memory**: Same as Adam (no extra state for orthogonalization)

**Best Results**: Transformer architectures, especially from-scratch training

## References

**Muon: An optimizer for hidden layers in neural networks**  
Jordan et al., 2024
