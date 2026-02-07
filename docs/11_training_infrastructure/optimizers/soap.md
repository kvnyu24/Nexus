# SOAP: Shampoo with Adam Optimizer Preconditioning

## Overview

SOAP (Vyas et al., 2024) combines Adam's adaptive learning rates with Shampoo's Kronecker-factored preconditioning, providing better conditioning than Adam while being more memory-efficient than full Shampoo.

## Mathematical Foundation

### Preconditioning Basics

Standard Adam update:
$$\\theta_{t+1} = \\theta_t - \\alpha \\frac{m_t}{\\sqrt{v_t} + \\epsilon}$$

SOAP preconditioned update (for 2D parameters):
$$\\theta_{t+1} = \\theta_t - \\alpha P_L^{-1/4} \\frac{m_t}{\\sqrt{v_t} + \\epsilon} P_R^{-1/4}$$

Where $P_L$ and $P_R$ are left/right preconditioners approximating the Hessian structure.

### Kronecker Factorization

For weight matrix $W \\in \\mathbb{R}^{m \\times n}$:
- Left preconditioner: $P_L \\in \\mathbb{R}^{m \\times m}$
- Right preconditioner: $P_R \\in \\mathbb{R}^{n \\times n}$
- Full preconditioner ≈ $P_L \\otimes P_R$ (Kronecker product)

**Memory**: $O(m + n)$ instead of $O(mn)$

### Preconditioner Updates

Every $k$ steps (typically 10):
$$P_L \\leftarrow \\beta P_L + (1-\\beta) g g^T$$
$$P_R \\leftarrow \\beta P_R + (1-\\beta) g^T g$$

Where $g$ is the gradient and $\\beta$ is the preconditioner decay (typically 0.95).

## Implementation

### Basic Usage

```python
from nexus.training.optimizers import SOAP

optimizer = SOAP(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.0,
    precondition_frequency=10,
    max_precond_dim=1024,
    preconditioner_decay=0.95,
    merge_dims=True,
    precondition_1d=False
)

for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### Hyperparameters

**Learning Rate**: 1e-3 (similar to Adam)  
**Precondition Frequency**: 10 (balance accuracy vs overhead)  
**Max Precond Dim**: 1024 (larger dims use standard Adam)  
**Preconditioner Decay**: 0.95 (EMA for preconditioner updates)

### When to Use

**Best for**:
- Transformer models (linear layers benefit most)
- Models with large weight matrices
- Training from scratch

**Not for**:
- Small models (<10M params)
- Primarily convolutional architectures
- Fine-tuning (Adam sufficient)

## Performance

**Convergence**: 10-30% faster than Adam on transformers  
**Memory**: ~2x parameter size for preconditioners  
**Compute**: <5% overhead (preconditioning amortized)

## References

**SOAP: Improving and Stabilizing Shampoo using Adam**  
Nikhil Vyas et al., 2024  
https://arxiv.org/abs/2409.11321
