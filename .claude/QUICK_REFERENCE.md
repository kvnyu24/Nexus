# Nexus Quick Reference

## Adding New Algorithms - Quick Commands

### Option 1: Natural Language (Recommended)
```
"Add SAC algorithm to the repository"
"Implement FlashAttention-3"
"Create TD3 with documentation"
```

### Option 2: Direct Skill Invocation
```
/add-module    # For implementation
/add-docs      # For documentation
```

## File Paths Cheat Sheet

### Implementations

| Category | Path | Example |
|----------|------|---------|
| RL - Value-based | `nexus/models/rl/value_based/` | `dqn.py` |
| RL - Policy Gradient | `nexus/models/rl/policy_gradient/` | `sac.py`, `td3.py` |
| RL - Offline | `nexus/models/rl/offline/` | `iql.py`, `cql.py` |
| RL - Alignment | `nexus/models/rl/alignment/` | `dpo.py`, `grpo.py` |
| RL - Multi-Agent | `nexus/models/rl/multi_agent/` | `mappo.py`, `qmix.py` |
| RL - Model-Based | `nexus/models/rl/model_based/` | `dreamerv3.py` |
| RL - Planning | `nexus/models/rl/planning/` | `mcts.py` |
| Attention | `nexus/components/attention/` | `flash_attention.py` |
| SSM | `nexus/components/ssm/` | `mamba.py`, `s4.py` |
| CV - Detection | `nexus/models/cv/detection/` | `detr.py`, `yolov10.py` |
| CV - Segmentation | `nexus/models/cv/segmentation/` | `sam.py` |
| CV - NeRF | `nexus/models/cv/nerf/` | `gaussian_splatting.py` |
| Generative - Diffusion | `nexus/models/generative/diffusion/` | `dit.py` |
| NLP - Reasoning | `nexus/models/nlp/reasoning/` | `cot.py`, `react.py` |
| NLP - RAG | `nexus/models/nlp/rag/` | `self_rag.py`, `crag.py` |
| NLP - PEFT | `nexus/models/compression/peft/` | `lora.py`, `qlora.py` |
| NLP - Quantization | `nexus/models/compression/quantization/` | `gptq.py`, `awq.py` |

### Documentation

| Category | Path | Example |
|----------|------|---------|
| RL - Value-based | `docs/01_reinforcement_learning/value_based/` | `dqn.md` |
| RL - Policy Gradient | `docs/01_reinforcement_learning/policy_gradient/` | `sac.md` |
| RL - Offline | `docs/01_reinforcement_learning/offline_rl/` | `iql.md` |
| RL - Alignment | `docs/01_reinforcement_learning/alignment/` | `dpo.md` |
| Attention | `docs/02_attention_mechanisms/` | `flash_attention.md` |
| SSM | `docs/03_state_space_models/` | `mamba.md` |
| Hybrid Architectures | `docs/04_hybrid_architectures/` | `jamba.md` |
| Positional Encodings | `docs/05_positional_encodings/` | `rope.md` |
| CV - Vision Transformers | `docs/08_computer_vision/vision_transformers/` | `vit.md` |
| CV - Object Detection | `docs/08_computer_vision/object_detection/` | `detr.md` |
| CV - Segmentation | `docs/08_computer_vision/segmentation/` | `sam.md` |
| CV - NeRF/3D | `docs/08_computer_vision/nerf_3d/` | `gaussian_splatting.md` |
| Generative - Diffusion | `docs/09_generative_models/diffusion/` | `dit.md` |
| NLP - Reasoning | `docs/10_nlp_llm/reasoning/` | `cot.md` |
| NLP - RAG | `docs/10_nlp_llm/rag/` | `self_rag.md` |
| NLP - PEFT | `docs/10_nlp_llm/peft/` | `lora.md` |

## Template Snippets

### Minimal Module (RL)
```python
from nexus.core.base import NexusModule
import torch

class MyAlgorithm(NexusModule):
    def __init__(self, config: dict):
        super().__init__(config)
        # Initialize components

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass
        pass

    def compute_loss(self, batch: dict) -> torch.Tensor:
        # Compute loss
        pass

    def update(self, batch: dict) -> dict:
        loss = self.compute_loss(batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}
```

### Minimal Component (Attention/SSM)
```python
import torch.nn as nn

class MyComponent(nn.Module):
    def __init__(self, dim: int, **kwargs):
        super().__init__()
        self.dim = dim
        # Initialize layers

    def forward(self, x):
        # Forward pass
        return x
```

### Documentation Sections (All 10 Required)
```markdown
# Algorithm Name

## 1. Overview & Motivation
## 2. Theoretical Background
## 3. Mathematical Formulation
## 4. High-Level Intuition
## 5. Implementation Details
## 6. Code Walkthrough
## 7. Optimization Tricks
## 8. Experiments & Results
## 9. Common Pitfalls
## 10. References
```

## Common Hyperparameters

### RL Algorithms
```python
config = {
    'state_dim': 17,        # Environment state dimension
    'action_dim': 6,        # Action space dimension
    'hidden_dim': 256,      # Hidden layer size
    'learning_rate': 3e-4,  # Learning rate
    'gamma': 0.99,          # Discount factor
    'batch_size': 256,      # Batch size
    'tau': 0.005,           # Target network update rate
}
```

### Attention Mechanisms
```python
config = {
    'dim': 512,             # Model dimension
    'num_heads': 8,         # Number of attention heads
    'dropout': 0.1,         # Dropout probability
    'bias': True,           # Use bias in projections
}
```

### Diffusion Models
```python
config = {
    'image_size': 256,      # Input image size
    'timesteps': 1000,      # Number of diffusion steps
    'beta_schedule': 'linear',  # Noise schedule
    'loss_type': 'l2',      # Loss function
}
```

## Verification Commands

```bash
# Test import
python -c "from nexus.models.{path}.{file} import {ClassName}"

# Run tests
python tests/test_{algorithm}.py
pytest tests/test_{algorithm}.py -v

# Check documentation renders
cat docs/{category}/{algorithm}.md

# Verify RESEARCH_TODO updated
grep "{Algorithm}" RESEARCH_TODO.md
```

## Git Workflow

```bash
# Create feature branch (optional)
git checkout -b add-{algorithm}

# Add files
git add nexus/models/{path}/{algorithm}.py
git add docs/{category}/{algorithm}.md
git add configs/{category}/{algorithm}.yaml
git add RESEARCH_TODO.md

# Commit with clear message
git commit -m "Add {Algorithm} implementation and documentation

- Implement {algorithm} in nexus/models/{path}
- Add comprehensive 10-section documentation
- Include configuration example
- Update RESEARCH_TODO.md"

# Push (if using branch)
git push origin add-{algorithm}
```

## Documentation Quality Checklist

- [ ] All 10 sections present and complete
- [ ] Mathematical formulas use LaTeX (`$$...$$`)
- [ ] At least 3 code examples
- [ ] Implementation path is correct
- [ ] Links to papers work (prefer arxiv)
- [ ] References section has 5+ sources
- [ ] Hyperparameters table filled out
- [ ] 3+ common pitfalls listed
- [ ] Benchmark results included (or noted as unavailable)
- [ ] Cross-references to related algorithms
- [ ] Quick reference section at end
- [ ] Added to category README

## Troubleshooting

### Import Error
```bash
# Check file exists
ls nexus/models/{path}/{algorithm}.py

# Check __init__.py exists
ls nexus/models/{path}/__init__.py

# Verify syntax
python -m py_compile nexus/models/{path}/{algorithm}.py
```

### Documentation Not Rendering
```bash
# Check markdown syntax
mdl docs/{category}/{algorithm}.md  # If you have mdl installed

# View in VS Code with preview
code docs/{category}/{algorithm}.md
```

### Tests Failing
```bash
# Run with verbose output
pytest tests/test_{algorithm}.py -v -s

# Run specific test
pytest tests/test_{algorithm}.py::test_creation -v
```

## Resources

- **Main Documentation:** [docs/README.md](../docs/README.md)
- **RESEARCH_TODO:** [RESEARCH_TODO.md](../RESEARCH_TODO.md)
- **Implementation Examples:** Browse `nexus/models/` directories
- **Documentation Examples:** Browse `docs/` directories

## Tips

1. **Start with examples** - Look at similar algorithms in the same category
2. **Use type hints** - Makes code more maintainable
3. **Write docstrings** - Include paper references with arxiv links
4. **Test imports** - Verify before committing
5. **Follow conventions** - Match style of existing code
6. **Cross-reference** - Link related algorithms in docs
7. **Update README** - Add to category README after creating docs
8. **Commit atomically** - Group related changes together

---

*Last updated: 2026-02-06*
