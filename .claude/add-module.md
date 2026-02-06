# Add Module Skill

## Description
Guided workflow for adding a new AI algorithm implementation to the Nexus repository. This skill helps you scaffold a complete module with proper structure, base classes, configuration, and integration.

## When to Use
- Adding a new RL algorithm (value-based, policy gradient, offline, alignment, etc.)
- Adding a new attention mechanism variant
- Adding a new SSM architecture
- Adding a new CV model (detection, segmentation, NeRF, etc.)
- Adding a new generative model
- Adding a new NLP/LLM method (RAG, PEFT, reasoning, etc.)

## Workflow

### Step 1: Gather Information
Ask the user for:
1. **Algorithm name** (e.g., "TD3", "FlashAttention-3", "Grounding DINO")
2. **Category** (RL/Attention/SSM/CV/Generative/NLP/etc.)
3. **Subcategory** (if applicable, e.g., "policy_gradient", "object_detection")
4. **Key papers** (arxiv links or titles)
5. **Key features** (what makes it unique/special)

### Step 2: Determine File Structure
Based on category, determine the correct location:

**Reinforcement Learning:**
- Value-based: `nexus/models/rl/dqn/` or `nexus/models/rl/value_based/`
- Policy gradient: `nexus/models/rl/policy_gradient/`
- Offline RL: `nexus/models/rl/offline/`
- Alignment: `nexus/models/rl/alignment/`
- Multi-agent: `nexus/models/rl/multi_agent/`
- Model-based: `nexus/models/rl/model_based/`
- Exploration: `nexus/models/rl/exploration/`
- Sequence-based: `nexus/models/rl/sequence/`
- Reward modeling: `nexus/models/rl/reward_models/`
- Planning: `nexus/models/rl/planning/`

**Attention Mechanisms:**
- `nexus/components/attention/`

**State Space Models:**
- `nexus/components/ssm/`

**Computer Vision:**
- Vision Transformers: `nexus/models/cv/`
- Object Detection: `nexus/models/cv/detection/`
- Segmentation: `nexus/models/cv/segmentation/`
- NeRF/3D: `nexus/models/cv/nerf/`

**Generative Models:**
- Diffusion: `nexus/models/generative/diffusion/`
- Audio/Video: `nexus/models/generative/audio_video/`
- GANs: `nexus/models/generative/gan/`
- VAE: `nexus/models/generative/vae/`

**NLP/LLM:**
- Reasoning: `nexus/models/nlp/reasoning/`
- RAG: `nexus/models/nlp/rag/`
- PEFT: `nexus/models/compression/peft/`
- Quantization: `nexus/models/compression/quantization/`
- Structured Generation: `nexus/models/nlp/structured/`

### Step 3: Create Module Structure

**For RL algorithms, create:**
```python
# nexus/models/rl/{category}/{algorithm_name}.py

from nexus.core.base import NexusModule
import torch
import torch.nn as nn

class {AlgorithmName}(NexusModule):
    """
    {Algorithm Full Name}

    Paper: {paper_title} ({year})
    Link: {arxiv_link}

    Key Features:
    - {feature 1}
    - {feature 2}
    - {feature 3}
    """

    def __init__(self, config: dict):
        super().__init__(config)

        # Extract key hyperparameters
        self.state_dim = config.get('state_dim')
        self.action_dim = config.get('action_dim')
        self.hidden_dim = config.get('hidden_dim', 256)
        self.learning_rate = config.get('learning_rate', 3e-4)

        # Initialize networks
        self._build_networks()

    def _build_networks(self):
        """Build neural network components."""
        # TODO: Implement network architecture
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # TODO: Implement forward pass
        pass

    def compute_loss(self, batch: dict) -> torch.Tensor:
        """Compute training loss."""
        # TODO: Implement loss computation
        pass

    def update(self, batch: dict) -> dict:
        """Perform one training step."""
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
        }, path)

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
```

**For Attention/SSM components, create:**
```python
# nexus/components/{attention|ssm}/{algorithm_name}.py

import torch
import torch.nn as nn

class {AlgorithmName}(nn.Module):
    """
    {Algorithm Full Name}

    Paper: {paper_title} ({year})
    Link: {arxiv_link}

    Key Features:
    - {feature 1}
    - {feature 2}
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        # TODO: Initialize components

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch, seq_len, dim]
            mask: Optional attention mask
        Returns:
            Output tensor [batch, seq_len, dim]
        """
        # TODO: Implement forward pass
        pass
```

**For CV models, create:**
```python
# nexus/models/cv/{category}/{algorithm_name}.py

from nexus.core.base import NexusModule
import torch
import torch.nn as nn

class {AlgorithmName}(NexusModule):
    """
    {Algorithm Full Name}

    Paper: {paper_title} ({year})
    Link: {arxiv_link}
    """

    def __init__(self, config: dict):
        super().__init__(config)

        self.num_classes = config.get('num_classes', 80)
        self.input_size = config.get('input_size', 640)

        # Build backbone, neck, head
        self._build_model()

    def _build_model(self):
        """Build model components."""
        # TODO: Implement architecture
        pass

    def forward(self, images: torch.Tensor) -> dict:
        """
        Args:
            images: [batch, 3, H, W]
        Returns:
            Dictionary with predictions
        """
        # TODO: Implement forward pass
        pass
```

### Step 4: Add Configuration Example

Create a config file in `configs/{category}/{algorithm_name}.yaml`:

```yaml
# Model configuration for {AlgorithmName}
model:
  name: "{algorithm_name}"
  type: "{category}"

  # Architecture
  hidden_dim: 256
  num_layers: 3

  # Training
  learning_rate: 3e-4
  batch_size: 256

  # Algorithm-specific
  # TODO: Add specific hyperparameters

# Training configuration
training:
  max_steps: 1000000
  eval_interval: 10000
  save_interval: 50000

# Environment configuration (for RL)
env:
  name: "HalfCheetah-v4"
  num_envs: 1
```

### Step 5: Update RESEARCH_TODO.md

Add the algorithm to the appropriate section in `/Users/kevinyu/Projects/Nexus/RESEARCH_TODO.md`:

```markdown
- [EXISTS] {AlgorithmName} — {brief description} ({venue} {year})
```

### Step 6: Verify Integration

1. **Check imports work:**
   ```bash
   cd /Users/kevinyu/Projects/Nexus
   python -c "from nexus.models.{path}.{algorithm_name} import {AlgorithmName}; print('Import successful!')"
   ```

2. **Test instantiation:**
   ```python
   config = {
       'state_dim': 17,
       'action_dim': 6,
       'hidden_dim': 256,
   }
   model = {AlgorithmName}(config)
   print(f"Model created: {model}")
   ```

3. **Verify it extends NexusModule:**
   ```python
   from nexus.core.base import NexusModule
   assert isinstance(model, NexusModule)
   ```

### Step 7: Create Placeholder Tests

Create `tests/test_{algorithm_name}.py`:

```python
import pytest
import torch
from nexus.models.{path}.{algorithm_name} import {AlgorithmName}

def test_{algorithm_name}_creation():
    """Test model instantiation."""
    config = {
        'state_dim': 17,
        'action_dim': 6,
        'hidden_dim': 256,
    }
    model = {AlgorithmName}(config)
    assert model is not None

def test_{algorithm_name}_forward():
    """Test forward pass."""
    config = {
        'state_dim': 17,
        'action_dim': 6,
        'hidden_dim': 256,
    }
    model = {AlgorithmName}(config)

    x = torch.randn(32, config['state_dim'])
    output = model(x)

    assert output.shape[0] == 32
    assert output.shape[1] == config['action_dim']

# TODO: Add more comprehensive tests
```

## Important Guidelines

1. **Always extend NexusModule** for models that need training
2. **Use config dict pattern** for all hyperparameters
3. **Follow existing code style** in similar modules
4. **Add type hints** for all function arguments
5. **Write clear docstrings** with paper references
6. **Include arxiv links** in docstrings
7. **Implement save/load methods** for persistence
8. **Add proper error handling** for edge cases
9. **Use existing components** when possible (e.g., MLP, CNN blocks)
10. **Test imports** before considering complete

## After Module Creation

Once the module is created, use the `add-docs` skill to create comprehensive documentation following the 10-section template.

## Example Usage

```
User: "I want to add SAC (Soft Actor-Critic) to the repository"