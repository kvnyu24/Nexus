"""
Process Reward Model (PRM) for Step-by-Step Verification
Paper: "Let's Verify Step by Step" (OpenAI, 2023)

PRM provides step-level verification for reasoning tasks:
- Assigns rewards to each step in a reasoning process
- Enables fine-grained credit assignment for multi-step problems
- Trained on human feedback at the step level
- More effective than outcome-only rewards for complex reasoning
- Used in OpenAI's mathematical reasoning systems

Key features:
- Step-level reward prediction
- Better credit assignment than outcome reward models (ORM)
- Enables verification during generation
- Can be used for best-of-N sampling or beam search
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List
from ....core.base import NexusModule


class ProcessRewardModel(NexusModule):
    """
    Process Reward Model for step-level verification.

    Predicts a reward/correctness score for each step in a multi-step
    reasoning process. Can be used with language models for mathematical
    reasoning, coding, or other sequential decision-making tasks.

    Args:
        config: Configuration dictionary with:
            - input_dim: Dimension of step embeddings (default: 768)
            - hidden_dim: Hidden layer size (default: 512)
            - n_layers: Number of transformer layers (default: 4)
            - n_heads: Number of attention heads (default: 8)
            - dropout: Dropout rate (default: 0.1)
            - max_steps: Maximum number of steps (default: 512)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.input_dim = config.get("input_dim", 768)
        self.hidden_dim = config.get("hidden_dim", 512)
        self.n_layers = config.get("n_layers", 4)
        self.n_heads = config.get("n_heads", 8)
        self.dropout = config.get("dropout", 0.1)
        self.max_steps = config.get("max_steps", 512)

        # Input projection
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)

        # Positional encoding for steps
        self.step_embedding = nn.Embedding(self.max_steps, self.hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.n_heads,
            dim_feedforward=4 * self.hidden_dim,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)

        # Reward prediction head
        self.reward_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(
        self,
        step_embeddings: torch.Tensor,
        step_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict rewards for each step.

        Args:
            step_embeddings: Embeddings of reasoning steps [batch, n_steps, input_dim]
            step_mask: Mask for valid steps [batch, n_steps] (True = valid)

        Returns:
            Step-level rewards [batch, n_steps]
        """
        batch_size, n_steps, _ = step_embeddings.shape

        # Project inputs
        x = self.input_projection(step_embeddings)

        # Add positional encodings
        positions = torch.arange(n_steps, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.step_embedding(positions)

        # Transformer encoding
        if step_mask is not None:
            # Create attention mask (True = attend, False = ignore)
            x = self.transformer(x, src_key_padding_mask=~step_mask)
        else:
            x = self.transformer(x)

        # Predict rewards
        rewards = self.reward_head(x).squeeze(-1)  # [batch, n_steps]

        # Mask invalid steps
        if step_mask is not None:
            rewards = rewards.masked_fill(~step_mask, 0.0)

        return rewards

    def compute_loss(
        self,
        step_embeddings: torch.Tensor,
        target_rewards: torch.Tensor,
        step_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute training loss.

        Args:
            step_embeddings: Step embeddings [batch, n_steps, input_dim]
            target_rewards: Target rewards/labels [batch, n_steps]
            step_mask: Mask for valid steps [batch, n_steps]

        Returns:
            Loss (scalar)
        """
        predicted_rewards = self.forward(step_embeddings, step_mask)

        # Binary cross-entropy for correctness labels
        loss = F.binary_cross_entropy_with_logits(
            predicted_rewards,
            target_rewards,
            reduction='none'
        )

        # Apply mask
        if step_mask is not None:
            loss = loss * step_mask.float()
            loss = loss.sum() / step_mask.float().sum().clamp(min=1.0)
        else:
            loss = loss.mean()

        return loss

    def score_trajectory(
        self,
        step_embeddings: torch.Tensor,
        step_mask: Optional[torch.Tensor] = None,
        aggregation: str = 'mean',
    ) -> torch.Tensor:
        """
        Score an entire reasoning trajectory.

        Args:
            step_embeddings: Step embeddings [batch, n_steps, input_dim]
            step_mask: Mask for valid steps [batch, n_steps]
            aggregation: How to aggregate step rewards ('mean', 'sum', 'min', 'product')

        Returns:
            Trajectory scores [batch]
        """
        step_rewards = self.forward(step_embeddings, step_mask)

        if step_mask is not None:
            # Only consider valid steps
            masked_rewards = step_rewards.masked_fill(~step_mask, 0.0)
            n_valid_steps = step_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        else:
            masked_rewards = step_rewards
            n_valid_steps = torch.tensor(step_rewards.size(1), device=step_rewards.device)

        if aggregation == 'mean':
            scores = masked_rewards.sum(dim=1) / n_valid_steps.squeeze(1)
        elif aggregation == 'sum':
            scores = masked_rewards.sum(dim=1)
        elif aggregation == 'min':
            scores = masked_rewards.min(dim=1)[0]
        elif aggregation == 'product':
            # Product of probabilities (assuming rewards are logits)
            probs = torch.sigmoid(step_rewards)
            if step_mask is not None:
                probs = probs.masked_fill(~step_mask, 1.0)
            scores = probs.prod(dim=1)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

        return scores


class OutcomeRewardModel(NexusModule):
    """
    Outcome Reward Model (ORM) for final-answer verification.

    Simpler than PRM, only evaluates the final outcome without
    considering intermediate steps.

    Args:
        config: Configuration dictionary with:
            - input_dim: Dimension of input embeddings (default: 768)
            - hidden_dim: Hidden layer size (default: 512)
            - dropout: Dropout rate (default: 0.1)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.input_dim = config.get("input_dim", 768)
        self.hidden_dim = config.get("hidden_dim", 512)
        self.dropout = config.get("dropout", 0.1)

        # Simple feedforward network
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, output_embedding: torch.Tensor) -> torch.Tensor:
        """
        Predict reward for final outcome.

        Args:
            output_embedding: Final output embedding [batch, input_dim]

        Returns:
            Outcome reward [batch]
        """
        return self.network(output_embedding).squeeze(-1)

    def compute_loss(
        self,
        output_embedding: torch.Tensor,
        target_reward: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute training loss.

        Args:
            output_embedding: Output embedding [batch, input_dim]
            target_reward: Target reward/label [batch]

        Returns:
            Loss (scalar)
        """
        predicted_reward = self.forward(output_embedding)
        loss = F.binary_cross_entropy_with_logits(predicted_reward, target_reward)
        return loss
