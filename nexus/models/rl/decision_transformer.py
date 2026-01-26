"""
Decision Transformer
Paper: "Decision Transformer: Reinforcement Learning via Sequence Modeling" (Chen et al., 2021)

Decision Transformer frames RL as sequence modeling by:
- Conditioning on returns-to-go (desired future returns)
- Using a GPT-style transformer to model (R, s, a) sequences
- Enabling offline RL without Bellman backup or policy gradients
- Achieving strong performance on offline RL benchmarks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, Union
from ...core.base import NexusModule
import numpy as np
import math


class CausalSelfAttention(nn.Module):
    """Causal self-attention for autoregressive modeling."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        max_seq_len: int = 1024
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    """Transformer block with causal attention."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        max_seq_len: int = 1024
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = CausalSelfAttention(hidden_dim, num_heads, dropout, max_seq_len)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class DecisionTransformer(NexusModule):
    """
    Decision Transformer for offline reinforcement learning.

    The model takes as input a sequence of (return-to-go, state, action) tuples
    and predicts the action for each timestep.

    Args:
        config: Configuration dictionary with:
            - state_dim: Dimension of state space
            - action_dim: Dimension of action space
            - hidden_dim: Transformer hidden dimension (default: 128)
            - num_layers: Number of transformer layers (default: 3)
            - num_heads: Number of attention heads (default: 1)
            - max_ep_len: Maximum episode length (default: 1000)
            - max_seq_len: Context length for transformer (default: 20)
            - dropout: Dropout rate (default: 0.1)
            - action_tanh: Whether to apply tanh to actions (default: True)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.state_dim = config["state_dim"]
        self.action_dim = config["action_dim"]
        self.hidden_dim = config.get("hidden_dim", 128)
        self.max_ep_len = config.get("max_ep_len", 1000)
        self.max_seq_len = config.get("max_seq_len", 20)
        self.action_tanh = config.get("action_tanh", True)

        num_layers = config.get("num_layers", 3)
        num_heads = config.get("num_heads", 1)
        dropout = config.get("dropout", 0.1)

        # Embedding layers for different modalities
        self.embed_timestep = nn.Embedding(self.max_ep_len, self.hidden_dim)
        self.embed_return = nn.Linear(1, self.hidden_dim)
        self.embed_state = nn.Linear(self.state_dim, self.hidden_dim)
        self.embed_action = nn.Linear(self.action_dim, self.hidden_dim)

        self.embed_ln = nn.LayerNorm(self.hidden_dim)

        # Transformer
        # Each timestep has 3 tokens: return, state, action
        self.transformer = nn.ModuleList([
            TransformerBlock(
                self.hidden_dim,
                num_heads,
                dropout,
                max_seq_len=3 * self.max_seq_len
            )
            for _ in range(num_layers)
        ])

        # Prediction heads
        self.predict_state = nn.Linear(self.hidden_dim, self.state_dim)
        self.predict_action = nn.Sequential(
            nn.Linear(self.hidden_dim, self.action_dim),
            nn.Tanh() if self.action_tanh else nn.Identity()
        )
        self.predict_return = nn.Linear(self.hidden_dim, 1)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the Decision Transformer.

        Args:
            states: (batch, seq_len, state_dim)
            actions: (batch, seq_len, action_dim)
            returns_to_go: (batch, seq_len, 1)
            timesteps: (batch, seq_len)
            attention_mask: Optional mask (batch, seq_len)

        Returns:
            state_preds: Predicted states
            action_preds: Predicted actions
            return_preds: Predicted returns
        """
        batch_size, seq_len = states.shape[0], states.shape[1]

        # Ensure returns_to_go has correct shape
        if returns_to_go.dim() == 2:
            returns_to_go = returns_to_go.unsqueeze(-1)

        # Embed each modality
        time_embeddings = self.embed_timestep(timesteps)

        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        return_embeddings = self.embed_return(returns_to_go) + time_embeddings

        # Stack tokens: (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # Shape: (batch, 3 * seq_len, hidden_dim)
        stacked_inputs = torch.stack(
            [return_embeddings, state_embeddings, action_embeddings], dim=2
        ).reshape(batch_size, 3 * seq_len, self.hidden_dim)

        stacked_inputs = self.embed_ln(stacked_inputs)

        # Process through transformer
        x = stacked_inputs
        for block in self.transformer:
            x = block(x)

        # Reshape back: separate predictions for each modality
        x = x.reshape(batch_size, seq_len, 3, self.hidden_dim)

        # Get predictions from appropriate positions
        # return prediction comes from state token (index 1)
        # state prediction comes from action token (index 2) - not typically used
        # action prediction comes from return token (index 0) and state token (index 1)
        return_preds = self.predict_return(x[:, :, 1])  # predict return from state
        state_preds = self.predict_state(x[:, :, 2])     # predict next state from action
        action_preds = self.predict_action(x[:, :, 1])   # predict action from state

        return state_preds, action_preds, return_preds

    def get_action(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """
        Get action prediction for the last timestep (for evaluation).

        Args:
            states: (batch, seq_len, state_dim)
            actions: (batch, seq_len, action_dim)
            returns_to_go: (batch, seq_len, 1)
            timesteps: (batch, seq_len)

        Returns:
            action: Predicted action for the last timestep
        """
        _, action_preds, _ = self.forward(states, actions, returns_to_go, timesteps)
        return action_preds[:, -1]


class DecisionTransformerAgent(NexusModule):
    """
    Agent wrapper for Decision Transformer with training utilities.

    Args:
        config: Same as DecisionTransformer, plus:
            - learning_rate: Learning rate (default: 1e-4)
            - weight_decay: Weight decay (default: 1e-4)
            - warmup_steps: LR warmup steps (default: 1000)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.model = DecisionTransformer(config)
        self.max_seq_len = config.get("max_seq_len", 20)
        self.state_dim = config["state_dim"]
        self.action_dim = config["action_dim"]

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get("learning_rate", 1e-4),
            weight_decay=config.get("weight_decay", 1e-4)
        )

        # Learning rate scheduler
        self.warmup_steps = config.get("warmup_steps", 1000)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda step: min(1.0, step / self.warmup_steps)
        )

        # State for autoregressive generation
        self.reset_history()

    def reset_history(self):
        """Reset the history buffer for a new episode."""
        self.state_history = []
        self.action_history = []
        self.return_history = []
        self.timestep_history = []

    def select_action(
        self,
        state: Union[np.ndarray, torch.Tensor],
        target_return: float,
        timestep: int
    ) -> np.ndarray:
        """
        Select action given current state and target return.

        Args:
            state: Current state
            target_return: Desired return-to-go
            timestep: Current timestep in episode

        Returns:
            action: Selected action
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Add to history
        self.state_history.append(state)
        self.return_history.append(torch.tensor([[target_return]]))

        if len(self.action_history) == 0:
            # First step: use zero action
            self.action_history.append(torch.zeros(1, self.action_dim))
        self.timestep_history.append(torch.tensor([timestep]))

        # Prepare context (last max_seq_len steps)
        context_len = min(len(self.state_history), self.max_seq_len)

        states = torch.cat(self.state_history[-context_len:], dim=0).unsqueeze(0)
        actions = torch.cat(self.action_history[-context_len:], dim=0).unsqueeze(0)
        returns = torch.cat(self.return_history[-context_len:], dim=0).unsqueeze(0)
        timesteps = torch.cat(self.timestep_history[-context_len:], dim=0).unsqueeze(0)

        # Pad if needed
        if states.shape[1] < self.max_seq_len:
            pad_len = self.max_seq_len - states.shape[1]
            states = F.pad(states, (0, 0, pad_len, 0))
            actions = F.pad(actions, (0, 0, pad_len, 0))
            returns = F.pad(returns, (0, 0, pad_len, 0))
            timesteps = F.pad(timesteps, (pad_len, 0))

        # Get action
        with torch.no_grad():
            action = self.model.get_action(states, actions, returns, timesteps)

        action_np = action[0].cpu().numpy()

        # Add predicted action to history
        self.action_history.append(torch.FloatTensor(action_np).unsqueeze(0))

        return action_np

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Update the model on a batch of trajectories.

        Args:
            batch: Dictionary containing:
                - states: (batch, seq_len, state_dim)
                - actions: (batch, seq_len, action_dim)
                - returns_to_go: (batch, seq_len, 1)
                - timesteps: (batch, seq_len)
                - attention_mask: (batch, seq_len)

        Returns:
            Dictionary with loss
        """
        states = batch["states"]
        actions = batch["actions"]
        returns_to_go = batch["returns_to_go"]
        timesteps = batch["timesteps"]
        attention_mask = batch.get("attention_mask")

        # Forward pass
        _, action_preds, _ = self.model(
            states, actions, returns_to_go, timesteps, attention_mask
        )

        # Compute loss (only on valid positions)
        action_target = actions.detach()

        if attention_mask is not None:
            loss = F.mse_loss(action_preds, action_target, reduction='none')
            loss = (loss * attention_mask.unsqueeze(-1)).sum() / attention_mask.sum()
        else:
            loss = F.mse_loss(action_preds, action_target)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()
        self.scheduler.step()

        return {
            "loss": loss.item(),
            "lr": self.scheduler.get_last_lr()[0]
        }
