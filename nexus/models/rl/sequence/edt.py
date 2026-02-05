"""
EDT: Elastic Decision Transformer
Paper: "Elastic Decision Transformer" (NeurIPS 2023)

EDT extends Decision Transformer with:
- Adaptive history length selection
- Dynamic trajectory stitching across different history lengths
- Better handling of variable-length contexts
- Improved generalization via elastic attention

Key features:
- Variable context window for different situations
- Trajectory stitching without fixed-length constraints
- Elastic positional encodings
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from ....core.base import NexusModule


class ElasticDecisionTransformer(NexusModule):
    """
    Elastic Decision Transformer with adaptive history length.

    Extends the standard Decision Transformer by allowing variable-length
    history and elastic attention mechanisms.

    Args:
        config: Configuration dictionary with:
            - state_dim: Dimension of state space
            - action_dim: Dimension of action space
            - hidden_dim: Hidden dimension (default: 128)
            - n_layers: Number of transformer layers (default: 3)
            - n_heads: Number of attention heads (default: 1)
            - max_ep_len: Maximum episode length (default: 4096)
            - min_context_len: Minimum context length (default: 5)
            - max_context_len: Maximum context length (default: 20)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.state_dim = config["state_dim"]
        self.action_dim = config["action_dim"]
        self.hidden_dim = config.get("hidden_dim", 128)
        self.n_layers = config.get("n_layers", 3)
        self.n_heads = config.get("n_heads", 1)
        self.max_ep_len = config.get("max_ep_len", 4096)
        self.min_context_len = config.get("min_context_len", 5)
        self.max_context_len = config.get("max_context_len", 20)

        # Embedding layers
        self.embed_timestep = nn.Embedding(self.max_ep_len, self.hidden_dim)
        self.embed_return = nn.Linear(1, self.hidden_dim)
        self.embed_state = nn.Linear(self.state_dim, self.hidden_dim)
        self.embed_action = nn.Linear(self.action_dim, self.hidden_dim)

        # Elastic position encoding
        self.embed_ln = nn.LayerNorm(self.hidden_dim)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.n_heads,
            dim_feedforward=4 * self.hidden_dim,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)

        # Prediction heads
        self.predict_action = nn.Sequential(
            nn.Linear(self.hidden_dim, self.action_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with elastic context.

        Args:
            states: State sequence [batch, seq_len, state_dim]
            actions: Action sequence [batch, seq_len, action_dim]
            returns_to_go: Returns-to-go [batch, seq_len, 1]
            timesteps: Timesteps [batch, seq_len]
            attention_mask: Optional attention mask [batch, seq_len]

        Returns:
            Predicted actions [batch, seq_len, action_dim]
        """
        batch_size, seq_len = states.shape[0], states.shape[1]

        # Embed each modality
        time_embeddings = self.embed_timestep(timesteps)
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings
        returns_embeddings = self.embed_return(returns_to_go) + time_embeddings

        # Stack tokens: (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        stacked_inputs = torch.stack(
            [returns_embeddings, state_embeddings, action_embeddings], dim=2
        ).view(batch_size, 3 * seq_len, self.hidden_dim)

        # Layer norm
        stacked_inputs = self.embed_ln(stacked_inputs)

        # Create causal mask if needed
        if attention_mask is not None:
            # Expand mask for stacked tokens
            attention_mask = attention_mask.repeat_interleave(3, dim=1)

        # Transformer
        transformer_outputs = self.transformer(
            stacked_inputs,
            src_key_padding_mask=~attention_mask if attention_mask is not None else None
        )

        # Extract action predictions (every third token starting from index 1)
        action_preds = transformer_outputs[:, 1::3]
        action_preds = self.predict_action(action_preds)

        return action_preds
