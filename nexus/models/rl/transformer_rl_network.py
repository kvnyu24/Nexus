from typing import Dict, Any, Optional, List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...core.base import NexusModule

class TransformerRLNetwork(NexusModule):
    """Advanced Transformer-based RL architecture incorporating latest research advances:
    - Vision Transformer (ViT) based state encoding for both vector and image inputs
    - Quantile value prediction with distributional RL
    - Multi-task auxiliary predictions including inverse dynamics, forward dynamics
    - Contrastive representation learning
    - Attention-based policy with scaled dot product attention
    - Prioritized experience replay with importance sampling
    - Noisy Networks for exploration
    - Dueling networks architecture
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Core dimensions
        self.state_dim = config["state_dim"]
        self.action_dim = config["action_dim"]
        self.hidden_dim = config.get("hidden_dim", 512)
        self.num_heads = config.get("num_heads", 8)
        self.num_layers = config.get("num_layers", 4)
        self.dropout = config.get("dropout", 0.1)
        
        # Vision Transformer state encoder
        self.patch_size = config.get("patch_size", 16)
        self.num_patches = (self.state_dim // self.patch_size) ** 2
        self.patch_dim = self.patch_size * self.patch_size * 3  # For RGB images
        
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.hidden_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout,
            batch_first=True,
            norm_first=True  # Pre-LN for better training stability
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(self.patch_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        )
        
        # Scaled dot-product attention policy network
        self.policy_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True
        )
        
        # Dueling network architecture
        self.advantage_stream = nn.Sequential(
            NoisyLinear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            NoisyLinear(self.hidden_dim, self.action_dim)
        )
        
        self.value_stream = nn.Sequential(
            NoisyLinear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            NoisyLinear(self.hidden_dim, 1)
        )
        
        # Quantile value prediction
        self.num_quantiles = config.get("num_quantiles", 32)
        self.quantile_net = nn.Sequential(
            NoisyLinear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            NoisyLinear(self.hidden_dim, self.num_quantiles)
        )
        
        # Auxiliary prediction networks with residual connections
        self.inverse_dynamics = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim)
        )
        
        self.forward_dynamics = nn.Sequential(
            nn.Linear(self.hidden_dim + self.action_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.state_dim)
        )
        
        # Contrastive learning head
        self.contrastive_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 128)  # Projection dimension
        )
        
        # Prioritized experience replay
        bank_size = config.get("bank_size", 50000)
        self.register_buffer("experience_bank", torch.zeros(bank_size, self.hidden_dim))
        self.register_buffer("priorities", torch.ones(bank_size))
        self.register_buffer("bank_ptr", torch.zeros(1, dtype=torch.long))
        self.alpha = config.get("priority_alpha", 0.6)
        self.beta = config.get("priority_beta", 0.4)
        self.priority_epsilon = config.get("priority_epsilon", 1e-6)
        
        # Temperature parameter for contrastive learning
        self.temperature = config.get("temperature", 0.07)
        
    def update_experience_bank(
        self,
        features: torch.Tensor,
        priorities: Optional[torch.Tensor] = None,
        indices: Optional[torch.Tensor] = None
    ) -> None:
        if indices is None:
            batch_size = features.size(0)
            ptr = int(self.bank_ptr.item())
            
            if ptr + batch_size > self.experience_bank.size(0):
                ptr = 0
                
            self.experience_bank[ptr:ptr + batch_size] = features.detach()
            if priorities is not None:
                self.priorities[ptr:ptr + batch_size] = priorities.detach() + self.priority_epsilon
            self.bank_ptr[0] = (ptr + batch_size) % self.experience_bank.size(0)
        else:
            self.experience_bank[indices] = features.detach()
            if priorities is not None:
                self.priorities[indices] = priorities.detach() + self.priority_epsilon
                
    def forward(
        self,
        states: Union[torch.Tensor, Dict[str, torch.Tensor]],
        action_mask: Optional[torch.Tensor] = None,
        prev_states: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        # Handle different input types
        if isinstance(states, dict):
            vector_states = states.get("vector")
            image_states = states.get("image")
            
            if image_states is not None:
                # Process image input through ViT
                B = image_states.shape[0]
                patches = image_states.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
                patches = patches.contiguous().view(B, -1, self.patch_dim)
                
                # Add class token and positional embeddings
                cls_tokens = self.cls_token.expand(B, -1, -1)
                patches = torch.cat([cls_tokens, patches], dim=1)
                patches = patches + self.pos_embedding
                
                state_features = self.state_encoder(patches)
                state_features = state_features[:, 0]  # Use CLS token features
            else:
                state_features = self.state_encoder(vector_states)
        else:
            state_features = self.state_encoder(states)
            
        # Scaled dot-product attention policy
        attn_out, attn_weights = self.policy_attention(
            state_features,
            state_features,
            state_features
        )
        
        # Dueling network
        advantage = self.advantage_stream(attn_out)
        value = self.value_stream(attn_out)
        action_logits = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        
        if action_mask is not None:
            action_logits = action_logits.masked_fill(~action_mask, float('-inf'))
            
        # Quantile value prediction
        quantiles = self.quantile_net(state_features)
        value = quantiles.mean(dim=-1, keepdim=True)
        
        outputs = {
            "action_logits": action_logits,
            "value": value,
            "quantiles": quantiles,
            "state_features": state_features
        }
        
        if return_attention:
            outputs["attention_weights"] = attn_weights
            
        # Contrastive learning
        if self.training:
            proj_features = self.contrastive_head(state_features)
            proj_features = F.normalize(proj_features, dim=-1)
            outputs["proj_features"] = proj_features
            
        # Auxiliary predictions
        if prev_states is not None and actions is not None:
            prev_features = self.state_encoder(prev_states)
            
            # Inverse dynamics with residual connection
            combined_features = torch.cat([prev_features, state_features], dim=-1)
            pred_actions = self.inverse_dynamics(combined_features)
            
            # Forward dynamics with residual connection
            action_embeddings = F.one_hot(actions, self.action_dim).float()
            dynamics_input = torch.cat([prev_features, action_embeddings], dim=-1)
            pred_states = self.forward_dynamics(dynamics_input)
            pred_states = pred_states + prev_states  # Residual connection
            
            outputs.update({
                "pred_actions": pred_actions,
                "pred_states": pred_states
            })
            
        return outputs

class NoisyLinear(nn.Module):
    """Noisy Linear layer for exploration"""
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
        
    def reset_parameters(self):
        mu_range = 1 / self.in_features ** 0.5
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / self.in_features ** 0.5)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / self.out_features ** 0.5)
        
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
        
    def _scale_noise(self, size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return F.linear(
                x,
                self.weight_mu + self.weight_sigma * self.weight_epsilon,
                self.bias_mu + self.bias_sigma * self.bias_epsilon
            )
        else:
            return F.linear(x, self.weight_mu, self.bias_mu)