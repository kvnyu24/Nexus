from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_llm import BaseLLM, BaseLLMConfig
import math

class QwenConfig(BaseLLMConfig):
    """Configuration class for Qwen model"""
    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 4096,
        num_layers: int = 32,
        num_heads: int = 32,
        intermediate_size: Optional[int] = None,
        max_seq_length: int = 32768,
        rope_theta: float = 10000.0,
        use_dynamic_ntk: bool = True,
        use_logn_attn: bool = True,
        use_flash_attn: bool = True,
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            max_seq_length=max_seq_length,
            **kwargs
        )
        self.rope_theta = rope_theta
        self.use_dynamic_ntk = use_dynamic_ntk
        self.use_logn_attn = use_logn_attn
        self.use_flash_attn = use_flash_attn

class QwenRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_length: int = 32768, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_length = max_seq_length
        self.theta = theta
        
        # Initialize rotary embeddings
        position = torch.arange(max_seq_length).unsqueeze(1)
        freqs = torch.exp(
            -torch.arange(0, dim, 2) * (math.log(theta) / dim)
        ).unsqueeze(0)
        emb = position * freqs
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
        
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len]
        )

class QwenAttention(nn.Module):
    def __init__(self, config: QwenConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_heads
        
        # Multi-Query Attention projections
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        
        # Rotary embeddings
        self.rotary_emb = QwenRotaryEmbedding(
            self.head_dim,
            max_seq_length=config.max_seq_length,
            theta=config.rope_theta
        )
        
        # LogN attention scaling
        self.logn_scaling = config.use_logn_attn
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        batch_size, seq_length = hidden_states.shape[:2]
        
        # Project queries, keys, values
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Apply rotary embeddings
        cos, sin = self.rotary_emb(value_states, seq_length)
        query_states, key_states = self._apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )
        
        # Reshape and scale
        query_states = query_states.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention with optional LogN scaling
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1))
        if self.logn_scaling:
            attn_weights = attn_weights / math.log(seq_length)
        
        # Apply attention mask and softmax
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # Compute output
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        outputs = (attn_output,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += ((key_states, value_states),)
            
        return outputs

class QwenBlock(nn.Module):
    def __init__(self, config: QwenConfig):
        super().__init__()
        self.attention = QwenAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.SiLU(),
            nn.Linear(config.intermediate_size, config.hidden_size)
        )
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

class QwenModel(BaseLLM):
    def __init__(self, config: Dict[str, Any]):
        if not isinstance(config, QwenConfig):
            config = QwenConfig(**config)
        super().__init__(config)
        
        self.layers = nn.ModuleList([
            QwenBlock(config) for _ in range(config.num_layers)
        ])
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
