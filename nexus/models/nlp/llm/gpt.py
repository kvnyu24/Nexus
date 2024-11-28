from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_llm import BaseLLM, BaseLLMConfig
from nexus.core.base import NexusModule

class GPTConfig(BaseLLMConfig):
    """Configuration class for GPT model"""
    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        intermediate_size: Optional[int] = None,
        max_seq_length: int = 1024,
        layer_norm_epsilon: float = 1e-5,
        activation_function: str = "gelu",
        resid_dropout: float = 0.1,
        embed_dropout: float = 0.1,
        attn_dropout: float = 0.1,
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
        self.layer_norm_epsilon = layer_norm_epsilon
        self.activation_function = activation_function
        self.resid_dropout = resid_dropout
        self.embed_dropout = embed_dropout
        self.attn_dropout = attn_dropout

class GPTAttention(NexusModule):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_heads
        
        self.c_attn = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.c_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.resid_dropout = nn.Dropout(config.resid_dropout)
        
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

class GPTBlock(NexusModule):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPTAttention(config)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU() if config.activation_function == "gelu" else nn.ReLU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.resid_dropout)
        )

class GPTModel(BaseLLM):
    def __init__(self, config: Dict[str, Any]):
        # Convert dict config to GPTConfig if needed
        if not isinstance(config, GPTConfig):
            config = GPTConfig(**config)
        super().__init__(config)
        
        # GPT-specific dropout
        self.drop = nn.Dropout(config.embed_dropout)
        
        # Override transformer layers with GPT-specific blocks
        self.layers = nn.ModuleList([
            GPTBlock(config) for _ in range(config.num_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        
        # Initialize weights using GPT-specific method
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def _prepare_causal_attention_mask(
        self,
        batch_size: int,
        seq_length: int,
        dtype: torch.dtype,
        device: torch.device
    ) -> torch.Tensor:
        mask = torch.triu(
            torch.ones((seq_length, seq_length), dtype=dtype, device=device) * -float("inf"),
            diagonal=1
        )
        return mask.unsqueeze(0).expand(batch_size, -1, -1) 