from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from .base_llm import BaseLLM, BaseLLMConfig
from .llama import BaseLLMBlock
from nexus.core.base import NexusModule
from ....core.initialization import WeightInitMixin


class BloomLayerNorm(NexusModule):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        variance = (x - mean).pow(2).mean(-1, keepdim=True)
        x = (x - mean) / torch.sqrt(variance + self.eps)
        return self.weight * x + self.bias


class BloomConfig(BaseLLMConfig):
    """Configuration class for BLOOM model"""
    def __init__(
        self,
        vocab_size: int = 250880,
        hidden_size: int = 1024,
        num_layers: int = 24,
        num_heads: int = 16,
        use_alt_layernorm: bool = True,
        apply_residual_connection_post_layernorm: bool = False,
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            **kwargs
        )
        self.use_alt_layernorm = use_alt_layernorm
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm

class BloomAttention(NexusModule):
    def __init__(self, config: BloomConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_heads
        
        self.query_key_value = nn.Linear(
            config.hidden_size, 
            3 * config.hidden_size,
            bias=True
        )
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_dropout = nn.Dropout(config.dropout)

class BloomBlock(BaseLLMBlock):
    def __init__(self, config: BloomConfig):
        super().__init__(config)
        self.attention = BloomAttention(config)
        
        # BLOOM uses a different layernorm implementation
        LayerNormClass = nn.LayerNorm if not config.use_alt_layernorm else BloomLayerNorm
        self.input_layernorm = LayerNormClass(config.hidden_size)
        self.post_attention_layernorm = LayerNormClass(config.hidden_size)
        
        self.apply_residual_connection_post_layernorm = (
            config.apply_residual_connection_post_layernorm
        )

class BloomModel(WeightInitMixin, BaseLLM):
    def __init__(self, config: Dict[str, Any]):
        if not isinstance(config, BloomConfig):
            config = BloomConfig(**config)
        super().__init__(config)

        # Override transformer layers with BLOOM-specific blocks
        self.layers = nn.ModuleList([
            BloomBlock(config) for _ in range(config.num_layers)
        ])

        # Initialize weights using LLM preset
        self.init_weights_llm() 