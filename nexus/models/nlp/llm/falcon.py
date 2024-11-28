from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_llm import BaseLLM, BaseLLMConfig
from .llama import BaseLLMBlock
from nexus.core.base import NexusModule

class FalconConfig(BaseLLMConfig):
    """Configuration class for Falcon model"""
    def __init__(
        self,
        vocab_size: int = 65024,
        hidden_size: int = 2048,
        num_layers: int = 32,
        num_heads: int = 32,
        parallel_attn: bool = True,
        alibi: bool = True,
        multi_query: bool = True,
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            **kwargs
        )
        self.parallel_attn = parallel_attn
        self.alibi = alibi
        self.multi_query = multi_query

class FalconAttention(NexusModule):
    def __init__(self, config: FalconConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_heads
        self.multi_query = config.multi_query
        
        # Multi-query attention uses a single key/value head
        kv_heads = 1 if self.multi_query else self.num_heads
        
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.head_dim * kv_heads, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.head_dim * kv_heads, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

class FalconBlock(BaseLLMBlock):
    def __init__(self, config: FalconConfig):
        super().__init__(config)
        self.attention = FalconAttention(config)
        self.parallel_attn = config.parallel_attn
        
        if self.parallel_attn:
            # Single layer norm for parallel attention
            self.norm = nn.LayerNorm(config.hidden_size)
        else:
            # Two layer norms for sequential attention
            self.norm_1 = nn.LayerNorm(config.hidden_size)
            self.norm_2 = nn.LayerNorm(config.hidden_size)

class FalconModel(BaseLLM):
    def __init__(self, config: Dict[str, Any]):
        if not isinstance(config, FalconConfig):
            config = FalconConfig(**config)
        super().__init__(config)
        
        # Override transformer layers with Falcon-specific blocks
        self.layers = nn.ModuleList([
            FalconBlock(config) for _ in range(config.num_layers)
        ])
        
        # Initialize weights
        self.apply(self._init_weights) 