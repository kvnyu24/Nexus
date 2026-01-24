import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from .base_llm import BaseLLM, BaseLLMConfig
import math
from nexus.core.base import NexusModule
from ....core.initialization import WeightInitMixin

class LlamaConfig(BaseLLMConfig):
    """Configuration class for LLaMA model"""
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        num_layers: int = 32,
        num_heads: int = 32,
        intermediate_size: Optional[int] = None,
        max_seq_length: int = 2048,
        rope_scaling: Optional[float] = None,
        rope_theta: float = 10000.0,
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
        self.rope_scaling = rope_scaling
        self.rope_theta = rope_theta

class LlamaRotaryEmbedding(NexusModule):
    def __init__(self, dim: int, max_seq_length: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_length = max_seq_length
        self.theta = theta
        
        # Create position embeddings cache
        position = torch.arange(max_seq_length).unsqueeze(1)
        freqs = torch.exp(
            -torch.arange(0, dim, 2) * (math.log(theta) / dim)
        ).unsqueeze(0)
        emb = position * freqs
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

class LlamaAttention(NexusModule):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_heads
        
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        self.rotary_emb = LlamaRotaryEmbedding(
            self.head_dim,
            max_seq_length=config.max_seq_length,
            theta=config.rope_theta
        )

class LlamaBlock(NexusModule):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.attention = LlamaAttention(config)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size, bias=False),
            nn.SiLU(),
            nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        )
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

class LlamaModel(WeightInitMixin, BaseLLM):
    def __init__(self, config: Dict[str, Any]):
        # Convert dict config to LlamaConfig
        if not isinstance(config, LlamaConfig):
            config = LlamaConfig(**config)
        super().__init__(config)

        # Override transformer layers with LLaMA-specific blocks
        self.layers = nn.ModuleList([
            LlamaBlock(config) for _ in range(config.num_layers)
        ])

        # Initialize weights using LLM preset
        self.init_weights_llm()
            
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # Call parent class forward with LLaMA-specific arguments
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )
        
        # Add LLaMA-specific outputs
        if use_cache:
            outputs["past_key_values"] = self._get_past_key_values()
            
        return outputs 