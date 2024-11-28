import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union, Tuple
from ....core.base import NexusModule
import math

class BaseLLMConfig:
    """Configuration class for Base LLM"""
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        intermediate_size: int = None,
        max_seq_length: int = 2048,
        dropout: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = True
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size or 4 * hidden_size
        self.max_seq_length = max_seq_length
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings

class BaseLLMAttention(NexusModule):
    def __init__(self, config: BaseLLMConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_heads
        
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        batch_size, seq_length, _ = hidden_states.shape
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        query_states = query_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention
        attention_scores = torch.matmul(query_states, key_states.transpose(-2, -1))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_states)
        context_layer = context_layer.transpose(1, 2).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        
        outputs = (self.o_proj(context_layer),)
        if output_attentions:
            outputs += (attention_probs,)
            
        return outputs

class BaseLLMBlock(NexusModule):
    def __init__(self, config: BaseLLMConfig):
        super().__init__()
        self.attention = BaseLLMAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.dropout)
        )
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attention_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions
        )
        hidden_states = residual + attention_outputs[0]
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += attention_outputs[1:]
            
        return outputs 

class BaseLLM(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Convert dict config to BaseLLMConfig
        self.config = BaseLLMConfig(**config)
        
        # Core components
        self.embeddings = nn.ModuleDict({
            "word_embeddings": nn.Embedding(
                self.config.vocab_size,
                self.config.hidden_size,
                padding_idx=self.config.pad_token_id
            ),
            "position_embeddings": nn.Embedding(
                self.config.max_seq_length,
                self.config.hidden_size
            )
        })
        
        # Transformer layers
        self.layers = nn.ModuleList([
            BaseLLMBlock(self.config) for _ in range(self.config.num_layers)
        ])
        
        # Output head
        self.norm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_epsilon)
        self.output = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        
        if self.config.tie_word_embeddings:
            self.output.weight = self.embeddings["word_embeddings"].weight
            
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
                
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_length = input_ids.shape
        
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
            
        # Get embeddings
        hidden_states = self.embeddings["word_embeddings"](input_ids)
        position_embeddings = self.embeddings["position_embeddings"](position_ids)
        hidden_states = hidden_states + position_embeddings
        
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        # Process through transformer layers
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions
            )
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_attentions += (layer_outputs[1],)
                
        # Final layer norm and output
        hidden_states = self.norm(hidden_states)
        logits = self.output(hidden_states)
        
        return {
            "logits": logits,
            "hidden_states": hidden_states,
            "all_hidden_states": all_hidden_states,
            "attentions": all_attentions
        }