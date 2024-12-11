from typing import Dict, Any, Optional, List, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_llm import BaseLLM, BaseLLMConfig
from nexus.core.base import NexusModule

class GPT4OConfig(BaseLLMConfig):
    """Configuration class for GPT-4O with enhanced reasoning and RLHF capabilities"""
    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 1024,
        num_layers: int = 24,
        num_heads: int = 16,
        num_reasoning_steps: int = 4,  # Increased for more thorough reasoning
        reasoning_hidden_size: Optional[int] = None,
        use_structured_reasoning: bool = True,
        max_reasoning_length: int = 1024,  # Increased for longer context
        reasoning_dropout: float = 0.1,
        use_memory_efficient_attention: bool = True,
        use_parallel_attention: bool = True,
        num_cot_iterations: int = 3,  # Chain of thought iterations
        rlhf_reward_scale: float = 1.0,
        use_value_head: bool = True,
        use_critic_head: bool = True,
        **kwargs
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            **kwargs
        )
        self.num_reasoning_steps = num_reasoning_steps
        self.reasoning_hidden_size = reasoning_hidden_size or hidden_size
        self.use_structured_reasoning = use_structured_reasoning
        self.max_reasoning_length = max_reasoning_length
        self.reasoning_dropout = reasoning_dropout
        self.use_memory_efficient_attention = use_memory_efficient_attention
        self.use_parallel_attention = use_parallel_attention
        self.num_cot_iterations = num_cot_iterations
        self.rlhf_reward_scale = rlhf_reward_scale
        self.use_value_head = use_value_head
        self.use_critic_head = use_critic_head

class ReasoningStep(NexusModule):
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        # Enhanced multi-query attention with rotary embeddings
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # Improved parallel cross-attention with relative position bias
        self.cross_q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.cross_k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.cross_v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.cross_o = nn.Linear(hidden_size, hidden_size, bias=False)
        self.rel_pos_bias = nn.Parameter(torch.zeros(1, num_heads, 128, 128))

        # Enhanced feed-forward with mixture of experts
        self.feed_forward = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Linear(hidden_size * 4, hidden_size * 2),
                nn.GLU(dim=-1)
            ) for _ in range(4)  # 4 expert networks
        ])
        self.router = nn.Linear(hidden_size, 4)

        # Improved layer norms with better initialization
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm3 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        
        # Enhanced gating with highway connections
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
        
        # Value estimation for RLHF
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Dict[str, torch.Tensor]:
        bsz, seq_len, _ = hidden_states.shape
        
        # Enhanced self-attention with rotary embeddings and cached keys/values
        normed_states = self.norm1(hidden_states)
        q = self._shape(self.q_proj(normed_states), seq_len, bsz) * self.scale
        k = self._shape(self.k_proj(normed_states), seq_len, bsz)
        v = self._shape(self.v_proj(normed_states), seq_len, bsz)

        if past_key_values is not None:
            k = torch.cat([past_key_values[0], k], dim=2)
            v = torch.cat([past_key_values[1], v], dim=2)

        self_attn = torch.matmul(q, k.transpose(-2, -1))
        
        # Add relative position bias and rotary embeddings
        if position_bias is not None:
            self_attn = self_attn + position_bias
        self_attn = self_attn + self.rel_pos_bias[:, :, :seq_len, :seq_len]

        if attention_mask is not None:
            self_attn = self_attn.masked_fill(attention_mask.unsqueeze(1), float('-inf'))
            
        self_attn_weights = F.softmax(self_attn, dim=-1)
        self_attn_weights = self.dropout(self_attn_weights)
        self_attn_output = torch.matmul(self_attn_weights, v)
        self_attn_output = self_attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        hidden_states = hidden_states + self.dropout(self.o_proj(self_attn_output))

        # Enhanced parallel cross-attention with improved routing
        normed_states = self.norm2(hidden_states)
        cross_q = self._shape(self.cross_q(normed_states), seq_len, bsz) * self.scale
        cross_k = self._shape(self.cross_k(context), context.size(1), bsz)
        cross_v = self._shape(self.cross_v(context), context.size(1), bsz)

        cross_attn = torch.matmul(cross_q, cross_k.transpose(-2, -1))
        cross_attn_weights = F.softmax(cross_attn, dim=-1)
        cross_attn_weights = self.dropout(cross_attn_weights)
        cross_attn_output = torch.matmul(cross_attn_weights, cross_v)
        cross_attn_output = cross_attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        
        # Enhanced gating with highway connections
        gate_input = torch.cat([hidden_states, cross_attn_output], dim=-1)
        gate_value = self.gate(gate_input)
        hidden_states = hidden_states + self.dropout(gate_value * self.cross_o(cross_attn_output))

        # Mixture of experts feed-forward
        normed_states = self.norm3(hidden_states)
        router_logits = self.router(normed_states)
        router_probs = F.softmax(router_logits, dim=-1)
        
        ff_output = torch.zeros_like(hidden_states)
        for i, expert in enumerate(self.feed_forward):
            ff_output += router_probs[:, :, i:i+1] * expert(normed_states)
            
        hidden_states = hidden_states + self.dropout(ff_output)

        # Value estimation for RLHF
        value = self.value_head(hidden_states).squeeze(-1)

        outputs = {
            "hidden_states": hidden_states,
            "self_attention": self_attn_weights,
            "cross_attention": cross_attn_weights,
            "gate_values": gate_value,
            "router_probs": router_probs,
            "value": value
        }

        if use_cache:
            outputs["past_key_values"] = (k, v)

        return outputs

class GPT4O(BaseLLM):
    def __init__(self, config: Dict[str, Any]):
        if not isinstance(config, GPT4OConfig):
            config = GPT4OConfig(**config)
        super().__init__(config)
        
        # Enhanced reasoning components with chain of thought
        self.reasoning_steps = nn.ModuleList([
            ReasoningStep(
                config.hidden_size,
                config.num_heads,
                config.reasoning_dropout
            )
            for _ in range(config.num_reasoning_steps)
        ])
        
        # Improved step embeddings with learned positional encoding
        self.step_embeddings = nn.Parameter(
            torch.randn(config.num_reasoning_steps, 1, config.hidden_size)
        )
        
        # Chain of thought components
        self.cot_controller = nn.GRUCell(config.hidden_size, config.hidden_size)
        self.cot_query = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Enhanced projection layers
        self.reasoning_proj = nn.Sequential(
            nn.Linear(config.hidden_size, config.reasoning_hidden_size),
            nn.LayerNorm(config.reasoning_hidden_size),
            nn.GELU()
        )
        
        self.final_proj = nn.Sequential(
            nn.Linear(config.reasoning_hidden_size, config.hidden_size * 2),
            nn.LayerNorm(config.hidden_size * 2),
            nn.GELU(),
            nn.Linear(config.hidden_size * 2, config.vocab_size)
        )
        
        # RLHF components
        if config.use_critic_head:
            self.critic = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.Tanh(),
                nn.Linear(config.hidden_size, 1)
            )
        
        # Enhanced memory system
        self.memory_size = config.max_reasoning_length
        self.memory_bank = nn.Parameter(
            torch.randn(1, self.memory_size, config.hidden_size)
        )
        self.memory_proj = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU()
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_reasoning_steps: bool = False,
        use_memory: bool = True,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        use_cache: bool = False,
        return_value: bool = False
    ) -> Dict[str, torch.Tensor]:
        # Initial processing
        base_outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_states = base_outputs["hidden_states"]
        
        # Initialize reasoning process
        reasoning_outputs = []
        attention_maps = []
        memory_states = []
        value_estimates = []
        current_state = self.reasoning_proj(hidden_states)
        
        # Initialize chain of thought state
        cot_state = torch.zeros_like(current_state[:, 0])
        
        # Expand memory bank to batch size
        if use_memory:
            batch_memory = self.memory_bank.expand(hidden_states.size(0), -1, -1)
            memory_mask = torch.ones(
                hidden_states.size(0),
                self.memory_size,
                device=hidden_states.device,
                dtype=torch.bool
            )
        
        # Progressive reasoning with chain of thought
        for step_idx, reasoning_step in enumerate(self.reasoning_steps):
            # Update chain of thought state
            cot_state = self.cot_controller(
                current_state[:, 0],
                cot_state
            )
            
            # Generate reasoning query
            cot_query = self.cot_query(cot_state)
            
            # Combine step embedding with CoT query
            step_embed = self.step_embeddings[step_idx].expand(
                hidden_states.size(0), -1, -1
            )
            step_state = current_state + step_embed + cot_query.unsqueeze(1)
            
            # Integrate memory if enabled
            if use_memory:
                memory_context = self.memory_proj(
                    torch.cat([step_state, batch_memory], dim=-1)
                )
                step_state = step_state + memory_context
            
            # Apply reasoning with caching
            step_output = reasoning_step(
                step_state,
                context=hidden_states,
                attention_mask=attention_mask,
                position_bias=None,
                past_key_values=past_key_values[step_idx] if past_key_values else None,
                use_cache=use_cache
            )
            
            current_state = step_output["hidden_states"]
            if use_memory:
                # Update memory bank with attention-weighted averaging
                memory_update = torch.matmul(
                    step_output["self_attention"],
                    current_state
                )
                batch_memory = torch.cat([
                    batch_memory[:, 1:],
                    memory_update[:, -1:] 
                ], dim=1)
                memory_states.append(batch_memory)
            
            reasoning_outputs.append(current_state)
            attention_maps.append({
                "self": step_output["self_attention"],
                "cross": step_output["cross_attention"],
                "gate": step_output["gate_values"],
                "router": step_output["router_probs"]
            })
            value_estimates.append(step_output["value"])
            
            if use_cache:
                if past_key_values is None:
                    past_key_values = []
                past_key_values.append(step_output["past_key_values"])
        
        # Final prediction with residual connection and value head
        logits = self.final_proj(current_state) + self.reasoning_proj(hidden_states)
        
        if hasattr(self, 'critic'):
            value = self.critic(current_state.mean(dim=1))
        else:
            value = torch.stack(value_estimates, dim=1).mean(dim=1)
        
        outputs = {
            "logits": logits,
            "hidden_states": current_state,
            "base_hidden_states": hidden_states
        }
        
        if return_value:
            outputs["value"] = value
            
        if return_reasoning_steps:
            outputs.update({
                "reasoning_steps": reasoning_outputs,
                "attention_maps": attention_maps,
                "memory_states": memory_states if use_memory else None,
                "cot_states": cot_state
            })
            
        if use_cache:
            outputs["past_key_values"] = past_key_values
            
        return outputs