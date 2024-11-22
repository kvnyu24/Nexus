from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_llm import BaseLLM, BaseLLMConfig

class GPT4OConfig(BaseLLMConfig):
    """Configuration class for GPT-4O with reasoning capabilities"""
    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 1024,
        num_layers: int = 24,
        num_heads: int = 16,
        num_reasoning_steps: int = 3,
        reasoning_hidden_size: Optional[int] = None,
        use_structured_reasoning: bool = True,
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

class ReasoningStep(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        context: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Self-attention for current reasoning state
        normed_states = self.norm1(hidden_states)
        self_attn_output, self_attn_weights = self.self_attention(
            normed_states, normed_states, normed_states,
            need_weights=True
        )
        hidden_states = hidden_states + self_attn_output
        
        # Cross-attention with context
        normed_states = self.norm2(hidden_states)
        cross_attn_output, cross_attn_weights = self.cross_attention(
            normed_states, context, context,
            need_weights=True
        )
        hidden_states = hidden_states + cross_attn_output
        
        # Feed-forward
        hidden_states = hidden_states + self.feed_forward(self.norm3(hidden_states))
        
        return {
            "hidden_states": hidden_states,
            "self_attention": self_attn_weights,
            "cross_attention": cross_attn_weights
        }

class GPT4O(BaseLLM):
    def __init__(self, config: Dict[str, Any]):
        if not isinstance(config, GPT4OConfig):
            config = GPT4OConfig(**config)
        super().__init__(config)
        
        # Reasoning components
        self.reasoning_steps = nn.ModuleList([
            ReasoningStep(config.hidden_size, config.num_heads)
            for _ in range(config.num_reasoning_steps)
        ])
        
        # Step embeddings to differentiate reasoning stages
        self.step_embeddings = nn.Parameter(
            torch.randn(config.num_reasoning_steps, 1, config.hidden_size)
        )
        
        # Additional projection layers
        self.reasoning_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.final_proj = nn.Linear(config.hidden_size, config.vocab_size)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_reasoning_steps: bool = False
    ) -> Dict[str, torch.Tensor]:
        # Initial processing through base LLM
        base_outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_states = base_outputs["hidden_states"]
        
        # Apply reasoning steps
        reasoning_outputs = []
        attention_maps = []
        current_state = self.reasoning_proj(hidden_states)
        
        for step_idx, reasoning_step in enumerate(self.reasoning_steps):
            # Add step embedding
            step_embed = self.step_embeddings[step_idx].expand(
                hidden_states.size(0), -1, -1
            )
            step_state = current_state + step_embed
            
            # Apply reasoning
            step_output = reasoning_step(
                step_state,
                context=hidden_states,
                attention_mask=attention_mask
            )
            
            current_state = step_output["hidden_states"]
            reasoning_outputs.append(current_state)
            attention_maps.append({
                "self": step_output["self_attention"],
                "cross": step_output["cross_attention"]
            })
        
        # Final prediction
        logits = self.final_proj(current_state)
        
        outputs = {
            "logits": logits,
            "hidden_states": current_state,
            "base_hidden_states": hidden_states
        }
        
        if return_reasoning_steps:
            outputs.update({
                "reasoning_steps": reasoning_outputs,
                "attention_maps": attention_maps
            })
            
        return outputs 