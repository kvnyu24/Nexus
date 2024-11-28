from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
from ...core.base import NexusModule
from ...components.attention import FlashAttention, MultiHeadSelfAttention

class ReasoningStep:
    def __init__(self, hidden_size: int):
        self.attention = MultiHeadSelfAttention(hidden_size=hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # Self-attention over current reasoning state
        attended = self.attention(x)
        
        # If we have additional context, attend to it
        if context is not None:
            context_attended = self.attention(x, context)
            attended = attended + context_attended
            
        # Transform through FFN
        x = self.norm(x + attended)
        x = x + self.ffn(x)
        
        return {
            "hidden_states": x,
            "attention_weights": self.attention.last_attention_weights
        }

class ChainOfThoughtModule(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.num_steps = config["num_reasoning_steps"]
        self.hidden_size = config["hidden_size"]
        
        # Create reasoning steps
        self.reasoning_steps = nn.ModuleList([
            ReasoningStep(self.hidden_size) 
            for _ in range(self.num_steps)
        ])
        
        # Optional: Add step embeddings to differentiate reasoning stages
        self.step_embeddings = nn.Parameter(
            torch.randn(self.num_steps, 1, self.hidden_size)
        )
        
        # Output projection
        self.output_proj = nn.Linear(self.hidden_size, config["vocab_size"])
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        batch_size = hidden_states.size(0)
        reasoning_outputs = []
        attention_maps = []
        
        current_state = hidden_states
        
        # Perform multiple reasoning steps
        for step_idx, reasoning_step in enumerate(self.reasoning_steps):
            # Add step embedding to current state
            step_embed = self.step_embeddings[step_idx].expand(batch_size, -1, -1)
            step_state = current_state + step_embed
            
            # Apply reasoning step
            step_output = reasoning_step(
                step_state,
                context=hidden_states  # Original input as context
            )
            
            current_state = step_output["hidden_states"]
            reasoning_outputs.append(current_state)
            attention_maps.append(step_output["attention_weights"])
            
        # Final prediction
        logits = self.output_proj(current_state)
        
        return {
            "logits": logits,
            "reasoning_steps": reasoning_outputs,
            "attention_maps": attention_maps
        }

class ReasoningLLM(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Base LLM configuration
        self.vocab_size = config["vocab_size"]
        self.hidden_size = config["hidden_size"]
        
        # Initialize base LLM components
        self.token_embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config["max_seq_length"], self.hidden_size)
        )
        
        # Chain of thoughts module
        self.reasoning_module = ChainOfThoughtModule(config)
        
        # Output head
        self.output = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        
        # Tie weights
        self.output.weight = self.token_embedding.weight
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_reasoning_steps: bool = False
    ) -> Dict[str, torch.Tensor]:
        # Get embeddings
        hidden_states = self.token_embedding(input_ids)
        hidden_states = hidden_states + self.position_embedding[:, :input_ids.size(1), :]
        
        # Apply reasoning module
        reasoning_outputs = self.reasoning_module(
            hidden_states,
            attention_mask=attention_mask,
            return_all_steps=return_reasoning_steps
        )
        
        # Generate logits
        logits = self.output(reasoning_outputs["hidden_states"])
        
        outputs = {
            "logits": logits,
            "hidden_states": reasoning_outputs["hidden_states"],
            "final_thought": reasoning_outputs["final_thought"]
        }
        
        if return_reasoning_steps:
            outputs["reasoning_steps"] = reasoning_outputs["all_thoughts"]
            
        return outputs 