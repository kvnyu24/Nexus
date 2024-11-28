import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from ...core.base import NexusModule
from .memory import AgentMemoryStream
from .environment import VirtualEnvironment

class ProactiveAgent(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Core dimensions
        self.hidden_dim = config["hidden_dim"]
        self.num_actions = config["num_actions"]
        
        # Environment monitoring
        self.state_encoder = nn.Sequential(
            nn.Linear(config["state_dim"], self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Dropout(config.get("dropout", 0.1))
        )
        
        # Enhanced proactive components
        self.initiative_threshold = config.get("initiative_threshold", 0.5)
        self.opportunity_detector = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Safety assessment (following DecisionMakingModule pattern)
        self.safety_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Action generation with uncertainty
        self.action_planner = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_actions * 2)  # Mean and variance
        )
        
        # Memory components
        self.memory_stream = AgentMemoryStream(config)
        self.context_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=config.get("num_heads", 8),
            dropout=config.get("dropout", 0.1)
        )
        
        # Feature bank for experience replay
        self.register_buffer(
            "experience_bank",
            torch.zeros(config.get("bank_size", 1000), self.hidden_dim)
        )
        self.register_buffer("bank_ptr", torch.zeros(1, dtype=torch.long))
        
    def update_experience_bank(self, features: torch.Tensor):
        """Update experience bank following EnhancedReID pattern"""
        batch_size = features.size(0)
        ptr = int(self.bank_ptr)
        
        if ptr + batch_size > self.experience_bank.size(0):
            ptr = 0
            
        self.experience_bank[ptr:ptr + batch_size] = features.detach()
        self.bank_ptr[0] = (ptr + batch_size) % self.experience_bank.size(0)
        
    def assess_safety(self, state_encoding: torch.Tensor) -> torch.Tensor:
        """Assess action safety"""
        return self.safety_head(state_encoding)
        
    def forward(
        self,
        environment_state: torch.Tensor,
        social_context: Optional[torch.Tensor] = None,
        memory_context: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        safety_threshold: float = 0.8
    ) -> Dict[str, torch.Tensor]:
        # Encode current state
        encoded_state = self.state_encoder(environment_state)
        
        # Process memory and context
        memory_outputs = self.memory_stream(
            encoded_state,
            context=social_context
        )
        
        # Update experience bank
        self.update_experience_bank(encoded_state)
        
        # Detect action opportunities
        opportunity_score = self.opportunity_detector(
            torch.cat([
                encoded_state,
                memory_outputs["memory_encoding"]
            ], dim=-1)
        )
        
        # Assess safety
        safety_score = self.assess_safety(encoded_state)
        
        # Generate action plan when safe and opportunity detected
        if opportunity_score > self.initiative_threshold and safety_score > safety_threshold:
            # Enhanced context attention
            context_features, attention_weights = self.context_attention(
                encoded_state.unsqueeze(0),
                torch.cat([
                    memory_outputs["episodic_memory"].unsqueeze(0),
                    self.experience_bank.unsqueeze(0)
                ], dim=1),
                memory_outputs["semantic_memory"].unsqueeze(0)
            )
            
            # Plan actions with uncertainty
            action_output = self.action_planner(
                torch.cat([
                    encoded_state,
                    context_features.squeeze(0),
                    memory_outputs["memory_encoding"]
                ], dim=-1)
            )
            
            # Split into mean and variance
            action_mean, action_var = torch.chunk(action_output, 2, dim=-1)
            action_logits = action_mean + torch.randn_like(action_mean) * action_var.sigmoid()
        else:
            action_logits = torch.zeros(
                encoded_state.size(0),
                self.num_actions,
                device=encoded_state.device
            )
            attention_weights = None
            action_var = torch.zeros_like(action_logits)
            
        outputs = {
            "action_logits": action_logits,
            "action_uncertainty": action_var.sigmoid(),
            "opportunity_score": opportunity_score,
            "safety_score": safety_score,
            "memory_outputs": memory_outputs
        }
        
        if return_attention:
            outputs["attention_weights"] = attention_weights
            
        return outputs

    def act(
        self,
        environment_state: torch.Tensor,
        social_context: Optional[torch.Tensor] = None,
        memory_context: Optional[torch.Tensor] = None,
        safety_threshold: float = 0.8
    ) -> Dict[str, torch.Tensor]:
        """Generate actions based on environmental observations"""
        outputs = self.forward(
            environment_state,
            social_context=social_context,
            memory_context=memory_context,
            safety_threshold=safety_threshold
        )
        
        # Select actions considering safety and uncertainty
        safe_mask = (
            (outputs["safety_score"] > safety_threshold) &
            (outputs["opportunity_score"] > self.initiative_threshold) &
            (outputs["action_uncertainty"].mean(dim=-1, keepdim=True) < 0.5)
        ).float()
        
        actions = torch.argmax(outputs["action_logits"], dim=-1)
        actions = actions * safe_mask.squeeze(-1)
        
        return {
            "actions": actions,
            "opportunity_score": outputs["opportunity_score"],
            "safety_score": outputs["safety_score"],
            "action_uncertainty": outputs["action_uncertainty"],
            "action_logits": outputs["action_logits"]
        } 