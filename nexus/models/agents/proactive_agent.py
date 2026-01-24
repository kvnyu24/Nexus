import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
from ...core.base import NexusModule
from ...core.mixins import FeatureBankMixin
from .memory import AgentMemoryStream
from .environment import VirtualEnvironment
from ..nlp import ChainOfThoughtModule, EnhancedRAGModule

class ProactiveAgent(NexusModule, FeatureBankMixin):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Core dimensions
        self.hidden_dim = config["hidden_dim"]
        self.num_actions = config["num_actions"]
        
        # Environment monitoring with enhanced perception
        self.state_encoder = nn.Sequential(
            nn.Linear(config["state_dim"], self.hidden_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim)
        )
        
        # Enhanced proactive components with meta-learning
        self.initiative_threshold = nn.Parameter(torch.tensor(config.get("initiative_threshold", 0.5)))
        self.opportunity_detector = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 2),  # Opportunity score and confidence
            nn.Sigmoid()
        )
        
        # Multi-level safety assessment
        self.safety_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.LayerNorm(self.hidden_dim // 2),
                nn.Linear(self.hidden_dim // 2, 1),
                nn.Sigmoid()
            ) for _ in range(3)  # Physical, social, and ethical safety
        ])
        
        # Enhanced action planning with uncertainty and value estimation
        self.action_planner = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, self.hidden_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Dropout(config.get("dropout", 0.1)),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, self.num_actions * 3)  # Mean, variance, and value
        )
        
        # Advanced reasoning components
        self.chain_of_thought = ChainOfThoughtModule(config)
        self.rag_module = EnhancedRAGModule(config)
        
        # Enhanced memory system
        self.memory_stream = AgentMemoryStream(config)
        self.context_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=config.get("num_heads", 8),
            dropout=config.get("dropout", 0.1)
        )
        
        # Expanded experience bank with prioritized replay using mixin
        bank_size = config.get("bank_size", 1000)
        self.register_feature_bank("experience", bank_size, self.hidden_dim)
        self.register_feature_bank("priority", bank_size, 1)

        # Adaptive thresholds
        self.register_buffer("safety_threshold_history", torch.zeros(100))
        self.safety_update_momentum = config.get("safety_momentum", 0.95)
        
    def assess_safety(self, state_encoding: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Multi-level safety assessment"""
        safety_scores = {
            "physical": self.safety_heads[0](state_encoding),
            "social": self.safety_heads[1](state_encoding),
            "ethical": self.safety_heads[2](state_encoding)
        }
        
        # Compute weighted aggregate safety score
        aggregate_safety = sum(score * 0.33 for score in safety_scores.values())
        
        # Update adaptive safety threshold
        with torch.no_grad():
            self.safety_threshold_history = torch.roll(self.safety_threshold_history, -1)
            self.safety_threshold_history[-1] = aggregate_safety.mean()
        
        return aggregate_safety, safety_scores
        
    def forward(
        self,
        environment_state: torch.Tensor,
        social_context: Optional[torch.Tensor] = None,
        memory_context: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        safety_threshold: float = 0.8
    ) -> Dict[str, torch.Tensor]:
        # Enhanced state encoding with error checking
        if not torch.isfinite(environment_state).all():
            raise ValueError("Invalid environment state detected")
            
        encoded_state = self.state_encoder(environment_state)
        
        # Process memory and context with chain-of-thought reasoning
        memory_outputs = self.memory_stream(encoded_state, context=social_context)
        reasoning_outputs = self.chain_of_thought(encoded_state, memory_outputs["memory_encoding"])
        
        # Update experience bank using mixin
        self.update_feature_bank("experience", encoded_state)
        importance_scores = reasoning_outputs.get("importance_scores", None)
        if importance_scores is not None:
            self.update_feature_bank("priority", importance_scores)
        
        # Enhanced opportunity detection with confidence
        opportunity_output = self.opportunity_detector(
            torch.cat([
                encoded_state,
                memory_outputs["memory_encoding"],
                reasoning_outputs["reasoning_state"]
            ], dim=-1)
        )
        opportunity_score, confidence = opportunity_output.chunk(2, dim=-1)
        
        # Multi-level safety assessment
        safety_score, safety_components = self.assess_safety(encoded_state)
        
        # Adaptive safety threshold based on historical data
        adaptive_threshold = self.safety_threshold_history.mean() * self.safety_update_momentum + \
                           safety_threshold * (1 - self.safety_update_momentum)
        
        # Generate action plan when safe and opportunity detected
        if (opportunity_score > self.initiative_threshold).any() and (safety_score > adaptive_threshold).any():
            # Enhanced context processing
            context_features, attention_weights = self.context_attention(
                encoded_state.unsqueeze(0),
                torch.cat([
                    memory_outputs["episodic_memory"].unsqueeze(0),
                    self.get_feature_bank("experience").unsqueeze(0)
                ], dim=1),
                memory_outputs["semantic_memory"].unsqueeze(0)
            )
            
            # Plan actions with uncertainty and value estimation
            action_output = self.action_planner(
                torch.cat([
                    encoded_state,
                    context_features.squeeze(0),
                    memory_outputs["memory_encoding"],
                    reasoning_outputs["reasoning_state"]
                ], dim=-1)
            )
            
            # Split into mean, variance, and value
            action_mean, action_var, action_value = torch.chunk(action_output, 3, dim=-1)
            
            # Sample actions with reparameterization
            eps = torch.randn_like(action_mean)
            action_logits = action_mean + eps * action_var.sigmoid()
        else:
            action_logits = torch.zeros(
                encoded_state.size(0),
                self.num_actions,
                device=encoded_state.device
            )
            attention_weights = None
            action_var = torch.zeros_like(action_logits)
            action_value = torch.zeros_like(action_logits)
            
        outputs = {
            "action_logits": action_logits,
            "action_uncertainty": action_var.sigmoid(),
            "action_value": action_value,
            "opportunity_score": opportunity_score,
            "opportunity_confidence": confidence,
            "safety_score": safety_score,
            "safety_components": safety_components,
            "memory_outputs": memory_outputs,
            "reasoning_outputs": reasoning_outputs,
            "adaptive_safety_threshold": adaptive_threshold
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
        """Generate robust actions with safety guarantees"""
        try:
            outputs = self.forward(
                environment_state,
                social_context=social_context,
                memory_context=memory_context,
                safety_threshold=safety_threshold
            )
            
            # Enhanced action selection with multiple criteria
            safe_mask = (
                (outputs["safety_score"] > outputs["adaptive_safety_threshold"]) &
                (outputs["opportunity_score"] > self.initiative_threshold) &
                (outputs["opportunity_confidence"] > 0.5) &
                (outputs["action_uncertainty"].mean(dim=-1, keepdim=True) < 0.5)
            ).float()
            
            # Value-weighted action selection
            action_scores = outputs["action_logits"] + outputs["action_value"]
            actions = torch.argmax(action_scores, dim=-1)
            actions = actions * safe_mask.squeeze(-1)
            
            return {
                "actions": actions,
                "opportunity_score": outputs["opportunity_score"],
                "opportunity_confidence": outputs["opportunity_confidence"],
                "safety_score": outputs["safety_score"],
                "safety_components": outputs["safety_components"],
                "action_uncertainty": outputs["action_uncertainty"],
                "action_value": outputs["action_value"],
                "action_logits": outputs["action_logits"],
                "adaptive_threshold": outputs["adaptive_safety_threshold"]
            }
            
        except Exception as e:
            # Fallback to safe default behavior
            device = environment_state.device
            batch_size = environment_state.size(0)
            return {
                "actions": torch.zeros(batch_size, device=device, dtype=torch.long),
                "error": str(e)
            }