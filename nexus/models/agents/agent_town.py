import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional
from ...core.base import NexusModule
from .interaction import InteractionModule
from .environment import VirtualEnvironment

class AgentBehavior(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.hidden_dim = config["hidden_dim"]
        self.num_actions = config["num_actions"]
        
        self.behavior_net = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_actions)
        )
        
    def forward(self, state: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([state, context], dim=-1)
        return self.behavior_net(combined)

class SocialAgent(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.hidden_dim = config["hidden_dim"]
        self.memory_size = config.get("memory_size", 1000)
        
        # Core components
        self.state_encoder = nn.Linear(config["state_dim"], self.hidden_dim)
        self.behavior = AgentBehavior(config)
        self.memory = nn.Parameter(torch.zeros(self.memory_size, self.hidden_dim))
        self.memory_attention = nn.MultiheadAttention(self.hidden_dim, 4)
        
    def forward(self, state: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Encode current state
        encoded_state = self.state_encoder(state)
        
        # Query memory
        memory_context, attention = self.memory_attention(
            encoded_state.unsqueeze(0),
            self.memory.unsqueeze(0),
            self.memory.unsqueeze(0)
        )
        
        # Generate behavior
        actions = self.behavior(encoded_state, memory_context.squeeze(0))
        
        return {
            "actions": actions,
            "memory_attention": attention
        }

class AgentTown(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Core configuration
        self.num_agents = config["num_agents"]
        self.hidden_dim = config["hidden_dim"]
        self.state_dim = config["state_dim"]
        
        # Initialize components
        self.environment = VirtualEnvironment(config)
        self.interaction_module = InteractionModule(config)
        
        # Create agents
        self.agents = nn.ModuleList([
            SocialAgent(config) for _ in range(self.num_agents)
        ])
        
    def forward(
        self,
        states: torch.Tensor,
        agent_masks: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        # Process each agent
        agent_actions = []
        agent_interactions = []
        
        for i, agent in enumerate(self.agents):
            if agent_masks is None or agent_masks[i]:
                # Get agent behavior
                outputs = agent(states[i])
                agent_actions.append(outputs["actions"])
                
                # Process interactions
                interactions = self.interaction_module(
                    states[i],
                    states,
                    outputs["actions"]
                )
                agent_interactions.append(interactions)
        
        # Update environment
        env_state = self.environment.step(
            torch.stack(agent_actions),
            torch.stack(agent_interactions)
        )
        
        return {
            "actions": torch.stack(agent_actions),
            "interactions": torch.stack(agent_interactions),
            "environment_state": env_state
        } 