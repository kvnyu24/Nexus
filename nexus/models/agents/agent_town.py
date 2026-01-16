import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
from ...core.base import NexusModule
from .interaction import InteractionModule
from .environment import VirtualEnvironment
from .proactive_agent import ProactiveAgent
from ..nlp import ChainOfThoughtModule, EnhancedRAGModule
from nexus.utils.logging import Logger

class AgentBehavior(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.hidden_dim = config["hidden_dim"]
        self.num_actions = config["num_actions"]
        
        # Enhanced behavior network with residual connections
        self.behavior_net = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ),
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
        ])
        self.output = nn.Linear(self.hidden_dim, self.num_actions)
        
    def forward(self, state: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([state, context], dim=-1)
        features = combined
        for layer in self.behavior_net:
            features = layer(features) + features
        return self.output(features)

class SocialAgent(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.hidden_dim = config["hidden_dim"]
        self.memory_size = config.get("memory_size", 1000)
        
        # Enhanced core components
        self.state_encoder = nn.Sequential(
            nn.Linear(config["state_dim"], self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )
        self.behavior = AgentBehavior(config)
        
        # Improved memory system
        self.episodic_memory = nn.Parameter(torch.zeros(self.memory_size, self.hidden_dim))
        self.semantic_memory = nn.Parameter(torch.zeros(self.memory_size // 2, self.hidden_dim))
        self.memory_attention = nn.MultiheadAttention(self.hidden_dim, 8, dropout=0.1)
        
        # Memory management
        self.memory_update = nn.GRU(self.hidden_dim, self.hidden_dim)
        self.memory_compress = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        
        # Reasoning components
        self.chain_of_thought = ChainOfThoughtModule(config)
        self.rag = EnhancedRAGModule(config)
        
    def forward(self, state: torch.Tensor, timestep: Optional[int] = None) -> Dict[str, torch.Tensor]:
        # Encode current state
        encoded_state = self.state_encoder(state)
        
        # Enhanced memory querying
        episodic_context, episodic_attention = self.memory_attention(
            encoded_state.unsqueeze(0),
            self.episodic_memory.unsqueeze(0),
            self.episodic_memory.unsqueeze(0)
        )
        
        semantic_context, semantic_attention = self.memory_attention(
            encoded_state.unsqueeze(0),
            self.semantic_memory.unsqueeze(0),
            self.semantic_memory.unsqueeze(0)
        )
        
        # Combine memory contexts
        memory_context = self.memory_compress(
            torch.cat([episodic_context.squeeze(0), semantic_context.squeeze(0)], dim=-1)
        )
        
        # Apply reasoning
        reasoning_output = self.chain_of_thought(encoded_state, memory_context)
        knowledge_output = self.rag(encoded_state)
        
        # Generate behavior with enhanced context
        actions = self.behavior(
            encoded_state,
            memory_context + reasoning_output["thought_vector"] + knowledge_output["retrieved_info"]
        )
        
        # Update memories if timestep provided
        if timestep is not None:
            self._update_memories(encoded_state, timestep)
        
        return {
            "actions": actions,
            "episodic_attention": episodic_attention,
            "semantic_attention": semantic_attention,
            "reasoning": reasoning_output,
            "knowledge": knowledge_output
        }
        
    def _update_memories(self, state: torch.Tensor, timestep: int):
        # Update episodic memory
        idx = timestep % self.memory_size
        self.episodic_memory.data[idx] = state
        
        # Periodically update semantic memory
        if timestep % 10 == 0:
            semantic_idx = (timestep // 10) % (self.memory_size // 2)
            _, hidden = self.memory_update(
                state.unsqueeze(0),
                self.semantic_memory[semantic_idx].unsqueeze(0)
            )
            self.semantic_memory.data[semantic_idx] = hidden.squeeze(0)

class AgentTown(NexusModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger = Logger(self.__class__.__name__)

        # Enhanced configuration
        self.num_agents = config["num_agents"]
        self.hidden_dim = config["hidden_dim"]
        self.state_dim = config["state_dim"]
        self.max_agents = config.get("max_agents", self.num_agents * 2)
        
        # Initialize enhanced components
        self.environment = VirtualEnvironment(config)
        self.interaction_module = InteractionModule(config)
        
        # Dynamic agent management
        self.agents = nn.ModuleDict({
            f"agent_{i}": SocialAgent(config) for i in range(self.num_agents)
        })
        
        # Enhanced proactive agents
        self.proactive_agents = nn.ModuleDict({
            f"proactive_{i}": ProactiveAgent(config) 
            for i in range(config.get("num_proactive_agents", 2))
        })
        
        # Global state tracking
        self.global_state = nn.Parameter(torch.zeros(1, self.hidden_dim))
        self.state_update = nn.GRU(self.hidden_dim, self.hidden_dim)
        
    def forward(
        self,
        states: torch.Tensor,
        agent_masks: Optional[torch.Tensor] = None,
        timestep: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        batch_size = states.size(0)
        
        # Process each agent with enhanced error handling
        agent_actions = []
        agent_interactions = []
        agent_states = []
        
        for agent_id, agent in self.agents.items():
            idx = int(agent_id.split('_')[1])
            if agent_masks is None or agent_masks[idx]:
                try:
                    # Get agent behavior
                    outputs = agent(states[idx], timestep)
                    agent_actions.append(outputs["actions"])
                    agent_states.append(outputs["reasoning"]["thought_vector"])
                    
                    # Process interactions with safety checks
                    if states.size(0) > 1:
                        interactions = self.interaction_module(
                            states[idx],
                            states,
                            outputs["actions"]
                        )
                        agent_interactions.append(interactions)
                except Exception as e:
                    self.logger.error(f"Error processing agent {agent_id}: {str(e)}")
                    continue
        
        if not agent_actions:
            raise RuntimeError("No valid agent actions generated")
            
        # Update environment with enhanced state tracking
        env_state = self.environment.step(
            torch.stack(agent_actions),
            torch.stack(agent_interactions) if agent_interactions else None
        )
        
        # Update global state
        _, self.global_state = self.state_update(
            torch.stack(agent_states).mean(0).unsqueeze(0),
            self.global_state
        )
        
        # Process proactive agents with enhanced context
        proactive_actions = []
        proactive_initiatives = []
        proactive_impacts = []
        
        for agent_id, agent in self.proactive_agents.items():
            try:
                outputs = agent.act(
                    states,
                    social_context=torch.stack(agent_interactions) if agent_interactions else None,
                    global_state=self.global_state,
                    threshold=self.config.get("initiative_threshold", 0.5)
                )
                proactive_actions.append(outputs["actions"])
                proactive_initiatives.append(outputs["initiative_score"])
                proactive_impacts.append(outputs.get("impact_assessment", torch.zeros(1)))
            except Exception as e:
                self.logger.error(f"Error processing proactive agent {agent_id}: {str(e)}")
                continue
        
        return {
            "actions": torch.stack(agent_actions),
            "interactions": torch.stack(agent_interactions) if agent_interactions else None,
            "environment_state": env_state,
            "proactive_actions": torch.stack(proactive_actions) if proactive_actions else None,
            "proactive_initiatives": torch.stack(proactive_initiatives) if proactive_initiatives else None,
            "proactive_impacts": torch.stack(proactive_impacts) if proactive_impacts else None,
            "global_state": self.global_state
        }