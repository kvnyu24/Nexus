from nexus.models.agents import InteractionModule, DialogueManager
import torch

# Configuration
config = {
    "hidden_dim": 256,
    "num_heads": 8,
    "num_interaction_types": 5,
    "vocab_size": 50000,
    "max_seq_length": 512,
    "dropout": 0.1
}

# Initialize modules
interaction_module = InteractionModule(config)
dialogue_manager = DialogueManager(config)

# Sample data
batch_size = 4
seq_length = 64
num_agents = 10

# Create dummy data
agent_state = torch.randn(batch_size, config["hidden_dim"])
other_agents = torch.randn(batch_size, num_agents, config["hidden_dim"])
dialogue_history = torch.randint(0, config["vocab_size"], (batch_size, seq_length))

# Process interactions
interaction_outputs = interaction_module(
    agent_state=agent_state,
    other_agents_states=other_agents
)

# Generate dialogue
dialogue_outputs = dialogue_manager(
    dialogue_history=dialogue_history,
    agent_state=agent_state
)

print("Interaction logits shape:", interaction_outputs["interaction_logits"].shape)
print("Dialogue logits shape:", dialogue_outputs["logits"].shape) 