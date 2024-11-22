from nexus.models.agents import AgentTown
from nexus.training import Trainer
from nexus.utils.metrics import MetricsCalculator
from nexus.data import Dataset
import torch
import torch.nn.functional as F

# Configuration
config = {
    "num_agents": 10,
    "hidden_dim": 256,
    "state_dim": 64,
    "num_actions": 20,
    "memory_size": 1000,
    "learning_rate": 1e-4
}

# Initialize model and trainer
model = AgentTown(config)
trainer = Trainer(
    model=model,
    optimizer="adam",
    learning_rate=config["learning_rate"]
)

# Create a simple dataset with random data
agent_town_dataset = Dataset({
    "states": torch.randn(1000, config["state_dim"]),
    "agent_masks": torch.ones(1000, config["num_agents"]),
    "target_actions": torch.randint(0, config["num_actions"], (1000,)),
    "target_interactions": torch.randn(1000, config["num_agents"], config["num_agents"])
})

# Training loop
def train_step(batch):
    # Forward pass
    outputs = model(
        states=batch["states"],
        agent_masks=batch["agent_masks"]
    )
    
    # Calculate losses
    action_loss = F.cross_entropy(
        outputs["actions"].view(-1, config["num_actions"]),
        batch["target_actions"].view(-1)
    )
    
    interaction_loss = F.mse_loss(
        outputs["interactions"],
        batch["target_interactions"]
    )
    
    # Combined loss
    total_loss = action_loss + 0.5 * interaction_loss
    
    return {
        "loss": total_loss,
        "action_loss": action_loss.item(),
        "interaction_loss": interaction_loss.item()
    }

# Train model
trainer.train(
    train_dataset=agent_town_dataset,  # Your dataset implementation
    batch_size=32,
    num_epochs=100,
    train_step_fn=train_step
) 