import numpy as np
from nexus.models.rl import PRMAgent
from nexus.utils.metrics import MetricsCalculator
import torch
# Environment configuration
state_bounds = np.array([
    [-10, 10],  # x bounds
    [-10, 10]   # y bounds
])

# Configure agent
config = {
    "state_dim": 2,
    "num_samples": 1000,
    "max_neighbors": 10,
    "connection_radius": 2.0,
    "learning_rate": 1e-3
}

# Initialize agent and metrics
agent = PRMAgent(config)
metrics = MetricsCalculator()

# Sample roadmap nodes
agent.sample_nodes(state_bounds)

# Training loop
num_episodes = 100
for episode in range(num_episodes):
    # Generate random start and goal
    start_state = np.random.uniform(state_bounds[:, 0], state_bounds[:, 1])
    goal_state = np.random.uniform(state_bounds[:, 0], state_bounds[:, 1])
    
    # Plan path
    path, cost = agent.plan_path(start_state, goal_state)
    
    # Create training batch
    states = torch.FloatTensor(np.array(path))
    values = torch.FloatTensor([-cost] * len(path)).unsqueeze(1)
    
    batch = {
        "states": states,
        "values": values
    }
    
    # Update agent
    metrics = agent.update(batch)
    
    print(f"Episode {episode + 1}, Path Length: {len(path)}, Cost: {cost:.2f}, Loss: {metrics['loss']:.4f}") 