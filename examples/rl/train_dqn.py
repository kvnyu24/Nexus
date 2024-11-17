import gym
from nexus.models.rl import DQNAgent
from nexus.data.replay_buffer import ReplayBuffer
from nexus.utils.metrics import MetricsCalculator
import torch
import numpy as np

# Create environment
env = gym.make('CartPole-v1')

# Configure agent
config = {
    "state_dim": env.observation_space.shape[0],
    "action_dim": env.action_space.n,
    "hidden_dim": 128,
    "gamma": 0.99,
    "epsilon": 1.0,
    "epsilon_min": 0.01,
    "epsilon_decay": 0.995,
    "learning_rate": 1e-3
}

# Initialize agent and replay buffer
agent = DQNAgent(config)
replay_buffer = ReplayBuffer(capacity=10000)
metrics = MetricsCalculator()

# Training parameters
num_episodes = 500
batch_size = 64
target_update_frequency = 10

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        # Select and perform action
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        
        # Store transition
        replay_buffer.push(state, action, reward, next_state, done)
        total_reward += reward
        state = next_state
        
        # Train agent
        if len(replay_buffer) >= batch_size:
            batch = replay_buffer.sample(batch_size)
            metrics = agent.update(batch)
            
    # Update target network periodically
    if episode % target_update_frequency == 0:
        agent.update_target_network()
        
    print(f"Episode {episode + 1}, Total Reward: {total_reward}, Loss: {metrics.get('loss', 0):.4f}")

# Save trained agent
torch.save(agent.state_dict(), "dqn_cartpole.pth") 