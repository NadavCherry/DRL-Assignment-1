import gym
from dqn_agent import DQNAgent
from utils import plot_rewards, plot_losses
from configs.cartpole_config import hyperparameters

# Load CartPole hyperparameters
hidden_layers = hyperparameters["hidden_layers"]
buffer_size = hyperparameters["buffer_size"]
batch_size = hyperparameters["batch_size"]
lr = hyperparameters["lr"]
gamma = hyperparameters["gamma"]
epsilon = hyperparameters["epsilon"]
epsilon_min = hyperparameters["epsilon_min"]
epsilon_decay = hyperparameters["epsilon_decay"]
sync_freq = hyperparameters["sync_freq"]
episodes = hyperparameters["episodes"]

# Initialize environment
env = gym.make("CartPole-v1")

# Initialize DQN agent
agent = DQNAgent(
    env=env,
    hidden_layers=hidden_layers,
    buffer_size=buffer_size,
    batch_size=batch_size,
    lr=lr,
    gamma=gamma,
    epsilon=epsilon,
    epsilon_min=epsilon_min,
    epsilon_decay=epsilon_decay,
    sync_freq=sync_freq,
)

# Train the agent
rewards, losses = agent.train_agent(episodes)

# Plot metrics
plot_rewards(rewards)
plot_losses(losses)

# Test the trained agent
avg_reward = agent.test_agent(num_episodes=100)
print(f"Average reward over 100 test episodes: {avg_reward:.2f}")
