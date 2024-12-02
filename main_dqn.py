import gym
from pytorch_lightning import Trainer
from dqn_agent import DQNLightning
from configs.cartpole_config import hyperparameters
from utils import plot_rewards, plot_losses

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
env = gym.make("CartPole-v1", new_step_api=True)

# Initialize DQN agent
agent = DQNLightning(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    hidden_layers=hidden_layers,
    buffer_size=buffer_size,
    batch_size=batch_size,
    lr=lr,
    gamma=gamma,
    epsilon=epsilon,
    epsilon_min=epsilon_min,
    epsilon_decay=epsilon_decay,
    sync_freq=sync_freq,
    env=env  # Pass the environment
)

# Initialize PyTorch Lightning Trainer
trainer = Trainer(
    accelerator="auto",
    max_epochs=episodes,
    log_every_n_steps=10,
)

# Train the agent
trainer.fit(agent)

# Test the trained agent
avg_reward = agent.test_agent(env, num_episodes=100)
print(f"Average reward over 100 test episodes: {avg_reward:.2f}")

# Plot metrics
plot_rewards(agent.total_rewards)
plot_losses(agent.losses)
