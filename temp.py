import glob
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import gym
import random
from collections import deque
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import neptune
from dotenv import load_dotenv

# Custom Dataset
class ReplayDataset(Dataset):
    def __init__(self, buffer):
        self.buffer = buffer

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        state, action, reward, next_state, done = self.buffer[idx]
        return (torch.tensor(state, dtype=torch.float32),
                torch.tensor(action, dtype=torch.int64),
                torch.tensor(reward, dtype=torch.float32),
                torch.tensor(next_state, dtype=torch.float32),
                torch.tensor(done, dtype=torch.bool))


# Neural Network
class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers):
        super(DQNNetwork, self).__init__()
        layers = []
        input_size = state_size
        for hidden_layer in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_layer))
            layers.append(nn.ReLU())
            input_size = hidden_layer
        layers.append(nn.Linear(input_size, action_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# Lightning Module
class DQNAgent(pl.LightningModule):
    def __init__(self, state_size, action_size, hidden_layers, lr=0.001, gamma=0.99, batch_size=512, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        super(DQNAgent, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.lr = lr

        # Epsilon-greedy parameters
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.q_network = DQNNetwork(state_size, action_size, hidden_layers)
        self.target_network = DQNNetwork(state_size, action_size, hidden_layers)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.loss_fn = nn.MSELoss()
        self.memory = deque(maxlen=10000)

        # Load environment variables
        load_dotenv()

        # Retrieve and clean up the API token
        api_token = os.getenv("NEPTUNE_API_TOKEN")
        if not api_token:
            raise ValueError("NEPTUNE_API_TOKEN not found in environment variables.")
        api_token = api_token.strip()  # Remove leading/trailing whitespace or newlines

        self.run = neptune.init_run(
            project="nadavcherry/dp1",
            capture_hardware_metrics=True,
            api_token=api_token,
            tags=f"DRL-1",
        )

        # Log parameters
        self.run["parameters"] = {
            "state_size": state_size,
            "action_size": action_size,
            "hidden_layers": hidden_layers,
            "learning_rate": lr,
            "gamma": gamma,
            "batch_size": batch_size,
            "epsilon": epsilon,
            "epsilon_min": epsilon_min,
            "epsilon_decay": epsilon_decay,
        }

    def forward(self, state):
        return self.q_network(state)

    def sample_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state)
            return torch.argmax(q_values).item()

    def training_step(self, batch, batch_idx):
        states, actions, rewards, next_states, dones = batch
        q_values = self.q_network(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones.float()) * self.gamma * next_q_values

        loss = self.loss_fn(q_values, target_q_values)
        self.log("loss", loss)
        self.run["loss"].log(loss)  # Log to Neptune
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.q_network.parameters(), lr=self.lr)

    def train_dataloader(self):
        dataset = ReplayDataset(self.memory)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def on_train_epoch_end(self):
        """Update the target network and decay epsilon after each epoch."""
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.run["epsilon"].log(self.epsilon)  # Log to Neptune

    def update_target_network(self):
        """Update the target network to match the Q-network."""
        self.target_network.load_state_dict(self.q_network.state_dict())


# Data Collection
def collect_data(agent, env, episodes=100):
    total_rewards = []
    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = agent.sample_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.memory.append((state, action, reward, next_state, done))
            state = next_state
            episode_reward += reward
        total_rewards.append(episode_reward)
    print(f"Average reward in collected episodes: {np.mean(total_rewards):.2f}")
    agent.run["total_rewards"].log(total_rewards)  # Log to Neptune


def train_agent(agent, env, episodes=500, data_collection_episodes=10, update_target_every=10):
    """
    Train the agent by alternating between data collection and training.

    Args:
        agent: The DQNAgent.
        env: The environment.
        episodes: Total number of episodes for training.
        data_collection_episodes: Number of episodes to collect data before each training iteration.
        update_target_every: Frequency (in episodes) of updating the target network.
    """

    # Function to find the last checkpoint
    def get_last_checkpoint(log_dir="lightning_logs"):
        checkpoints = glob.glob(os.path.join(log_dir, "**", "checkpoints", "last.ckpt"), recursive=True)
        return max(checkpoints, key=os.path.getctime) if checkpoints else None

    last_ckpt = get_last_checkpoint()

    total_collected_episodes = 0
    for episode in range(0, episodes, data_collection_episodes):
        # Collect data by interacting with the environment
        collect_data(agent, env, episodes=data_collection_episodes)
        total_collected_episodes += data_collection_episodes

        # Create Trainer
        trainer = pl.Trainer(
            max_epochs=10,
            default_root_dir="lightning_logs",
            log_every_n_steps=1,
            enable_checkpointing=True
        )

        # Train the agent with checkpoint resumption
        trainer.fit(agent, ckpt_path=last_ckpt)

        # Update checkpoint path after training
        last_ckpt = get_last_checkpoint()

        # Log progress
        print(f"Episodes completed: {total_collected_episodes}, Epsilon: {agent.epsilon:.4f}")



def test_agent(agent, env, episodes=10):
    total_rewards = []

    # Save current epsilon and set to 0.0 for testing
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0

    for _ in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            env.render()
            action = agent.sample_action(state)  # No need to pass epsilon
            state, reward, done, _ = env.step(action)
            total_reward += reward
        total_rewards.append(total_reward)

    # Restore the original epsilon
    agent.epsilon = original_epsilon

    env.close()
    print(f"Average reward over {episodes} episodes: {np.mean(total_rewards)}")



if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    hidden_layers = [64, 64, 64]
    agent = DQNAgent(state_size, action_size, hidden_layers)
    train_agent(agent, env, episodes=500, data_collection_episodes=10, update_target_every=10)
    test_agent(agent, env)
    agent.run.stop()
