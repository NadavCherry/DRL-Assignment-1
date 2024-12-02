import random
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from replay_buffer import ReplayBuffer
from network import DQNNetwork


class DQNDataset(Dataset):
    """Dataset for replay buffer samples."""
    def __init__(self, replay_buffer, batch_size):
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size

    def __len__(self):
        # Ensure at least `batch_size` samples are in the replay buffer
        return min(len(self.replay_buffer), self.batch_size)

    def __getitem__(self, index):
        if len(self.replay_buffer) < self.batch_size:
            raise ValueError("Not enough samples in replay buffer to sample a batch.")
        return self.replay_buffer.sample(1)[0]  # Return one sample



class DQNLightning(pl.LightningModule):
    def __init__(self, state_dim, action_dim, hidden_layers, buffer_size, batch_size,
                 lr, gamma, epsilon, epsilon_min, epsilon_decay, sync_freq, env):
        super().__init__()
        self.env = env  # Save environment
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_layers = hidden_layers
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.sync_freq = sync_freq

        # Initialize networks
        self.q_network = DQNNetwork(self.state_dim, self.action_dim, self.hidden_layers)
        self.target_network = DQNNetwork(self.state_dim, self.action_dim, self.hidden_layers)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(self.buffer_size)

        # Optimizer and loss function
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

        # Metrics
        self.total_rewards = []
        self.losses = []

    def forward(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        return self.q_network(state_tensor)

    def sample_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                return torch.argmax(self.forward(state)).item()

    def training_step(self, batch, batch_idx):
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Compute Q-values and targets
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        # Compute loss
        loss = self.loss_fn(q_values, target_q_values)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return self.optimizer

    def train_dataloader(self):
        dataset = DQNDataset(self.replay_buffer, self.batch_size)
        return DataLoader(dataset, batch_size=self.batch_size)

    def on_train_epoch_end(self):
        # Sync target network
        if self.current_epoch % self.sync_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def test_agent(self, env, num_episodes=10):
        total_rewards = []
        for _ in range(num_episodes):
            state = env.reset()
            total_reward = 0
            done = False
            while not done:
                action = self.sample_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
            total_rewards.append(total_reward)
        return sum(total_rewards) / len(total_rewards)
