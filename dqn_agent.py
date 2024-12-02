import random
import numpy as np
import torch
import torch.nn as nn
from network import DQNNetwork
from replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(self, env, hidden_layers, buffer_size, batch_size, lr, gamma, epsilon, epsilon_min, epsilon_decay, sync_freq):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.sync_freq = sync_freq

        # Neural networks
        self.q_network = DQNNetwork(self.state_dim, self.action_dim, hidden_layers)
        self.target_network = DQNNetwork(self.state_dim, self.action_dim, hidden_layers)
        self.target_network.load_state_dict(self.q_network.state_dict())  # Sync target with Q-network
        self.target_network.eval()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Optimizer and loss function
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # Metrics
        self.total_rewards = []
        self.losses = []

    def sample_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                return torch.argmax(self.q_network(state_tensor)).item()

    def train_agent(self, num_episodes):
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            done = False

            while not done:
                action = self.sample_action(state)
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward

                # Add to replay buffer
                self.replay_buffer.add((state, action, reward, next_state, done))
                state = next_state

                # Train step
                if len(self.replay_buffer) >= self.batch_size:
                    loss = self.train_step()
                    self.losses.append(loss)

                # Sync target network
                if len(self.losses) % self.sync_freq == 0:
                    self.target_network.load_state_dict(self.q_network.state_dict())

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.total_rewards.append(total_reward)

        return self.total_rewards, self.losses

    def train_step(self):
        # Sample a batch
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Compute Q-values
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

        return loss.item()

    def test_agent(self, num_episodes=10):
        total_rewards = []
        for _ in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            while not done:
                action = self.sample_action(state)
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
            total_rewards.append(total_reward)
        return np.mean(total_rewards)

    def play(self):
        state = self.env.reset()
        done = False
        while not done:
            self.env.render()
            action = self.sample_action(state)
            state, reward, done, _ = self.env.step(action)
        self.env.close()
