import gym
import torch
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import Trainer
# from pytorch_lightning.core.lightning import LightningModule
from torch.nn import MSELoss
from torch.optim import Adam
from dqn_agent import DQNLightning
from configs.cartpole_config import hyperparameters
from utils import plot_rewards, plot_losses

# Ensure code entry point is protected
if __name__ == "__main__":
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

    # Ensure the environment reset is handled after termination
    class DQNLightningFixed(DQNLightning):
        # Initialize lists for tracking
        total_rewards = []  # To track rewards per episode
        losses = []  # To track losses per training step

        def train_dataloader(self):
            # Creating a dummy dataset for demonstration
            train_data = TensorDataset(
                torch.rand(1000, env.observation_space.shape[0]),
                torch.randint(0, env.action_space.n, (1000,))
            )
            train_loader = DataLoader(
                train_data, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True
            )
            return train_loader

        def forward(self, state):
            """ Forward pass through the Q-network """
            return self.q_network(state)


        def training_step(self, batch, batch_idx):
            state, action = batch
            q_values = self(state)
            # Your loss and optimization logic
            # For example:
            target = torch.rand_like(q_values)  # Dummy target for illustration
            loss = MSELoss()(q_values, target)

            # Log the loss
            self.losses.append(loss.item())  # Store loss for plotting

            self.log("train_loss", loss)
            return loss

        def test_agent(self, env, num_episodes=100):
            """ Test the trained agent """
            total_rewards = []
            for episode in range(num_episodes):
                reset_output = env.reset()
                state, _ = reset_output if isinstance(reset_output, tuple) else (reset_output, {})

                done = False
                total_reward = 0
                while not done:
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    action = self(state_tensor).argmax().item()
                    step_output = env.step(action)
                    next_state, reward, terminated, truncated, _ = step_output
                    done = terminated or truncated
                    total_reward += reward
                    state = next_state
                total_rewards.append(total_reward)
            self.total_rewards = total_rewards  # Store total rewards for plotting
            return sum(total_rewards) / len(total_rewards)

        # def test_agent(self, env, num_episodes=100):
        #     """ Test the trained agent """
        #     total_rewards = []
        #     for episode in range(num_episodes):
        #         # Reset environment and get initial state
        #         reset_output = env.reset()
        #         if isinstance(reset_output, tuple):
        #             state, _ = reset_output  # Unpack state and additional info
        #         else:
        #             state = reset_output  # For older Gym versions
        #
        #         if state is None or len(state) != env.observation_space.shape[0]:
        #             raise ValueError(f"Invalid state returned by env.reset(): {state}")
        #
        #         done = False
        #         total_reward = 0
        #         while not done:
        #             # Ensure state is valid
        #             if state is None or len(state) != env.observation_space.shape[0]:
        #                 raise ValueError(f"Invalid state encountered during episode: {state}")
        #
        #             state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        #             action = self(state_tensor).argmax().item()
        #
        #             # Perform environment step
        #             step_output = env.step(action)
        #             if isinstance(step_output, tuple) and len(step_output) == 5:
        #                 next_state, reward, terminated, truncated, _ = step_output
        #             else:
        #                 raise ValueError(f"Unexpected output from env.step(): {step_output}")
        #
        #             done = terminated or truncated
        #             total_reward += reward
        #
        #             if done:
        #                 break
        #
        #             state = next_state
        #
        #         total_rewards.append(total_reward)
        #
        #     # Calculate average reward
        #     avg_reward = sum(total_rewards) / len(total_rewards)
        #     return avg_reward

    # Initialize DQN agent
    agent = DQNLightningFixed(
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
        env=env
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
    avg_reward = agent.test_agent(env, num_episodes=1000)
    print(f"Average reward over 100 test episodes: {avg_reward:.2f}")

    # Plot metrics
    plot_rewards(agent.total_rewards)
    plot_losses(agent.losses)
