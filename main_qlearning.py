import numpy as np
import matplotlib.pyplot as plt
from environment import initialize_frozenlake
from qlearning_agent import QLearningAgent
from utils import plot_q_table_values, plot_q_values_map, plot_average_rewards, plot_average_steps
from configs.frozenlake_config import hyperparameters

# Load FrozenLake hyperparameters
alpha = hyperparameters["alpha"]
gamma = hyperparameters["gamma"]
epsilon = hyperparameters["epsilon"]
epsilon_decay = hyperparameters["epsilon_decay"]
epsilon_min = hyperparameters["epsilon_min"]
episodes = hyperparameters["episodes"]
max_steps = hyperparameters["max_steps"]

# Initialize environment
env = initialize_frozenlake(is_slippery=False)

# Initialize Q-learning agent
agent = QLearningAgent(
    state_space=env.observation_space.n,
    action_space=env.action_space.n,
    alpha=alpha,
    gamma=gamma,
    epsilon=epsilon,
    epsilon_decay=epsilon_decay,
    epsilon_min=epsilon_min,
)

# Training variables
rewards = []
steps_to_goal = []
qtable_snapshots = {}

# Training loop
for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    steps = 0

    for step in range(max_steps):
        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Update the Q-table
        agent.update(state, action, reward, next_state, done)
        state = next_state

        total_reward += reward
        steps += 1

        if done:
            break

    # Track rewards and steps
    rewards.append(total_reward)
    steps_to_goal.append(steps if total_reward > 0 else max_steps)

    # Decay epsilon
    agent.decay_epsilon()

    # Save Q-tables at specific episodes
    if episode in [0, 500, 2000, 4999]:
        qtable_snapshots[episode] = agent.q_table.copy()
        plot_q_table_values(agent.q_table, episode)
        plot_q_values_map(agent.q_table, env, int(np.sqrt(env.observation_space.n)), f"Q-Values and Policy at Episode {episode}")

# Plot training metrics
plot_average_rewards(rewards, interval=100)
plot_average_steps(steps_to_goal, interval=100)

# Test the agent
success_rate = agent.test_agent(env, max_steps, num_episodes=100)
print(f"Success rate over 100 test episodes: {success_rate:.2f}%")
env.close()
