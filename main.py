import gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from matplotlib import MatplotlibDeprecationWarning

warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)




# Initialize FrozenLake environment with the new step API
env = gym.make('FrozenLake-v1', new_step_api=True)

# Parameters
alpha = 0.1            # Learning rate
gamma = 0.99           # Discount factor
epsilon = 1.0          # Initial epsilon for ε-greedy
epsilon_decay = 0.995  # Decay rate for epsilon
epsilon_min = 0.01     # Minimum epsilon
episodes = 5000        # Number of episodes
max_steps = 100        # Max steps per episode

# Initialize Q-table with zeros
q_table = np.zeros((env.observation_space.n, env.action_space.n))

# Track rewards and steps for analysis
rewards = []
steps_to_goal = []

# Q-learning algorithm
for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    steps = 0

    for step in range(max_steps):
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Update Q-value
        best_next_action = np.argmax(q_table[next_state])
        q_table[state, action] = q_table[state, action] + alpha * (
            reward + gamma * q_table[next_state, best_next_action] - q_table[state, action]
        )

        state = next_state
        total_reward += reward
        steps += 1

        if done:
            break

    # Track rewards and steps
    rewards.append(total_reward)
    steps_to_goal.append(steps if reward > 0 else max_steps)

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Save Q-table snapshots
    if episode + 1 in [500, 2000, 5000]:
        plt.figure(figsize=(8, 6))
        sns.heatmap(q_table, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title(f"Q-Table at Episode {episode + 1}")
        plt.xlabel("Actions")
        plt.ylabel("States")
        plt.show()

# Plot reward per episode
plt.figure(figsize=(10, 6))
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Reward Per Episode")
plt.grid()
plt.show()

# Plot average steps to goal over last 100 episodes
avg_steps = [np.mean(steps_to_goal[i:i+100]) for i in range(0, len(steps_to_goal), 100)]
plt.figure(figsize=(10, 6))
plt.plot(range(0, len(steps_to_goal), 100), avg_steps)
plt.xlabel("Episode (x100)")
plt.ylabel("Average Steps to Goal")
plt.title("Average Steps to Goal Over Last 100 Episodes")
plt.grid()
plt.show()

# Test the trained agent
success_rate = 0
test_episodes = 100
for _ in range(test_episodes):
    state = env.reset()
    for _ in range(max_steps):
        action = np.argmax(q_table[state])
        state, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            success_rate += reward
            break

print(f"Success rate over {test_episodes} episodes: {success_rate / test_episodes * 100:.2f}%")
env.close()
