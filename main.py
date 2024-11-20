import gym
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

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

# Q-learning algorithm
for episode in range(episodes):
    state = env.reset()  # Reset environment
    # env.render()  # Render the environment
    total_reward = 0     # Track total reward for the episode

    for step in range(max_steps):
        # Choose action using ε-greedy policy
        # env.render()  # Render the environment
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Random action
        else:
            action = np.argmax(q_table[state])  # Best action from Q-table

        next_state, reward, terminated, truncated, info = env.step(action)

        # Combine `terminated` and `truncated` flags for episode termination
        done = terminated or truncated

        # Update Q-value using the Q-learning update rule
        best_next_action = np.argmax(q_table[next_state])
        q_table[state, action] = q_table[state, action] + alpha * (
            reward + gamma * q_table[next_state, best_next_action] - q_table[state, action]
        )

        state = next_state  # Move to the next state
        total_reward += reward

        if done:
            break

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Print progress every 500 episodes
    if (episode + 1) % 500 == 0:
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

# Test the trained agent
success_rate = 0
test_episodes = 100
for _ in range(test_episodes):
    state = env.reset()
    for _ in range(max_steps):
        action = np.argmax(q_table[state])
        state, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            success_rate += reward
            break

print(f"Success rate over {test_episodes} episodes: {success_rate / test_episodes * 100:.2f}%")
env.close()
