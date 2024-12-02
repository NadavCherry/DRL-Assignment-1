import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from Q_Learning import QLearningAgent
from environment import initialize_environment
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def qtable_directions_map(qtable, map_size):
    """
    Create direction and value maps for visualization.
    """
    actions = ['←', '↓', '→', '↑']  # Left, Down, Right, Up
    qtable_val_max = np.max(qtable, axis=1).reshape(map_size, map_size)
    qtable_directions = np.array([actions[np.argmax(qtable[state])] for state in range(len(qtable))]).reshape(map_size, map_size)
    return qtable_val_max, qtable_directions


def plot_q_table_values(qtable, episode):
    """
    Plot the numerical Q-table values as a heatmap.
    """
    plt.figure(figsize=(10, 6))
    sns.heatmap(qtable, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title(f"Q-Table Values at Episode {episode}")
    plt.xlabel("Actions")
    plt.ylabel("States")
    plt.show()


def plot_q_values_map(qtable, env, map_size, title):
    """
    Plot the Q-value heatmap and the policy map.
    :param qtable: The Q-table.
    :param env: The Gym environment.
    :param map_size: The size of the map (e.g., 4 for a 4x4 FrozenLake map).
    :param title: Title for the plot.
    """
    qtable_val_max, qtable_directions = qtable_directions_map(qtable, map_size)

    # Get the last rendered frame of the environment
    last_frame = env.render()

    # Handle cases where multiple frames are returned
    if isinstance(last_frame, (list, np.ndarray)) and len(last_frame) > 1:
        last_frame = last_frame[-1]  # Use the last frame

    # Validate the frame shape
    if len(last_frame.shape) != 3:
        raise ValueError(f"Unexpected frame shape: {last_frame.shape}. Expected (height, width, 3).")

    # Plot the last frame and policy
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    ax[0].imshow(last_frame)
    ax[0].axis("off")
    ax[0].set_title("Last Frame")

    sns.heatmap(
        qtable_val_max,
        annot=qtable_directions,
        fmt="",
        ax=ax[1],
        cmap=sns.light_palette("#79C", as_cmap=True),
        linewidths=0.7,
        linecolor="black",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": "xx-large"},
    ).set(title="The Policy")

    # Format the heatmap
    for _, spine in ax[1].spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("black")

    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    plt.show()


# Hyperparameters
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.999
epsilon_min = 0.01
episodes = 5000
max_steps = 100

# Initialize environment
env = initialize_environment(is_slippery=False)

# Initialize agent
agent = QLearningAgent(
    state_space=env.observation_space.n,
    action_space=env.action_space.n,
    alpha=alpha,
    gamma=gamma,
    epsilon=epsilon,
    epsilon_decay=epsilon_decay,
    epsilon_min=epsilon_min,
)

# Track rewards and steps for analysis
rewards = []
steps_to_goal = []
qtable_snapshots = {}  # Dictionary to store Q-tables at specific episodes

# Training loop
for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    steps = 0

    for step in range(max_steps):
        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Update the agent
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

    # Save Q-table and plots at specific episodes
    if episode in [0, 500, 2000, 4999]:
        qtable_snapshots[episode] = agent.q_table.copy()
        plot_q_table_values(agent.q_table, episode)  # Plot Q-table values
        plot_q_values_map(agent.q_table, env, int(np.sqrt(env.observation_space.n)), f"Q-Values and Policy at Episode {episode}")

# Save Q-tables for later inspection
for key, qtable in qtable_snapshots.items():
    print(f"\nQ-table at Episode {key}:\n")
    print(qtable)

# Plot average reward per 100 episodes
avg_rewards = [np.mean(rewards[i:i + 100]) for i in range(0, len(rewards), 100)]
plt.figure(figsize=(10, 6))
plt.plot(range(0, len(rewards), 100), avg_rewards, label="Average Reward (per 100 episodes)")
plt.xlabel("Episode (x100)")
plt.ylabel("Average Reward")
plt.title("Average Reward Per 100 Episodes")
plt.grid()
plt.legend()
plt.show()

# Plot average steps to goal per 100 episodes
avg_steps = [np.mean(steps_to_goal[i:i + 100]) for i in range(0, len(steps_to_goal), 100)]
plt.figure(figsize=(10, 6))
plt.plot(range(0, len(steps_to_goal), 100), avg_steps, label="Average Steps to Goal (per 100 episodes)", color="orange")
plt.xlabel("Episode (x100)")
plt.ylabel("Average Steps to Goal")
plt.title("Average Steps to Goal Per 100 Episodes")
plt.grid()
plt.legend()
plt.show()

# Test the trained agent
success_rate = 0
test_episodes = 100
for _ in range(test_episodes):
    state = env.reset()
    for _ in range(max_steps):
        action = np.argmax(agent.q_table[state])
        state, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            success_rate += reward
            break

print(f"Success rate over {test_episodes} episodes: {success_rate / test_episodes * 100:.2f}%")
env.close()
