import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_rewards(rewards):
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Reward Per Episode")
    plt.grid()
    plt.show()


def plot_losses(losses):
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Loss Per Training Step")
    plt.grid()
    plt.show()


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



def plot_average_rewards(rewards, window_size=100):
    """
    Plot the average rewards over a specified window size.
    :param rewards: A list of rewards obtained during training.
    :param window_size: The window size for calculating averages (default is 100 episodes).
    """
    avg_rewards = [np.mean(rewards[i:i + window_size]) for i in range(0, len(rewards), window_size)]
    plt.figure(figsize=(10, 6))
    plt.plot(range(0, len(rewards), window_size), avg_rewards, label="Average Reward (per 100 episodes)")
    plt.xlabel(f"Episode (x{window_size})")
    plt.ylabel("Average Reward")
    plt.title("Average Reward Per 100 Episodes")
    plt.grid()
    plt.legend()
    plt.show()


def plot_average_steps(steps_to_goal, window_size=100):
    """
    Plot the average steps to goal over a specified window size.
    :param steps_to_goal: A list of steps taken to reach the goal during training.
    :param window_size: The window size for calculating averages (default is 100 episodes).
    """
    avg_steps = [np.mean(steps_to_goal[i:i + window_size]) for i in range(0, len(steps_to_goal), window_size)]
    plt.figure(figsize=(10, 6))
    plt.plot(range(0, len(steps_to_goal), window_size), avg_steps, label="Average Steps to Goal (per 100 episodes)", color="orange")
    plt.xlabel(f"Episode (x{window_size})")
    plt.ylabel("Average Steps to Goal")
    plt.title("Average Steps to Goal Per 100 Episodes")
    plt.grid()
    plt.legend()
    plt.show()
