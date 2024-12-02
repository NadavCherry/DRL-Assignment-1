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

def plot_q_table_values(qtable, episode):
    plt.figure(figsize=(10, 6))
    sns.heatmap(qtable, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title(f"Q-Table Values at Episode {episode}")
    plt.xlabel("Actions")
    plt.ylabel("States")
    plt.show()

# Additional functions for plotting metrics can go here
