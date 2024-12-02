import numpy as np


class QLearningAgent:
    def __init__(self, state_space, action_space, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01):
        """
        Initialize the Q-Learning agent.

        :param state_space: Number of states in the environment.
        :param action_space: Number of actions in the environment.
        :param alpha: Learning rate.
        :param gamma: Discount factor.
        :param epsilon: Initial epsilon for ε-greedy policy.
        :param epsilon_decay: Decay rate for epsilon after each episode.
        :param epsilon_min: Minimum epsilon for ε-greedy policy.
        """
        self.state_space = state_space
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = np.zeros((state_space, action_space))

    def choose_action(self, state):
        """
        Choose an action based on the epsilon-greedy policy.
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)  # Explore
        return np.argmax(self.q_table[state])  # Exploit

    def update(self, state, action, reward, next_state, done):
        """
        Update the Q-value based on the action taken and the resulting reward.
        """
        best_next_action = np.argmax(self.q_table[next_state])
        target = reward if done else reward + self.gamma * self.q_table[next_state, best_next_action]
        self.q_table[state, action] += self.alpha * (target - self.q_table[state, action])

    def decay_epsilon(self):
        """
        Decay epsilon to reduce exploration over time.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
