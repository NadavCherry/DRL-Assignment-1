import numpy as np

class QLearningAgent:
    def __init__(self, state_space, action_space, alpha, gamma, epsilon, epsilon_decay, epsilon_min):
        self.state_space = state_space
        self.action_space = action_space
        self.q_table = np.zeros((state_space, action_space))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, done):
        max_next_q = np.max(self.q_table[next_state])
        target = reward + (1 - done) * self.gamma * max_next_q
        self.q_table[state, action] += self.alpha * (target - self.q_table[state, action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def test_agent(self, env, max_steps, num_episodes=100):
        total_rewards = []
        for _ in range(num_episodes):
            state = env.reset()
            total_reward = 0
            for _ in range(max_steps):
                action = np.argmax(self.q_table[state])
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
            total_rewards.append(total_reward)
        return np.mean(total_rewards)
