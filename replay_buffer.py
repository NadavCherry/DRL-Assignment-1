import random
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity

    def add(self, experience):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            raise ValueError("Not enough experiences in the buffer to sample.")
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
