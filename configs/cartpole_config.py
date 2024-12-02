hyperparameters = {
    "hidden_layers": [128, 128],  # Number of neurons in hidden layers
    "buffer_size": 10000,        # Replay buffer size
    "batch_size": 64,            # Batch size for training
    "lr": 1e-3,                  # Learning rate
    "gamma": 0.99,               # Discount factor
    "epsilon": 1.0,              # Initial epsilon for ε-greedy policy
    "epsilon_min": 0.01,         # Minimum epsilon value
    "epsilon_decay": 0.995,      # Decay rate for epsilon
    "sync_freq": 100,            # Frequency for syncing target network
    "episodes": 500,             # Number of training episodes
    "max_steps": 200             # Maximum steps per episode
}
