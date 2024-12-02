import gym


def initialize_frozenlake(is_slippery=True):
    """
    Initialize the FrozenLake-v1 environment.
    """
    return gym.make("FrozenLake-v1", new_step_api=True, is_slippery=is_slippery, render_mode="rgb_array")


def initialize_cartpole():
    """
    Initialize the CartPole-v1 environment.
    """
    return gym.make("CartPole-v1")
