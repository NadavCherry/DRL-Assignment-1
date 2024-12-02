import gym


def initialize_environment(is_slippery=True):
    """
    Initialize the FrozenLake-v1 environment with proper render mode.
    """
    env = gym.make('FrozenLake-v1', new_step_api=True, is_slippery=is_slippery, render_mode='rgb_array')
    return env
