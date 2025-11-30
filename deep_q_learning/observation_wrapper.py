import gymnasium as gym
import numpy as np


class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        # Original bounds (CartPole has inf bounds)
        low = env.observation_space.low
        high = env.observation_space.high

        # Replace +inf / -inf with finite numbers
        # These ranges are commonly used in practice
        low = np.where(np.isinf(low), -5.0, low)
        high = np.where(np.isinf(high),  5.0, high)

        self.min_value = low
        self.max_value = high

        # Define new normalized observation space [0,1]
        self.observation_space = gym.spaces.Box(
            low=np.zeros_like(low),
            high=np.ones_like(high),
            dtype=np.float32
        )

    def observation(self, state):
        state = np.clip(state, self.min_value, self.max_value)
        normalized = (state - self.min_value) / (self.max_value - self.min_value)
        return normalized.astype(np.float32)
