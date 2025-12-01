import gymnasium as gym
import numpy as np


class ObservationWrapper(gym.ObservationWrapper):
    """
    Custom wrapper to normalize CartPole observations into the range [0,1].

    Why do this?
    ------------
    In the default Gym setting, CartPole observations include:
      - Cart velocity     (unbounded → ±inf)
      - Pole angle rate   (unbounded → ±inf)

    Neural networks train more reliably when inputs are bounded
    and on similar scales. This wrapper:
      1️⃣ Clamps infinite values to finite physical limits
      2️⃣ Normalizes state features to [0,1]
    """

    def __init__(self, env):
        super().__init__(env)

        # ----------------------------
        # Extract original observation bounds
        # CartPole returns:
        # low  = [-4.8,  -inf, -0.418, -inf]
        # high = [ 4.8,  +inf,  0.418, +inf]
        # ----------------------------
        low = env.observation_space.low
        high = env.observation_space.high

        # Replace +-inf with reasonable numeric bounds.
        # This prevents division by inf and extreme scaling issues.
        low = np.where(np.isinf(low), -5.0, low)
        high = np.where(np.isinf(high),  5.0, high)

        # Save min/max for scaling new observations
        self.min_value = low
        self.max_value = high

        # Define a NEW observation space that is strictly in [0,1]
        # This maintains Gym's metadata consistency.
        self.observation_space = gym.spaces.Box(
            low=np.zeros_like(low),
            high=np.ones_like(high),
            dtype=np.float32
        )

    # ----------------------------------------------------
    # Called automatically on every env.step() and env.reset()
    # ----------------------------------------------------
    def observation(self, state):
        """
        Normalize observation:
        1️⃣ Clip to physical bounds
        2️⃣ Scale into range [0,1] using:
            normalized = (state - min) / (max - min)
        """
        state = np.clip(state, self.min_value, self.max_value)
        normalized = (state - self.min_value) / (self.max_value - self.min_value)
        return normalized.astype(np.float32)
