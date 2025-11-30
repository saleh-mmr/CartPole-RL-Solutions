import gymnasium as gym
import numpy as np


class RewardWrapper(gym.RewardWrapper):
    """
    Reward shaping specifically for CartPole-v1.
    - Strong reward for keeping the pole upright
    - Reward for keeping the cart near center
    - Gentle reward for reducing angular velocity
    - Small step penalty to encourage quick stabilization
    - Smooth terminal bonus for very good balance
    - Works with normalized observations [0,1]
    """

    def __init__(self, env):
        super().__init__(env)

        # Extract original (real) bounds, clamp infinities
        low  = np.where(np.isinf(env.observation_space.low),  -5.0, env.observation_space.low)
        high = np.where(np.isinf(env.observation_space.high),  5.0, env.observation_space.high)

        self.x_min, self.xdot_min, self.th_min, self.thdot_min = low
        self.x_max, self.xdot_max, self.th_max, self.thdot_max = high


    # ----------------------------------------------------
    # Convert normalized state [0,1] -> real coordinates
    # ----------------------------------------------------
    def _denormalize(self, s):
        x      = self.x_min    + s[0] * (self.x_max    - self.x_min)
        x_dot  = self.xdot_min + s[1] * (self.xdot_max - self.xdot_min)
        theta  = self.th_min   + s[2] * (self.th_max   - self.th_min)
        th_dot = self.thdot_min+ s[3] * (self.thdot_max- self.thdot_min)
        return x, x_dot, theta, th_dot


    # ----------------------------------------------------
    # Shaping reward specifically crafted for CartPole
    # ----------------------------------------------------
    def reward(self, state):
        x, x_dot, theta, th_dot = self._denormalize(state)

        # 1. Strong upright reward (main task)
        upright_reward = 1.0 - abs(theta) / 0.2095     # 12 degrees
        upright_reward = max(upright_reward, 0.0)
        upright_reward *= 2.0

        # 2. Centering reward (keeps cart from drifting)
        center_reward = 1.0 - abs(x) / 2.4
        center_reward = max(center_reward, 0.0)
        center_reward *= 0.5

        # 3. Stability reward (low angular velocity)
        stability_reward = np.exp(-abs(th_dot) * 2.0) * 0.5

        # 4. Small step penalty (prevents idle wobbling)
        step_penalty = -0.05

        # 5. Bonus when everything is very stable
        bonus = 0.0
        if abs(theta) < 0.03 and abs(x) < 0.2 and abs(th_dot) < 0.2:
            bonus = 1.0

        return upright_reward + center_reward + stability_reward + bonus + step_penalty
