import gymnasium as gym
import numpy as np


class RewardWrapper(gym.RewardWrapper):
    """
    Custom reward shaping for CartPole-v1.

    Goal of reward shaping:
    ----------------------
    Improve learning speed by giving more informative feedback
    instead of the default sparse reward:
        reward = +1 for every step before failure

    This shaping encourages:
      ✓ Keeping pole upright
      ✓ Keeping cart near center
      ✓ Low angular velocity
      ✓ Quick stabilization
      ✓ Large bonus near perfect balance

    IMPORTANT:
      - This modifies the original environment reward function!
      - Agent may learn shaping objective rather than true task objective.
      - Works ONLY with normalized observations [0,1].
    """

    def __init__(self, env):
        super().__init__(env)

        # Extract and clamp infinite observation bounds
        # Original CartPole:
        # [-4.8, -inf, -0.418, -inf] to [4.8, inf, 0.418, inf]
        low  = np.where(np.isinf(env.observation_space.low),  -5.0, env.observation_space.low)
        high = np.where(np.isinf(env.observation_space.high),  5.0, env.observation_space.high)

        # Store real-world bounds for later denormalization
        self.x_min, self.xdot_min, self.th_min, self.thdot_min = low
        self.x_max, self.xdot_max, self.th_max, self.thdot_max = high

    # ----------------------------------------------------
    # Normalize → Real-world unit conversion
    # ----------------------------------------------------
    def _denormalize(self, s):
        """
        Convert normalized inputs [0,1] back to their physical ranges.
        Required because shaping uses physics knowledge.
        """
        x      = self.x_min    + s[0] * (self.x_max    - self.x_min)
        x_dot  = self.xdot_min + s[1] * (self.xdot_max - self.xdot_min)
        theta  = self.th_min   + s[2] * (self.th_max   - self.th_min)
        th_dot = self.thdot_min+ s[3] * (self.thdot_max- self.thdot_min)
        return x, x_dot, theta, th_dot

    # ----------------------------------------------------
    # Custom reward shaping function
    # ----------------------------------------------------
    def reward(self, state):
        # Convert normalized state into real-world units
        x, x_dot, theta, th_dot = self._denormalize(state)

        # 1️⃣ Pole upright reward (main objective)
        upright_reward = 1.0 - abs(theta) / 0.2095  # 12 degrees limit
        upright_reward = max(upright_reward, 0.0)
        upright_reward *= 2.0  # stronger weight

        # 2️⃣ Centering reward (cart position)
        center_reward = 1.0 - abs(x) / 2.4
        center_reward = max(center_reward, 0.0)
        center_reward *= 0.5

        # 3️⃣ Stability: small angular velocity rewarded
        stability_reward = np.exp(-abs(th_dot) * 2.0) * 0.5

        # 4️⃣ Time penalty to encourage faster control
        step_penalty = -0.05

        # 5️⃣ Big bonus for near-perfect control
        bonus = 0.0
        if abs(theta) < 0.03 and abs(x) < 0.2 and abs(th_dot) < 0.2:
            bonus = 1.0

        return upright_reward + center_reward + stability_reward + bonus + step_penalty
