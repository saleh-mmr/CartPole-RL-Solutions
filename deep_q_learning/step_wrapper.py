import gymnasium as gym

from observation_wrapper import ObservationWrapper
# from reward_wrapper import RewardWrapper  # ❌ Removed to return to original reward


class StepWrapper(gym.Wrapper):
    """
    Wrapper applied to each environment step.

    Purpose:
    --------
    - Modify the observation before returning it to the agent.
    - Reward shaping could be applied here (was removed for correctness).

    Currently:
    - Only observation normalization fully enabled
    - Reward remains the original CartPole reward (+1 per step)
    """

    def __init__(self, env):
        super().__init__(env)

        # Attach only the observation wrapper
        # This keeps RL training aligned with official CartPole reward structure.
        self.observation_wrapper = ObservationWrapper(env)

        # Reward wrapper removed to train using the true reward signal
        # self.reward_wrapper = RewardWrapper(env)

    # ----------------------------------------------------
    # Override the step() function of the environment
    # ----------------------------------------------------
    def step(self, action):
        """
        Take a step in the environment:

        Parameters:
        -----------
        action : int
            Action chosen by the agent.

        Returns:
        --------
        modified_state : np.array
            Normalized observation ∈ [0,1] passed to the agent.
        reward : float
            Original environment reward (shaping removed).
        done : bool
            Episode termination flag.
        truncation : bool
            Timeout flag (max steps reached).
        info : dict
            Extra environment feedback.
        """

        # Perform original CartPole step
        state, reward, done, truncation, info = self.env.step(action)

        # Normalize observation only
        modified_state = self.observation_wrapper.observation(state)

        return modified_state, reward, done, truncation, info

    # ----------------------------------------------------
    # Override reset() so the initial state is normalized
    # ----------------------------------------------------
    def reset(self, seed=None):
        """
        Reset environment and normalize the first state.
        """
        state, info = self.env.reset(seed=seed)

        # Normalize initial observation
        modified_state = self.observation_wrapper.observation(state)
        return modified_state, info
