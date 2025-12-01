from collections import deque
import numpy as np
import torch
from config import device


class ReplayMemory:
    """
    Experience Replay buffer storing past transitions for off-policy learning.

    Why experience replay?
    ----------------------
    - Breaks temporal correlations in observed data
    - Improves sample efficiency by reusing past experiences
    - Stabilizes training compared to learning only from latest transition

    Stored transition format:
        (state, action, next_state, reward, done, truncated)

    Implemented using separate deques for faster append/pop operations.
    """

    def __init__(self, capacity):
        """
        Parameters:
        -----------
        capacity : int
            Maximum buffer size. Old samples are removed automatically
            once capacity is exceeded.
        """
        self.capacity = capacity

        # Each component of the transition is stored separately
        # for efficient sampling and tensor conversion.
        self.states = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.next_states = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.dones = deque(maxlen=capacity)  # TRUE = terminal episode end
        self.truncs = deque(maxlen=capacity)  # TRUE = timeout termination

    def store(self, state, action, next_state, reward, done, trunc):
        """
        Store a new transition into the replay buffer.

        FIFO behavior: if full, oldest transitions are automatically discarded.
        """
        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.dones.append(done)
        self.truncs.append(trunc)

    def sample(self, batch_size):
        """
        Randomly sample a batch of transitions for training.

        Returns tensors directly on the correct device (CPU/GPU).
        """
        # Generate random unique indices
        indices = np.random.choice(len(self), size=batch_size, replace=False)

        # Convert selected transitions into batched torch tensors

        states = torch.as_tensor(
            np.array([self.states[i] for i in indices]),
            dtype=torch.float32,
            device=device
        )

        next_states = torch.as_tensor(
            np.array([self.next_states[i] for i in indices]),
            dtype=torch.float32,
            device=device
        )

        actions = torch.as_tensor(
            [self.actions[i] for i in indices],
            dtype=torch.long,
            device=device
        )

        rewards = torch.as_tensor(
            [self.rewards[i] for i in indices],
            dtype=torch.float32,
            device=device
        )

        # Done flags are boolean tensors
        dones = torch.as_tensor(
            [self.dones[i] for i in indices],
            dtype=torch.bool,
            device=device
        )

        # Truncation flags are kept separately
        truncs = torch.as_tensor(
            [self.truncs[i] for i in indices],
            dtype=torch.bool,
            device=device
        )

        return states, actions, next_states, rewards, dones, truncs

    def __len__(self):
        """
        Current size of memory (number of stored transitions).
        """
        return len(self.dones)
