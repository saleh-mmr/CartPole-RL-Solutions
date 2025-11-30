from collections import deque
import numpy as np
import torch
from config import device


class ReplayMemory:
    def __init__(self, capacity):
        """
        Experience Replay Memory using deques to store transitions.
        """
        self.capacity = capacity
        self.states = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.next_states = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)
        self.dones = deque(maxlen=capacity)
        self.truncs = deque(maxlen=capacity)  # store truncation flags separately

    def store(self, state, action, next_state, reward, done, trunc):
        """
        Append a transition to memory.
        """
        self.states.append(state)
        self.actions.append(action)
        self.next_states.append(next_state)
        self.rewards.append(reward)
        self.dones.append(done)
        self.truncs.append(trunc)

    def sample(self, batch_size):
        """
        Randomly sample transitions from memory and return them as tensors on device.
        """
        indices = np.random.choice(len(self), size=batch_size, replace=False)

        # Convert lists of numpy arrays to a single tensor
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

        dones = torch.as_tensor(
            [self.dones[i] for i in indices],
            dtype=torch.bool,
            device=device
        )

        truncs = torch.as_tensor(
            [self.truncs[i] for i in indices],
            dtype=torch.bool,
            device=device
        )

        return states, actions, next_states, rewards, dones, truncs

    def __len__(self):
        """
        Number of stored transitions.
        """
        return len(self.dones)
