import numpy as np
import torch
from torch import nn, optim

from config import seed, device
from dqn_network import DQNNetwork
from replay_memory import ReplayMemory



class DQNAgent:
    """
    DQN Agent Class
    """
    def __init__(self, env, epsilon_max, epsilon_min, epsilon_decay,
                 clip_grad_norm, learning_rate, discount, memory_capacity):

        # Loss logging
        self.loss_history = []
        self.running_loss = 0
        self.learned_counts = 0

        # RL hyperparameters
        self.epsilon_max = epsilon_max    # acts as current epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.discount = discount

        self.action_space = env.action_space
        self.action_space.seed(seed)

        self.observation_space = env.observation_space
        self.replay_memory = ReplayMemory(capacity=memory_capacity)

        # Networks
        input_dim = self.observation_space.shape[0]
        output_dim = self.action_space.n

        self.main_network = DQNNetwork(num_actions=output_dim, input_dim=input_dim).to(device)
        self.target_network = DQNNetwork(num_actions=output_dim, input_dim=input_dim).to(device).eval()
        self.target_network.load_state_dict(self.main_network.state_dict())

        self.clip_grad_norm = clip_grad_norm
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=learning_rate)


    def select_action(self, state):
        """
        Epsilon-greedy action selection.
        """
        # Exploration
        if np.random.random() < self.epsilon_max:
            return self.action_space.sample()

        # Exploitation
        if not torch.is_tensor(state):
            state = torch.as_tensor(state, dtype=torch.float32, device=device)

        with torch.no_grad():
            q_values = self.main_network(state)
            return torch.argmax(q_values).item()


    def learn(self, batch_size, episode_done):
        """
        Train using a sampled batch from replay memory.
        """
        states, actions, next_states, rewards, dones, truncs = self.replay_memory.sample(batch_size)

        # Prepare shapes
        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)
        truncs = truncs.unsqueeze(1)

        # Q(s, a)
        predicted_q = self.main_network(states).gather(1, actions)

        # ---------- DOUBLE DQN ----------
        with torch.no_grad():
            # Action selection by main network
            next_actions = self.main_network(next_states).argmax(dim=1, keepdim=True)

            # Action evaluation by target network
            next_target_q = self.target_network(next_states).gather(1, next_actions)

            # True terminal states only
            true_terminal_mask = dones  # truncation does NOT terminate value
            next_target_q[true_terminal_mask] = 0.0

        # Bellman target
        targets = rewards + self.discount * next_target_q

        # Loss
        loss = self.criterion(predicted_q, targets)

        # Log loss
        self.running_loss += loss.item()
        self.learned_counts += 1

        if episode_done:
            episode_loss = self.running_loss / self.learned_counts
            self.loss_history.append(episode_loss)
            self.running_loss = 0
            self.learned_counts = 0

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), self.clip_grad_norm)
        self.optimizer.step()


    def hard_update(self):
        """
        Copy main network weights to target network.
        """
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.target_network.eval()


    def update_epsilon(self):
        """
        Decay epsilon.
        """
        self.epsilon_max = max(self.epsilon_min, self.epsilon_max * self.epsilon_decay)


    def save(self, path):
        """
        Save main network weights.
        """
        torch.save(self.main_network.state_dict(), path)
