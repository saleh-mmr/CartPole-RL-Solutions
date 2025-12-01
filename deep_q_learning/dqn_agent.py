import numpy as np
import torch
from torch import nn, optim

from config import seed, device
from dqn_network import DQNNetwork
from replay_memory import ReplayMemory


class DQNAgent:
    """
    DQN Agent implementing Double DQN with a soft-updated target network.
    """

    def __init__(
        self,
        env,
        epsilon_max,
        epsilon_min,
        epsilon_decay,
        clip_grad_norm,
        learning_rate,
        discount,
        memory_capacity,
        tau,  # NEW: soft update rate for target network
    ):
        # -----------------------
        # Logging-related fields
        # -----------------------
        # Stores average loss per episode for plotting and analysis.
        self.loss_history = []
        # Running sum of loss within a single episode (for averaging later).
        self.running_loss = 0
        # Number of gradient steps taken within the current episode.
        self.learned_counts = 0

        # -----------------------
        # RL Hyperparameters
        # -----------------------
        # NOTE: epsilon_max here is used as the *current* epsilon value.
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay  # multiplicative decay factor per episode
        self.discount = discount            # discount factor γ
        self.tau = tau                      # NEW: Polyak averaging coefficient for soft updates

        # -----------------------
        # Environment spaces
        # -----------------------
        self.action_space = env.action_space
        # Seed the action space for reproducible random actions.
        self.action_space.seed(seed)

        self.observation_space = env.observation_space

        # Replay buffer to store transitions (s, a, r, s', done, trunc).
        self.replay_memory = ReplayMemory(capacity=memory_capacity)

        # -----------------------
        # Q-Networks (Main & Target)
        # -----------------------
        # Input dimension = size of state vector, e.g. 4 for CartPole.
        input_dim = self.observation_space.shape[0]
        # Output dimension = number of discrete actions.
        output_dim = self.action_space.n

        # Main network: used for action selection & learning.
        self.main_network = DQNNetwork(
            num_actions=output_dim,
            input_dim=input_dim
        ).to(device)

        # Target network: used to compute stable target Q-values.
        # eval() ensures layers like dropout/batchnorm (if any) are in eval mode.
        self.target_network = DQNNetwork(
            num_actions=output_dim,
            input_dim=input_dim
        ).to(device).eval()

        # Initialize target network with the same weights as the main network.
        self.target_network.load_state_dict(self.main_network.state_dict())

        # -----------------------
        # Optimization setup
        # -----------------------
        self.clip_grad_norm = clip_grad_norm          # gradient clipping threshold
        self.criterion = nn.SmoothL1Loss()            # Huber loss (standard for DQN)
        self.optimizer = optim.Adam(
            self.main_network.parameters(),
            lr=learning_rate
        )

    # ---------------------------------------------------------------------
    # ACTION SELECTION
    # ---------------------------------------------------------------------
    def select_action(self, state):
        """
        Epsilon-greedy action selection.

        With probability epsilon (epsilon_max), take a random action (exploration).
        Otherwise, choose the action with the highest predicted Q-value (exploitation).
        """
        # -----------------------
        # Exploration
        # -----------------------
        if np.random.random() < self.epsilon_max:
            # Random action sampled from environment action space.
            return self.action_space.sample()

        # -----------------------
        # Exploitation
        # -----------------------
        # Ensure state is a torch tensor on the correct device.
        if not torch.is_tensor(state):
            state = torch.as_tensor(state, dtype=torch.float32, device=device)

        # No gradient needed for action selection.
        with torch.no_grad():
            q_values = self.main_network(state)
            # Choose the action index with max Q-value.
            return torch.argmax(q_values).item()

    # ---------------------------------------------------------------------
    # LEARNING / TRAINING STEP
    # ---------------------------------------------------------------------
    def learn(self, batch_size, episode_done):
        """
        Perform one gradient step using a sampled batch from replay memory.

        Uses Double DQN:
        - Action selection via main_network
        - Action evaluation via target_network
        """

        # Sample a batch of transitions:
        # states, actions, next_states, rewards, dones, truncs
        states, actions, next_states, rewards, dones, truncs = self.replay_memory.sample(batch_size)

        # -----------------------
        # Prepare shapes
        # -----------------------
        # Actions: [batch] -> [batch, 1] so we can gather along action dimension.
        actions = actions.unsqueeze(1)
        # Rewards and done flags also go to shape [batch, 1].
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)
        truncs = truncs.unsqueeze(1)  # stored but not used in Bellman target here.

        # -----------------------
        # Q(s, a): Current estimates from main network
        # -----------------------
        # main_network(states) -> [batch, num_actions]
        # .gather(1, actions) -> [batch, 1] selecting Q(s,a) for the taken actions.
        predicted_q = self.main_network(states).gather(1, actions)

        # -----------------------
        # Double DQN target computation
        # -----------------------
        with torch.no_grad():
            # 1) Action selection for next state using MAIN network (argmax over actions).
            #    This avoids overestimation bias (Double DQN trick).
            next_actions = self.main_network(next_states).argmax(dim=1, keepdim=True)

            # 2) Action evaluation for next state using TARGET network.
            #    We take the Q-value corresponding to the chosen action above.
            next_target_q = self.target_network(next_states).gather(1, next_actions)

            # 3) Properly handle terminal states:
            #    For true terminal states (done == True), future Q-value = 0.
            true_terminal_mask = dones  # truncation does NOT terminate value here.
            next_target_q[true_terminal_mask] = 0.0

        # -----------------------
        # Bellman target:
        #   target = r + γ * Q_target(s', a')
        # -----------------------
        targets = rewards + self.discount * next_target_q

        # -----------------------
        # Loss between current Q(s,a) and the target
        # -----------------------
        loss = self.criterion(predicted_q, targets)

        # -----------------------
        # Loss logging (averaged per episode)
        # -----------------------
        self.running_loss += loss.item()
        self.learned_counts += 1

        # If the episode just finished, compute average loss for that episode.
        if episode_done:
            episode_loss = self.running_loss / self.learned_counts
            self.loss_history.append(episode_loss)
            # Reset for next episode.
            self.running_loss = 0
            self.learned_counts = 0

        # -----------------------
        # Backpropagation
        # -----------------------
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients to avoid exploding gradients.
        torch.nn.utils.clip_grad_norm_(self.main_network.parameters(), self.clip_grad_norm)

        # Apply the gradient update step.
        self.optimizer.step()

    # ---------------------------------------------------------------------
    # TARGET NETWORK UPDATES
    # ---------------------------------------------------------------------
    # NOTE: The old hard update is commented out because we now use soft updates.
    #
    # def hard_update(self):
    #     """
    #     HARD UPDATE (OLD):
    #     Copy main network weights to target network instantly.
    #     This can cause instability if done too infrequently.
    #     """
    #     self.target_network.load_state_dict(self.main_network.state_dict())
    #     self.target_network.eval()

    def soft_update(self):
        """
        SOFT UPDATE (NEW):

        Polyak averaging / soft target update:
            θ_target ← τ * θ_main + (1 - τ) * θ_target

        This leads to smoother, more stable updates of the target network
        compared to a hard copy every N steps.
        """
        for target_param, policy_param in zip(
            self.target_network.parameters(),
            self.main_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
            )

    # ---------------------------------------------------------------------
    # EPSILON UPDATE
    # ---------------------------------------------------------------------
    def update_epsilon(self):
        """
        Decay epsilon (exploration rate) multiplicatively per episode.

        epsilon ← max(epsilon_min, epsilon * epsilon_decay)
        """
        self.epsilon_max = max(self.epsilon_min, self.epsilon_max * self.epsilon_decay)

    # ---------------------------------------------------------------------
    # MODEL SAVING
    # ---------------------------------------------------------------------
    def save(self, path):
        """
        Save main network weights to the given file path.
        (Target network can always be reconstructed from main network.)
        """
        torch.save(self.main_network.state_dict(), path)
