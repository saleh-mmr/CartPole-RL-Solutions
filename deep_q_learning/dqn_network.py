from torch import nn


class DQNNetwork(nn.Module):
    """
    Neural network used by both:
      - Main Q-network (for action selection)
      - Target Q-network (for stable target values)

    Architecture:
      Input Layer → FC1 (64 units) → ReLU →
      FC2 (64 units) → ReLU →
      Output Layer → Q-values for each action

    This is a standard MLP architecture suitable for low-dimensional
    environments like CartPole.
    """

    def __init__(self, num_actions, input_dim):
        """
        Parameters:
        ----------
        num_actions : int
            Size of the action space (number of discrete actions).
        input_dim : int
            Dimension of the observation vector (state features).
        """
        super(DQNNetwork, self).__init__()

        # ---------------------------
        # Fully Connected (FC) model
        # ---------------------------
        # Sequential layers for easier readability.
        # Activations use ReLU for good gradient flow.
        self.FC = nn.Sequential(
            nn.Linear(input_dim, 64),  # Layer 1: takes state input
            nn.ReLU(inplace=True),

            nn.Linear(64, 64),         # Layer 2: hidden layer
            nn.ReLU(inplace=True),

            nn.Linear(64, num_actions) # Output: Q-values for each action
        )

        # ---------------------------
        # Weight Initialization
        # ---------------------------
        # Kaiming (He) initialization improves training stability
        # with ReLU activations by scaling weights appropriately.
        for module in self.FC:
            if isinstance(module, nn.Linear):
                # He initialization
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                # Bias initialized to zero
                nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Forward pass through the Q-network.

        Parameters:
        ----------
        x : Tensor
            Input state(s) as a tensor [batch_size, input_dim]

        Returns:
        -------
        Q-values for each possible action [batch_size, num_actions]
        """
        Q = self.FC(x)
        return Q  # shape: [batch, num_actions]
