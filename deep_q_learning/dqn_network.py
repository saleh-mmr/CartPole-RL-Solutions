from torch import nn


class DQNNetwork(nn.Module):

    def __init__(self, num_actions, input_dim):
        super(DQNNetwork, self).__init__()
        self.FC = nn.Sequential(
            nn.Linear(input_dim,64),
            nn.ReLU(inplace=True),
            nn.Linear(64,64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_actions)
        )

        # Initialize the FC layer weights using He Initialization
        for module in self.FC:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                nn.init.zeros_(module.bias)


    def forward(self, x):
        Q = self.FC(x)
        return Q

