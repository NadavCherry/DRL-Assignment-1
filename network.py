import torch.nn as nn

class DQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layers):
        super(DQNNetwork, self).__init__()
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, action_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
