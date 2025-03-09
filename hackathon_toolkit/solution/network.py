import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.fc(x)

class Critic(nn.Module):
    def __init__(self, state_size, hidden_size=32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc(x)