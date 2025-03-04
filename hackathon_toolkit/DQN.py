import torch
import torch.nn as nn

class DQN(nn.Module):
    """
    Deep Q Network to model the Q function.
    """
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        #assume 4*7 in dim output
        return x

class MixingNetwork(nn.Module):
    def __init__(self, num_agents, hidden_size=32):
        super(MixingNetwork, self).__init__()
        self.hyper_w1 = nn.Linear(num_agents, num_agents * hidden_size, bias=True)
        self.hyper_w2 = nn.Linear(num_agents * hidden_size, 1, bias=True)

    def forward(self, q_values):
        w1 = torch.relu(self.hyper_w1(q_values))
        w2 = self.hyper_w2(w1)
        return w2