import torch, torch.nn as nn, torch.nn.functional as F

class PPONet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, act_dim=6):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # policy head
        self.pi  = nn.Linear(hidden_dim, act_dim)
        # value head
        self.v   = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.pi(x)
        value  = self.v(x).squeeze(-1)
        return logits, value
