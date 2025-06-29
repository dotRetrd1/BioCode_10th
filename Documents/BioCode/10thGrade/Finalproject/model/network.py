import torch
import torch.nn as nn
import torch.nn.functional as F


class PreyPolicyNet(nn.Module):
    """
       3x3 view  (27 inputs)  - PreyEnv / PredatorEnv
       5x5 view  (200 inputs) - JointEnv
    """
    def __init__(self, input_dim: int = 27,
                 hidden_dim: int = 64,
                 output_dim: int = 6):
        super().__init__()
        self.fc1    = nn.Linear(input_dim, hidden_dim)
        self.fc2    = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.output(x)          # logits (softmax applied by agent code)
