# its so stupid that I had to do this :<
import torch
from model.ppo_agent import PPONet

class AdaptNet(torch.nn.Module):
    def __init__(self, base: PPONet):
        super().__init__()
        self.base   = base
        self.in_dim = (
            base.fc1.in_features if hasattr(base, "fc1")
            else base.input_dim   # fallback for any custom net
        )

    def forward(self, x: torch.Tensor):
        if x.numel() != self.in_dim:
            if x.numel() > self.in_dim:              # cut off extras
                x = x[:self.in_dim]
            else:                                    # zero-pad
                pad = torch.zeros(self.in_dim - x.numel(),
                                    dtype=x.dtype, device=x.device)
                x = torch.cat([x, pad])
        return self.base(x)
