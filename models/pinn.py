import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# per-sample MLP
class BreathMLP(nn.Module):
    def __init__(self, hidden: int = 128, n_layers: int = 4) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(4, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers.append(nn.Linear(hidden, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, t_norm: torch.Tensor, class_idx: int) -> torch.Tensor:
        t = t_norm.view(-1, 1)
        oh = torch.zeros(t.shape[0], 3, device=t.device)
        oh[:, class_idx] = 1.0
        x = torch.cat([t, oh], dim=1)
        return self.net(x)
    

