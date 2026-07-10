from __future__ import annotations

import torch
from torch import Tensor, nn


class CalibrationHead(nn.Module):
    def __init__(self, in_features: int, rank: int = 8) -> None:
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, 1, bias=True)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, features: Tensor) -> Tensor:
        hidden = self.down(features)
        logit: Tensor = self.up(hidden).squeeze(-1)
        return logit

    def bias_for(self, features: Tensor) -> float:
        with torch.no_grad():
            return float(self.forward(features).mean().item())
