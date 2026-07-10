from __future__ import annotations

import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim.lr_scheduler import LambdaLR

from oncosentinel.execute.calibration_head import CalibrationHead
from oncosentinel.knowledge.settings_schema import CalibrationConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _schedule(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return float(0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))


def train_calibration(
    head: CalibrationHead,
    features: Tensor,
    labels: Tensor,
    config: CalibrationConfig,
    seed: int = 0,
) -> list[float]:
    set_seed(seed)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = LambdaLR(
        optimizer, lr_lambda=lambda step: _schedule(step, config.warmup_steps, config.steps)
    )
    loss_fn = nn.BCEWithLogitsLoss()
    history: list[float] = []
    head.train()
    for _ in range(config.steps):
        optimizer.zero_grad()
        logits = head(features)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        history.append(float(loss.item()))
    return history


def save_checkpoint(head: CalibrationHead, seed: int, path: Path) -> None:
    payload = {"state_dict": head.state_dict(), "seed": seed}
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(head: CalibrationHead, path: Path) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    head.load_state_dict(payload["state_dict"])
    seed = int(payload["seed"])
    set_seed(seed)
    return seed
