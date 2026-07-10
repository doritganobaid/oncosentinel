from __future__ import annotations

import torch

from oncosentinel.execute.calibration_head import CalibrationHead
from oncosentinel.execute.calibration_trainer import set_seed, train_calibration
from oncosentinel.knowledge.settings_schema import CalibrationConfig


def _overfit_config() -> CalibrationConfig:
    return CalibrationConfig(lr=5e-2, lora_rank=8, steps=800, warmup_steps=10)


def test_calibration_head_overfits_single_batch() -> None:
    set_seed(0)
    features = torch.randn(12, 16)
    labels = (torch.rand(12) > 0.5).float()
    head = CalibrationHead(in_features=16, rank=8)
    history = train_calibration(head, features, labels, _overfit_config(), seed=0)
    assert history[-1] < history[0]
    assert history[-1] < 0.1


def test_gradient_flows_to_all_parameters() -> None:
    set_seed(1)
    features = torch.randn(12, 16)
    labels = (torch.rand(12) > 0.5).float()
    head = CalibrationHead(in_features=16, rank=8)
    train_calibration(
        head, features, labels, CalibrationConfig(lr=5e-2, steps=10, warmup_steps=2), seed=1
    )
    head.zero_grad()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(head(features), labels)
    loss.backward()
    for parameter in head.parameters():
        assert parameter.grad is not None
        assert float(parameter.grad.abs().sum()) > 0.0
