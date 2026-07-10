from __future__ import annotations

import numpy as np
import torch

from oncosentinel.execute.calibration_head import CalibrationHead
from oncosentinel.execute.calibration_trainer import train_calibration
from oncosentinel.execute.pipeline import Orchestrator
from oncosentinel.knowledge.cohort_registry import CANDIDATE_REGIMENS
from oncosentinel.knowledge.cohort_synthesis import generate_cohort
from oncosentinel.knowledge.settings_loader import default_configs_dir, load_config
from oncosentinel.monitor.adaptive import Biomarker


def test_two_step_calibration_decreases_loss() -> None:
    config = load_config("_smoke", default_configs_dir())
    cohort = config.cohorts[0]
    cases = generate_cohort(cohort, limit=24)
    features = torch.tensor(np.stack([c.features.omics for c in cases]), dtype=torch.float32)
    labels = torch.tensor(
        np.asarray([c.response_label for c in cases], dtype=np.float32), dtype=torch.float32
    )
    head = CalibrationHead(in_features=features.shape[1], rank=config.calibration.lora_rank)
    history = train_calibration(head, features, labels, config.calibration, seed=0)
    assert len(history) == config.calibration.steps == 2
    assert all(np.isfinite(history))
    assert history[-1] < history[0]


def test_end_to_end_orchestration_with_monitor() -> None:
    config = load_config("_smoke", default_configs_dir())
    orchestrator = Orchestrator.from_config(config)
    cohort = config.cohorts[0]
    case = generate_cohort(cohort, limit=1)[0].features
    stream = [
        Biomarker("TROPONIN", 0.42),
        Biomarker("RECIST", 1.0),
        Biomarker("ECHO", 0.30),
    ]
    result = orchestrator.run(case, list(CANDIDATE_REGIMENS), stream)
    assert result.recommendation.regimen
    assert result.evidence.verify()
    assert len(result.evidence.records) > 0
    assert len(result.kl_values) == len(stream)
    assert all(value >= 0.0 for value in result.kl_values)
