from __future__ import annotations

import math
from pathlib import Path

import attrs

from oncosentinel.execute.pipeline import Orchestrator
from oncosentinel.knowledge.cohort_registry import CANDIDATE_REGIMENS
from oncosentinel.knowledge.cohort_synthesis import generate_cohort
from oncosentinel.knowledge.ledger import EvidenceChain
from oncosentinel.knowledge.settings_loader import default_configs_dir, load_config

KNOWN_REGIMENS = {regimen.name for regimen in CANDIDATE_REGIMENS} | {"escalate_to_tumor_board"}


def _run(case_index: int):  # type: ignore[no-untyped-def]
    config = load_config("_smoke", default_configs_dir())
    orchestrator = Orchestrator.from_config(config)
    cohort = config.cohorts[0]
    case = generate_cohort(cohort, limit=case_index + 1)[case_index].features
    return orchestrator.run(case, list(CANDIDATE_REGIMENS))


def test_pipeline_is_deterministic() -> None:
    first = _run(0)
    second = _run(0)
    assert first.recommendation.regimen == second.recommendation.regimen
    assert first.recommendation.frontier_size == second.recommendation.frontier_size
    assert math.isclose(
        first.recommendation.hv_argmax, second.recommendation.hv_argmax, abs_tol=1e-12
    )
    assert first.recommendation.regimen in KNOWN_REGIMENS


def test_evidence_chain_round_trips_and_verifies(tmp_path: Path) -> None:
    result = _run(1)
    assert result.evidence.verify()
    assert len(result.evidence.records) > 0
    path = tmp_path / "evidence.jsonl"
    result.evidence.write(path)
    reloaded = EvidenceChain.load(path)
    assert reloaded.verify()
    assert len(reloaded.records) == len(result.evidence.records)


def test_tampered_evidence_fails_verification() -> None:
    result = _run(2)
    records = result.evidence.records
    if records:
        records[0] = attrs.evolve(records[0], output={"tampered": True})
    assert not result.evidence.verify()
