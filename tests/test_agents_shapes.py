from __future__ import annotations

import math

from oncosentinel.analyze.cardiotox import CardiotoxicityRiskAgent
from oncosentinel.analyze.immune import ImmuneContextureAgent
from oncosentinel.analyze.omics import OmicsReasoningAgent
from oncosentinel.knowledge.cohort_registry import DEFAULT_ANCHORS
from oncosentinel.knowledge.cohort_synthesis import generate_cohort


def _case() -> object:
    cohort = next(c for c in DEFAULT_ANCHORS if c.name == "TCGA")
    return generate_cohort(cohort, limit=1)[0].features


def test_omics_posterior_in_unit_interval() -> None:
    case = _case()
    posterior = OmicsReasoningAgent.default().infer(case)
    assert 0.0 < posterior.mean < 1.0
    assert posterior.variance > 0.0


def test_immune_fractions_sum_to_one_and_embedding_shape() -> None:
    case = _case()
    contexture = ImmuneContextureAgent().infer(case)
    assert math.isclose(sum(contexture.til_fractions.values()), 1.0, abs_tol=1e-9)
    assert contexture.phenotype in {"hot", "cold"}
    assert contexture.embedding.shape == (4,)


def test_cardiac_posterior_range_and_determinism() -> None:
    case = _case()
    agent = CardiotoxicityRiskAgent.default()
    first = agent.infer(case)
    second = agent.infer(case)
    assert 0.0 <= first.posterior.stable_probability <= 1.0
    assert math.isclose(first.posterior.mean, second.posterior.mean, abs_tol=1e-12)
    assert first.contraindicated == second.contraindicated
