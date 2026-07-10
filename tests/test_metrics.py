from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import roc_auc_score

from oncosentinel.analyze.metrics import (
    auroc,
    binomial_ci,
    bootstrap_auroc_ci,
    interaction_ratio,
)
from oncosentinel.analyze.stats import delong_test


def test_auroc_matches_sklearn() -> None:
    scores = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9], dtype=np.float64)
    labels = np.array([0, 0, 1, 1, 0, 1], dtype=np.int_)
    assert math.isclose(auroc(scores, labels), float(roc_auc_score(labels, scores)), abs_tol=1e-12)


def test_interaction_ratio_reported_value() -> None:
    assert math.isclose(interaction_ratio(0.880, 0.779), 1.1297, abs_tol=1e-3)


def test_bootstrap_ci_brackets_point_estimate() -> None:
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, size=200).astype(np.int_)
    scores = labels + rng.normal(0.0, 0.6, size=200)
    point, lower, upper = bootstrap_auroc_ci(scores, labels, iters=500, seed=3)
    assert lower <= point <= upper


def test_binomial_ci_contains_estimate() -> None:
    lower, upper = binomial_ci(152, 1000)
    assert lower < 0.152 < upper


def test_delong_identical_scores_gives_zero_z() -> None:
    rng = np.random.default_rng(2)
    labels = rng.integers(0, 2, size=120).astype(np.int_)
    scores = labels + rng.normal(0.0, 0.7, size=120)
    z, p = delong_test(scores, scores, labels)
    assert math.isclose(z, 0.0, abs_tol=1e-9)
    assert math.isclose(p, 1.0, abs_tol=1e-9)
