from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import beta as beta_dist
from sklearn.metrics import roc_auc_score


def auroc(scores: NDArray[np.float64], labels: NDArray[np.int_]) -> float:
    if np.unique(labels).size < 2:
        return 0.5
    return float(roc_auc_score(labels, scores))


def bootstrap_auroc_ci(
    scores: NDArray[np.float64],
    labels: NDArray[np.int_],
    iters: int = 10000,
    seed: int = 0,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = scores.shape[0]
    point = auroc(scores, labels)
    estimates = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        idx = rng.integers(0, n, size=n)
        estimates[i] = auroc(scores[idx], labels[idx])
    lower = float(np.quantile(estimates, (1.0 - confidence) / 2.0))
    upper = float(np.quantile(estimates, 1.0 - (1.0 - confidence) / 2.0))
    return point, lower, upper


def interaction_ratio(combined: float, additive: float) -> float:
    if additive <= 0.0:
        raise ValueError("additive sum-of-parts must be positive")
    return float(combined / additive)


def mcid_crossing_rate(flags: NDArray[np.bool_]) -> float:
    if flags.size == 0:
        return 0.0
    return float(np.mean(flags))


def binomial_ci(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    if total == 0:
        return (0.0, 0.0)
    alpha = 1.0 - confidence
    lower = (
        0.0
        if successes == 0
        else float(beta_dist.ppf(alpha / 2.0, successes, total - successes + 1))
    )
    upper = (
        1.0
        if successes == total
        else float(beta_dist.ppf(1.0 - alpha / 2.0, successes + 1, total - successes))
    )
    return (lower, upper)


def brier_score(probabilities: NDArray[np.float64], labels: NDArray[np.int_]) -> float:
    return float(np.mean((probabilities - labels) ** 2))


def expected_calibration_error(
    probabilities: NDArray[np.float64], labels: NDArray[np.int_], bins: int = 10
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = probabilities.shape[0]
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        mask = (probabilities > low) & (probabilities <= high)
        count = int(np.sum(mask))
        if count == 0:
            continue
        confidence = float(np.mean(probabilities[mask]))
        accuracy = float(np.mean(labels[mask]))
        error += (count / total) * abs(confidence - accuracy)
    return float(error)


def ood_flag_precision(flags: NDArray[np.bool_], truth: NDArray[np.bool_]) -> float:
    flagged = int(np.sum(flags))
    if flagged == 0:
        return 0.0
    return float(np.sum(flags & truth) / flagged)
