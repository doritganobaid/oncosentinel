from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import friedmanchisquare, kruskal, norm


def _midrank(values: NDArray[np.float64]) -> NDArray[np.float64]:
    order = np.argsort(values)
    sorted_values = values[order]
    n = values.shape[0]
    ranks = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    out = np.empty(n, dtype=np.float64)
    out[order] = ranks
    return out


def _fast_delong(
    predictions: NDArray[np.float64], positives: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    m = positives
    n = predictions.shape[1] - m
    k = predictions.shape[0]
    positive = predictions[:, :m]
    negative = predictions[:, m:]
    tx = np.empty((k, m), dtype=np.float64)
    ty = np.empty((k, n), dtype=np.float64)
    tz = np.empty((k, m + n), dtype=np.float64)
    for r in range(k):
        tx[r] = _midrank(positive[r])
        ty[r] = _midrank(negative[r])
        tz[r] = _midrank(predictions[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    return aucs, np.atleast_2d(cov)


def delong_test(
    scores_a: NDArray[np.float64], scores_b: NDArray[np.float64], labels: NDArray[np.int_]
) -> tuple[float, float]:
    order = np.argsort(-labels.astype(np.int64))
    positives = int(np.sum(labels == 1))
    predictions = np.vstack([scores_a, scores_b])[:, order]
    aucs, cov = _fast_delong(predictions, positives)
    variance = cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1]
    if variance <= 0.0:
        return 0.0, 1.0
    z = float((aucs[0] - aucs[1]) / np.sqrt(variance))
    p = float(2.0 * (1.0 - norm.cdf(abs(z))))
    return z, p


def bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    threshold = alpha / max(1, len(p_values))
    return [p < threshold for p in p_values]


def friedman_test(per_method_ranks: list[NDArray[np.float64]]) -> tuple[float, float]:
    statistic, p_value = friedmanchisquare(*per_method_ranks)
    return float(statistic), float(p_value)


def kruskal_test(groups: list[NDArray[np.float64]]) -> tuple[float, float]:
    statistic, p_value = kruskal(*groups)
    return float(statistic), float(p_value)


def nemenyi_critical_difference(num_methods: int, num_blocks: int, q_alpha: float) -> float:
    return float(q_alpha * np.sqrt(num_methods * (num_methods + 1) / (6.0 * num_blocks)))
