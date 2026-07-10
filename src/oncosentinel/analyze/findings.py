from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import spearmanr

from oncosentinel.plan.hypervolume import non_dominated_front


def front_is_non_convex(points: NDArray[np.float64] | list[list[float]]) -> bool:
    front = non_dominated_front(np.asarray(points, dtype=np.float64))
    if front.shape[0] < 3:
        return False
    order = np.argsort(front[:, 0])
    ordered = front[order]
    for i in range(1, ordered.shape[0] - 1):
        left = ordered[i - 1]
        mid = ordered[i]
        right = ordered[i + 1]
        span = right[0] - left[0]
        if span <= 0.0:
            continue
        weight = (mid[0] - left[0]) / span
        chord_y = left[1] + weight * (right[1] - left[1])
        if mid[1] < chord_y - 1e-9:
            return True
    return False


def non_convex_fraction(flags: list[bool]) -> float:
    if not flags:
        return 0.0
    return float(np.mean(np.asarray(flags, dtype=np.float64)))


def tool_call_scaling(
    complexity: NDArray[np.float64], tool_counts: NDArray[np.float64]
) -> tuple[float, float]:
    if (
        complexity.size < 2
        or float(np.std(complexity)) < 1e-12
        or float(np.std(tool_counts)) < 1e-12
    ):
        return 0.0, 0.0
    rho = float(spearmanr(complexity, tool_counts).correlation)
    slope = float(np.polyfit(complexity, tool_counts, 1)[0])
    return rho, slope


def complexity_index(omics_dim: int, cardiac_dim: int, cancer_type_count: int) -> float:
    return float(np.log(omics_dim + cardiac_dim + cancer_type_count))


def kl_phase_transition(taus: NDArray[np.float64], pareto_irs: NDArray[np.float64]) -> float:
    return float(taus[int(np.argmax(pareto_irs))])


def bayesian_update_dominance(switch_sources: list[str]) -> dict[str, float]:
    total = len(switch_sources)
    if total == 0:
        return {"cardio": 0.0, "efficacy": 0.0, "balanced": 0.0}
    counts = {"cardio": 0, "efficacy": 0, "balanced": 0}
    for source in switch_sources:
        counts[source] = counts.get(source, 0) + 1
    return {key: value / total for key, value in counts.items()}
