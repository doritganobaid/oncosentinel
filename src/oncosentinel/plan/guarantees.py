from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from oncosentinel.plan.pareto import recommend

GENERALIZATION_PP_AT_REFERENCE = 11.0
REFERENCE_TOTAL_VARIATION = 0.08
REGRET_PER_STEP_PP = 4.7


def pareto_bayesian_optimum(points: NDArray[np.float64], reference: tuple[float, float]) -> int:
    return recommend(points, reference).argmax_index


def is_unique_optimum(contributions: NDArray[np.float64], tolerance: float = 1e-9) -> bool:
    finite = contributions[np.isfinite(contributions)]
    if finite.shape[0] < 2:
        return bool(finite.shape[0] == 1)
    ordered = np.sort(finite)[::-1]
    return bool(ordered[0] - ordered[1] > tolerance)


def planner_complexity(n_regimens: int) -> int:
    if n_regimens <= 1:
        return n_regimens
    return int(math.ceil(n_regimens * math.log2(n_regimens)))


def generalization_bound(
    total_variation: float, pp_at_reference: float = GENERALIZATION_PP_AT_REFERENCE
) -> float:
    if total_variation < 0.0:
        raise ValueError("total variation distance must be non-negative")
    coefficient = pp_at_reference / REFERENCE_TOTAL_VARIATION
    return float(coefficient * total_variation)


def regret_bound(steps: int = 1, per_step_pp: float = REGRET_PER_STEP_PP) -> float:
    if steps < 0:
        raise ValueError("steps must be non-negative")
    return float(per_step_pp * steps)
