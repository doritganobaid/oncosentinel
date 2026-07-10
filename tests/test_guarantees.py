from __future__ import annotations

import math

import numpy as np

from oncosentinel.plan.guarantees import (
    generalization_bound,
    is_unique_optimum,
    pareto_bayesian_optimum,
    planner_complexity,
    regret_bound,
)


def test_generalization_bound_matches_reference_point() -> None:
    assert math.isclose(generalization_bound(0.08), 11.0, abs_tol=1e-9)
    assert generalization_bound(0.16) > generalization_bound(0.08)
    assert generalization_bound(0.0) == 0.0


def test_regret_bound_per_step_value() -> None:
    assert math.isclose(regret_bound(1), 4.7, abs_tol=1e-9)
    assert math.isclose(regret_bound(2), 9.4, abs_tol=1e-9)


def test_planner_complexity_is_n_log_n() -> None:
    assert planner_complexity(1) == 1
    assert planner_complexity(8) == math.ceil(8 * math.log2(8))


def test_unique_optimum_detection() -> None:
    assert is_unique_optimum(np.array([0.3, 0.1, 0.05]))
    assert not is_unique_optimum(np.array([0.3, 0.3, 0.05]))


def test_optimum_index_lies_on_front() -> None:
    points = np.array([[0.3, 0.6], [0.5, 0.4], [0.1, 0.9]], dtype=np.float64)
    index = pareto_bayesian_optimum(points, (0.0, 0.0))
    assert index in (0, 1, 2)
