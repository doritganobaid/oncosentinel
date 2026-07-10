from __future__ import annotations

import math

import numpy as np

from oncosentinel.plan.hypervolume import (
    hypervolume,
    hypervolume_contribution,
    non_dominated_front,
)
from oncosentinel.plan.pareto import recommend


def test_single_point_hypervolume_is_rectangle() -> None:
    assert math.isclose(hypervolume([(3.0, 1.0)], (0.0, 0.0)), 3.0)


def test_two_point_hypervolume_golden_value() -> None:
    assert math.isclose(hypervolume([(3.0, 1.0), (1.0, 2.0)], (0.0, 0.0)), 4.0)


def test_dominated_point_does_not_change_hypervolume() -> None:
    base = [(3.0, 1.0), (1.0, 2.0)]
    with_dominated = base + [(0.5, 0.5)]
    assert math.isclose(hypervolume(base, (0.0, 0.0)), hypervolume(with_dominated, (0.0, 0.0)))


def test_non_dominated_front_excludes_dominated() -> None:
    points = np.array([[3.0, 1.0], [1.0, 2.0], [0.5, 0.5], [2.0, 1.5]], dtype=np.float64)
    front = non_dominated_front(points)
    assert front.shape[0] == 3
    assert not any(np.allclose(row, [0.5, 0.5]) for row in front)


def test_dominated_point_has_zero_contribution() -> None:
    points = np.array([[3.0, 1.0], [1.0, 2.0], [0.5, 0.5]], dtype=np.float64)
    contributions = hypervolume_contribution(points, (0.0, 0.0))
    assert math.isclose(contributions[2], 0.0, abs_tol=1e-12)
    assert contributions[0] > 0.0 and contributions[1] > 0.0


def test_recommend_selects_front_member() -> None:
    points = np.array([[0.3, 0.6], [0.5, 0.4], [0.1, 0.9], [0.05, 0.05]], dtype=np.float64)
    selection = recommend(points, (0.0, 0.0))
    assert selection.argmax_index in selection.front_indices.tolist()
    assert 3 not in selection.front_indices.tolist()
    assert selection.frontier_size == 3
