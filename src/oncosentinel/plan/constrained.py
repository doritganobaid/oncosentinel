from __future__ import annotations

import attrs
import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from oncosentinel.plan.mcid import McidAnchors
from oncosentinel.plan.pareto import ParetoSelection, recommend


@attrs.frozen
class ConstrainedSelection:
    argmax_index: int
    hv_argmax: float
    frontier_size: int
    feasible_indices: NDArray[np.int_]
    mcid_crossing: bool
    refused: bool


@attrs.frozen
class RegimenDeltas:
    orr_mean_pp: NDArray[np.float64]
    orr_std_pp: NDArray[np.float64]
    lvef_mean_pp: NDArray[np.float64]
    lvef_std_pp: NDArray[np.float64]


def _coverage_probabilities(
    deltas: RegimenDeltas, anchors: McidAnchors
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    p_orr = 1.0 - norm.cdf((anchors.orr_pp - deltas.orr_mean_pp) / deltas.orr_std_pp)
    p_lvef = norm.cdf((anchors.lvef_drop_pp - deltas.lvef_mean_pp) / deltas.lvef_std_pp)
    return np.asarray(p_orr, dtype=np.float64), np.asarray(p_lvef, dtype=np.float64)


def constrained_recommend(
    points: NDArray[np.float64],
    reference: tuple[float, float],
    deltas: RegimenDeltas,
    anchors: McidAnchors,
    coverage_delta: float,
) -> ConstrainedSelection:
    arr = np.asarray(points, dtype=np.float64)
    p_orr, p_lvef = _coverage_probabilities(deltas, anchors)
    threshold = 1.0 - coverage_delta
    feasible = (p_orr >= threshold) & (p_lvef >= threshold)
    feasible_indices = np.flatnonzero(feasible)
    refused = feasible_indices.size == 0
    candidate_indices = np.arange(arr.shape[0]) if refused else feasible_indices
    selection: ParetoSelection = recommend(arr[candidate_indices], reference)
    global_argmax = int(candidate_indices[selection.argmax_index])
    crossing = (not refused) and anchors.crossing(
        float(deltas.orr_mean_pp[global_argmax]),
        float(deltas.lvef_mean_pp[global_argmax]),
    )
    return ConstrainedSelection(
        argmax_index=global_argmax,
        hv_argmax=selection.hv_argmax,
        frontier_size=selection.frontier_size,
        feasible_indices=feasible_indices,
        mcid_crossing=crossing,
        refused=refused,
    )
